from typing import Tuple, List
from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from torch.utils.data import DataLoader

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset
from models.utils.recon_model import ReconModel
from utils.logger import logger
from torch.functional import F
import wandb
from models.rec_vae import RecVAE
from models.rec_vae_er import RecVAEER
from continual.buffer import Buffer
from utils.spectral_analysis import laplacian_analysis

# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae --kl_weight 1 --batch_size 64 --n_epochs 30 --latent_space 32 --approach joint
class RecVAEConER(RecVAEER):
    NAME = 'rec-vae-coner'
    VARIATIONAL = True
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        RecVAE.add_model_args(parser)
        Buffer.add_args(parser)
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        parser.add_argument('--fmap_dim', type=int, default=20,
                            help='Number of eigenvectors to take to build functional maps.')
        parser.add_argument('--er_mode', action='store_true',
                            help='Normal ER activated.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super().__init__(args=args, dataset=dataset)

        self.spectral_buffer = Buffer(self.args, self.config.device)
        self.buffer_evectors = []

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        if self.spectral_buffer.num_seen_examples < self.spectral_buffer.buffer_size:
            self.spectral_buffer.add_data(examples=x, labels=y)

        if not self.args.er_mode or self.joint:
            self.buffer.empty()

        return super().train_on_batch(x, y, task)

    def validate(self):
        if not self.spectral_buffer.is_full():
            return
        self.net_eval()
        # running mse
        rec_err = self.mini_validation()
        if self.args.wandb and self.cur_task <= 1:
            wandb.log({'reconstruction_error': rec_err})

        # running consolidation error
        if self.cur_task > 1:
            self.buffer_evectors.append(self.compute_buffer_evects())
            cerr = self.get_consolidation_error(details=False)
            self.buffer_evectors.pop()
            if self.args.wandb:
                wandb.log({'consolidation_error': cerr, 'reconstruction_error': rec_err})
        self.net_train()

    @torch.no_grad()
    def mini_validation(self):
        total = []
        X, y = self.spectral_buffer.get_all_data()
        X = X.to(self.device)
        outs = self.forward(X)
        rec_errs = self.anomaly_score(outs, X)
        total.extend(rec_errs.tolist())
        return sum(total) / len(total)

    def train_on_task(self, task_loader: DataLoader, task: int):
        self.net_train()
        for e in range(self.args.n_epochs):
            self.validate()
            # if self.args.wandb:
            #     wandb.log({"epoch": e, "lr": self.scheduler.get_last_lr()[0]})
            keep_progress = True  # if e == self.args.n_epochs - 1 else False
            progress = logger.get_tqdm(task_loader,
                                       f'TRAIN on task {task+1}/{self.dataset.n_tasks} - epoch {e+1}/{self.args.n_epochs}',
                                       leave=keep_progress)
            for x, y in progress:
                loss = self.train_on_batch(x.to(self.device), y.to(self.device), task)
                progress.set_postfix({'loss': loss})
                if self.args.wandb:
                    wandb.log({"loss": loss, "epoch": e, "lr": self.scheduler.get_last_lr()[0], "task": task})
            self.scheduler_step()

        self.buffer_evectors.append(self.compute_buffer_evects())
        if task > 1:
            cerr = self.get_consolidation_error(details=False)
            logger.log(f'Consolidation error: {cerr:.4f}')

        if task == 1:
            self.buffer_evectors.append(self.compute_buffer_evects())
            cerr = self.get_consolidation_error(details=False)
            self.buffer_evectors.pop()
            logger.log(f'Consolidation error: {cerr:.4f}')

        # if task == self.dataset.n_tasks - 1:
        #     exit()

    def train_on_dataset(self):
        logger.log(vars(self.args))
        # logger.log(self.net)
        loader = self.dataset.joint_loader if self.joint else self.dataset.task_loader
        for i, task_dl in enumerate(loader()):
            self.cur_task = i
            self.opt = self.get_opt()
            self.scheduler = self.get_scheduler()
            self.train_on_task(task_dl, i)
            self.full_log['knowledge'][str(i)] = self.dataset.last_seen_classes.copy()
            # evaluate on test
            self.test_step(self.dataset.test_loader(), i)

    @torch.no_grad()
    def compute_buffer_evects(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        self.net_eval()
        outputs, latent_mu, latent_logvar, z = self.forward(inputs)
        self.net_train()
        energy, eigenvalues, eigenvectors, L = laplacian_analysis(latent_mu, logvars=latent_logvar, norm_lap=True,
                                                                  knn=self.args.knn_laplace)
        return eigenvectors[:, :self.args.fmap_dim]

    @torch.no_grad()
    def get_consolidation_error(self, details=False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        evects = self.buffer_evectors
        n_vects = self.args.fmap_dim

        ncols = len(evects) - 1
        figsize = (6*ncols, 6)
        fig, ax = plt.subplots(1, ncols, figsize=figsize)
        plt.suptitle(f'\nKnn Norm Laplacian | {n_vects} eigenvects | {len(evects[0])} data')
        mask = torch.eye(n_vects) == 0
        c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
        c_product = torch.ones((n_vects, n_vects), device=self.device, dtype=torch.double)
        for i, ev in enumerate(evects[:-1]):
            c = ev[:, :n_vects].T @ evects[i + 1][:, :n_vects]
            if i == 0:
                c_product = c.clone()
            else:
                c_product = c_product @ c
            oode = torch.square(c[mask]).sum().item()
            sns.heatmap(c.cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[i], cbar=True if i + 1 == ncols else False)
            ax[i].set_title(f'FMap Task {i} => {i + 1} | oode={oode:.4f}')

        if details: plt.show()
        else: plt.close()

        figsize = (6 * 3, 8)
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        plt.suptitle(f'\nCompare differences of 0->Last and consecutive product')

        oode = torch.square(c_0_last[mask]).sum().item()
        sns.heatmap(c_0_last.cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[0], cbar=False)
        ax[0].set_title(f'FMap Task 0 => {len(evects) - 1}\n oode={oode:.4f}')
        oode = torch.square(c_product[mask]).sum().item()
        sns.heatmap(c_product.cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[1], cbar=False)
        ax[1].set_title(f'FMap Diagonal Product\n oode={oode:.4f}')
        diff = (c_0_last - c_product).abs()
        sns.heatmap(diff.cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[2], cbar=True)
        ax[2].set_title(f'Absolute Differences | sum: {diff.sum().item():.4f}')
        if details: plt.show()
        else: plt.close()

        return diff.sum().item()


