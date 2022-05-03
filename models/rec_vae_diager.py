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
from utils.batch_norm_freeze import bn_untrack_stats
from copy import deepcopy


# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae --kl_weight 1 --batch_size 64 --n_epochs 30 --latent_space 32 --approach joint
class RecVAEDiagER(RecVAEER):
    NAME = 'rec-vae-diager'
    VARIATIONAL = True
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        RecVAE.add_model_args(parser)
        Buffer.add_args(parser)
        parser.add_argument('--con_weight', type=float, default=0,
                            help='Weight of consolidation.')
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
        self.diagonal_buffer: Buffer

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        real_batch_size = x.shape[0]
        if self.args.er_mode and not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data()
            x = torch.cat((x, buf_inputs))
            y = torch.cat((y, buf_labels))

        # todo: add not augmented data in buffer
        # add data every time to the buffer
        self.buffer.add_data(examples=x[:real_batch_size], labels=y[:real_batch_size])

        for param in self.net.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                assert False, 'param nan or inf'

        self.opt.zero_grad()
        outputs, latent_mu, latent_logvar, z = self.forward(x, task)
        loss_reconstruction = self.reconstruction_loss(x, outputs)
        loss_kl = self.kld_loss(latent_mu, latent_logvar)
        loss = loss_reconstruction + self.args.kl_weight * loss_kl
        wlogs = {'rec_loss': loss_reconstruction.item(), 'kl_loss': loss_kl.item()}

        if torch.isnan(loss) or torch.isinf(loss):
            # assert False, 'loss nan or inf'
            pass

        if task > 1:  # and self.args.con_weight > 0:
            with bn_untrack_stats(self.net):
                inputs, labels = self.spectral_buffer.get_all_data()
                outputs, latent_mu, latent_logvar, z = self.forward(inputs)
                energy, eigenvalues, eigenvectors, L, (A, D, dists) = laplacian_analysis(latent_mu, logvars=None,
                                                                                         norm_lap=True,
                                                                                         knn=self.args.knn_laplace,
                                                                                         n_pairs=self.args.fmap_dim)
                self.buffer_evectors.append(eigenvectors)
                c_loss = self.get_off_diagonal_error()
                # self.buffer_evectors.pop()
                if torch.isnan(c_loss) or torch.isinf(c_loss):
                    assert False, 'c_loss nan or inf'
                loss += self.args.con_weight * c_loss
                wlogs['con_loss'] = c_loss.item()

        loss.backward()

        for param in self.net.parameters():
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f'\nEigenvectors there is None: {torch.isnan(eigenvectors.grad).any().item()}'
                      f'\nL there is None: any {torch.isnan(L.grad).any().item()} all {torch.isnan(L.grad).all().item()}'
                      f'\nA there is None: any {torch.isnan(A.grad).any().item()} all {torch.isnan(A.grad).all().item()}'
                      f'\nDistances there is None: any {torch.isnan(dists.grad).any().item()} all {torch.isnan(dists.grad).all().item()}'
                      f'\nLatents there is None: any {torch.isnan(latent_mu.grad).any().item()} all {torch.isnan(latent_mu.grad).all().item()}')
                assert False, 'grad nan or inf'
        if task > 1:
            self.buffer_evectors.pop()

        self.opt.step()
        if self.args.wandb:
            wandb.log(wlogs)

        return loss.item()

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
            with torch.no_grad():
                self.buffer_evectors.append(self.compute_buffer_evects())
                cerr = self.get_consolidation_error()
                self.buffer_evectors.pop()
            if self.args.wandb:
                wandb.log({'consolidation_error': cerr.item(), 'reconstruction_error': rec_err})
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
                                       f'TRAIN on task {task + 1}/{self.dataset.n_tasks} - epoch {e + 1}/{self.args.n_epochs}',
                                       leave=keep_progress)
            for x, y in progress:
                loss = self.train_on_batch(x.to(self.device), y.to(self.device), task)
                progress.set_postfix({'loss': loss})
                if self.args.wandb:
                    wandb.log({"loss": loss, "epoch": e,
                               "lr": self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.args.lr,
                               "task": task})
            self.scheduler_step()

        with torch.no_grad():
            self.net_eval()
            evects = self.compute_buffer_evects()
            self.net_train()
        self.buffer_evectors.append(evects)
        if task > 1:
            self.spectral_buffer = deepcopy(self.buffer)
            cerr = self.get_consolidation_error()
            logger.log(f'Consolidation error: {cerr.item():.4f}')

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

    def compute_buffer_evects(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        outputs, latent_mu, latent_logvar, z = self.forward(inputs)
        energy, eigenvalues, eigenvectors, L, _ = laplacian_analysis(latent_mu, logvars=latent_logvar, norm_lap=True,
                                                                     knn=self.args.knn_laplace)
        return eigenvectors[:, :self.args.fmap_dim]

    def get_off_diagonal_error(self):
        # ((evects_tmen1@evects_t * torch.diag(~torch.ones(len(evects_t))))**2).sum()
        evects = self.buffer_evectors
        n_vects = self.args.fmap_dim
        c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
        c_product = torch.ones((n_vects, n_vects), device=self.device, dtype=torch.double)
        for i, ev in enumerate(evects[:-1]):
            c = ev[:, :n_vects].T @ evects[i + 1][:, :n_vects]
            if i == 0:
                c_product = c.clone()
            else:
                c_product = c_product @ c

        diff = (c_0_last - c_product).abs()
        return diff.sum()

    # def get_consolidation_error(self, details=False):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     evects = self.buffer_evectors
    #     n_vects = self.args.fmap_dim
    #
    #     ncols = len(evects) - 1
    #     figsize = (6*ncols, 6)
    #     fig, ax = plt.subplots(1, ncols, figsize=figsize)
    #     plt.suptitle(f'\nKnn Norm Laplacian | {n_vects} eigenvects | {len(evects[0])} data')
    #     mask = torch.eye(n_vects) == 0
    #     c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
    #     c_product = torch.ones((n_vects, n_vects), device=self.device, dtype=torch.double)
    #     for i, ev in enumerate(evects[:-1]):
    #         c = ev[:, :n_vects].T @ evects[i + 1][:, :n_vects]
    #         if i == 0:
    #             c_product = c.clone()
    #         else:
    #             c_product = c_product @ c
    #         oode = torch.square(c[mask]).sum().item()
    #         sns.heatmap(c.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[i], cbar=True if i + 1 == ncols else False)
    #         ax[i].set_title(f'FMap Task {i} => {i + 1} | oode={oode:.4f}')
    #
    #     if details: plt.show()
    #     else: plt.close()
    #
    #     figsize = (6 * 3, 8)
    #     fig, ax = plt.subplots(1, 3, figsize=figsize)
    #     plt.suptitle(f'\nCompare differences of 0->Last and consecutive product')
    #
    #     oode = torch.square(c_0_last[mask]).sum().item()
    #     sns.heatmap(c_0_last.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[0], cbar=False)
    #     ax[0].set_title(f'FMap Task 0 => {len(evects) - 1}\n oode={oode:.4f}')
    #     oode = torch.square(c_product[mask]).sum().item()
    #     sns.heatmap(c_product.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[1], cbar=False)
    #     ax[1].set_title(f'FMap Diagonal Product\n oode={oode:.4f}')
    #     diff = (c_0_last - c_product).abs()
    #     sns.heatmap(diff.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[2], cbar=True)
    #     ax[2].set_title(f'Absolute Differences | sum: {diff.sum().item():.4f}')
    #     if details: plt.show()
    #     else: plt.close()
    #
    #     # if self.args.wandb:
    #     #     wandb.log({"fmap": wandb.Image(diff.cpu().detach().numpy())})
    #
    #     return diff.sum()
