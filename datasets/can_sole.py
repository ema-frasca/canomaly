import os
from typing import List

import numpy as np
import pandas as pd
import torch
from argparse import Namespace, ArgumentParser
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.writer import writer
from torchvision.datasets import MNIST


class CanSOLE(CanomalyDataset):
    NAME = 'can-sole'
    N_CLASSES = 2

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--classes_per_task', type=int, choices=[1], required=True,
                            help='Classes per task. This also determines the number of tasks.')
        writer.dir_args.append('classes_per_task')

    def __init__(self, args: Namespace):
        super(CanSOLE, self).__init__(args)

        self.classes_per_task = args.classes_per_task
        self.n_tasks = self.N_CLASSES // args.classes_per_task - 1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        # x_train, y_train = map(lambda x: pd.read_csv(x, index_col=['year', 'month', 'cod_cliente_dbma', 'label_group']),
        #                        [os.path.join(self.config.data_dir, 'sole', 'training_X.csv'),
        #                         os.path.join(self.config.data_dir, 'sole', 'training_y.csv')])
        # x_test, y_test = map(lambda x: pd.read_csv(x, index_col=['year', 'month', 'cod_cliente_dbma', 'label_group']),
        #                      [os.path.join(self.config.data_dir, 'sole', 'test_X.csv'),
        #                       os.path.join(self.config.data_dir, 'sole', 'test_y.csv')])
        # self.train_dataset = RegressionColumnarDataset(x_train, [x for x in x_test if 'group' in x], y_train)
        # self.test_dataset = RegressionColumnarDataset(x_test, [x for x in x_test if 'group' in x], y_test)

        self.train_dataset = MNIST(self.config.data_dir, train=True, download=False, transform=self.transform)
        self.test_dataset = MNIST(self.config.data_dir, train=False, download=False, transform=self.transform)
        # unflatten self.train_dataset[0][0].reshape(28,28,1)
        self.INPUT_SHAPE = self.train_dataset[0][0].shape[0]

    def _get_subset(self, labels: List[int], train=True):
        self.last_seen_classes = labels
        base_ds = self.train_dataset if train else self.test_dataset
        idxes = torch.isin(base_ds.targets.flatten(), torch.tensor(labels)).nonzero(as_tuple=True)[0]
        return Subset(base_ds, idxes)

    def _get_task_dataset(self):
        for step in range(self.n_tasks):
            labels = [step * self.classes_per_task + cl for cl in range(self.classes_per_task)]
            yield self._get_subset(labels)

    def _get_joint_dataset(self):
        labels = [0, 1, 2,3 ,4,5,6,7,8] #[cl for cl in range(self.N_CLASSES - self.classes_per_task)]
        return self._get_subset(labels)

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)


class RegressionColumnarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cats: list, y: pd.Series):
        self.dfcats = df[cats]  # type: pandas.core.frame.DataFrame
        self.dfconts = df.drop(cats, axis=1)  # type: pandas.core.frame.DataFrame
        if len(cats):
            self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64)  # tpye: numpy.ndarray
        self.conts = torch.Tensor(np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(
            np.float64))  # tpye: numpy.ndarray
        self.targets = torch.Tensor(y.values.flatten())

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        return self.conts[idx], self.targets[idx]

    # def __getitem__(self, idx):
    #     return [self.cats[idx], self.conts[idx], self.y[idx]]
