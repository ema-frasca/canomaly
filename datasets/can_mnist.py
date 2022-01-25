from typing import List

import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.writer import writer


class CanMNIST(CanomalyDataset):
    NAME = 'can-mnist'
    N_CLASSES = 10
    INPUT_SHAPE = (1, 28, 28)

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--classes_per_task', type=int, choices=[1, 2], required=True,
                            help='Classes per task. This also determines the number of tasks.')
        writer.dir_args.append('classes_per_task')

    def __init__(self, args: Namespace):
        super(CanMNIST, self).__init__(args)

        self.classes_per_task = args.classes_per_task
        self.n_tasks = self.N_CLASSES // args.classes_per_task - 1

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset = MNIST(self.config.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = MNIST(self.config.data_dir, train=False, download=True, transform=self.transform)

    def _get_subset(self, labels: List[int], train=True):
        self.last_seen_classes = labels
        base_ds = self.train_dataset if train else self.test_dataset
        idxes = torch.isin(base_ds.targets, torch.tensor(labels)).nonzero(as_tuple=True)[0]
        return Subset(base_ds, idxes)

    def _get_task_dataset(self):
        for step in range(self.n_tasks):
            labels = [step * self.classes_per_task + cl for cl in range(self.classes_per_task)]
            yield self._get_subset(labels)

    def _get_joint_dataset(self):
        labels = [cl for cl in range(self.N_CLASSES - self.classes_per_task)]
        return self._get_subset(labels)
