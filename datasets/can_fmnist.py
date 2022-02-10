from typing import List

import numpy as np
import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset

from datasets.transforms.rotation import Rotation
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.writer import writer


class CanFMNIST(CanomalyDataset):
    NAME = 'can-fmnist'
    N_CLASSES = 10
    INPUT_SHAPE = (1, 28, 28)

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--classes_per_task', type=int, choices=[1, 2], required=True,
                            help='Classes per task. This also determines the number of tasks.')
        parser.add_argument('--permute_dataset', action='store_true', help='Permute randomly the label.')
        parser.add_argument('--add_rotation', action='store_true', help='Add max 90 degrees rotation.')
        parser.add_argument('--min_max_rotation', type=tuple, help='Min max rotation degrees.')
        writer.dir_args.append('classes_per_task')

    def __init__(self, args: Namespace):
        super(CanFMNIST, self).__init__(args)

        self.classes_per_task = args.classes_per_task
        self.n_tasks = self.N_CLASSES // args.classes_per_task - 1

        transform_list = []
        if args.add_rotation:
            deg_min = args.min_max_rotation[0] if args.min_max_rotation is not None else -90
            deg_max = args.min_max_rotation[1] if args.min_max_rotation is not None else 90
            transform_list.append(Rotation(deg_min=deg_min, deg_max=deg_max))

        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

        self.transform_test = transforms.Compose([transforms.ToTensor()])

        self.train_dataset = FashionMNIST(self.config.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = FashionMNIST(self.config.data_dir, train=False, download=True,
                                         transform=self.transform_test)

        self.class_order_arr = self.train_dataset.targets.unique()
        if args.permute_dataset:
            self.class_order_arr = np.random.permutation(self.class_order_arr)

    def _get_subset(self, labels: List[int], train=True):
        self.last_seen_classes = labels
        base_ds = self.train_dataset if train else self.test_dataset
        idxes = torch.isin(base_ds.targets, torch.tensor(labels)).nonzero(as_tuple=True)[0]
        return Subset(base_ds, idxes)

    def _get_task_dataset(self):
        for step in range(self.n_tasks):
            idx_labels = [step * self.classes_per_task + cl for cl in range(self.classes_per_task)]
            labels = self.class_order_arr[idx_labels].tolist()
            yield self._get_subset(labels)

    def _get_joint_dataset(self):
        labels = [cl for cl in range(self.N_CLASSES - self.classes_per_task)]
        return self._get_subset(labels)
