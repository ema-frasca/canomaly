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


class ForgFMNIST(CanomalyDataset):
    NAME = 'forg-fmnist'
    INPUT_SHAPE = (1, 28, 28)
    N_CLASSES = 10
    MACRO_CLASSES = [[7], [5], [1], [8], [0, 2, 3, 4, 6], [9]]
    N_GROUPS = len(MACRO_CLASSES)
    # MACRO_CLASSES = [[0, 2, 4, 6], [1], [3], [5, 7, 9], [8]]
    ## Classes:

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        # parser.add_argument('--classes_per_task', type=int, choices=[1, 2], required=True,
        #                     help='Classes per task. This also determines the number of tasks.')
        parser.add_argument('--ano_group', type=int, choices=[i for i in range(ForgFMNIST.N_GROUPS)],
                            default=ForgFMNIST.N_GROUPS-1, help='Group-class that will stay as anomaly.')
        writer.dir_args.append('ano_group')
        parser.add_argument('--add_rotation', action='store_true', help='Add degrees rotation.')
        parser.add_argument('--min_max_rotation', type=tuple, default=(-15, 15), help='Min max rotation degrees.')

    def __init__(self, args: Namespace):
        super(ForgFMNIST, self).__init__(args)

        self.classes_per_task = 1
        self.n_tasks = self.N_GROUPS - 1
        if args.approach == 'joint' and not args.per_task:
            self.n_tasks = 1

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

        self.anomaly_group = self.args.ano_group
        self.train_groups = [i for i in range(self.N_GROUPS) if i != self.anomaly_group]
        self.train_quantity = {label: 6000 // len(group) for group in self.MACRO_CLASSES for label in group }

        # for macro_label, group in enumerate(self.MACRO_CLASSES):
            # test_indexes = torch.isin(self.test_dataset.targets, torch.tensor(group)).nonzero(as_tuple=True)[0]
            # self.test_dataset.targets[test_indexes] = macro_label

            # n_classes = len(group)
            # for label in group:
            #     indexes = (self.train_dataset.targets == label).nonzero(as_tuple=True)[0]
            #     split_th = 6000 // n_classes
            #     in_idxes = indexes[:split_th]
            #     out_idxes = indexes[split_th:]
            #     self.train_dataset.targets[in_idxes] = macro_label
            #     self.train_dataset.targets[out_idxes] = self.N_CLASSES


    def _get_subset(self, labels: List[int], train=True):
        self.last_seen_classes = labels
        base_ds = self.train_dataset if train else self.test_dataset
        index_list = []
        for label in labels:
            idxes = (base_ds.targets == label).nonzero(as_tuple=True)[0][:self.train_quantity[label]]
            index_list.append(idxes)
        idxes = torch.cat(index_list)
        return Subset(base_ds, idxes)

    def _get_task_dataset(self):
        for group_idx in self.train_groups:
            yield self._get_subset(self.MACRO_CLASSES[group_idx])

    def _get_joint_dataset(self):
        if not self.args.per_task:
            yield self._get_subset([label for group_idx in self.train_groups for label in self.MACRO_CLASSES[group_idx]])
            return
        for step in range(self.n_tasks):
            yield self._get_subset([label for group_idx in self.train_groups[:step+1] for label in self.MACRO_CLASSES[group_idx]])
