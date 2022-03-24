from typing import List

import numpy as np
import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

from datasets.transforms.rotation import Rotation
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.writer import writer


class RecFMNIST(CanomalyDataset):
    NAME = 'rec-fmnist'
    INPUT_SHAPE = (1, 28, 28)
    N_CLASSES = 10
    MACRO_CLASSES = [[0, 2, 3, 4, 6], [1], [5, 7, 9], [8]]
    # MACRO_CLASSES = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    N_GROUPS = len(MACRO_CLASSES)

    # MACRO_CLASSES = [[0, 2, 4, 6], [1], [3], [5, 7, 9], [8]]
    ## Classes:

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--add_rotation', action='store_true', help='Add degrees rotation.')
        parser.add_argument('--min_max_rotation', type=tuple, default=(-15, 15), help='Min max rotation degrees.')
        parser.add_argument('--poison_perc', default=0, type=float)

    def __init__(self, args: Namespace):
        super(RecFMNIST, self).__init__(args)

        self.classes_per_task = 1
        self.n_tasks = self.N_GROUPS
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

        self.train_groups = [i for i in range(self.N_GROUPS)]
        # train max: 6k | test max: 1k
        self.train_quantity = {label: 6000 // (len(group)) for group in self.MACRO_CLASSES for label in group}
        self.test_quantity = {label: 1000 // (len(group)) for group in self.MACRO_CLASSES for label in group}
        self.classes_seen = []

    def _get_subset(self, labels: List[int], train=True):
        self.classes_seen.append(labels)
        self.last_seen_classes = labels
        base_ds = self.train_dataset if train else self.test_dataset
        index_list = []
        for label in labels:
            idxes = (base_ds.targets == label).nonzero(as_tuple=True)[0][:(self.train_quantity[label]
                                                                           * (1 - self.args.poison_perc)).__round__()]
            index_list.append(idxes)
        if self.args.poison_perc > 0:
            poison_idx = self._random_sample_from_other_classes()
            index_list.extend(poison_idx)
        idxes = torch.cat(index_list)
        return Subset(base_ds, idxes)

    def _get_task_dataset(self):
        for group_idx in self.train_groups:
            yield self._get_subset(self.MACRO_CLASSES[group_idx])

    def _get_test_dataset(self):
        index_list = []
        for label in range(self.N_CLASSES):
            idxes = (self.test_dataset.targets == label).nonzero(as_tuple=True)[0][:self.test_quantity[label]]
            index_list.append(idxes)
        idxes = torch.cat(index_list)
        return Subset(self.test_dataset, idxes)

    def test_loader(self):
        return DataLoader(self._get_test_dataset(), batch_size=self.args.batch_size, shuffle=False)

    def _random_sample_from_other_classes(self):
        # last
        classes_not_seen = list(set(sublist for x in self.MACRO_CLASSES for sublist in x).difference(
            set(sublist for x in self.classes_seen for sublist in x)))
        base_ds = self.train_dataset
        total_quantity = (sum(self.train_quantity[classe] for classe in self.last_seen_classes) *
                          self.args.poison_perc).__round__()
        single_label_quantity = total_quantity // len(classes_not_seen)
        poison_index_list = []
        for label in classes_not_seen:
            total_idxes = (base_ds.targets == label).nonzero(as_tuple=True)[0]
            idxes = total_idxes[torch.randperm(total_idxes.size()[0])][:single_label_quantity.__round__()]
            poison_index_list.append(idxes)
        return poison_index_list
        # np.random.choice([x for x in classes_seen])

    def _get_joint_dataset(self):
        if not self.args.per_task:
            yield self._get_subset(
                [label for group_idx in self.train_groups for label in self.MACRO_CLASSES[group_idx]])
            return
        for step in range(self.n_tasks):
            yield self._get_subset(
                [label for group_idx in self.train_groups[:step + 1] for label in self.MACRO_CLASSES[group_idx]])
