import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from typing import List

from datasets.utils.canomaly_dataset import CanomalyDataset
from itertools import product


class MyCeleba(CelebA):
    def __init__(self, target_idxes, **kwargs):
        super().__init__(**kwargs)
        self.target_idxes = target_idxes
        targets = product([0, 1], repeat=len(target_idxes))
        self.trans_dict = {t: num for num, t in enumerate(targets)}
        self.targets = [self.trans_dict[tuple(x.tolist())] for x in self.attr[:, target_idxes]]

    def __getitem__(self, item):
        X, target = super().__getitem__(item)
        real_target = self.trans_dict[tuple(target[self.target_idxes].tolist())]
        return X, real_target


class RecCelebA(CanomalyDataset):
    NAME = 'rec-celeba'
    IMG_SIZE = 64
    INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
    TARGET_IDXES = [9, 20]
    N_CLASSES = len([x for x in product([0, 1], repeat=len(TARGET_IDXES))])
    MACRO_CLASSES = [[i] for i in range(N_CLASSES)]
    N_GROUPS = len(MACRO_CLASSES)

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--poison_perc', default=0, type=float)

    def __init__(self, args: Namespace):
        super().__init__(args)

        self.classes_per_task = 1
        self.n_tasks = self.N_GROUPS
        if args.approach == 'joint' and not args.per_task:
            self.n_tasks = 1

        transform_list = [
            transforms.Resize(self.IMG_SIZE),
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(transform_list)

        # if ssl doesn't work:
        # import ssl
        # ssl._create_default_https_context = ssl._create_unverified_context

        self.train_dataset = MyCeleba(root=self.config.data_dir, split='train', download=True, transform=self.transform,
                                      target_idxes=self.TARGET_IDXES)
        self.test_dataset = MyCeleba(root=self.config.data_dir, split='test', download=True, transform=self.transform,
                                     target_idxes=self.TARGET_IDXES)

        self.train_groups = [i for i in range(self.N_GROUPS)]
        self.train_quantity = {label: 5000 // len(group) for group in self.MACRO_CLASSES for label in group}
        self.test_quantity = {label: -1  # 1000 // (len(group))
                              for group in self.MACRO_CLASSES for label in group}

        self.classes_seen = []

    def _get_subset(self, labels: List[int], train=True):
        self.classes_seen.append(labels)
        self.last_seen_classes = labels
        # idxes = (self.train_dataset.attr[:, 20] == 1).nonzero(as_tuple=True)[0]
        # return Subset(self.train_dataset, idxes)
        base_ds = self.train_dataset if train else self.test_dataset
        index_list = []
        for label in labels:
            idxes = (torch.Tensor(base_ds.targets) == label).nonzero(as_tuple=True)[0][
                    :(self.train_quantity[label]
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
        # idxes = (self.test_dataset.attr[:, 20] == 1).nonzero(as_tuple=True)[0]
        # return Subset(self.train_dataset, idxes)
        index_list = []
        for label in range(self.N_CLASSES):
            idxes = (torch.Tensor(self.test_dataset.targets) == label).nonzero(as_tuple=True)[0][
                    :self.test_quantity[label]]
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
            total_idxes = (torch.Tensor(base_ds.targets) == label).nonzero(as_tuple=True)[0]
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
