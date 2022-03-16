import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Subset
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.writer import writer
import matplotlib.pyplot as plt


class ForgCelebA(CanomalyDataset):
    NAME = 'forg-celeba'
    IMG_SIZE = 64
    INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
    N_CLASSES = 10
    MACRO_CLASSES = [[i] for i in range(N_CLASSES)]
    N_GROUPS = len(MACRO_CLASSES)

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        parser.add_argument('--ano_group', type=int, choices=[i for i in range(ForgCelebA.N_GROUPS)],
                            default=ForgCelebA.N_GROUPS - 1, help='Group-class that will stay as anomaly.')
        writer.dir_args.append('ano_group')

    def __init__(self, args: Namespace):
        super(ForgCelebA, self).__init__(args)

        self.classes_per_task = 1
        self.n_tasks = self.N_GROUPS - 1
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

        self.train_dataset = CelebA(self.config.data_dir, split='train', download=True, transform=self.transform)
        self.test_dataset = CelebA(self.config.data_dir, split='test', download=True, transform=self.transform)

        self.anomaly_group = self.args.ano_group
        self.train_groups = [i for i in range(self.N_GROUPS) if i != self.anomaly_group]
        self.train_quantity = {label: 5000 // len(group) for group in self.MACRO_CLASSES for label in group}

    def _get_subset(self, labels: list[int], train=True):
        self.last_seen_classes = labels
        idxes = (self.train_dataset.attr[:, 20] == 1).nonzero(as_tuple=True)[0]
        return Subset(self.train_dataset, idxes)
        base_ds = self.train_dataset if train else self.test_dataset
        index_list = []
        for label in labels:
            idxes = (torch.tensor(base_ds.targets) == label).nonzero(as_tuple=True)[0][:self.train_quantity[label]]
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
