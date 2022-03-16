import torch
from argparse import Namespace, ArgumentParser
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Subset, Dataset, DataLoader
from datasets.utils.canomaly_dataset import CanomalyDataset
from PIL import Image
import os
from utils.writer import writer
import matplotlib.pyplot as plt


class MVTecAD_dataset(Dataset):
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    def __init__(self, path: str, train=True, transform: any = None, class_names: list[str] = None):
        assert class_names is None or torch.tensor([cname in self.CLASS_NAMES for cname in class_names]).all(), \
            f'MVTecAD: class_name: {class_names}, should be in {self.CLASS_NAMES}'
        self.dataset_path = os.path.join(path, 'MVTecAD')
        assert os.path.exists(self.dataset_path), 'MVTecAD: dataset not found'
        self.train = train
        self.class_names = class_names
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform

        # load dataset
        if class_names is None:
            class_names = self.CLASS_NAMES
        self.data, self.targets, self.names = [], [], []
        for cname in class_names:
            data, targets = self.load_dataset_folder(cname)
            names = [cname] * len(data)
            self.data.extend(data)
            self.targets.extend(targets)
            self.names.extend(names)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

    def load_dataset_folder(self, class_name: str):
        partition = 'train' if self.train else 'test'
        x, y = [], []

        class_dir = os.path.join(self.dataset_path, class_name, partition)

        img_types = sorted(os.listdir(class_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(class_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_path_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.png')])
            x.extend(img_path_list)

            # load labels
            if img_type == 'good':
                y.extend([0] * len(img_path_list))
            else:
                y.extend([1] * len(img_path_list))

        return list(x), list(y)


class MVTecAD(CanomalyDataset):
    NAME = 'mvtecad'
    IMG_SIZE = 128
    INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
    N_CLASSES = len(MVTecAD_dataset.CLASS_NAMES)
    MACRO_CLASSES = [['bottle'], ['cable']]
    N_GROUPS = len(MACRO_CLASSES)

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace):
        super(MVTecAD, self).__init__(args)

        self.n_tasks = self.N_GROUPS
        if args.approach == 'joint' and not args.per_task:
            self.n_tasks = 1

        transform_list = [
            transforms.Resize(self.IMG_SIZE),
            transforms.CenterCrop(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(transform_list)

        self.train_datasets, self.test_datasets = [], []
        if args.approach == 'joint':
            if args.per_task:
                cum_names = []
                for cnames in self.MACRO_CLASSES:
                    cum_names.extend(cnames)
                    self.train_datasets.append(MVTecAD_dataset(self.config.data_dir,
                                                               train=True,
                                                               transform=self.transform,
                                                               class_names=cum_names))
                    self.test_datasets.append(MVTecAD_dataset(self.config.data_dir,
                                                              train=False,
                                                              transform=self.transform,
                                                              class_names=cum_names))
            else:
                cum_names = [cname for cnames in self.MACRO_CLASSES for cname in cnames]
                self.train_datasets = [MVTecAD_dataset(self.config.data_dir,
                                                       train=True,
                                                       transform=self.transform,
                                                       class_names=cum_names)]
                self.test_datasets = [MVTecAD_dataset(self.config.data_dir,
                                                      train=False,
                                                      transform=self.transform,
                                                      class_names=cum_names)]
        else:
            cum_names = []
            for cnames in self.MACRO_CLASSES:
                cum_names.extend(cnames)
                self.train_datasets.append(MVTecAD_dataset(self.config.data_dir,
                                                           train=True,
                                                           transform=self.transform,
                                                           class_names=cnames))
                self.test_datasets.append(MVTecAD_dataset(self.config.data_dir,
                                                          train=False,
                                                          transform=self.transform,
                                                          class_names=cum_names))

        dl = torch.utils.data.DataLoader(self.train_datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                         drop_last=True)
        self.last_step = None

    def _get_task_dataset(self):
        for step, dataset in enumerate(self.train_datasets):
            self.last_seen_classes = self.MACRO_CLASSES[step]
            self.last_step = step
            yield dataset

    def _get_joint_dataset(self):
        for dataset in self._get_task_dataset():
            yield dataset

    def _get_test_dataset(self):
        assert self.last_step is not None, 'TEST dataset requested without training'
        return self.test_datasets[self.last_step]

    def test_loader(self):
        return DataLoader(self._get_test_dataset(), batch_size=self.args.batch_size, shuffle=True)
