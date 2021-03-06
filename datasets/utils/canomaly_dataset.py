from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader, Dataset
from utils.config import config
from typing import Tuple, Iterator


class CanomalyDataset:
    NAME: str = None
    N_CLASSES: int = None
    INPUT_SHAPE: Tuple[int, int, int] = None

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace):
        self.config = config
        self.args = args

        self.classes_per_task = None
        self.n_tasks = None

        self.train_dataset = None
        self.test_dataset = None
        self.transform = None
        self.transform_test = None

        self.last_seen_classes: list[int] = []

    @abstractmethod
    def _get_task_dataset(self) -> Iterator[Dataset]:
        pass

    @abstractmethod
    def _get_joint_dataset(self) -> Dataset:
        pass

    def _get_test_dataset(self) -> Dataset:
        return self.test_dataset

    def _get_mini_test_dataset(self) -> Dataset:
        return self.test_dataset

    def task_loader(self):
        for ds in self._get_task_dataset():
            yield DataLoader(ds, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

    def joint_loader(self):
        for ds in self._get_joint_dataset():
            yield DataLoader(ds, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

    def test_loader(self):
        return DataLoader(self._get_test_dataset(), batch_size=self.args.batch_size, shuffle=False)

    def mini_test_loader(self):
        return DataLoader(self._get_mini_test_dataset(), batch_size=self.args.batch_size, shuffle=False)
