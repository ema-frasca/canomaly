import numpy as np
import os
import random
import torch


class Config:
    def __init__(self, seed: int = None):
        local = False if 'SSH_CONNECTION' in os.environ else True
        self.local = local
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.storage_dir = './storage/' if local else '/nas/softechict-nas-2/efrascaroli/canomaly-data/'
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        self.data_dir = self.storage_dir + 'data/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.models_dir = self.storage_dir + 'models/'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.checkpoints_dir = self.storage_dir + 'checkpoints/'
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.results_dir = self.storage_dir + 'results/'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.logs_dir = self.storage_dir + 'logs/'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if seed is None:
            seed = torch.seed() % 2**32
        self.set_seed(seed)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed


config = Config()
