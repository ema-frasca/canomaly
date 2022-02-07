import numpy as np
import os
import random
import torch
from utils import create_dir


class Config:
    def __init__(self, seed: int = None):
        local = False if 'SSH_CONNECTION' in os.environ else True
        self.local = local
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.storage_dir = create_dir('./storage/' if local else '/nas/softechict-nas-1/rbenaglia/canomaly-data/')

        self.data_dir = create_dir(self.storage_dir + 'data/')
        self.models_dir = create_dir(self.storage_dir + 'models/')
        self.checkpoints_dir = create_dir(self.storage_dir + 'checkpoints/')
        self.results_dir = create_dir(self.storage_dir + 'results/')
        self.logs_dir = create_dir(self.storage_dir + 'logs/')

        self.seed = 0
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
