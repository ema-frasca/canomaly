import sys
from tqdm import tqdm
from typing import Iterable


class Logger:
    def __init__(self):
        self.default_out = sys.stdout

    def log(self, message: object):
        print(message, file=self.default_out)

    def get_tqdm(self, iter_obj: Iterable, desc: str):
        return tqdm(iter_obj, desc=desc.ljust(20), file=self.default_out, ncols=150)


logger = Logger()
