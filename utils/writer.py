import os
from utils import create_dir
from utils.config import config


class Writer:
    DEFAULT_NAME = 'logs'

    def __init__(self):
        self.dir_args: list[str] = []
        self.name_arg = ''

    def write_log(self, log: dict, name: str = None, extension='pyd'):
        target_dir = config.logs_dir
        for dir_arg in self.dir_args:
            target_dir = create_dir(os.path.join(target_dir, dir_arg + '-' + log[dir_arg]))

        filename = name
        if not filename and self.name_arg in log:
            filename = log[self.name_arg]
        if not filename:
            filename = self.DEFAULT_NAME
        filename += '.' + extension

        with open(os.path.join(target_dir, filename), 'a') as f:
            f.write(str(log) + '\n')


writer = Writer()
