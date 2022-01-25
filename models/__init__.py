import os
import importlib
from typing import Type
from models.utils.canomaly_model import CanomalyModel


def get_all_models():
    return [model.split('.')[0].replace('_', '-') for model in os.listdir('models')
            if '__' not in model and model.endswith('.py')]


def get_model(model: str) -> Type[CanomalyModel]:
    mod = importlib.import_module('models.' + model.replace('-', '_'))
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('-', '')]
    return getattr(mod, class_name)
