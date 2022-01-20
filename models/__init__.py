import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if '__' not in model and model.endswith('.py')]


def get_model(args):
    mod = importlib.import_module('models.' + args.model)
    class_name = {x.lower(): x for x in mod.__dir__()}[args.model.replace('_', '')]
    model_class = getattr(mod, class_name)
    return model_class(args)
