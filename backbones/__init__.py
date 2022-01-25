import importlib
from typing import Type
from backbones.utils.encoder import Encoder
from backbones.utils.decoder import Decoder


def get_decoder(dataset: str) -> Type[Decoder]:
    decoder_name = dataset.replace('can-', '').upper() + '_decoder'
    mod = importlib.import_module('backbones.' + decoder_name)
    class_name = {x.lower(): x for x in mod.__dir__()}[decoder_name.replace('_', '').lower()]
    return getattr(mod, class_name)


def get_encoder(dataset: str) -> Type[Encoder]:
    encoder_name = dataset.replace('can-', '').upper() + '_encoder'
    mod = importlib.import_module('backbones.' + encoder_name)
    class_name = {x.lower(): x for x in mod.__dir__()}[encoder_name.replace('_', '').lower()]
    return getattr(mod, class_name)
