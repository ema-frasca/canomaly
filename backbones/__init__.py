import importlib
from typing import Type, List
from backbones.utils.encoder import Encoder
from backbones.utils.decoder import Decoder

string_to_erase = ['can-', 'group-', 'forg-']


def replace_str(base: str, to_replace: List[str], replacement: str):
    text = base
    for repl in to_replace:
        text = text.replace(repl, replacement)
    return text


def get_decoder(dataset: str) -> Type[Decoder]:
    decoder_name = replace_str(dataset, string_to_erase, '').upper() + '_decoder'
    mod = importlib.import_module('backbones.' + decoder_name)
    class_name = {x.lower(): x for x in mod.__dir__()}[decoder_name.replace('_', '').lower()]
    return getattr(mod, class_name)


def get_encoder(dataset: str) -> Type[Encoder]:
    encoder_name = replace_str(dataset, string_to_erase, '').upper() + '_encoder'
    mod = importlib.import_module('backbones.' + encoder_name)
    class_name = {x.lower(): x for x in mod.__dir__()}[encoder_name.replace('_', '').lower()]
    return getattr(mod, class_name)
