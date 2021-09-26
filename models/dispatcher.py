import argparse

from models.collection.basic_decoder import BasicDecoder

models = {
    'basic_decoder': BasicDecoder
}


def add_arguments_of_models(parent_parser: argparse.ArgumentParser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    for model_cls in models.values():
        parser = model_cls.add_arguments(parser)
    return parser
