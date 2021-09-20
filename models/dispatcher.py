import torch

from models.collection.basic_decoder import BasicDecoder

models = {
    'basic_decoder': BasicDecoder
}

optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}
