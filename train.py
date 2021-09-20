import argparse
from typing import cast

import pytorch_lightning as pl
from clearml import Task

from config.general import PL_RANDOM_SEED
from models.dispatcher import optimizers, models
from models.pl_wrapper import PLWrapper
from utils.data import get_embedding_size
from utils.datasets import CaptionsDataModule
from utils.general import parse_generic_args
from utils.vocab import get_vocabulary


class TrainArgs(argparse.Namespace):
    debug: bool
    fold: int
    batch_size: int
    model: str
    gpus: str
    optimizer: str
    optimizer_args: str


def train_model(args: TrainArgs):
    # Create vocabulary & datamodule
    vocabulary = get_vocabulary(fold=args.fold)
    datamodule = CaptionsDataModule(fold=args.fold, vocabulary=vocabulary,
                                    batch_size=args.batch_size)

    # Get image embedding size
    embedding_size = get_embedding_size()

    # Create the model
    model_cls = models[args.model]
    nn_model = model_cls(vocabulary=vocabulary, hid_size=embedding_size)

    # Get optimizer metadata
    optimizer_args = parse_generic_args(args.optimizer_args)
    optimizer_cls = optimizers[args.optimizer]

    # Create an optimizer based on the arguments
    optimizer = optimizer_cls(nn_model.parameters(), **optimizer_args)

    # Wrap everything in a Pytorch Lightning wrapper and begin training
    pl_model = PLWrapper(model=nn_model, optimizer=optimizer)

    # Set random seed
    pl.seed_everything(PL_RANDOM_SEED)

    # Prepare Trainer params
    params = {
        'deterministic': True,
        'fast_dev_run': 2 if args.debug else False
    }
    if args.gpus is not None:
        params['gpus'] = args.gpus

    # Create & fit the model using Pytorch Lightning's Trainer
    trainer = pl.Trainer(**params)
    trainer.fit(pl_model, datamodule=datamodule)

    if not args.debug:
        trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    task = Task.init(project_name='Image Caption Generator', task_name="basic decoder")

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model', type=str, default='basic_decoder')
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--optimizer-args', type=str, default='lr=1e-3')

    args = parser.parse_args()
    args = cast(TrainArgs, args)

    train_model(args)
