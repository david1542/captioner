import argparse

import pytorch_lightning as pl
from clearml import Task

from config.general import PL_RANDOM_SEED, PROJECT_NAME
from models import dispatcher
from utils.datasets import CaptionsDataModule
from utils.vocab import get_vocabulary


def train_model(args: argparse.Namespace):
    # Create vocabulary & datamodule
    vocabulary = get_vocabulary(fold=args.fold)
    datamodule = CaptionsDataModule(fold=args.fold, vocabulary=vocabulary,
                                    batch_size=args.batch_size)

    # Create the model
    model_cls = dispatcher.models[args.model]
    model = model_cls.from_arguments(vocabulary=vocabulary, args=vars(args))

    # Set random seed
    pl.seed_everything(PL_RANDOM_SEED)

    # Create & fit the model using Pytorch Lightning's Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run is None:
        trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = dispatcher.add_arguments_of_models(parser)

    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--task-name', type=str, required=True)
    parser.add_argument('--model', type=str, default='basic_decoder')
    parser.add_argument('--disable-clearml', action='store_true')

    args = parser.parse_args()
    if not args.disable_clearml:
        task = Task.init(project_name=PROJECT_NAME, task_name=args.task_name)

    train_model(args)
