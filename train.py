import argparse

import pytorch_lightning as pl
from clearml import Task

from config.general import PL_RANDOM_SEED, PROJECT_NAME
from models.dispatcher import models
from utils.data import get_embedding_size
from utils.datasets import CaptionsDataModule
from utils.vocab import get_vocabulary


def train_model(args: argparse.Namespace):
    # Create vocabulary & datamodule
    vocabulary = get_vocabulary(fold=args.fold)
    datamodule = CaptionsDataModule(fold=args.fold, vocabulary=vocabulary,
                                    batch_size=args.batch_size)

    # Get image embedding size
    embedding_size = get_embedding_size()

    # Create the model
    model_cls = models[args.model]
    model = model_cls(vocabulary=vocabulary, hid_size=embedding_size,
                      learning_rate=args.learning_rate)

    # Set random seed
    pl.seed_everything(PL_RANDOM_SEED)

    # Create & fit the model using Pytorch Lightning's Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)

    if not args.debug:
        trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    task = Task.init(project_name=PROJECT_NAME, task_name="basic decoder")

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model', type=str, default='basic_decoder')
    parser.add_argument('--learning-rate', type=str, default='1e-3')

    args = parser.parse_args()
    train_model(args)
