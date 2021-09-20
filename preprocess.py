import argparse

from scripts.create_folds import create_folds
from scripts.extract_embeddings import extract_embeddings


def run_pipeline(args):
    # Create folds
    create_folds(n_splits=args.n_splits)

    # Extract embeddings
    extract_embeddings(save_every=args.save_every)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)

    args = parser.parse_args()

    # Run the preprocess pipeline
    run_pipeline(args)
