import argparse

from scripts.create_folds import create_folds
from scripts.download_data import download_data
from scripts.extract_embeddings import extract_embeddings


def run_pipeline(args):
    if args.download_data:
        download_data(clean=args.clean_data)

    # Create folds
    create_folds(n_splits=args.n_splits)

    # Extract embeddings
    extract_embeddings(save_every=args.save_every)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-data', action='store_true')
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)

    args = parser.parse_args()

    # Run the preprocess pipeline
    run_pipeline(args)
