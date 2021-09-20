import argparse

import pandas as pd
from sklearn import model_selection

from config.general import PL_RANDOM_SEED
from config.paths import ORIGINAL_TRAIN_PATH, ORIGINAL_VALID_PATH, FOLDS_DATA_PATH


def create_folds(n_splits=5, random_state=PL_RANDOM_SEED):
    df_train = pd.read_csv(ORIGINAL_TRAIN_PATH, sep='\n', names=['image_id'])
    df_valid = pd.read_csv(ORIGINAL_VALID_PATH, sep='\n', names=['image_id'])

    df = pd.concat([df_train, df_valid])
    df['kfold'] = -1

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=n_splits)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, 'kfold'] = f

    df.to_csv(FOLDS_DATA_PATH, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int, default=5)

    args = parser.parse_args()
    create_folds(n_splits=args.n_splits)
