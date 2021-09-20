import string

import pandas as pd
import torch

from config.paths import RAW_LEMMATIZED_CAPTIONS_PATH, RAW_CAPTIONS_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_EMBEDDINGS_MAP_PATH


def load_raw_captions(lemmatized=False):
    captions_path = RAW_LEMMATIZED_CAPTIONS_PATH if lemmatized else RAW_CAPTIONS_PATH
    captions = pd.read_csv(captions_path, sep='\t', names=['image_id', 'caption'])

    # Extract caption numbers from the image ids
    caption_numbers = captions['image_id'].str.slice(-1)

    # Create a dedicated column for caption number
    captions.loc[:, 'caption_number'] = caption_numbers

    # Remove #0, #1, etc. from the image ids.
    captions.loc[:, 'image_id'] = captions['image_id'].str.slice(stop=-2)

    # Remove space and dot at the end of the captions
    captions.loc[:, 'caption'] = captions['caption'].str.slice(stop=-2)

    # Remove image ids that don't exist
    excluded_image_ids = ['2258277193_586949ec62.jpg.1']
    captions = captions.drop(captions[captions['image_id'].isin(excluded_image_ids)].index)
    return captions


def load_captions(lemmatized=False):
    captions = load_raw_captions(lemmatized=lemmatized)
    captions = preprocess_captions(captions)
    return captions


def preprocess_captions(captions: pd.DataFrame):
    captions.loc[:, 'caption'] = captions['caption'].str.lower()
    captions.loc[:, 'caption'] = captions['caption'].str.replace('[{}]'.format(string.punctuation), '')
    return captions


def load_embeddings():
    embeddings = torch.load(IMAGE_EMBEDDINGS_PATH)
    mappings = pd.read_csv(IMAGE_EMBEDDINGS_MAP_PATH, index_col='image_id')
    return embeddings, mappings


def get_embedding_size():
    embeddings, _ = load_embeddings()
    return embeddings.shape[1]
