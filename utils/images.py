import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from config.paths import FLICKER_IMAGES_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_EMBEDDINGS_MAP_PATH
from utils.data import load_captions


def display_random_image(df: pd.DataFrame, lemmatized=False):
    captions = load_captions(lemmatized=lemmatized)
    image_ids = df['image_id'].values
    random_id = np.random.choice(image_ids)
    image_path = os.path.join(FLICKER_IMAGES_PATH, random_id)

    image_captions = captions.loc[captions['image_id'] == random_id, ['caption', 'caption_number']].values
    image_captions = [f'#{caption_number}: {caption}' for caption, caption_number in image_captions]
    title = '\n'.join(image_captions)

    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(title, wrap=True)


class ImageEmbedding:
    def __init__(self):
        self.image_embeddings = torch.load(IMAGE_EMBEDDINGS_PATH)
        self.image_embeddings_map = pd.read_csv(IMAGE_EMBEDDINGS_MAP_PATH, index_col='image_id')

    def get_embedding(self, image_id: str):
        index = self.image_embeddings_map.loc[image_id]
        return self.image_embeddings[index]

    def get_embedding_size(self):
        return len(self.image_embeddings[0])
