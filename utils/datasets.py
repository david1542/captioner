import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from PIL import Image
import pytorch_lightning as pl

from config.paths import FOLDS_DATA_PATH, ORIGINAL_TEST_PATH
from utils.data import load_captions
from utils.images import ImageEmbedding
from utils.vocab import Vocabulary, get_tokens


class ImageDataset(data.Dataset):
    def __init__(self, path: str, transform: Compose):
        self.path = path
        self.transform = transform
        self.image_list = os.listdir(path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        file_name = self.image_list[index]
        file_path = os.path.join(self.path, file_name)

        image = Image.open(file_path)
        image = self.transform(image)
        return image, file_name

    def get_by_filename(self, file_name: str):
        return self[self.image_list.index(file_name)]


class Collector:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def collect(self, batch):
        embedding_size = len(batch[0][0])
        embeddings = torch.zeros(len(batch), embedding_size)
        batch_tokens = []
        image_ids = []
        for i, t in enumerate(batch):
            embedding, caption, image_id = t
            embeddings[i] = embedding
            batch_tokens.append(caption)
            image_ids.append(image_id)
        batch_tokens = self.vocabulary.to_matrix(batch_tokens)

        return embeddings, batch_tokens, image_ids


class CaptionsDataset(data.Dataset):
    def __init__(self, captions: pd.DataFrame):
        self.captions = captions
        self.image_embeddings = ImageEmbedding()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        row = self.captions.iloc[index]
        image_id = row['image_id']

        caption = row['caption']
        embedding = self.image_embeddings.get_embedding(image_id=image_id).flatten()
        return embedding, caption, image_id

    def get_embedding_size(self):
        return self.image_embeddings.get_embedding_size()


class CaptionsDataModule(pl.LightningDataModule):
    def __init__(self, fold: int, vocabulary: Vocabulary, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.fold = fold
        self.datasets = {}

        self.collector = Collector(vocabulary=vocabulary)
        self.vocabulary = vocabulary

        self.reference = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.fold is None:
            raise Exception('You must supply fold number')

        # Load captions
        captions = load_captions()
        folds_mapping = pd.read_csv(FOLDS_DATA_PATH, index_col='image_id')

        images_ids = {
            'train': folds_mapping.loc[folds_mapping['kfold'] != self.fold].index.tolist(),
            'valid': folds_mapping.loc[folds_mapping['kfold'] == self.fold].index.tolist(),
            'test': pd.read_csv(ORIGINAL_TEST_PATH, sep='\n', names=['image_id'])['image_id'].tolist()
        }
        for set_name in ['train', 'valid', 'test']:
            set_image_ids = images_ids[set_name]
            set_captions = captions[captions['image_id'].isin(set_image_ids)].reset_index(drop=True)
            self.datasets[set_name] = CaptionsDataset(captions=set_captions)

        # Set reference sample
        random_image_id = np.random.choice(images_ids['valid'])
        ref_captions = captions[captions['image_id'] == random_image_id]
        self.datasets['ref'] = CaptionsDataset(captions=ref_captions)

    def train_dataloader(self) -> Dict[str, DataLoader]:
        ref_dataloader = data.DataLoader(dataset=self.datasets['ref'], batch_size=1,
                                         collate_fn=self.collector.collect)
        train_dataloader = data.DataLoader(dataset=self.datasets['train'], batch_size=self.batch_size,
                                           shuffle=True, collate_fn=self.collector.collect)
        return {'train': train_dataloader, 'ref': ref_dataloader}

    def val_dataloader(self) -> DataLoader:
        return data.DataLoader(dataset=self.datasets['valid'], batch_size=self.batch_size,
                               collate_fn=self.collector.collect)

    def test_dataloader(self) -> DataLoader:
        return data.DataLoader(dataset=self.datasets['test'], batch_size=self.batch_size,
                               collate_fn=self.collector.collect)
