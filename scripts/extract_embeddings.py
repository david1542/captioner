import argparse
import os

from utils.datasets import ImageDataset
import pandas as pd

import torch
import torch.nn as nn
from config.paths import FLICKER_IMAGES_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_EMBEDDINGS_MAP_PATH
from tqdm import tqdm
from torch.utils import data

from torchvision import models, transforms


def save_artifacts(embeddings, image_ids):
    embeddings = torch.cat(embeddings)
    image_ids = pd.DataFrame(image_ids, columns=['image_id']).set_index('image_id')

    update_files = os.path.exists(IMAGE_EMBEDDINGS_PATH) and \
                   os.path.exists(IMAGE_EMBEDDINGS_MAP_PATH)

    if update_files:
        existing_mappings = pd.read_csv(IMAGE_EMBEDDINGS_MAP_PATH, index_col='image_id')
        existing_embeddings = torch.load(IMAGE_EMBEDDINGS_PATH)

        image_ids = pd.concat([existing_mappings, image_ids])
        embeddings = torch.cat([existing_embeddings, embeddings])

    # Save embeddings and mapping
    torch.save(embeddings, IMAGE_EMBEDDINGS_PATH)
    image_ids.to_csv(IMAGE_EMBEDDINGS_MAP_PATH)


def extract_embeddings(save_every: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # Skip the final FC layer

    model = model.to(device)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    batch_size = 16
    embedding_size = 512
    dataset = ImageDataset(path=FLICKER_IMAGES_PATH, transform=data_transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Set model to evaluation
    model.eval()

    embeddings = []
    image_ids = []

    # Forward images
    for i, batch in enumerate(tqdm(dataloader)):
        images, batch_ids = batch
        images = images.to(device)
        outputs = model(images)

        embeddings.append(outputs)
        image_ids.extend(batch_ids)

        if (i + 1) % save_every == 0:
            print(f'Saving intermediate results. Batch number: {i + 1}')
            save_artifacts(embeddings, image_ids)
            embeddings = []
            image_ids = []

    if len(embeddings) > 0:
        print(f'Saving leftovers...')
        save_artifacts(embeddings, image_ids)
        embeddings = []
        image_ids = []

    image_ids = pd.read_csv(IMAGE_EMBEDDINGS_MAP_PATH).reset_index().set_index('image_id')
    image_ids.to_csv(IMAGE_EMBEDDINGS_MAP_PATH)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-every', type=int, default=10)

    args = parser.parse_args()
    extract_embeddings(save_every=args.save_every)
