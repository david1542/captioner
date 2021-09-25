import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from pytorch_lightning import LightningModule

from config.paths import FLICKER_IMAGES_PATH
from utils.vocab import Vocabulary


def evaluate_model(model: LightningModule, embeddings: torch.Tensor, captions: torch.Tensor,
                   vocabulary: Vocabulary, max_length=25):
    predicted = model.predict(embeddings, max_length=max_length)
    actual = vocabulary.to_lines(captions)

    # calculate BLEU score
    bleu_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    scores = [bleu_1, bleu_2, bleu_3, bleu_4]
    return scores


def evaluate_on_image(model: LightningModule, sample, vocabulary, max_length=20):
    embedding, caption, image_id = sample
    image_path = os.path.join(FLICKER_IMAGES_PATH, image_id)

    embedding = embedding.unsqueeze(0)
    caption = caption.unsqueeze(0)

    predicted_caption = model.predict(embedding, max_length=max_length)[0]
    actual_caption = vocabulary.to_lines(caption.cpu().numpy())[0]

    title = '\n'.join([
        r'$\bf{' + 'Predicted:' + '}$ ' + predicted_caption,
        r'$\bf{' + 'Actual:' + '}$ ' + actual_caption,
    ])

    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(title)
    plt.show()
