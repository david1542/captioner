import torch
from nltk.translate.bleu_score import corpus_bleu
from pytorch_lightning import LightningModule

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
