import pytorch_lightning as pl
from clearml import Logger as ClearMLLogger

from utils.evaluate import evaluate_model


class BaseModule(pl.LightningModule):
    def __init__(self, vocabulary, eval_max_length=25):
        super().__init__()
        self.vocabulary = vocabulary
        self.eval_max_length = eval_max_length

    def training_step(self, batch, *args):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, *args):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss)
        self.evaluate_bleu('val', batch)
        return loss

    def test_step(self, batch, *args):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss)
        self.evaluate_bleu('test', batch)
        return loss

    def evaluate_bleu(self, phase, batch):
        embeddings, captions, _ = batch
        bleu_scores = evaluate_model(self, embeddings=embeddings, captions=captions,
                                     vocabulary=self.vocabulary, max_length=self.eval_max_length)

        logger: ClearMLLogger = self.logger.experiment
        graph_name = f'{phase}_bleu_scores'
        for i, score in enumerate(bleu_scores):
            series_name = f'bleu_{i}_gram'
            logger.report_scalar(graph_name, series_name, value=score)