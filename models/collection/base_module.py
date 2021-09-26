import pytorch_lightning as pl
from utils.evaluate import evaluate_model, evaluate_on_image

base_config = {
    'eval_image_rate': 100,
    'eval_max_length': 25,
    'learning_rate': 1e-3,
    'image_emb_size': 128
}


class BaseModule(pl.LightningModule):
    def __init__(self, vocabulary, *args, **kwargs):
        super().__init__()
        self.vocabulary = vocabulary
        self.eval_max_length = kwargs['eval_max_length']
        self.eval_image_rate = kwargs['eval_image_rate']
        self.learning_rate = kwargs['learning_rate']

    def training_step(self, batch, *args):
        loss = self._calculate_loss('train', batch['train'], check_bleu_metrics=False)

        if self.global_step > 0 and self.global_step % self.eval_image_rate == 0:
            self._evaluate_image(batch['ref'])
        return loss

    def validation_step(self, batch, *args):
        return self._calculate_loss('valid', batch)

    def test_step(self, batch, *args):
        return self._calculate_loss('test', batch)

    def _calculate_loss(self, phase, batch, check_bleu_metrics=True):
        embeddings, captions, _ = batch
        loss = self.compute_loss(embeddings, captions)
        self.log(f'{phase}_loss', loss)

        if check_bleu_metrics:
            self._evaluate_bleu(phase, batch)
        return loss

    def _evaluate_bleu(self, phase, batch):
        embeddings, captions, _ = batch
        bleu_scores = evaluate_model(self, embeddings=embeddings, captions=captions,
                                     vocabulary=self.vocabulary, max_length=self.eval_max_length)
        score_data = {f'bleu_{i}_gram': value for i, value in enumerate(bleu_scores)}
        self.log(f'{phase}_bleu_scores', score_data)

    def _evaluate_image(self, batch):
        # Take the first sample from the reference batch
        embeddings, captions, image_ids = batch
        sample = (embeddings[0], captions[0], image_ids[0])

        # Forward the sample to the evaluate_on_image method
        evaluate_on_image(self, sample, self.vocabulary, self.eval_max_length)

    @staticmethod
    def add_arguments(parser):
        for key, value in base_config.items():
            arg_name = f"--{key.replace('_', '-')}"
            parser.add_argument(arg_name, type=type(value), default=value)
        return parser

    @classmethod
    def get_config_from_args(cls, args):
        base_args_keys = list(base_config.keys())
        config = {**base_config}
        for key in base_args_keys:
            if key in args:
                config[key] = args[key]
        return config

    @classmethod
    def from_arguments(cls, vocabulary, args):
        config = cls.get_config_from_args(args)
        return cls(vocabulary=vocabulary, **config)
