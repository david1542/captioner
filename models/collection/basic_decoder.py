import torch
from torch import nn
import torch.nn.functional as F

from models.collection.base_module import BaseModule
from models.device import device
from utils.data import get_embedding_size

base_config = {
    'dropout_input_rate': 0.5,
    'dropout_hidden_rate': 0.2,
    'hidden_size': 64,
}


class BasicDecoder(BaseModule):
    def __init__(self, vocabulary, *args, **kwargs):
        super().__init__(vocabulary, *args, **kwargs)
        self.learning_rate = kwargs['learning_rate']
        self.image_emb_size = get_embedding_size()

        self.emb_out = nn.Embedding(len(self.vocabulary), kwargs['hidden_size'])

        self.dropout_input = nn.Dropout(p=kwargs['dropout_input_rate'])
        self.dropout_hidden = nn.Dropout(p=kwargs['dropout_hidden_rate'])

        self.dec0 = nn.GRUCell(kwargs['hidden_size'], self.image_emb_size)
        self.logits = nn.Linear(self.image_emb_size, len(self.vocabulary))

    def forward(self, embeddings, captions):
        """ Apply model in training mode """
        embeddings = self.dropout_input(embeddings)
        return self.decode(embeddings, captions)

    def decode(self, initial_state, out_tokens):
        """ Iterate over reference tokens (out_tokens) with decode_step """
        batch_size = out_tokens.shape[0]
        state = initial_state

        # initial logits: always predict BOS
        onehot_bos = F.one_hot(torch.full([batch_size], self.vocabulary.bos_ix, dtype=torch.int64),
                               num_classes=len(self.vocabulary)).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)

        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_step(self, prev_state, prev_tokens):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(inp_voc)]
        """
        embedded_tokens = self.emb_out(prev_tokens)  # batch_size X emb_size
        embedded_tokens = self.dropout_hidden(embedded_tokens)

        new_dec_state = self.dec0(embedded_tokens, prev_state)
        output_logits = self.logits(new_dec_state)
        return new_dec_state, output_logits

    def decode_inference(self, initial_state, max_length):
        """ Generate solutions from model (greedy version) """
        batch_size = len(initial_state)
        state = initial_state
        outputs = [torch.full([batch_size], self.vocabulary.bos_ix, dtype=torch.int64,
                              device=device)]
        all_states = [initial_state]

        for i in range(max_length):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)
        return torch.stack(outputs, dim=1), all_states

    def predict(self, embeddings, max_length=20):
        captions, _ = self.create_captions(embeddings, max_length)
        return captions

    def create_captions(self, embeddings, max_length):
        out_ids, states = self.decode_inference(embeddings, max_length)
        return self.vocabulary.to_lines(out_ids.cpu().numpy()), states

    def compute_loss(self, embeddings, captions):
        outputs = self(embeddings, captions)
        outputs = outputs.view(-1, len(self.vocabulary))
        captions = captions.view(-1)

        return F.cross_entropy(outputs, captions)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_arguments(parser):
        parser = super(BasicDecoder, BasicDecoder).add_arguments(parser)
        parser.add_argument('--drop-input-rate', type=float)
        parser.add_argument('--drop-hidden-rate', type=float)
        return parser

    @classmethod
    def get_config_from_args(cls, args):
        base_args_keys = list(base_config.keys())
        parent_config = super().get_config_from_args(args)
        config = {**parent_config, **base_config}
        for key in base_args_keys:
            if key in args:
                config[key] = args[key]
        return config

