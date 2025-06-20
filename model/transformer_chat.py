import math
import torch.nn as nn
from torch.nn import Transformer
from .utils import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.transformer = Transformer(d_model, nhead,
                                       num_encoder_layers, num_decoder_layers,
                                       dim_feedforward, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.embedding.embedding_dim))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim))
        out = self.transformer(src_emb, tgt_emb,
                               src_mask, tgt_mask,
                               None, src_key_padding_mask, tgt_key_padding_mask, None)
        return self.fc_out(out)