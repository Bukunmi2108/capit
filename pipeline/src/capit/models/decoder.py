"""Decoder: a recurrent neural network for generating captions."""

from __future__ import annotations
import torch
from torch import nn
from capit.config import config
from capit.models.attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.embed_dim,
        decoder_dim: int = config.decoder_dim,
        attention_dim: int = config.attention_dim,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        self.attention = BahdanauAttention(config.encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + config.encoder_dim, decoder_dim)
        self.init_h = nn.Linear(config.encoder_dim, decoder_dim)
        self.init_c = nn.Linear(config.encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, config.encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h, c
    
    def forward(
        self, features: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        embeddings = self.embedding(captions)
        h, c = self.init_hidden(features)
        decode_lengths = (lengths - 1).tolist()
        if any(a < b for a, b in zip(decode_lengths, decode_lengths[1:])):
            raise ValueError(f"forward expects lengths sorted descending; got {lengths.tolist()}")
        T = max(decode_lengths)
        batch_size, num_pixels, _ = features.size()
        vocab_size = self.fc.out_features
        logits = features.new_zeros(batch_size, T, vocab_size)
        alphas = features.new_zeros(batch_size, T, num_pixels)
        for t in range(T):
            n = sum(l > t for l in decode_lengths)
            context, alpha = self.attention(features[:n], h[:n])
            beta = torch.sigmoid(self.f_beta(h[:n]))
            context = beta * context
            x = torch.cat([embeddings[:n, t], context], dim=1)
            h, c = self.decode_step(x, (h[:n], c[:n]))
            logits[:n, t] = self.fc(self.dropout(h))
            alphas[:n, t] = alpha
        return logits, alphas, decode_lengths

    @torch.no_grad()
    def greedy(
        self, features: torch.Tensor, start_id: int, end_id: int, max_len: int = 50
    ) -> list[list[int]]:
        was_training = self.training
        self.eval()
        batch_size = features.size(0)
        h, c = self.init_hidden(features)
        prev = features.new_full((batch_size,), start_id, dtype=torch.long)
        outputs: list[list[int]] = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        for _ in range(max_len):
            context, _ = self.attention(features, h)
            context = torch.sigmoid(self.f_beta(h)) * context
            x = torch.cat([self.embedding(prev), context], dim=1)
            h, c = self.decode_step(x, (h, c))
            prev = self.fc(h).argmax(dim=1)
            for i in range(batch_size):
                if not done[i]:
                    tok = int(prev[i])
                    if tok == end_id:
                        done[i] = True
                    else:
                        outputs[i].append(tok)
            if all(done):
                break
        if was_training:
            self.train()
        return outputs

