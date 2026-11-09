"""Transformer components implemented with basic PyTorch tensor operations."""

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("Embedding dimension must be divisible by head count.")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x, padding_mask, causal=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.n_head, self.head_size)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5
        if causal:
            # Query i can read only keys j <= i, never the next-token target.
            future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        # Mask padded keys while retaining normalized rows at every query.
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = (attention @ v).transpose(1, 2).contiguous()
        return self.projection(context.reshape(batch, length, width)), attention


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_attention = nn.LayerNorm(n_embd)
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        self.ln_ffn = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, padding_mask):
        context, attention = self.attention(self.ln_attention(x), padding_mask)
        x = x + context
        return x + self.feedforward(self.ln_ffn(x)), attention


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, block_size=32, n_embd=64, n_head=2, n_layer=4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [EncoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.block_size:
            raise ValueError("Expected token IDs of shape [batch, 1..block_size].")
        padding_mask = token_ids.eq(0)
        if padding_mask.all(dim=1).any():
            raise ValueError("Each sequence must have at least one non-padding token.")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_maps = []
        for block in self.blocks:
            x, attention = block(x, padding_mask)
            attention_maps.append(attention)
        return self.ln_final(x), attention_maps


class SpeakerClassifier(nn.Module):
    def __init__(self, encoder, n_embd=64, n_hidden=100, n_output=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output)
        )

    def forward(self, token_ids):
        hidden, _ = self.encoder(token_ids)
        valid = token_ids.ne(0).unsqueeze(-1)
        # Average words only: padding is not part of the speech segment.
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1)
        return self.classifier(pooled)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden=100):
        super().__init__()
        self.ln_attention = nn.LayerNorm(n_embd)
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        self.ln_ffn = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_embd)
        )

    def forward(self, x, padding_mask):
        context, attention = self.attention(self.ln_attention(x), padding_mask, causal=True)
        x = x + context
        return x + self.feedforward(self.ln_ffn(x)), attention


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, block_size=32, n_embd=64, n_head=2,
                 n_layer=4, n_hidden=100):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, n_hidden) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.block_size:
            raise ValueError("Expected token IDs of shape [batch, 1..block_size].")
        padding_mask = token_ids.eq(0)
        if padding_mask.all(dim=1).any():
            raise ValueError("Each sequence must have at least one non-padding token.")
        # Left padding can leave early causal queries without an unmasked key.
        if (padding_mask[:, :-1] & ~padding_mask[:, 1:]).any():
            raise ValueError("Decoder inputs must use right padding only.")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_maps = []
        for block in self.blocks:
            x, attention = block(x, padding_mask)
            attention_maps.append(attention)
        return self.ln_final(x), attention_maps


class DecoderLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=32, n_embd=64, n_head=2,
                 n_layer=4, n_hidden=100):
        super().__init__()
        self.decoder = TransformerDecoder(
            vocab_size, block_size, n_embd, n_head, n_layer, n_hidden
        )
        # Untied output weights: every position predicts over the vocabulary.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, token_ids, targets=None):
        hidden, attention_maps = self.decoder(token_ids)
        logits = self.lm_head(hidden)
        if targets is None:
            return logits, attention_maps
        if targets.shape != token_ids.shape:
            raise ValueError("Targets must have the same [batch, length] shape as inputs.")
        # The dataset already shifts targets by one; do not shift a second time.
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
