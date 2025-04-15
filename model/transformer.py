# model/transformer.py

import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        # Residual + LayerNorm + Attention
        x = x + self.attn(self.ln1(x))
        # Residual + LayerNorm + MLP
        x = x + self.ff(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class Config:
    vocab_size = 50257        # GPT-2 tokenizer vocab size
    block_size = 128          # max sequence length
    n_layer = 6               # number of transformer blocks
    n_head = 6                # number of attention heads
    n_embd = 384              # embedding dimension (should be divisible by n_head)

class IronMindGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm before output
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head — projects back to vocab
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        # Get embeddings
        tok_emb = self.token_embedding(idx)               # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)            # (1, T, n_embd)

        x = tok_emb + pos_emb                             # (B, T, n_embd)
        x = self.blocks(x)                                # (B, T, n_embd)
        x = self.ln_f(x)                                  # (B, T, n_embd)

        logits = self.head(x)                             # (B, T, vocab_size)
        return logits
