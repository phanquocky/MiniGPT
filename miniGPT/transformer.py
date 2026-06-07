import torch.nn as nn
from miniGPT.attention import MultiHeadAttention
from miniGPT.layers import FeedForward, LayerNorm

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            context_lengh=config['context_length'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            qkv_bias=config['qkv_bias'])
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config['embedding_dim'])
        self.norm2 = LayerNorm(config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
