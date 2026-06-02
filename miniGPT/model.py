import torch
import torch.nn as nn
from miniGPT.transformer import Transformer
from miniGPT.layers import LayerNorm

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.position_embedding = nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout_embedding = nn.Dropout(config['dropout'])
        self.transformer_blocks = nn.Sequential(*[Transformer(config) for _ in range(config['num_layers'])])
        self.final_norm = LayerNorm(config['embedding_dim'])
        self.output_head = nn.Linear(config['embedding_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        position_embeddings = self.position_embedding(torch.arange(seq_length))
        x = self.dropout_embedding(token_embeddings + position_embeddings)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits