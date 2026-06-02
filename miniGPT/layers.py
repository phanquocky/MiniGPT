import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        x = self.scale * normalized + self.shift
        return x
     
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embedding_dim'], 4 * config['embedding_dim']),
            nn.GELU(),
            nn.Linear(4 * config['embedding_dim'], config['embedding_dim'])
        )
    def forward(self, x):
        return self.layers(x)