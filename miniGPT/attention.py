import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_lengh, dropout, qkv_bias=False):
        super().__init__()
        print("initializing multi head attention with d_in:", d_in, "d_out:", d_out, "num_heads:", num_heads, "context_length:", context_lengh, "dropout:", dropout)
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Dropout = nn.Dropout(dropout)
        self.OutProj = nn.Linear(d_out, d_out) # output projection to combine heads
        self.register_buffer('mask', torch.tril(torch.ones(context_lengh, context_lengh), diagonal=0)) # precompute mask for max context length of 1000

    def forward(self,x):
        batch_size, num_tokens, d_in = x.shape
        querys = self.W_query(x) # shape: (batch_size, num_tokens, d_out = num_heads * head_size)
        keys = self.W_key(x) # shape: (batch_size, num_tokens, d_out = num_heads * head_size)
        values = self.W_value(x) # shape: (batch_size, num_tokens, d_out = num_heads * head_size)
        # print("querys shape:", querys.shape)
        # print("keys shape:", keys.shape)
        # print("values shape:", values.shape)
       
        querys = querys.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1,2) # shape: (batch_size, num_heads, num_tokens, head_size)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1,2) # shape: (batch_size, num_heads, num_tokens, head_size)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_size).transpose(1,2) # shape: (batch_size, num_heads, num_tokens, head_size)

        attn_scores = querys @ keys.transpose(-2,-1) # (batch_size, num_heads, num_tokens, head_size) @ (batch_size, num_heads, head_size, num_tokens) --> (batch_size, num_heads, num_tokens, num_tokens)
        # print("before masked: ", attn_scores)
        masked_attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens] == 0, float('-inf')) # (batch_size, num_heads, num_tokens, num_tokens)
        # print("mask: ", self.mask)
        # print("after masked: ", masked_attn_scores)
        attn_weights = torch.softmax(masked_attn_scores / (keys.shape[-1] ** 0.5), dim=-1) # (batch_size, num_heads, num_tokens, num_tokens)
        attn_weights = self.Dropout(attn_weights) # (batch_size, num_heads, num_tokens, num_tokens)
        context_vector = attn_weights @ values # (batch_size, num_heads, num_tokens, num_tokens) @ (batch_size, num_heads, num_tokens, head_size) --> (batch_size, num_heads, num_tokens, head_size)

        context_vector = context_vector.transpose(1,2) # (batch_size, num_tokens, num_heads, head_size)
        # print("context: ", context_vector)
        #combine heads
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.num_heads * self.head_size) # (batch_size, num_tokens, d_out)
        # print("combined context: ", context_vector)

        context_vector = self.OutProj(context_vector) # (batch_size, num_tokens, d_out)
        return context_vector
