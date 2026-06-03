import tiktoken
import torch

class tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def get_tokenizer(self):
        return self.tokenizer

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor =  torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor
    
    def token_ids_to_text(self, ids):
        return self.tokenizer.decode(ids.squeeze(0).tolist())