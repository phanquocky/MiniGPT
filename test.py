import torch
from miniGPT.tokenizer import tokenizer
from miniGPT.config import model_config
from miniGPT.model import GPT2

def generate_text_simple(model, idx, max_new_token, context_size):
    for _ in range(max_new_token):
        idx_con = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_con)
        
        logits = logits[:, -1, :]
        prob = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(prob, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim= 1)
    return idx


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded tensor: ", encoded_tensor)

model = GPT2(model_config)
out = generate_text_simple(model, encoded_tensor, 6, model_config['context_length'])
print("OUT: ", out)

out = tokenizer.decode(out.squeeze(0).tolist())
print("OUT: ", out)