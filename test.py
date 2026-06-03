import torch
from miniGPT.tokenizer import tokenizer
from miniGPT.config import model_config
from miniGPT.model import GPT2
from miniGPT.dataloader import create_dataloader

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


# start_context = "Hello, I am"
# tokenizer = tokenizer()
# tokenids = tokenizer.text_to_token_ids(start_context)
# print("tokenids: ", tokenids)
# model = GPT2(model_config)
# out = generate_text_simple(model, tokenids, 6, model_config['context_length'])
# print("OUT: ", tokenizer.token_ids_to_text(out))

with open('./the-verdict.txt', "r", encoding="utf-8") as f:
    txt = f.read()

dataloader = create_dataloader(txt)
for i, data in iter(dataloader):
    print("I: ", i, ", Data: ", data)