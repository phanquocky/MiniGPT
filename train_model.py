import torch
from miniGPT.model import GPT2
from miniGPT.config import model_config

model = GPT2(model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    input = f.read()

# training config, you can change config in self.train_model() function
''' train_ratio = 0.8
    batch_size = 2
    num_epochs = 10
    eval_freq = 5
    eval_iter = 5
    lr = 0.0004
    weight_decay = 0.1'''

model.train_model(input, device)