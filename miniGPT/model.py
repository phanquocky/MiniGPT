import torch
import torch.nn as nn
from miniGPT.transformer import Transformer
from miniGPT.layers import LayerNorm
from miniGPT.tokenizer import tokenizer
from miniGPT.dataloader import create_dataloader
import numpy as np

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # model tools
        self.tokenizer = tokenizer()
        self.model_config = config

        # model weights
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
    
    def calc_loss_batch(self, input_batch, target_batch, device):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = self(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
        return loss

    def calc_loss_loader(self, data_loader, device, num_batches = None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        
        if num_batches == None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input,target) in enumerate(data_loader):
            if (i < num_batches):
                loss = self.calc_loss_batch(input, target, device)
                total_loss += loss
            else:
                break

        return total_loss/num_batches

    def eval_model(self, train_loader, val_loader, device, eval_iter):
        self.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, device, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, device, num_batches=eval_iter)

        self.train()
        return train_loss, val_loss

    def train_model(self, txt, device):
        self.train()
        # config
        train_ratio = 0.8
        batch_size = 2
        num_epochs = 10
        eval_freq = 5
        eval_iter = 5
        lr = 0.0004
        weight_decay = 0.1
        train_losses, val_losses = [],[]
        global_step = -1

        # step1: create train, val dataloader
        
        split_idx = int(len(txt)*train_ratio)
        train_data = txt[:split_idx]
        val_data = txt[split_idx:]
        train_loader = create_dataloader(train_data, batch_size, max_length=self.model_config['context_length'],
                                         stride=self.model_config['context_length'], drop_last=True, shuffle=True,num_workers=0)
        val_loader = create_dataloader(val_data, batch_size, max_length=self.model_config['context_length'],
                                     stride=self.model_config['context_length'], drop_last=False, shuffle=False,num_workers=0)
      
        # step2: training loop
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            self.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch, device)
                loss.backward()
                optimizer.step()
                
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.eval_model(train_loader, val_loader, device, eval_iter)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch + 1} (Step: {global_step}): ",
                        f"Train loss {train_loss} ",
                        f"val_loss: {val_loss}")
                
            # generate_and_print_sample(model, tokenizer, device, start_context)

        return train_losses, val_losses

    def load_pretrain_weights(self, params):
        self.position_embedding.weight = assign(self.position_embedding.weight, params['wpe'])
        self.token_embedding.weight = assign(self.token_embedding.weight, params['wte'])
        
        for b in range(len(params['blocks'])):
            # weight self-attention block
            q_w, k_w, v_w = np.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis=-1)
            self.transformer_blocks[b].attention.W_query.weight = assign(self.transformer_blocks[b].attention.W_query.weight, q_w.T)
            self.transformer_blocks[b].attention.W_key.weight = assign(self.transformer_blocks[b].attention.W_key.weight, k_w.T)
            self.transformer_blocks[b].attention.W_value.weight = assign(self.transformer_blocks[b].attention.W_value.weight, v_w.T)

            # bias self-attention block
            b_w, b_k, b_v = np.split((params['blocks'][b]['attn']['c_attn'])['b'], 3, axis=-1)
            self.transformer_blocks[b].attention.W_query.bias = assign(self.transformer_blocks[b].attention.W_query.bias, b_w)
            self.transformer_blocks[b].attention.W_key.bias = assign(self.transformer_blocks[b].attention.W_key.bias, b_k)
            self.transformer_blocks[b].attention.W_value.bias = assign(self.transformer_blocks[b].attention.W_value.bias, b_v)

            # OutProj self-attention block
            self.transformer_blocks[b].attention.OutProj.weight = assign(self.transformer_blocks[b].attention.OutProj.weight,
                                                                        params['blocks'][b]['attn']['c_proj']['w'].T)
            self.transformer_blocks[b].attention.OutProj.bias = assign(self.transformer_blocks[b].attention.OutProj.bias,
                                                                        params['blocks'][b]['attn']['c_proj']['b'].T)
            
            # feed forward layers
            self.transformer_blocks[b].feed_forward.layers[0].weight = assign(self.transformer_blocks[b].feed_forward.layers[0].weight,
                                                                    params['blocks'][b]['mlp']['c_fc']['w'].T)
            self.transformer_blocks[b].feed_forward.layers[0].bias = assign(self.transformer_blocks[b].feed_forward.layers[0].bias,
                                                                    params['blocks'][b]['mlp']['c_fc']['b'])
            
            self.transformer_blocks[b].feed_forward.layers[2].weight = assign(self.transformer_blocks[b].feed_forward.layers[2].weight,
                                                                    params['blocks'][b]['mlp']['c_proj']['w'].T)
            self.transformer_blocks[b].feed_forward.layers[2].bias = assign(self.transformer_blocks[b].feed_forward.layers[2].bias,
                                                                    params['blocks'][b]['mlp']['c_proj']['b'])
            
            self.transformer_blocks[b].norm1.scale = assign(self.transformer_blocks[b].norm1.scale,
                                                        params['blocks'][b]['ln_1']['g'])
            self.transformer_blocks[b].norm1.shift = assign(self.transformer_blocks[b].norm1.scale,
                                                        params['blocks'][b]['ln_1']['b'])
            
            self.transformer_blocks[b].norm2.scale = assign(self.transformer_blocks[b].norm2.scale,
                                                        params['blocks'][b]['ln_2']['g'])
            self.transformer_blocks[b].norm2.shift = assign(self.transformer_blocks[b].norm2.scale,
                                                        params['blocks'][b]['ln_2']['b'])
        self.final_norm.scale = assign(self.final_norm.scale, params['g'])
        self.final_norm.shift = assign(self.final_norm.scale, params['b'])
        self.output_head.weight = assign(self.output_head.weight, params['wte'])

    def generate(self, context, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
        idx = self.tokenizer.text_to_token_ids(context)
        for _ in range(max_new_tokens):                      #1
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]

            if top_k is not None:                            #2
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]

                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                )

            if temperature > 0.0:                           #3
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                idx_next = torch.multinomial(
                    probs,
                    num_samples=1
                )
            else:                                            #4
                idx_next = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            if idx_next == eos_id:                          #5
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return self.tokenizer.token_ids_to_text(idx)
        

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
