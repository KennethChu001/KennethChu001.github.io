import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from utils import Transformer, cross_entropy_high_precision
import random

@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 15000
    p: int = 67
    frac_train: float = 0.4
    seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    n_ctx: int = 3
    act_type: str = 'ReLU'
    use_ln: bool = False
    save_every: int = 100
    log_every: int = 500

    @property
    def d_vocab(self): return self.p + 2 
    @property
    def d_mlp(self): return 4 * self.d_model
    @property
    def d_head(self): return self.d_model // self.num_heads
    @property
    def device(self): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_train_test_multitask(p, frac_train, seed=0):
    add_token = p
    mul_token = p + 1
    
    add_pairs = [(i, j, add_token) for i in range(p) for j in range(p)]
    mul_pairs = [(i, j, mul_token) for i in range(p) for j in range(p)]
    
    random.seed(seed)
    random.shuffle(add_pairs)
    random.shuffle(mul_pairs)
    
    div_add = int(frac_train * len(add_pairs))
    div_mul = int(frac_train * len(mul_pairs))
    
    train_data = add_pairs[:div_add] + mul_pairs[:div_mul]
    test_data = add_pairs[div_add:] + mul_pairs[div_mul:]
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data

def eval_fn(i, j, op, p):
    if op == p:
        return (i + j) % p
    else:
        return (i * j) % p

class TrainerCoGrok:
    def __init__(self, config):
        self.config = config
        self.model = Transformer(
            num_layers=config.num_layers, d_vocab=config.d_vocab,
            d_model=config.d_model, d_mlp=config.d_mlp, d_head=config.d_head,
            num_heads=config.num_heads, n_ctx=config.n_ctx,
            act_type=config.act_type, use_cache=False, use_ln=config.use_ln,
        )
        self.model.to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay, betas=(0.9, 0.98),
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / 10, 1)
        )
        self.train_data, self.test_data = gen_train_test_multitask(config.p, config.frac_train, config.seed)
        
        self.train_labels = torch.tensor([eval_fn(i, j, op, config.p) for i, j, op in self.train_data]).to(config.device)
        self.test_labels = torch.tensor([eval_fn(i, j, op, config.p) for i, j, op in self.test_data]).to(config.device)
        self.train_data_t = torch.tensor(self.train_data).to(config.device)
        self.test_data_t = torch.tensor(self.test_data).to(config.device)
        
        self.test_is_add = (self.test_data_t[:, 2] == config.p)
        self.test_is_mul = (self.test_data_t[:, 2] == config.p + 1)
        
        self.add_losses = []
        self.mul_losses = []

    def step(self, epoch):
        self.model.train()
        logits_train = self.model(self.train_data_t)[:, -1]
        train_loss = cross_entropy_high_precision(logits_train, self.train_labels)
        
        with torch.no_grad():
            self.model.eval()
            logits_test = self.model(self.test_data_t)[:, -1]
            
            logits_add = logits_test[self.test_is_add]
            labels_add = self.test_labels[self.test_is_add]
            add_loss = cross_entropy_high_precision(logits_add, labels_add)
            
            logits_mul = logits_test[self.test_is_mul]
            labels_mul = self.test_labels[self.test_is_mul]
            mul_loss = cross_entropy_high_precision(logits_mul, labels_mul)
            
        self.add_losses.append(add_loss.item())
        self.mul_losses.append(mul_loss.item())

        if epoch % self.config.log_every == 0:
            print(f"Epoch {epoch:>6d} | train: {train_loss.item():.4f} | add: {add_loss.item():.4f} | mul: {mul_loss.item():.4f}")

        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

def run_cogrokking_study():
    config = Config()
    trainer = TrainerCoGrok(config)
    
    print(f"Training Multi-task (p={config.p}) on {config.device}")
    for epoch in range(config.num_epochs):
        trainer.step(epoch)
            
    plt.figure(figsize=(10, 5))
    epochs = np.arange(len(trainer.add_losses))
    plt.semilogy(epochs, trainer.add_losses, label='Test Loss (Addition)')
    plt.semilogy(epochs, trainer.mul_losses, label='Test Loss (Multiplication)')
    plt.title('Co-Grokking: Multi-task Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('co_grokking_curves.png')
    print("Saved co_grokking_curves.png")

if __name__ == "__main__":
    run_cogrokking_study()
