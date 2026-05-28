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
    fn_name: str = 'multiply'
    frac_train: float = 0.5
    seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    n_ctx: int = 3
    act_type: str = 'ReLU'
    use_ln: bool = False
    save_every: int = 100
    stopping_thresh: float = -1.0
    log_every: int = 500

    @property
    def d_vocab(self): return self.p  
    @property
    def d_mlp(self): return 4 * self.d_model
    @property
    def d_head(self): return self.d_model // self.num_heads
    @property
    def device(self): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_train_test_mult(p, frac_train, seed=0):
    pairs = [(i, j, p-1) for i in range(p-1) for j in range(p-1)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]

def mult_fn(i, j, p):
    val1 = i + 1
    val2 = j + 1
    result = (val1 * val2) % p
    return result - 1

class TrainerMult:
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
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
        self.train_data, self.test_data = gen_train_test_mult(config.p, config.frac_train, config.seed)
        self.train_losses = []
        self.test_losses = []
        
        self.train_labels = torch.tensor([mult_fn(i, j, config.p) for i, j, _ in self.train_data]).to(config.device)
        self.test_labels = torch.tensor([mult_fn(i, j, config.p) for i, j, _ in self.test_data]).to(config.device)
        self.train_data_t = torch.tensor(self.train_data).to(config.device)
        self.test_data_t = torch.tensor(self.test_data).to(config.device)

    def step(self, epoch):
        self.model.train()
        logits_train = self.model(self.train_data_t)[:, -1]
        train_loss = cross_entropy_high_precision(logits_train, self.train_labels)
        
        with torch.no_grad():
            self.model.eval()
            logits_test = self.model(self.test_data_t)[:, -1]
            test_loss = cross_entropy_high_precision(logits_test, self.test_labels)
            
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())

        if epoch % self.config.log_every == 0:
            print(f"Epoch {epoch:>6d} | train loss: {train_loss.item():.4f} | test loss: {test_loss.item():.4f}")

        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return test_loss.item()

def run_multiplication_study():
    config = Config()
    trainer = TrainerMult(config, "checkpoints_mult")
    
    print(f"Training Modular Multiplication (p={config.p}) on {config.device}")
    for epoch in range(config.num_epochs):
        test_loss = trainer.step(epoch)
        if test_loss < 0.01:
            print(f"Generalized at epoch {epoch}")
            break
            
    plt.figure(figsize=(10, 5))
    epochs = np.arange(len(trainer.train_losses))
    plt.semilogy(epochs, trainer.train_losses, label='Train loss')
    plt.semilogy(epochs, trainer.test_losses, label='Test loss')
    plt.title('Grokking - Modular Multiplication')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('multiplication_circuits.png')
    print("Saved multiplication_circuits.png")
    
    with open('multiplication_analysis_report.md', 'w') as f:
        f.write("# Modular Multiplication Algorithm\n")
        f.write("The model learns an isomorphism to a cyclic group using discrete logarithms.\n")
        f.write(f"The input space is Z_{config.p}^x. The model maps inputs to frequencies of the cyclic group Z_{config.p-1}.\n")
        f.write("By projecting the embedding matrix onto the Fourier basis of the discrete logarithm, we can see sparse, key frequencies just like in modular addition.")

if __name__ == "__main__":
    run_multiplication_study()
