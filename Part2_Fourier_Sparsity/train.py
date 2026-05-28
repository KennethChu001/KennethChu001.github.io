import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils import Transformer, cross_entropy_high_precision, gen_train_test

@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 40_000
    p: int = 113
    fn_name: str = 'add'
    frac_train: float = 0.3
    seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    n_ctx: int = 3
    act_type: str = 'ReLU'
    use_ln: bool = False
    save_every: int = 100
    stopping_thresh: float = -1.0
    log_every: int = 1000

    @property
    def d_vocab(self): return self.p + 1
    @property
    def d_mlp(self): return 4 * self.d_model
    @property
    def d_head(self): return self.d_model // self.num_heads
    @property
    def device(self): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @property
    def fn(self):
        fns = {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
        }
        return fns[self.fn_name]

class Trainer:
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
        self.train_data, self.test_data = gen_train_test(config.p, config.frac_train, config.seed)
        self.train_losses = []
        self.test_losses = []
        self.saved_state_dicts = []
        self.saved_epochs = []
        
        # Pre-compute tensors
        self.train_labels = torch.tensor([config.fn(i, j) for i, j, _ in self.train_data]).to(config.device)
        self.test_labels = torch.tensor([config.fn(i, j) for i, j, _ in self.test_data]).to(config.device)
        self.train_data_t = torch.tensor(self.train_data).to(config.device)
        self.test_data_t = torch.tensor(self.test_data).to(config.device)

    def step(self, epoch):
        self.model.train()
        # Train loss
        logits_train = self.model(self.train_data_t)[:, -1]
        train_loss = cross_entropy_high_precision(logits_train, self.train_labels)
        
        # Test loss
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

    def save_checkpoint(self, epoch):
        sd = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.saved_state_dicts.append(sd)
        self.saved_epochs.append(epoch)

    def save_full_run_data(self):
        import dataclasses
        save_path = os.path.join(self.save_dir, 'full_run_data.pth')
        full_run_data = {
            'state_dicts': self.saved_state_dicts,
            'epochs': self.saved_epochs,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'config': dataclasses.asdict(self.config),
        }
        torch.save(full_run_data, save_path)
        print(f"Saved {len(self.saved_state_dicts)} checkpoints to {save_path}")

def plot_losses(train_losses, test_losses, save_path):
    import numpy as np
    plt.figure(figsize=(10, 5))
    epochs = np.arange(len(train_losses))
    plt.semilogy(epochs, train_losses, label='Train loss', alpha=0.8)
    plt.semilogy(epochs, test_losses, label='Test loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Grokking - Modular Addition (p=113)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    config = Config()
    save_dir = "checkpoints"
    trainer = Trainer(config, save_dir)
    print(f"Training on device: {config.device}")
    
    for epoch in range(config.num_epochs):
        if epoch % config.save_every == 0:
            trainer.save_checkpoint(epoch)
            
        test_loss = trainer.step(epoch)
        if config.stopping_thresh > 0 and test_loss < config.stopping_thresh:
            print(f"Test loss below threshold at epoch {epoch}.")
            break
            
    trainer.save_checkpoint(config.num_epochs)
    trainer.save_full_run_data()
    plot_losses(trainer.train_losses, trainer.test_losses, "loss_curves.png")
    print("Training complete! Plot saved to loss_curves.png")
