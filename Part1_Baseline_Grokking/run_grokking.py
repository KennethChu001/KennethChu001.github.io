import os, sys
from pathlib import Path
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import einops
import random
import time
from dataclasses import dataclass

SAVE_ROOT = Path(os.getcwd()) / 'large_files'
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class CONFIG:
    lr: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 15000
    p: int = 31
    fn_name: str = 'add'
    frac_train: float = 0.3
    seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    n_ctx: int = 3
    act_type: str = 'ReLU'
    use_ln: bool = False
    
    @property
    def d_vocab(self): return self.p + 1
    @property
    def d_mlp(self): return 4 * self.d_model
    @property
    def d_head(self): return self.d_model // self.num_heads
    
    save_every: int = 1000
    stopping_thresh: float = -1.0
    log_every: int = 1000
    
    @property
    def device(self):
        return t.device('cuda' if t.cuda.is_available() else 'cpu')
        
    @property
    def fn(self):
        fns = {'add': lambda x, y: (x + y) % self.p, 'subtract': lambda x, y: (x - y) % self.p}
        return fns[self.fn_name]

config = CONFIG()

def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.to(t.float64), dim=-1)
    prediction_logprobs = t.gather(logprobs, index=labels[:, None], dim=-1)
    return -t.mean(prediction_logprobs)

def gen_train_test(config):
    pairs = [(i, j, config.p) for i in range(config.p) for j in range(config.p)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    def give_name(self, name):
        self.name = name
    def add_hook(self, hook, dir='fwd'):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir == 'fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == 'bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
    def remove_hooks(self, dir='fwd'):
        if dir in ('fwd', 'both'):
            for h in self.fwd_hooks: h.remove()
            self.fwd_hooks = []
        if dir in ('bwd', 'both'):
            for h in self.bwd_hooks: h.remove()
            self.bwd_hooks = []
    def forward(self, x):
        return x

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))
    def forward(self, x):
        return x @ self.W_U

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))
    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
    def forward(self, x):
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(
            F.softmax(self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)), dim=-1)
        )
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        return t.einsum('df,bqf->bqd', self.W_O, z_flat)

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
    def forward(self, x):
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        return t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.hook_resid_pre(x))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=False):
        super().__init__()
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.pos_embed = PosEmbed(max_ctx=n_ctx, d_model=d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, model=[self])
            for _ in range(num_layers)
        ])
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

class Trainer:
    def __init__(self, config, save_root):
        self.config = config
        self.save_root = save_root
        self.model = Transformer(
            num_layers=config.num_layers, d_vocab=config.d_vocab,
            d_model=config.d_model, d_mlp=config.d_mlp, d_head=config.d_head,
            num_heads=config.num_heads, n_ctx=config.n_ctx,
            act_type=config.act_type, use_cache=False, use_ln=config.use_ln,
        )
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step / 10, 1))
        self.train_data, self.test_data = gen_train_test(config)
        self.train_losses = []
        self.test_losses = []

    def full_loss(self, data):
        x = t.tensor(data, device=self.config.device)
        logits = self.model(x)[:, -1]
        labels = t.tensor([self.config.fn(i, j) for i, j, _ in data], dtype=t.long, device=self.config.device)
        return cross_entropy_high_precision(logits, labels)

    def step(self, epoch):
        train_loss = self.full_loss(self.train_data)
        test_loss = self.full_loss(self.test_data)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        if epoch % self.config.log_every == 0:
            print(f"Epoch {epoch:>6d} | train loss {train_loss.item():.4f} | test loss {test_loss.item():.4f}")
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return train_loss, test_loss

trainer = Trainer(config, save_root=SAVE_ROOT)
print("Starting training...")
for epoch in range(config.num_epochs):
    train_loss, test_loss = trainer.step(epoch)
    if config.stopping_thresh > 0 and test_loss.item() < config.stopping_thresh:
        break

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
epochs = np.arange(len(trainer.train_losses))
ax.semilogy(epochs, trainer.train_losses, label='Train loss', alpha=0.8)
ax.semilogy(epochs, trainer.test_losses, label='Test loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title(f'Grokking — {config.fn_name} mod {config.p}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grokking_loss.png')
print("Saved grokking_loss.png")
