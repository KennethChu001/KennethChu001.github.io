import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Config
from utils import Transformer

def get_fourier_basis(p):
    basis = [torch.ones(p) / np.sqrt(p)]
    names = ['Const']
    for i in range(1, p // 2 + 1):
        cos_comp = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sin_comp = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        cos_comp /= cos_comp.norm()
        sin_comp /= sin_comp.norm()
        basis.append(cos_comp)
        basis.append(sin_comp)
        names.append(f'cos {i}')
        names.append(f'sin {i}')
    return torch.stack(basis, dim=0), names

def plot_embed_bars(fourier_embed, title, save_path):
    freqs = np.arange(1, (len(fourier_embed) - 1) // 2 + 1)
    cos_norms = fourier_embed[1::2]
    sin_norms = fourier_embed[2::2]
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    plt.bar(freqs - bar_width/2, cos_norms, bar_width, label='cos')
    plt.bar(freqs + bar_width/2, sin_norms, bar_width, label='sin')
    plt.xlabel('Frequency ($w_k$)')
    plt.ylabel('Norm')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    save_dir = "checkpoints"
    save_path = os.path.join(save_dir, 'full_run_data.pth')
    if not os.path.exists(save_path):
        print(f"{save_path} not found. Please run train.py first.")
        exit(1)
        
    full_run_data = torch.load(save_path, weights_only=False, map_location='cpu')
    config_dict = full_run_data['config']
    config = Config(**config_dict)
    
    model = Transformer(
        num_layers=config.num_layers, d_vocab=config.d_vocab,
        d_model=config.d_model, d_mlp=config.d_mlp, d_head=config.d_head,
        num_heads=config.num_heads, n_ctx=config.n_ctx,
        act_type=config.act_type, use_cache=False, use_ln=config.use_ln,
    )
    model.load_state_dict(full_run_data['state_dicts'][-1])
    
    p = config.p
    fourier_basis, fourier_basis_names = get_fourier_basis(p)
    
    # 1D DFT on Embedding matrix
    W_E = model.embed.W_E[:, :-1]  # Exclude the equality token
    fourier_embed = (W_E @ fourier_basis.T).norm(dim=0).detach().cpu().numpy()
    
    plot_embed_bars(fourier_embed, 'Norm of embedding of each Fourier Component', 'fourier_embedding.png')
    print("Saved fourier_embedding.png")
