import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Config
from utils import Transformer, cross_entropy_high_precision

def get_fourier_basis(p):
    basis = [torch.ones(p) / np.sqrt(p)]
    for i in range(1, p // 2 + 1):
        cos_comp = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        sin_comp = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        cos_comp /= cos_comp.norm()
        sin_comp /= sin_comp.norm()
        basis.append(cos_comp)
        basis.append(sin_comp)
    return torch.stack(basis, dim=0)

def fourier_2d_basis_term(x_index, y_index, fourier_basis):
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def get_component_cos_xpy(tensor, freq, fourier_basis, p):
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1, fourier_basis).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq, fourier_basis).flatten()
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction) / np.sqrt(2)
    return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, fourier_basis, p):
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1, fourier_basis).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq, fourier_basis).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction) / np.sqrt(2)
    return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)

if __name__ == "__main__":
    save_path = os.path.join("checkpoints", 'full_run_data.pth')
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
    
    p = config.p
    fourier_basis = get_fourier_basis(p)
    
    all_data = torch.tensor([(i, j, p) for i in range(p) for j in range(p)])
    labels = torch.tensor([config.fn(i, j) for i, j, _ in all_data])
    
    epochs = full_run_data['epochs']
    state_dicts = full_run_data['state_dicts']
    
    train_losses = []
    excluded_losses = []
    sum_sq_weights = []
    
    model.load_state_dict(state_dicts[-1])
    W_E = model.embed.W_E[:, :-1]
    fourier_embed = (W_E @ fourier_basis.T).norm(dim=0)
    freq_norms = []
    for freq in range(1, p // 2 + 1):
        freq_norms.append(fourier_embed[2*freq-1]**2 + fourier_embed[2*freq]**2)
    key_freqs = np.argsort(freq_norms)[-5:] + 1  # 1-indexed frequencies
    print("Key frequencies identified:", key_freqs)
    
    print("Evaluating checkpoints to map phases...")
    sampled_indices = range(0, len(state_dicts), max(1, len(state_dicts)//100))
    
    for idx in sampled_indices:
        model.load_state_dict(state_dicts[idx])
        model.eval()
        
        with torch.no_grad():
            logits = model(all_data)[:, -1, :-1]  
            loss = cross_entropy_high_precision(logits, labels).item()
            
            new_logits = logits.clone().reshape(p*p, p)
            for freq in key_freqs:
                new_logits -= get_component_cos_xpy(new_logits, freq, fourier_basis, p)
                new_logits -= get_component_sin_xpy(new_logits, freq, fourier_basis, p)
            
            excl_loss = cross_entropy_high_precision(new_logits, labels).item()
            ssw = sum(param.pow(2).sum().item() for param in model.parameters())
            
            train_losses.append(loss)
            excluded_losses.append(excl_loss)
            sum_sq_weights.append(ssw)
            
    plot_epochs = [epochs[i] for i in sampled_indices]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (log scale)', color=color)
    ax1.semilogy(plot_epochs, train_losses, color=color, label='Train Loss')
    ax1.semilogy(plot_epochs, excluded_losses, color='tab:orange', label='Excluded Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Sum of Squared Weights', color=color)  
    ax2.plot(plot_epochs, sum_sq_weights, color=color, label='L2 Norm of Weights')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title('Three Phases of Grokking (Memorization -> Circuit Formation -> Cleanup)')
    fig.tight_layout()  
    plt.savefig('phases_chart.png')
    plt.close()
    print("Saved phases_chart.png")
