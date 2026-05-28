import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import Transformer, cross_entropy_high_precision
import random

# Use ASCII characters to avoid UnicodeEncodeError in Windows console
def plot_bars(norms, title, save_path):
    freqs = np.arange(1, len(norms) + 1)
    plt.figure(figsize=(12, 6))
    plt.bar(freqs, norms, color='blue', alpha=0.7)
    plt.xlabel('Frequency (k)')
    plt.ylabel('Norm')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def get_primitive_root(p):
    # Find a primitive root modulo p
    # For p=67, we know 2 is a primitive root.
    return 2

def get_discrete_log_mapping(p, g):
    # Returns a dictionary mapping a -> x where g^x = a (mod p)
    # for a in {1, ..., p-1}, x in {0, ..., p-2}
    mapping = {}
    current = 1
    for x in range(p - 1):
        mapping[current] = x
        current = (current * g) % p
    return mapping

def get_discrete_log_fourier_basis(p, mapping):
    # Creates a Fourier basis over Z_{p-1} mapped to the original vocabulary indices
    # Vocab indices are 0 to p-2, representing numbers 1 to p-1.
    basis = [torch.ones(p - 1) / np.sqrt(p - 1)]
    for i in range(1, (p - 1) // 2 + 1):
        cos_comp = torch.zeros(p - 1)
        sin_comp = torch.zeros(p - 1)
        for a_idx in range(p - 1):
            a = a_idx + 1 # actual number
            x = mapping[a] # discrete log
            cos_comp[a_idx] = np.cos(2 * np.pi * i * x / (p - 1))
            sin_comp[a_idx] = np.sin(2 * np.pi * i * x / (p - 1))
        
        # Normalize
        if cos_comp.norm() > 1e-6:
            cos_comp /= cos_comp.norm()
        if sin_comp.norm() > 1e-6:
            sin_comp /= sin_comp.norm()
            
        basis.append(cos_comp)
        basis.append(sin_comp)
    return torch.stack(basis, dim=0)

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

def main():
    p = 67
    frac_train = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} with p={p}, frac_train={frac_train}")
    
    # Model Config
    d_vocab = p
    d_model = 128
    d_head = 32
    d_mlp = 512
    num_heads = 4
    num_layers = 1
    n_ctx = 3
    act_type = 'ReLU'
    use_ln = False
    
    model = Transformer(
        num_layers=num_layers, d_vocab=d_vocab,
        d_model=d_model, d_mlp=d_mlp, d_head=d_head,
        num_heads=num_heads, n_ctx=n_ctx,
        act_type=act_type, use_cache=False, use_ln=use_ln,
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3,
        weight_decay=1.0, betas=(0.9, 0.98),
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
    
    train_data, test_data = gen_train_test_mult(p, frac_train)
    train_labels = torch.tensor([mult_fn(i, j, p) for i, j, _ in train_data]).to(device)
    test_labels = torch.tensor([mult_fn(i, j, p) for i, j, _ in test_data]).to(device)
    train_data_t = torch.tensor(train_data).to(device)
    test_data_t = torch.tensor(test_data).to(device)
    
    print("Starting training...")
    for epoch in range(1, 15001):
        model.train()
        logits_train = model(train_data_t)[:, -1]
        train_loss = cross_entropy_high_precision(logits_train, train_labels)
        
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if epoch % 500 == 0:
            with torch.no_grad():
                model.eval()
                logits_test = model(test_data_t)[:, -1]
                test_loss = cross_entropy_high_precision(logits_test, test_labels)
            print(f"Epoch {epoch:5d} | Train: {train_loss.item():.6f} | Test: {test_loss.item():.6f}")
            if test_loss.item() < 0.01:
                print(f"Grokking threshold reached at epoch {epoch}!")
                break

    print("Training complete. Analyzing weights...")
    
    # Analyze discrete log fourier basis
    g = get_primitive_root(p)
    mapping = get_discrete_log_mapping(p, g)
    basis = get_discrete_log_fourier_basis(p, mapping).to(device)
    
    # Get embedding matrix (excluding the operator token p-1)
    W_E = model.embed.W_E[:, :-1]
    
    # Project embedding onto discrete log fourier basis
    # W_E is (d_model, p-1). basis is (p-1, p-1)
    projection = (W_E @ basis.T).norm(dim=0).detach().cpu().numpy()
    
    # Calculate norm of each frequency pair
    freq_norms = []
    for freq in range(1, (p - 1) // 2 + 1):
        idx_cos = 2 * freq - 1
        idx_sin = 2 * freq
        norm_sq = projection[idx_cos]**2 + projection[idx_sin]**2
        freq_norms.append(np.sqrt(norm_sq))
        
    # Plot it
    plot_bars(freq_norms, f'Norm of Embedding in Discrete Log Fourier Basis (p={p})', 'discrete_log_sparsity.png')
    
    # Top frequencies
    top_k = np.argsort(freq_norms)[-5:][::-1] + 1
    print("Top 5 dominant frequencies in discrete log space:", top_k)
    
if __name__ == '__main__':
    main()
