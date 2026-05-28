"""
Task 4 – Data-fraction sweep: "Epochs until Generalization" vs training-data fraction.
Uses p=31 and well-tuned hyperparameters so grokking actually happens.
Fractions tested: 10%, 30%, 60%, 90%.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Hyperparameters ────────────────────────────────────────────────────────────
P           = 31       # mod prime — small enough to train fast on CPU
D_MODEL     = 64
D_MLP       = 4 * D_MODEL
NUM_HEADS   = 4
D_HEAD      = D_MODEL // NUM_HEADS
LR          = 1e-3
WD          = 1.0       # weight-decay is the key driver of grokking
MAX_EPOCHS  = 25_000
GROK_THRESH = 0.05      # test-loss threshold to declare "grokked"
LOG_EVERY   = 500
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRACTIONS   = [0.10, 0.30, 0.60, 0.90]
PALETTE     = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

print(f"Device: {DEVICE} | p={P} | max_epochs={MAX_EPOCHS:,}")

# ── Minimal 1-layer transformer ────────────────────────────────────────────────
class GrokkingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab = P + 1
        self.W_E   = nn.Parameter(torch.randn(D_MODEL, vocab)       / D_MODEL**0.5)
        self.W_pos = nn.Parameter(torch.randn(3, D_MODEL)           / D_MODEL**0.5)
        self.W_K   = nn.Parameter(torch.randn(NUM_HEADS, D_HEAD, D_MODEL) / D_MODEL**0.5)
        self.W_Q   = nn.Parameter(torch.randn(NUM_HEADS, D_HEAD, D_MODEL) / D_MODEL**0.5)
        self.W_V   = nn.Parameter(torch.randn(NUM_HEADS, D_HEAD, D_MODEL) / D_MODEL**0.5)
        self.W_O   = nn.Parameter(torch.randn(D_MODEL, D_MODEL)     / D_MODEL**0.5)
        self.W_in  = nn.Parameter(torch.randn(D_MLP, D_MODEL)       / D_MODEL**0.5)
        self.b_in  = nn.Parameter(torch.zeros(D_MLP))
        self.W_out = nn.Parameter(torch.randn(D_MODEL, D_MLP)       / D_MODEL**0.5)
        self.b_out = nn.Parameter(torch.zeros(D_MODEL))
        self.W_U   = nn.Parameter(torch.randn(D_MODEL, vocab)       / vocab**0.5)
        self.register_buffer("mask", torch.tril(torch.ones(3, 3)))

    def forward(self, x):
        S   = x.shape[1]
        emb = torch.einsum("dv,bsv->bsd", self.W_E, F.one_hot(x, P + 1).float())
        emb = emb + self.W_pos[:S]
        # Multi-head attention
        k = torch.einsum("hfd,bsd->bshf", self.W_K, emb)
        q = torch.einsum("hfd,bsd->bshf", self.W_Q, emb)
        v = torch.einsum("hfd,bsd->bshf", self.W_V, emb)
        sc = torch.einsum("bqhf,bkhf->bqkh", q, k) / D_HEAD**0.5
        sc = sc - 1e9 * (1 - self.mask[:S, :S].unsqueeze(0).unsqueeze(-1))
        attn = F.softmax(sc, dim=2)
        z    = torch.einsum("bqkh,bkhf->bqhf", attn, v)
        z    = z.reshape(z.shape[0], z.shape[1], -1)          # concat heads
        resid = emb + z @ self.W_O.T
        # MLP
        pre  = resid @ self.W_in.T + self.b_in
        out  = resid + F.relu(pre) @ self.W_out.T + self.b_out
        return out @ self.W_U                                  # [B, S, vocab]


def ce_loss(logits, labels):
    logp = F.log_softmax(logits.double(), dim=-1)
    return -torch.mean(torch.gather(logp, 1, labels[:, None]))


def build_dataset():
    pairs = [(i, j) for i in range(P) for j in range(P)]
    random.seed(42)
    random.shuffle(pairs)
    return pairs


def run_fraction(frac, all_pairs):
    n      = max(2, int(frac * len(all_pairs)))
    train  = all_pairs[:n]
    test   = all_pairs[n:]
    print(f"\n-- Fraction {frac:.0%}  train={len(train)}  test={len(test)} --")

    x_tr = torch.tensor([[a, b, P] for a, b in train], dtype=torch.long, device=DEVICE)
    y_tr = torch.tensor([(a + b) % P for a, b in train], dtype=torch.long, device=DEVICE)
    x_te = torch.tensor([[a, b, P] for a, b in test],  dtype=torch.long, device=DEVICE)
    y_te = torch.tensor([(a + b) % P for a, b in test], dtype=torch.long, device=DEVICE)

    model = GrokkingTransformer().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))

    grok_ep      = None
    curve_ep     = []
    curve_tr     = []
    curve_te     = []

    for ep in range(MAX_EPOCHS):
        model.train()
        logits_tr = model(x_tr)[:, -1]
        loss_tr   = ce_loss(logits_tr, y_tr)
        loss_tr.backward()
        opt.step(); sched.step(); opt.zero_grad()

        if ep % LOG_EVERY == 0 or ep == MAX_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                loss_te = ce_loss(model(x_te)[:, -1], y_te).item() if len(test) > 0 else 0.0
            loss_tr_val = loss_tr.item()
            curve_ep.append(ep)
            curve_tr.append(loss_tr_val)
            curve_te.append(loss_te)

            if ep % (LOG_EVERY * 5) == 0:
                print(f"  ep {ep:6d} | train {loss_tr_val:.4f} | test {loss_te:.4f}")

            if grok_ep is None and loss_te < GROK_THRESH and len(test) > 0:
                grok_ep = ep
                print(f"  *** GROKKED @ epoch {ep:,} | test_loss={loss_te:.4f} ***")
                break

    if grok_ep is None:
        print(f"  No grokking in {MAX_EPOCHS:,} epochs (final test={curve_te[-1]:.4f})")

    return grok_ep, list(zip(curve_ep, curve_tr, curve_te))


# ── Run all fractions ──────────────────────────────────────────────────────────
all_pairs = build_dataset()
results   = {}
for frac in FRACTIONS:
    results[frac] = run_fraction(frac, all_pairs)


# ── Plotting ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

ax_bar  = fig.add_subplot(gs[0, 0])
ax_te   = fig.add_subplot(gs[0, 1])
ax_tr   = fig.add_subplot(gs[1, 0])
ax_both = fig.add_subplot(gs[1, 1])

for ax in [ax_bar, ax_te, ax_tr, ax_both]:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

# ── 1) Bar: Epochs to grokking ─────────────────────────────────────────────────
bar_vals = [results[f][0] if results[f][0] is not None else MAX_EPOCHS for f in FRACTIONS]
bar_cols = [PALETTE[i] if results[f][0] is not None else "#555" for i, f in enumerate(FRACTIONS)]
xlabels  = [f"{int(f*100)}%" for f in FRACTIONS]

bars = ax_bar.bar(xlabels, bar_vals, color=bar_cols, edgecolor="#30363d", linewidth=0.8, width=0.55)
for i, (h, f) in enumerate(zip(bar_vals, FRACTIONS)):
    lbl = f"{h:,}" if results[f][0] else f">{MAX_EPOCHS:,}\n(No Grok)"
    ax_bar.text(i, h + MAX_EPOCHS * 0.02, lbl, ha="center", va="bottom",
                fontsize=9, color="white")
ax_bar.set_ylim(0, MAX_EPOCHS * 1.25)
ax_bar.set_xlabel("Training Data Fraction", fontsize=11)
ax_bar.set_ylabel("Epochs until Generalization", fontsize=11)
ax_bar.set_title("Epochs to Grokking by Data Fraction", fontsize=12, fontweight="bold")
ax_bar.grid(axis="y", alpha=0.2, color="white")
grok_patch  = mpatches.Patch(color=PALETTE[2], label="Grokked")
nogrok_patch = mpatches.Patch(color="#555", label="Did not grok")
ax_bar.legend(handles=[grok_patch, nogrok_patch], facecolor="#21262d", labelcolor="white",
              edgecolor="#30363d")

# ── 2) Test loss curves ────────────────────────────────────────────────────────
for i, (frac, (ge, curve)) in enumerate(results.items()):
    ep, _, te = zip(*curve)
    ax_te.semilogy(ep, te, label=f"{int(frac*100)}%", color=PALETTE[i], linewidth=2)
    if ge is not None:
        ax_te.axvline(ge, color=PALETTE[i], linestyle="--", alpha=0.5, linewidth=1.2)
ax_te.axhline(GROK_THRESH, color="white", linestyle=":", linewidth=1.3,
              label=f"Threshold ({GROK_THRESH})")
ax_te.set_xlabel("Epoch", fontsize=11)
ax_te.set_ylabel("Test Loss (log)", fontsize=11)
ax_te.set_title("Test Loss Curves\n(dashed = grokking moment)", fontsize=12, fontweight="bold")
leg = ax_te.legend(title="Train Fraction", facecolor="#21262d", labelcolor="white",
                   edgecolor="#30363d")
leg.get_title().set_color("white")
ax_te.grid(alpha=0.2, color="white")

# ── 3) Train loss curves ───────────────────────────────────────────────────────
for i, (frac, (ge, curve)) in enumerate(results.items()):
    ep, tr, _ = zip(*curve)
    ax_tr.semilogy(ep, tr, label=f"{int(frac*100)}%", color=PALETTE[i], linewidth=2)
ax_tr.set_xlabel("Epoch", fontsize=11)
ax_tr.set_ylabel("Train Loss (log)", fontsize=11)
ax_tr.set_title("Train Loss Curves", fontsize=12, fontweight="bold")
leg = ax_tr.legend(title="Train Fraction", facecolor="#21262d", labelcolor="white",
                   edgecolor="#30363d")
leg.get_title().set_color("white")
ax_tr.grid(alpha=0.2, color="white")

# ── 4) Generalization gap (test-train) ─────────────────────────────────────────
for i, (frac, (ge, curve)) in enumerate(results.items()):
    ep, tr, te = zip(*curve)
    gap = [t - r for t, r in zip(te, tr)]
    ax_both.semilogy(ep, [abs(g) for g in gap], label=f"{int(frac*100)}%",
                     color=PALETTE[i], linewidth=2)
    if ge is not None:
        ax_both.axvline(ge, color=PALETTE[i], linestyle="--", alpha=0.5, linewidth=1.2)
ax_both.set_xlabel("Epoch", fontsize=11)
ax_both.set_ylabel("|Test − Train| Loss (log)", fontsize=11)
ax_both.set_title("Generalization Gap\n(collapses at grokking moment)", fontsize=12, fontweight="bold")
leg = ax_both.legend(title="Train Fraction", facecolor="#21262d", labelcolor="white",
                     edgecolor="#30363d")
leg.get_title().set_color("white")
ax_both.grid(alpha=0.2, color="white")

fig.suptitle(
    f"Grokking & Data Scarcity  —  (a+b) mod {P}  |  d={D_MODEL}  |  wd={WD}  |  lr={LR}",
    fontsize=14, fontweight="bold", color="white", y=0.98,
)
plt.savefig("grokking_fraction_v2.png", dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved grokking_fraction_v2.png")
