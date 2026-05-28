# Grokking Research Repository

This repository contains code and experimental results for reproducing and studying the phenomenon of **Grokking** (delayed generalization) in neural networks. 

## 🌐 Online Report
A highly polished, publicly accessible online report presenting our findings is available via **GitHub Pages**.

👉 **[View the Research Report Here](https://KennethChu001.github.io/)**

## Repository Structure

The research is divided into 6 distinct parts, covering different aspects of the grokking phenomenon. Code to run each part's experiments is included in its respective directory:

- **[📁 Part 1: Baseline Grokking](./Part1_Baseline_Grokking)** — Standard setup showing the baseline double-descent in time on modular addition.
- **[📁 Part 2: Fourier Sparsity](./Part2_Fourier_Sparsity)** — Analysis of the discrete Fourier transform and how the model discovers sparse Fourier circuits.
- **[📁 Part 3: Three Phases](./Part3_Three_Phases)** — The decomposition of training into Memorization, Circuit Formation, and Cleanup via restricted/excluded loss.
- **[📁 Part 4: Data Scarcity](./Part4_Data_Scarcity)** — Fraction sweep experiments identifying the transition threshold between non-grokking and grokking regimes.
- **[📁 Part 5: AI Alignment](./Part5_AI_Alignment)** — Discussion on how mechanistic interpretability of grokking informs AI Alignment and inner alignment testing.
- **[📁 Part 6: Modular Multiplication](./Part6_Modular_Multiplication)** — Grokking on a harder group structure requiring discrete-logarithm mapping, and multi-task co-grokking experiments.

## Running the Code
Each sub-directory contains Python scripts designed to run on PyTorch.
Refer to the `README.md` in each individual directory for execution instructions and hyperparameters.

## Serving the Report Locally
To view the report locally before pushing to GitHub Pages, you can use any static server from the `docs` folder:
```bash
npx serve docs
# or
python -m http.server 8000 -d docs
```
