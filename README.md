# Class-Dependent Gamma Focal Loss (CDG-FL)

A **stable and frequency-aware extension of Focal Loss** designed for long-tailed visual recognition tasks.  
CDG-FL replaces the single global focusing parameter (γ) with **bounded, class-dependent γ values** computed from empirical class frequencies, along with a **cosine warm-up** to prevent early gradient suppression.

This repository contains the implementation, experiments, and evaluation pipeline for CDG-FL.

## Motivation

Long-tailed datasets suffer because:
- Head classes dominate training
- Tail classes receive weak gradients
- Standard Cross-Entropy overfits head classes
- Vanilla Focal Loss uses a **single γ**, which is inflexible and unstable for heterogeneous class frequencies

**CDG-FL solves this by:**
- Assigning each class a **frequency-aware γc**
- Using a **piecewise log/linear mapping** to keep γc stable and bounded
- Applying a **cosine warm-up during early epochs** to avoid optimization collapse

## Key Idea

The focusing factor for each class is:

γ_c = clamp(raw_c, γ_min, γ_max)

Raw score:

raw_c = log(1/p_c) if p_c > τ  
raw_c = log(1/τ) + k(p_c − τ) if p_c ≤ τ

Warm-up:

γ_c(e) = w(e) * γ_c  
w(e) = 0.5 * (1 − cos(πe / E_w))

Final Loss:

L = −(1 − p_t)^(γ_c(e)) * log(p_t)


## Results

Balanced & long‑tailed benchmark results included in paper.

## Citation

  Class-Dependent Gamma Focal Loss,
  Jagati Brainard Philemon,
  year = 2025
