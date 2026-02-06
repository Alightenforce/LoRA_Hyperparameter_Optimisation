# LoRA Hyperparameter Optimization Project

This repository documents the systematic tuning of a Stable Diffusion LoRA. Through 7 iterations, I optimized the relationship between **Network Rank** and **Learning Rate** to achieve maximum stylistic fidelity without model degradation.

## 📊 The Optimization Journey
I tested seven distinct configurations to find the "Sweet Spot."

| Version | Rank ($r$)| Learning Rate | Total Checkpoints | Optimal checkpoint | Result |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **v1** | 32 | 1e-4 | 900 | TBC | TBC | ❌ |
| **v2** | 4 | 1e-4 | 500 | TBC | TBC | ⚠️ |
| **v3** | 16 | 1e-4 | 800 | TBC | TBC | ❌ |
| **v4** | 16 | 1e-4 | 800 | TBC | TBC | ⚠️ |
| **v5** | **32** | **1.5e-4** | 1200 | TBC | **The Winner: Sharp details, clean anatomy.** | ✅ |
| **v6** | 32 | 1e-4 | 400 | TBC | TBC | ⚠️ |
| **v7** | 32 | 1.5e-4 | 900 | TBC | TBC | ❌ |



## 📸 Visual Comparison
| High LR (v1) | Optimized (v5) | Low Rank (v3) |
| :---: | :---: | :---: |
| ![v1](samples/v1_result.jpg) | ![v5](samples/v5_result.jpg) | ![v3](samples/v3_result.jpg) |

## 🛠️ Configuration ($v5$)
The final optimized parameters are stored in `hparamsV5.yaml`.
- **Optimizer:** AdamW8bit
- **Scheduler:** Cosine with restarts
- **Resolution:** 512x512

## 📝 Technical Note
The training architecture for this project was scaffolded using AI to ensure best practices, while all hyperparameter experimentation, evaluation, and tuning were conducted manually to achieve the final result.
