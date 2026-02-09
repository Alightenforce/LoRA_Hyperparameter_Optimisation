# LoRA Hyperparameter Optimization Project

This repository documents the systematic tuning of a Stable Diffusion LoRA. Through 10 iterations, I optimized the relationship between **Network Rank** and **Learning Rate** to achieve maximum stylistic fidelity without model degradation.

## Optimising the model
I tested ten distinct configurations to find the "Sweet Spot."

| Version | Rank ($r$)| Learning Rate | Total Checkpoints | Optimal checkpoint | Result |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **v1** | 32 | 1e-4 | 900 | TBC | TBC | TBC |
| **v2** | 4 | 1e-4 | 500 | TBC | TBC | TBC |
| **v3** | 16 | 1e-4 | 800 | TBC | TBC | TBC |
| **v4** | 16 | 1e-4 | 800 | TBC | TBC | TBC |
| **v5** | 32 | 1.5e-4 | 1200 | TBC | TBC | TBC |
| **v6** | 32 | 1e-4 | 400 | TBC | TBC | TBC |
| **v7** | 32 | 1.5e-4 | 900 | TBC | TBC | TBC |
| **v8** | 64 | 1.5e-4 | 1200 | TBC | TBC | TBC |
| **v9** | 64 | 1e-4 | 1400 | TBC | TBC | TBC |
| **v10** | 64 | 5e-5 | 2500 | TBC | TBC | TBC |

## Visual Comparison
| Best Result (v8) | Second Best (v9) | Third Best (v10) |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/892829f5-9e91-4bdd-9a6d-7863b70cb3c7" width="400"> | <img src="https://github.com/user-attachments/assets/c2ae827c-c631-4b2d-8b21-83275e5d5d87" width="400"> | <img src="https://github.com/user-attachments/assets/c2604dc6-cc78-4d00-bb5e-99d56749f496" width="400"> |

## Configuration
The final optimized parameters are stored in `TBC`.
- **Optimizer:** AdamW8bit
- **Scheduler:** Cosine with restarts
- **Resolution:** 512x512

## Technical Note
The training architecture for this project was scaffolded using AI to ensure best practices, while all hyperparameter experimentation, evaluation, and tuning were conducted manually to achieve the final result.
