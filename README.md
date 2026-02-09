# LoRA Hyperparameter Optimization Project

This repository documents the systematic tuning of a Stable Diffusion LoRA. Through 10 iterations, I optimized the relationship between **Network Rank** and **Learning Rate** to achieve maximum stylistic fidelity without model degradation.

## Optimising the model
I tested ten distinct configurations to find the "Sweet Spot."

| Version | Rank ($r$)| Learning Rate | Total Checkpoints | Optimal checkpoint | Result |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **v1** | 32 | 1e-4 | 900 | TBC | TBC | 
| **v2** | 4 | 1e-4 | 500 | TBC | TBC | 
| **v3** | 16 | 1e-4 | 800 | TBC | TBC | 
| **v4** | 16 | 1e-4 | 800 | TBC | TBC | 
| **v5** | 32 | 1.5e-4 | 1200 | 750 | Refer to v7 | 
| **v6** | 32 | 1e-4 | 400 | TBC | TBC |
| **v7** | 32 | 1.5e-4 | 900 | 750 | Facial structure extremely accurate to the original; however, obvious eye issues caused by low rank | 
| **v8** | 64 | 1.5e-4 | 1200 | 800 | Extremely accurate to the original | 
| **v9** | 64 | 1e-4 | 1400 | 800 | Very accurate to the original; however, slight, noticeable burning | 
| **v10** | 64 | 5e-5 | 3000 | 2750 | Fairly accurate to the original | 

## Note
v1 - v4 has unpadded images and worse tagging

## Visual Comparison

### 1. Best Result (High Rank, Fast Learn Rate) (v8)
<img src="https://github.com/user-attachments/assets/892829f5-9e91-4bdd-9a6d-7863b70cb3c7" width="100%">

---

### 2. High Rank, Medium Learn Rate (v9)
<img src="https://github.com/user-attachments/assets/c2ae827c-c631-4b2d-8b21-83275e5d5d87" width="100%">

---

### 3. High Rank, Low Learn Rate (v10)
<img src="https://github.com/user-attachments/assets/c2604dc6-cc78-4d00-bb5e-99d56749f496" width="100%">

---

### 4. Medium Rank, Low Learn Rate (v7)
<img src="https://github.com/user-attachments/assets/cfc4a0f9-a91e-42a9-aada-144febbb5df4" width="100%"/>

## Configuration

The final optimized parameters are stored in `TBC`.
- **Optimizer:** AdamW8bit
- **Scheduler:** Cosine with restarts
- **Resolution:** 512x512

## Technical Note
The training architecture for this project was scaffolded using AI to ensure best practices, while all hyperparameter experimentation, evaluation, and tuning were conducted manually to achieve the final result.
