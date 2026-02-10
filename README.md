# LoRA Hyperparameter Optimization Project

This repository documents the systematic tuning of a Stable Diffusion LoRA. Through 10 iterations, I optimised the relationship between **Network Rank** and **Learning Rate** to achieve maximum stylistic fidelity without model degradation/burning.

## Optimising the model
I tested ten distinct configurations to find the model which could produce seamless replicas of Levi Ackerman from Attack on Titan

| Version | Rank ($r$)| Learning Rate | Total Checkpoints | Optimal checkpoint | Result |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **v1** | 32 | 1e-4 | 900 | TBC | TBC | 
| **v2** | 4 | 1e-4 | 500 | TBC | TBC | 
| **v3** | 16 | 1e-4 | 800 | 800 | Refer to v4 | 
| **v4** | 16 | 1e-4 | 800 | 800 | General artistic style is achieved, however, far from any sort of accurate portrayal of the character | 
| **v5** | 32 | 1.5e-4 | 1200 | 750 | Refer to v7| 
| **v6** | 32 | 1e-4 | 400 | TBC | Setting a control |
| **v7** | 32 | 1.5e-4 | 900 | 750 | Facial structure extremely accurate to the original; however, obvious eye issues caused by low rank | 
| **v8** | 64 | 1.5e-4 | 1200 | 800 | Extremely accurate to the original | 
| **v9** | 64 | 1e-4 | 1400 | 800 | Very accurate to the original; however, slight, noticeable burning | 
| **v10** | 64 | 5e-5 | 3000 | 2750 | Fairly accurate to the original | 

## Note
- During the training process, I realised the images I was using as inputs were centre-cropped by a 512x512 aspect ratio, and a lot of important pieces of the images were lost, which also messed with the tagging. As a result, v1 - v4 will have differing results compared to v5+ as I updated the tagging to match it more accurately and ensured images were padded such that the whole image was used in training. This is why v6 was a copy of v1, as I wanted to set a control version to keep track of improvements.

- Hires was also used as a trialled solution for v5 to fix the eye burning; however, initially, the primary issue of the eyes was caused by low rank and not too few pixels and using Hires didn't achieve any change in results. Despite this, it was kept for all subsequent runs and noticeably helped with eye burning with all the following models when there were too few pixels to render the eyes and pupils accurately.

## 1. Source Image vs. Best Result (High Rank, Fast Learn Rate) (v8)

| Source Image (Reference) | Best Result (v8) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/ed9e2c72-732b-4c31-810a-85024db51c04"  width="100%"> | <img src="https://github.com/user-attachments/assets/892829f5-9e91-4bdd-9a6d-7863b70cb3c7" width="100%"> |

---

### 2. High Rank, Medium Learn Rate (v9)
<img src="https://github.com/user-attachments/assets/c2ae827c-c631-4b2d-8b21-83275e5d5d87" width="100%">

---

### 3. High Rank, Low Learn Rate (v10)
<img src="https://github.com/user-attachments/assets/c2604dc6-cc78-4d00-bb5e-99d56749f496" width="100%">

---

### 4. Medium Rank, Low Learn Rate (v7)
<img src="https://github.com/user-attachments/assets/cfc4a0f9-a91e-42a9-aada-144febbb5df4" width="100%"/>

---

### 5. Low Rank, Fast Learn Rate (v4)
<img src="https://github.com/user-attachments/assets/e9d73914-1b0e-4c1b-852a-902ce9c01003" width="100%"/>

## Reproduction Settings
To replicate the results shown above, use the following parameters in your inference pipeline.

### **Prompting**
**Positive Prompt:**
> `levi_training_model, black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, looking at viewer, standing, serious`

**Negative Prompt:**
> `shiny eyes, glossy eyes, wet eyes, reflective eyes, eye shine, bad anatomy, poorly drawn eyes, ugly eyes, lowres, blurry, bad quality`

### **Generation Parameters**
| Parameter | Value |
| :--- | :--- |
| **LoRA Scale** | `0.9` |
| **Guidance Scale (CFG)** | `7.0` |
| **Inference Steps** | `30` |
| **Resolution** | 768x768 (512x512 base) |
| **Hires. Fix Factor** | `1.5x` |
| **Denoising Strength** | `0.35` |

## Configuration

The final optimized parameters are stored in `hparamsV8.yml`.
- **Optimizer:** AdamW8bit
- **Scheduler:** Cosine with restarts
- **Resolution**: 768x768 (Generated at 512x512 with 1.5x Hires. fix)

## Technical Environment
- **Dataset:** 25 Images
- **Hardware:** Google Colab T4 GPU
- **Training time**: 80 minutes for version 10 (3000 steps)

## Conclusion
- The first 4 versions were me experimenting around with different values for rank (and v3 being the same as v4 to see if any major differences occurred during training), and I quickly realised a **higher rank** resulted in better images. 
- In version 5, I decided to fasten the learning rate slightly, and I noticed extremely good improvements over previous models (also thanks to better input data and tagging); however, the eyes still had issues. As a result, I decided to use the same parameters for v7 to see if another training run could resolve the eye issues, but nothing changed, so I decided to increase the rank again.
- By version 8, the accuracy was nearly perfect with the source, so I decided to compare the learning rates of this newer, more accurate model and still concluded that a **faster learning rate was better** (at least for my character).

## Technical Note
The boilerplate was rapidly prototyped by generative AI to ensure best practices. However, all hyperparameter experimentation, evaluation, and tuning were conducted manually to achieve the final result.
