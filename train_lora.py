from google.colab import drive
drive.mount('/content/drive')
!pip install -q torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
!pip install -q xformers==0.0.33.post2 accelerate transformers bitsandbytes datasets ftfy
!git clone https://github.com/huggingface/diffusers.git
!pip install -e ./diffusers  
!pip install mediapipe

from accelerate.utils import write_basic_config
write_basic_config()

print("Environment Ready")

import os
from PIL import Image, ImageOps

INPUT_FOLDER = "/content/drive/MyDrive/training_images"
OUTPUT_FOLDER = "/content/drive/MyDrive/training_images_padded"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
processed_count = 0

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        try:
            img_path = os.path.join(INPUT_FOLDER, filename)
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            background = Image.new("RGB", (512, 512), (0, 0, 0))
            x_offset = (512 - img.width) // 2
            y_offset = (512 - img.height) // 2
            background.paste(img, (x_offset, y_offset))
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            background.save(save_path, quality=100)
            processed_count += 1

        except Exception as e:
            print(f"Error on {filename}: {e}")

print(f"\nPadded images.")

import os
import json
from PIL import Image


IMAGES_FOLDER = "/content/drive/MyDrive/training_images_padded"
TRIGGER_WORD = "levi_training_model"

metadata_path = os.path.join(IMAGES_FOLDER, "metadata.jsonl")

image_tags = {
    # Bloody/Combat scenes
    "Bloody3.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, blood on face, reverse grip, holding sword, profile, lowered view, intense expression, serious",
    "Bloody1.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, dual wielding, reverse grip, holding swords, action pose, looking at camera, dynamic angle, mid air, blood on face, injured, intense expression",
    "Bloody2.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, blood on face, injured, standing, looking at viewer, serious",

    # Generic/Standard Survey Corps Uniform
    "Generic1.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, white cravat, view from the side, standing, hands at sides, serious",
    "Generic2.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, white cravat, white pants, standing, looking at viewer, hands at sides, serious",
    "Generic3.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, profile, view from side, looking away, standing, serious",
    "Generic4.jpeg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, white cravat, profile, view from side, looking away, standing, serious",
    "Generic5.png": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, standing, looking to the side, hands at sides, serious",
    "Generic6.jpg": "black hair, undercut, gray eyes, male focus, angry, screaming, survey corps uniform, green cape, white cravat, looking at viewer, standing, hands at sides, open mouth",

    # Combat/Sword poses
    "GenericSword1.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, white pants, 3d maneuvering gear, dual wielding, reverse grip, holding swords, action pose, facing camera, dynamic angle, mid air, serious",
    "GenericSword2.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, white cravat, reverse grip, unsheathing sword, combat pose, looking at viewer, serious",

    # Formal/Suit outfits
    "Suit1.jpg": "black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, black pants, looking to the side, sitting on chair, serious",
    "Suit2.jpg": "black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, looking at viewer, standing, serious",
    "Suit3.jpg": "black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, looking at viewer, sitting, view from side, serious",
    "Suit4.jpg": "black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, view from side, looking down, obscured eyes, sitting, serious",

    # Casual/Shirt only (no jacket)
    "Shirt1.jpg": "black hair, undercut, gray eyes, male focus, survey corps uniform, brown jacket, no cape, white cravat, looking at viewer, serious",
    "Shirt2.jpg": "black hair, undercut, gray eyes, male focus, white shirt, rolled sleeves, looking at viewer, standing, arms raised in front, view from side, serious",
    "Shirt3.jpg": "black hair, undercut, gray eyes, male focus, white shirt, dual wielding, reverse grip, holding swords, looking at the side, view from side, mid air, 3d maneuvering gear, serious",
    "Shirt4.jpg": "black hair, undercut, gray eyes, male focus, white shirt, looking at viewer, view from side, mid air, serious",
    "Shirt5.jpeg": "black hair, undercut, gray eyes, male focus, white shirt, view from side, looking to the side, standing, serious",
    "Shirt6.jpg": "black hair, undercut, gray eyes, male focus, white shirt, looking at viewer, sitting on chair, view from side, serious",
    "Shirt7.jpg": "black hair, undercut, gray eyes, male focus, white shirt, profile, side view, standing, serious",
    "Shirt8.jpg": "black hair, undercut, gray eyes, male focus, white shirt, looking to the side, elevated view, serious",

    # No cape/minimal uniform
    "NoCape1.jpg": "black hair, undercut, gray eyes, male focus, white shirt, white cravat, looking at ahead, elevated view, serious",
    "NoCape2.jpg": "black hair, undercut, gray eyes, male focus, white shirt, rolled sleeves, white cravat, hands with towel, looking at viewer, view from side, standing, serious",
}

valid_images = []
print(f"Scanning {IMAGES_FOLDER}")

if not os.path.exists(IMAGES_FOLDER):
    print(f"Padded folder not found")
else:
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                path = os.path.join(IMAGES_FOLDER, filename)
                with Image.open(path) as img:
                    img.verify()
                description = image_tags.get(
                    filename,
                    "black hair, undercut, gray eyes, male focus, survey corps uniform, green cape, serious"
                )
                full_text = f"{TRIGGER_WORD}, {description}"

                entry = {"file_name": filename, "text": full_text}
                valid_images.append(json.dumps(entry))
                print(f"Mapped: {filename}")

            except Exception as e:
                print(f"Bad file {filename}: {e}")

    if len(valid_images) > 0:
        with open(metadata_path, "w") as f:
            for line in valid_images:
                f.write(line + "\n")
        print(f"\nMetadata saved ")
    else:
        print("No valid images found.")

import os
import shutil

SOURCE_DIR = "/content/drive/MyDrive/training_images_padded"
LOCAL_DIR = "/content/local_train_data"
OUTPUT_DIR = "/content/drive/MyDrive/my_first_lora_output_v5"


if os.path.exists(LOCAL_DIR):
    shutil.rmtree(LOCAL_DIR)
shutil.copytree(SOURCE_DIR, LOCAL_DIR)
print(f"Copied to {LOCAL_DIR}")


os.environ["MODEL_NAME"] = "runwayml/stable-diffusion-v1-5"
os.environ["OUTPUT_DIR"] = OUTPUT_DIR
os.environ["DATA_DIR"] = LOCAL_DIR

!accelerate launch /content/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --dataloader_num_workers=0 \
  --resolution=512 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=900 \
  --learning_rate=1.5e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --rank=32 \
  --checkpointing_steps=150 \
  --seed=1337 \
  --mixed_precision="fp16"

import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
from IPython.display import display

CHECKPOINT_STEP = 600      
OUTPUT_DIR = "/content/drive/MyDrive/my_first_lora_output_v5"

LORA_SCALE = 0.9
NUM_IMAGES = 2              
GUIDANCE_SCALE = 7.0
INFERENCE_STEPS = 30

ENABLE_HIRES_FIX = True
HIRES_STRENGTH = 0.35       
UPSCALE_FACTOR = 1.5        

model_folder = os.path.join(OUTPUT_DIR, f"checkpoint-{CHECKPOINT_STEP}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipe.load_lora_weights(model_folder, weight_name="pytorch_lora_weights.safetensors")

refiner = StableDiffusionImg2ImgPipeline(**pipe.components)

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', (cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, (i%cols*w, i//cols*h))
    return grid

prompt = (
"levi_training_model, black hair, undercut, gray eyes, male focus, black suit, white shirt, formal wear, looking at viewer, sitting, view from side, serious, highly detailed eyes, sharp focus"
)

neg_prompt = (
    "shiny eyes, glossy eyes, wet eyes, reflective eyes, eye shine, "
    "bad anatomy, poorly drawn eyes, ugly eyes, lowres, blurry, bad quality"
)

print(f"\nGenerating Base Images...")


base_images = pipe(
    prompt,
    negative_prompt=neg_prompt,
    num_inference_steps=INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    cross_attention_kwargs={"scale": LORA_SCALE},
    num_images_per_prompt=NUM_IMAGES,
    height=512, width=512  
).images

final_images = base_images

if ENABLE_HIRES_FIX:
    print(f"Running Hires...")
    hires_images = []
    
    for img in base_images:
        w, h = img.size
        new_w, new_h = int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)
        upscaled_img = img.resize((new_w, new_h), resample=Image.LANCZOS)

        refined_img = refiner(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=upscaled_img,
            num_inference_steps=30,    
            strength=HIRES_STRENGTH,    
            guidance_scale=GUIDANCE_SCALE,
            cross_attention_kwargs={"scale": LORA_SCALE}
        ).images[0]
        hires_images.append(refined_img)
    
    final_images = hires_images

grid = image_grid(final_images, rows=1, cols=NUM_IMAGES)
display(grid)

print("\nGeneration complete!")