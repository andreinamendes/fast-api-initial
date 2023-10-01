import torch
import os

from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL
from huggingface_hub import login

# Package Environment:
# conda install -c conda-forge transformers -y
# conda install -c conda-forge cudatoolkit -y
# conda install -c anaconda cudnn -y
# pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu118/torch_stable.html
# pip install --no-deps diffusers["torch"]

TOKEN='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'
login(TOKEN)

if torch.cuda.is_available():
  vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
  pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16")
  pipe.to("cuda")
  pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")

  prompt = 'uma criança no parquinho'

  image = pipe(prompt='pixel art, ' + prompt).images[0]

  width, heigth = image.size
  image = image.resize((int(width/8), int(heigth/8)), Image.NEAREST)
  image.show()
else:
  print("CUDA is not available")