from fastapi.responses import StreamingResponse
import numpy as np
import requests
import io
import time

from utils.schemas import promptCreate
from PIL import Image, UnidentifiedImageError

API_URL = 'https://api-inference.huggingface.co/models/nerijs/pixel-art-xl'
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def get_time_inference():
    prompts = {'um gato preto',
               'um passaro comendo',
               'uma pessoa sorrindo',
               'uma criança pulando',
               'uma giranfa tomando banho',
               'um elefante na selva',
               'um canguru correndo',
               'um copo em uma mesa preta',
               'uma abelha rainha sendo reverenciada por suas operárias',
               'um ventilador em uma cadeira',
               'uma cadeira rosa em uma sala branca',
               'insetos brigando',
               'uma mesa em formato de L',
               'uma criança em uma bicicleta',
               'uma arma sniper',
               'um menino de cabelos encaracolados',
               'colhendo limões para uma limonada',
               'uma barraca em um acampamento',
               'dois amigos andando de bicicleta',
               'um computador dos anos 90'}

    times = []

    for prompt in prompts:
        start = time.time()
        image = gen('pixel art, ' + prompt)
        end = time.time()

        if image is not None:
            times.append(end-start)

    print(f'Images generated: {len(times)}')
    print(f'Average time: {sum(times)/len(times)}')


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


async def gen(prompt: promptCreate):
  try:
    image_bytes = io.BytesIO(query({'inputs': prompt.prompt}))
    image = Image.open(image_bytes)
    
    width, heigth = image.size
    image = image.resize((int(width/8), int(heigth/8)), Image.NEAREST)
    
    return image
  except UnidentifiedImageError as _:
    print('Image could not be genereted')
  except Exception as e:
    print(e)