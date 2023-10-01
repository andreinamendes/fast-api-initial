import io

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from utils.schemas import promptCreate
from gen.models.pixelart_requests import gen as gen_pixel_art
from PIL import Image

async def gen(prompt: promptCreate):
  image = await gen_pixel_art(prompt=prompt)

  if not image:
      raise HTTPException(
          status_code=404, detail="Image could not be generateded.")
  else:
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")