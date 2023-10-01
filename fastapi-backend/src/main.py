from fastapi import FastAPI
from utils import schemas, services

app = FastAPI(tittle="Generate Pixel Art")


@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffussers PixelArt API"}


@app.get("/gen", status_code=200)
async def gen_pixelart(prompt: schemas.promptCreate):
    response = await services.gen(prompt=prompt)
    return response
