#https://medium.com/@lankinen/fastai-model-to-production-this-is-how-you-make-web-app-that-use-your-model-57d8999450cf

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

app = Starlette()

path=Path(r"C:\Users\Chuck\Downloads")
learn = load_learner(path)
defaults.device = torch.device('cpu')

@app.route("/")
def form(request):
    return HTMLResponse("""
        <h3>This app will classify Parrots, Eagles, Woodpeckers and Falcons<h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))

    pred_class,pred_idx,outputs = learn.predict(img)

    return JSONResponse({
        "prediction": pred_class.__dict__['obj'],
        "confidence": outputs.numpy()[pred_class.__dict__['data'].item()]*100
    })


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


