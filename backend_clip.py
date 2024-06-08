import os
import torch
from io import BytesIO
from typing import Optional, List
import requests
from fastapi import FastAPI, HTTPException
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import login

clip_id = os.getenv('CLIP_ID', 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

clip_model = CLIPModel.from_pretrained(clip_id)
processor = CLIPProcessor.from_pretrained(clip_id)

app = FastAPI()

class EncodeRequest(BaseModel):
    texts: Optional[List[str]] = None
    images: Optional[List[str]] = None

@app.post("/encode")
async def encode(request: EncodeRequest):
    texts = request.texts
    images = request.images

    if not any((texts, images)):
        raise HTTPException(status_code=400, detail="Please provide either 'texts' as list of strings or 'images' as list of Image URLs.")

    if all((texts, images)):
        raise HTTPException(status_code=400, detail="Please provide either texts or image URLs, not both.")

    if texts:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)

    if images:
        image_inputs = []
        for image_url in images:
            response = requests.get(image_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Unable to fetch image from {image_url}")
            image = Image.open(BytesIO(response.content))
            image_input = processor(images=image, return_tensors="pt")
            image_inputs.append(image_input)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs[0])
            for image_input in image_inputs[1:]:
                image_feature = clip_model.get_image_features(**image_input)
                image_features = torch.cat((image_features, image_feature), dim=0)
        features = image_features

    features /= features.norm(dim=-1, keepdim=True)
    return {"features": features.tolist()}


# TODO
# Запустив такой код на сервере с поддержкой GPU - можно получать векторы по ключевым кадрам видео (изображениям) и тексту.
# TODO
# Важно: сейчас такое приложение принимает URL-ссылки на ключевые кадры и скачивает их самостоятельно.
# Нужно переписать код таким образом, чтобы можно было передавать и сразу ключевой кадр в запросе, как байтовые данные.

