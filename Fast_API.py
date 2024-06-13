from fastapi import FastAPI, UploadFile, File, Form
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch
from typing import List
import logging

# Идентификатор модели CLIP
clip_id = 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
clip_model = CLIPModel.from_pretrained(clip_id)
processor = CLIPProcessor.from_pretrained(clip_id)

# Setup logging
logging.basicConfig(filename='api_requests.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

@app.post("/encode")
async def encode(images: List[UploadFile] = File(default=[]), texts: List[str] = Form(None)):
    image_inputs = []
    text_inputs = {}

    # Обработка изображений
    if images:
        for image_file in images:
            image = Image.open(BytesIO(await image_file.read()))
            image_input = processor(images=image, return_tensors="pt")
            image_inputs.append(image_input["pixel_values"])

    # Обработка текстов
    if texts and any(texts):
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    image_features = []
    text_features = []

    # Получение признаков изображений
    if image_inputs:
        with torch.no_grad():
            for image_input in image_inputs:
                features = clip_model.get_image_features(image_input)
                image_features.append(features.squeeze().tolist())  # Преобразуем тензор в список

    # Получение признаков текстов
    if text_inputs:
        with torch.no_grad():
            features = clip_model.get_text_features(**text_inputs)
            text_features = features.squeeze().tolist()  # Преобразуем тензор в список

    if not image_features and not text_features:
        return {"features": None}  # Возвращаем None, если нет векторов

    response = {"image_features": image_features}

    # Проверка и разделение текстовых признаков на отдельные тензоры
    if text_features:
        if isinstance(text_features[0], list):
            response["text_features"] = text_features
        else:
            response["text_features"] = [text_features]

    logging.info(f"Encoded response: {response}")
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the CLIP API. Use /encode for image and text encoding."}
