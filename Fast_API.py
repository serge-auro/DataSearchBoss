

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch
from typing import List, Optional

clip_id = 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
clip_model = CLIPModel.from_pretrained(clip_id)
processor = CLIPProcessor.from_pretrained(clip_id)

app = FastAPI()



@app.post("/encode")
async def encode(images: List[UploadFile] = File(), texts: List[str] = Form(None)):
#List[str] = Form(...)):
#Optional[List[str]] = Form(None)):

    image_inputs = []
    text_inputs = {}

    for image_file in images:
        image = Image.open(BytesIO(await image_file.read()))
        image_input = processor(images=image, return_tensors="pt")
        image_inputs.append(image_input["pixel_values"])


    if texts:
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    if image_inputs:
        with torch.no_grad():
            image_features = clip_model.get_image_features(torch.cat(image_inputs, dim=0))
            image_features = image_features.mean(dim=0, keepdim=True)

    if text_inputs:
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features.mean(dim=0, keepdim=True)

    if image_inputs and text_inputs:
        features = torch.cat((image_features, text_features), dim=0).mean(dim=0, keepdim=True)
        features /= features.norm(dim=-1, keepdim=True)
    elif image_inputs:
        features = image_features
    elif text_inputs:
        features = text_features
    else:
        raise HTTPException(status_code=400, detail="No valid images or texts provided.")

    return {"features": features.tolist()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the CLIP API. Use /encode for image and text encoding."}
