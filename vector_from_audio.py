import os
import torch
from io import BytesIO
from typing import Optional, List
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
import numpy as np

# Загрузка модели для аудио
audio_model_id = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(audio_model_id)
audio_model = Wav2Vec2Model.from_pretrained(audio_model_id)

app = FastAPI()


class AudioEncodeRequest(BaseModel):
    video_url: str


@app.post("/encode_audio")
async def encode_audio(request: AudioEncodeRequest):
    video_url = request.video_url
    if not video_url:
        raise HTTPException(status_code=400, detail="Please provide a video URL.")

    try:
        # Скачивание видеофайла
        response = requests.get(video_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Unable to fetch video from {video_url}")

        video_data = BytesIO(response.content)
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data.getbuffer())

        # Извлечение аудиодорожки из видеофайла
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)

        # Преобразование аудиофайла в массив numpy
        samplerate, data = wavfile.read(audio_path)

        if data.ndim > 1:
            # Если аудиодорожка имеет более одного канала, взять только первый канал
            data = data[:, 0]

        # Нормализация данных
        data = data.astype(np.float32) / 32768.0

        # Создание тензора
        audio_tensor = torch.tensor(data).unsqueeze(0)  # Добавление batch dimension

        # Преобразование аудио в вектора
        inputs = processor(audio_tensor, sampling_rate=samplerate, return_tensors="pt")
        with torch.no_grad():
            features = audio_model(**inputs).last_hidden_state

        features /= features.norm(dim=-1, keepdim=True)

        return {"features": features.squeeze(0).tolist()}

    finally:
        # Удаление временных файлов
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


# Тестирование
if __name__ == "__main__":
    import uvicorn

    test_url = "https://cdn-st.rutubelist.ru/media/b0/e9/ef285e0241139fc611318ed33071/fhd.mp4"
    test_request = AudioEncodeRequest(video_url=test_url)
    test_response = encode_audio(test_request)
    print(test_response)

    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)


# TODO:
# 1. Запустив такой код на сервере с поддержкой GPU, можно получать векторы по аудиодорожке из видеофайлов.
# 2. Сейчас такое приложение принимает URL-ссылки на видеофайлы и скачивает их самостоятельно.