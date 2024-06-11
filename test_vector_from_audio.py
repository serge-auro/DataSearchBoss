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
import uvicorn
import asyncio
from scipy.signal import resample
import time
from sklearn.decomposition import PCA

# Загрузка модели для аудио
audio_model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
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

        start_time = time.time()

        # Извлечение видеофайла
        video = VideoFileClip(video_path)

        # Вывод длины видео в секундах
        video_duration = video.duration
        print(f"Video duration: {video_duration} seconds")

        # Извлечение аудиодорожки из видеофайла
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)

        # Явное закрытие видео и аудио объектов
        audio.close()
        video.close()

        # Преобразование аудиофайла в массив numpy
        samplerate, data = wavfile.read(audio_path)
        if data.ndim > 1:
            # Если аудиодорожка имеет более одного канала, взять только первый канал
            # data = data[:, 0]
            # Если аудиодорожка имеет более одного канала, усреднить каналы
            data = np.mean(data, axis=1)

        # Нормализация данных
        data = data.astype(np.float32) / 32768.0

        # Приведение частоты дискретизации к 16000 Гц
        target_samplerate = 16000
        if samplerate != target_samplerate:
            num_samples = int(len(data) * float(target_samplerate) / samplerate)
            data = resample(data, num_samples)

        # Создание тензора
        audio_tensor = torch.tensor(data)

        # Преобразование аудио в вектора
        inputs = processor(audio_tensor, sampling_rate=target_samplerate, return_tensors="pt")
        input_values = inputs.input_values  # Получаем тензор входных значений
        with torch.no_grad():
            features = audio_model(input_values).last_hidden_state
        features /= features.norm(dim=-1, keepdim=True)

        # Преобразование тензора в numpy массив
        features_np = features.squeeze(0).numpy()

        # Применение PCA для усреднения векторов без потери важной информации
        pca = PCA(n_components=1024)
        aggregated_vector = pca.fit_transform(features_np.T).T.mean(axis=1)

        end_time = time.time()
        total_time = end_time - start_time
        print(f'Processing time: {total_time}')

        # return {"features": features.squeeze(0).tolist()}
        return {"features": aggregated_vector.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Удаление временных файлов
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def test_encode_audio():
    # test_url = 'https://cdn-st.rutubelist.ru/media/c7/ba/3a3dad294ee9befea47fb56ed0d5/fhd.mp4'
    # test_url = 'https://cdn-st.rutubelist.ru/media/39/6c/b31bc6864bef9d8a96814f1822ca/fhd.mp4'
    test_url = 'https://cdn-st.rutubelist.ru/media/0f/48/8a1ff7324073947a31e80f71d001/fhd.mp4'
    # test_url = "https://cdn-st.rutubelist.ru/media/b0/e9/ef285e0241139fc611318ed33071/fhd.mp4"
    test_request = AudioEncodeRequest(video_url=test_url)
    response = await encode_audio(test_request)
    features = response["features"]
    print(f"Размерность списка векторов: {len(features)}")
    # print(f"Размерность списка векторов: {len(features)} x {len(features[0]) if features else 0}")
    # print(response)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # Запуск сервера в отдельном потоке
    server_task = loop.create_task(run_server())
    # Выполнение теста
    loop.run_until_complete(test_encode_audio())
    # Завершение сервера после выполнения теста
    server_task.cancel()
    try:
        loop.run_until_complete(server_task)
    except asyncio.CancelledError:
        pass
    loop.close()
