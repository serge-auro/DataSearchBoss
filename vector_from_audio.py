import os
import torch
from io import BytesIO
from typing import Optional, List
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # Wav2Vec2Model
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
import numpy as np
from scipy.signal import resample
import time
from sklearn.decomposition import PCA
import logging
import json
import asyncio

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели для аудио
audio_model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(audio_model_id)
audio_model = Wav2Vec2ForCTC.from_pretrained(audio_model_id)

# Создание папки для сохранения результатов
if not os.path.exists("audio_processing"):
    os.makedirs("audio_processing")

# Чтение 100 записей из файла all_videos.json
with open("video_description/all_videos.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    start_video = 900
    end_video = 1000
    n100_records = list(data.items())[start_video:end_video]
    """
    На момент 13.06.2024 01:00 обработаны видеофайлы 0 - 999 (включительно).
    Для дальнейшей работы задать start_video = 1000, end_video = 1001 (минимум)
    """

new_audio_vectors = {}
new_audio_process = {}
new_audio_fail = {}

app = FastAPI()


class AudioEncodeRequest(BaseModel):
    video_url: str


@app.post("/encode_audio")
async def encode_audio(request: AudioEncodeRequest):
    video_url = request.video_url
    if not video_url:
        raise HTTPException(status_code=400, detail="Please provide a video URL.")

    video_path = "temp_video.mp4"
    audio_path = "temp_audio.wav"

    try:
        # Скачивание видеофайла
        response = requests.get(video_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Unable to fetch video from {video_url}")
        video_data = BytesIO(response.content)
        with open(video_path, "wb") as f:
            f.write(video_data.getbuffer())

        start_time = time.time()

        # Извлечение видеофайла
        video = VideoFileClip(video_path)

        # Вывод длины видео в секундах
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration} seconds")

        # Извлечение аудиодорожки из видеофайла
        audio = video.audio
        audio.write_audiofile(audio_path)

        # Явное закрытие видео и аудио объектов
        audio.close()
        video.close()

        # Преобразование аудиофайла в массив numpy
        samplerate, data = wavfile.read(audio_path)
        channels_count = 1 if data.ndim == 1 else data.shape[1]
        if data.ndim > 1:
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
        audio_tensor = torch.tensor(data)  # .unsqueeze(0)  # Добавление batch dimension - ломает программу

        # Преобразование аудио в вектора
        inputs = processor(audio_tensor, sampling_rate=target_samplerate, return_tensors="pt")
        input_values = inputs.input_values  # Получаем тензор входных значений
        with torch.no_grad():
            features = audio_model(input_values).last_hidden_state
        features /= features.norm(dim=-1, keepdim=True)

        # Преобразование тензора в numpy массив
        features_np = features.squeeze(0).numpy()

        # Padding или усечение данных до необходимого размера
        if features_np.shape[0] < 1024:
            padding = np.zeros((1024 - features_np.shape[0], features_np.shape[1]))
            features_np = np.vstack((features_np, padding))
        else:
            features_np = features_np[:1024, :]

        # Применение PCA для усреднения векторов без потери важной информации
        pca = PCA(n_components=1024)
        aggregated_vector = pca.fit_transform(features_np.T).T.mean(axis=1)

        # Преобразование вектора в нужный формат 1x1024
        aggregated_vector = aggregated_vector.reshape(1, -1)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f'Processing time: {total_time}')

        return {
            "features": aggregated_vector.tolist(),
            "processing_audio_time": total_time,
            "video_duration": video_duration,
            "channels_count": channels_count
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Удаление временных файлов
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


# Обработка записей
async def process_records(records):
    process_records = records
    count = start_video
    for key, value in process_records:
        logger.info(f'Video number, index: {count}, {key}')
        count += 1
        video_url = value["url"]
        try:
            request = AudioEncodeRequest(video_url=video_url)
            result = await encode_audio(request)
            new_audio_vectors[key] = result["features"]
            new_audio_process[key] = {
                "processing_audio_time": result["processing_audio_time"],
                "video_duration": result["video_duration"],
                "channels_count": result["channels_count"]
            }
        except Exception as e:
            new_audio_fail[key] = {"url": video_url, "error": str(e)}
    return new_audio_vectors, new_audio_process, new_audio_fail

program_start_time = time.time()

new_audio_vectors, new_audio_process, new_audio_fail = asyncio.run(process_records(n100_records))

# Чтение существующих данных из файла audio_vectors
try:
    with open("audio_processing/audio_vectors.json", "r") as file:
        audio_vectors = json.load(file)
except FileNotFoundError:
    audio_vectors = {}
# Дополнение данных новыми записями
audio_vectors.update(new_audio_vectors)
# Запись обновленных данных обратно в файл audio_vectors
with open("audio_processing/audio_vectors.json", "w") as file:
    json.dump(audio_vectors, file, indent=4)

# Чтение существующих данных из файла audio_process
try:
    with open("audio_processing/audio_process.json", "r") as file:
        audio_process = json.load(file)
except FileNotFoundError:
    audio_process = {}
# Дополнение данных новыми записями
audio_process.update(new_audio_process)
# Запись обновленных данных обратно в файл audio_process
with open("audio_processing/audio_process.json", "w") as file:
    json.dump(audio_process, file, indent=4)

# Чтение существующих данных из файла audio_fail
try:
    with open("audio_processing/audio_fail.json", "r") as file:
        audio_fail = json.load(file)
except FileNotFoundError:
    audio_fail = {}
# Дополнение данных новыми записями
audio_fail.update(new_audio_fail)
# Запись обновленных данных обратно в файл audio_fail
with open("audio_processing/audio_fail.json", "w") as file:
    json.dump(audio_fail, file, indent=4)

# Сохранение результатов в файлы
# with open("audio_processing/audio_vectors.json", "w") as file:
#     json.dump(audio_vectors, file)
#
# with open("audio_processing/audio_process.json", "w") as file:
#     json.dump(audio_process, file)
#
# with open("audio_processing/audio_fail.json", "w") as file:
#     json.dump(new_audio_fail, file)

program_end_time = time.time()
program_execution_time = program_end_time - program_start_time
logger.info(f'Program execution time: {program_execution_time}')

# Запуск приложения через Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# TODO:
# 1. Запустив такой код на сервере с поддержкой GPU, можно получать векторы по аудиодорожке из видеофайлов.
# 2. Сейчас такое приложение принимает URL-ссылки на видеофайлы и скачивает их самостоятельно.
#    Загружает временные файлы на диск и удаляет их в конце работы.