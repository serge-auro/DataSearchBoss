import os
import json
import time
import requests
import ffmpeg
import numpy as np
import whisper
from io import BytesIO
from moviepy.editor import VideoFileClip
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import asyncio
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()


class AudioEncodeRequest(BaseModel):
    video_url: str


video_path = "temp_video.mp4"
audio_path = "temp_audio.wav"


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
        with open(video_path, "wb") as f:
            f.write(video_data.getbuffer())
        # start_time = time.time()
        # Извлечение видеофайла
        video = VideoFileClip(video_path)
        # Вывод длины видео в секундах
        video_duration = video.duration
        logging.info(f"Video duration: {video_duration} seconds")
        # Извлечение аудиодорожки из видеофайла
        audio = video.audio
        audio.write_audiofile(audio_path)
        # Явное закрытие видео и аудио объектов
        audio.close()
        video.close()

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Error processing video")

    return {"video_duration": video_duration, "audio_path": audio_path}


# Функция для преобразования аудио в текст
def audio_to_text(audio_path):
    logging.info("Converting audio to text...")
    try:
        result = model.transcribe(audio_path, language='ru')
    except Exception as e:
        logging.error(f"Whisper model error: {e}")
        return None
    return result['text']


# Загрузка модели Whisper
model = whisper.load_model('small')

# Чтение URL видео из файла
with open('video_description/all_videos.json', 'r', encoding='utf-8') as f:
    videos = json.load(f)

# Создание папки для транскрипций, если она не существует
os.makedirs('whisper_transcriptions', exist_ok=True)

# Инициализация файлов для записи результатов
# transcriptions = {}
# none_transcriptions = {}
# transcriptions_fail = {}


# Функция для загрузки состояния из файла
def load_state(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


# Загрузка предыдущего состояния
transcriptions = load_state('whisper_transcriptions/transcriptions_1.json')
none_transcriptions = load_state('whisper_transcriptions/none_transcriptions_1.json')
transcriptions_fail = load_state('whisper_transcriptions/transcriptions_fail.json')


# Функция для сохранения состояния в файл
def save_state(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


program_start_time = time.time()


async def process_videos(videos):
    # Обработка первых 1500 записей
    for i, (video_id, video_info) in enumerate(list(videos.items())[1600:1620]):
        if video_id in transcriptions or video_id in none_transcriptions or video_id in transcriptions_fail:
            logging.info(f"Skipping already processed video ID: {video_id}")
            continue

        logging.info(f"Processing {i + 1}: {video_id}")
        video_url = video_info['url']

        try:
            start_time = time.time()

            # Скачивание видео и извлечение аудио
            result = await encode_audio(AudioEncodeRequest(video_url=video_url))
            video_duration = result['video_duration']
            audio_path = result['audio_path']

            if audio_path is None:
                none_transcriptions[video_id] = {
                    'url': video_url,
                    'processing_time': 0,
                    'video_duration': video_duration,
                    'channels_count': 0,
                    'transcription': None
                }
                save_state('whisper_transcriptions/none_transcriptions_1.json', none_transcriptions)
                continue

            # Преобразование аудио в текст
            transcription = audio_to_text(audio_path)

            if transcription is None:
                transcriptions_fail[video_id] = {
                    'url': video_url,
                    'error': 'Transcription failed'
                }
                save_state('whisper_transcriptions/transcriptions_fail.json', transcriptions_fail)
                continue

            end_time = time.time()
            processing_time = end_time - start_time

            logging.info(
                f"Video ID: {video_id}, Video URL: {video_url}, Duration: {video_duration}s, "
                f"Channels: 1, Processing Time: {processing_time}s")
            logging.info(f"Transcription: {transcription}")

            transcriptions[video_id] = {
                'url': video_url,
                'processing_time': processing_time,
                'video_duration': video_duration,
                'channels_count': 1,
                'transcription': transcription
            }
            save_state('whisper_transcriptions/transcriptions_1.json', transcriptions)

        except Exception as e:
            error_name = e.__class__.__name__
            error_description = str(e)
            logging.error(f"Error processing video {video_id}: {error_name}, {error_description}")
            transcriptions_fail[video_id] = {
                'url': video_url,
                'error': f"{error_name}: {error_description}"
            }
            save_state('whisper_transcriptions/transcriptions_fail.json', transcriptions_fail)

    return transcriptions, none_transcriptions, transcriptions_fail


asyncio.run(process_videos(videos))


# Удаление временных файлов
if os.path.exists(video_path):
    os.remove(video_path)
if os.path.exists(audio_path):
    os.remove(audio_path)


program_end_time = time.time()
program_execution_time = program_end_time - program_start_time
logging.info(f'Processing completed, program execution time: {program_execution_time:.2f}s')
