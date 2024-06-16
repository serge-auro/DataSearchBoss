import os
import logging
from moviepy.editor import VideoFileClip
import whisper
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Функция для преобразования аудио в текст
def audio_to_text(audio_path, model):
    logging.info("Converting audio to text...")
    try:
        result = model.transcribe(audio_path, language='ru')
    except Exception as e:
        logging.error(f"Whisper model error: {e}")
        return None
    return result['text'] if result['text'].strip() else None

# Функция для извлечения аудио и транскрибации
def encode_and_transcribe(video_path, model):
    if not os.path.exists(video_path):
        logging.error("Video file does not exist")
        return None, 0

    audio_path = "temp_audio.wav"
    start_time = time.time()

    try:
        # Извлечение аудиодорожки из видеофайла
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()

        # Транскрибация аудио
        transcription = audio_to_text(audio_path, model)
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return None, 0
    finally:
        # Удаление временного аудиофайла, если он существует
        if os.path.exists(audio_path):
            os.remove(audio_path)

    processing_time = time.time() - start_time
    return transcription, processing_time

# Пример вызова функции
# encode_and_transcribe('path_to_video.mp4', whisper_model)
