import json
import time
import os
import logging
import requests
import spacy
import whisper

from subtitles_extraction_easyocr_extra import get_subtitles
from translation import translate_text
from download_video_by_url_and_make_frames import create_thumbnails_for_video_message, get_video_duration
from whisper_extraction import encode_and_transcribe
from upload_only_VIDEO_vector import process_only_video_data, delete_frames
from key_words_extraction import extract_keywords
from MongoDB import VideoIndex


# переменная для хранения модели spaCy
nlp = spacy.load("en_core_web_sm")
# загрузка модели whisper
whisper_model = whisper.load_model('small')

# инициализация индекса видео
video_index = VideoIndex(db_name="video_database", collection_name="videos")


# Настройка журнала с именем 'HANDLE1_logging'
logger = logging.getLogger('HANDLE1_logging')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


vectors_file_path = 'Server_API.json'
statistics_file_path = 'statistics_Server_API.json'
unprocessed_videos_log = 'unprocessed_videos_Server_API.log'

def load_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            content = file.read().strip()
            if content:
                return json.loads(content)
            else:
                return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error loading JSON file {file_path}: {str(e)}")
            return {}

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def log_unprocessed_video(video_id):
    with open(unprocessed_videos_log, 'a', encoding='utf-8') as file:
        file.write(f"{video_id}\n")

def extract_video_id(url: str) -> str:
    # Разбиваем URL на части по слешу, берем предпоследнюю часть
    parts = url.strip('/').split('/')
    unique_id = parts[-2]
    return unique_id

def process_only_video_data(video_id, all_texts):
    url = "http://127.0.0.1:8000/encode"
    frames_dir = "frames"

    files = []
    file_handles = []
    try:
        for filename in os.listdir(frames_dir):
            if filename.startswith(f'key_frame_{video_id}_') and filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(frames_dir, filename)
                file_handle = open(file_path, 'rb')
                files.append(('images', (filename, file_handle, 'image/jpeg')))
                file_handles.append(file_handle)
        data = {'texts': all_texts}

        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("Получен успешный ответ")

            image_vectors = response.json().get('image_features', None)
            text_vector = response.json().get('text_features', None)
            result = True
        else:
            log_message = f"Failed to get a proper response. Status code: {response.status_code}\nResponse: {response.text}"
            print(log_message)
            logging.error(log_message)
            image_vectors = None
            text_vector = None
            result = False
    except Exception as e:
        log_message = f"Error during data processing: {str(e)}"
        print(log_message)
        logging.error(log_message)
        image_vectors = None
        text_vector = None
        result = False
    finally:
        for file_handle in file_handles:
            file_handle.close()

    return result, image_vectors, text_vector

def main_handle_videos():
    vectors = {}
    statistics = {}

    # Ввод ссылки на видео и описания
    video_url = input("Введите ссылку на видео: ")
    description = input("Введите описание видео (Если описания нет, нажмите Enter): ")

    all_texts = []
    # Перевод описания в русский язык
    if description is not None:
        description = translate_text(description)  # Перевод описания в русский язык
        all_texts.append(description)

    video_id = extract_video_id(video_url)  # Использование функции extract_video_id для получения ID видео

    start_time = time.time()

    output_folder = "frames"
    frames, video_duration, frames_count, video_path = create_thumbnails_for_video_message(video_id, video_url, output_folder)

    # блок извлечения субтитров
    subtitles, subtitles_processing_time = get_subtitles(video_path)


    cleaned_subtitles = None
    print(subtitles)
    if subtitles is not None:

        subtitles_by_keywords = extract_keywords(subtitles, nlp)
        print(subtitles_by_keywords)
        if subtitles_by_keywords is not None:
            subtitles_translated = translate_text(subtitles_by_keywords)
            if subtitles_translated is not None:
                cleaned_subtitles = extract_keywords(subtitles_translated, nlp)
                all_texts.append(cleaned_subtitles)



    audio_transcription_translated = None
    #извлечение аудиодорожки с помощью Whisper
    audio_transcription, audio_processing_time = encode_and_transcribe(video_path, whisper_model)
    print(audio_transcription)
    if audio_transcription is not None:
        audio_transcription_translated = translate_text(audio_transcription)
        if audio_transcription_translated is not None:
            print(audio_transcription_translated)
            all_texts.append(audio_transcription_translated)

    success, image_vectors, text_vector = process_only_video_data(video_id, all_texts)
    if success and image_vectors is not None:
        # Учитываем наличие векторов и сохраняем в правильном порядке
        description_vector, subtitle_vector, audio_vector = None, None, None
        index = 0
        if description:
            description_vector = text_vector[index] if index < len(text_vector) else None
            index += 1
        if cleaned_subtitles:
            subtitle_vector = text_vector[index] if index < len(text_vector) else None
            index += 1
        if audio_transcription_translated:
            audio_vector = text_vector[index] if index < len(text_vector) else None

        video_index.add_video(video_id, image_vectors, description_vector, subtitle_vector, audio_vector)

        delete_frames(output_folder, video_id)
        log_message = f"Successfully processed data for {video_id} and frames deleted."
        print(log_message)
        logging.info(log_message)

        #блок извлечения субтитров
        subtitles, subtitles_processing_time = get_subtitles(video_path)

        if subtitles:
            subtitles = translate_text(subtitles)

    else:
        log_message = f"Data for {video_id} was not processed, frames remain in the folder."
        print(log_message)
        logging.warning(log_message)
        log_unprocessed_video(video_id)

    end_time = time.time()
    total_time = end_time - start_time

    statistics[video_id] = {
        "processing_time": total_time,
        "video_duration": video_duration,
        "frames_count": frames_count,
        "description": description if description else None,
        "subtitles": cleaned_subtitles if cleaned_subtitles else None,
        "subtitles_processing_time": subtitles_processing_time,
        "audio_transription": audio_transcription_translated if audio_transcription_translated else None,
        "audio_transription_processing": audio_processing_time
    }

    log_message = f"Total execution time for {video_id}: {total_time} seconds"
    print(log_message)
    logging.info(log_message)

    log_message = f"Video duration for {video_id}: {video_duration} seconds, frames count: {frames_count}"
    print(log_message)
    logging.info(log_message)

    # Сохранение данных после обработки видео
    save_json(vectors, vectors_file_path)
    save_json(statistics, statistics_file_path)
    # Удаление видео после обработки
    os.unlink(video_path)

if __name__ == "__main__":
    try:
        main_handle_videos()
    except Exception as e:
        log_message = f"An error occurred: {str(e)}"
        print(log_message)
        logging.error(log_message)
