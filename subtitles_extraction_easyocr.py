import json
import logging
import os
import cv2
import requests
import time
import re
import easyocr

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к файлу all_videos.json и создание папки subtitles_extraction
all_videos_file = 'video_description/all_videos.json'
subtitles_extraction_dir = 'subtitles_extraction'
os.makedirs(subtitles_extraction_dir, exist_ok=True)

# Имена файлов для результатов
subtitles_json_file = os.path.join(subtitles_extraction_dir, 'subtitles_easyocr.json')
none_subtitles_json_file = os.path.join(subtitles_extraction_dir, 'none_subtitles_1.json')
subtitles_fail_json_file = os.path.join(subtitles_extraction_dir, 'subtitles_fail.json')

# Инициализация EasyOCR
reader = easyocr.Reader(['ru', 'en'])

# Функция для предобработки кадра
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return gray

# Функция для извлечения субтитров
def extract_subtitles_from_frame(frame):
    try:
        # Предобработка кадра
        processed_frame = preprocess_frame(frame)
        # Извлечение текста
        result = reader.readtext(processed_frame, detail=0)
        text = ' '.join(result)
        return text
    except Exception as e:
        logging.error(f"Ошибка извлечения субтитров: {str(e)}")
        return ""


# Исключения для буквенно-цифровых последовательностей
exceptions = {"3Д", "3д", "2Д", "2д", "2D", "3D", "4G", "5G", "H2O", "CO2", "R2D2", "C3PO",
              "B2B", "B2C", "G8", "G20", "2d", "3d", "4g", "5g", "h2o", "co2",
              "r2d2", "c3po", "b2b", "b2c", "g8", "g20"}


def clean_subtitles_text(text):
    try:
        # Убираем все знаки препинания и другие знаки кроме букв
        text = re.sub(r'[^A-Za-zА-Яа-я0-9\s]', '', text)

        # Убираем цифры или буквенно-цифровые записи, если они не в списке исключений
        words = text.split()
        cleaned_words = []
        for word in words:
            if word in exceptions:
                cleaned_words.append(word)
            elif not re.search(r'\d', word):
                cleaned_words.append(word)

        text = ' '.join(cleaned_words)

        # Убираем слова, состоящие из русских и английских букв одновременно
        text = re.sub(r'\b(?=[A-Za-zА-Яа-я]*[A-Za-z])(?=[A-Za-zА-Яа-я]*[А-Яа-я])[A-Za-zА-Яа-я]+\b', '', text)

        # Убираем слова, состоящие только из согласных
        vowels = 'аеёиоуыэюяaeiouy'
        text = ' '.join([word for word in text.split() if any(char in vowels for char in word.lower())])

        # Убираем слова, в которых есть более 3 согласных подряд
        text = re.sub(r'\b\w*[бвгджзйклмнпрстфхцчшщ]{4,}\w*\b', '', text, flags=re.IGNORECASE)

        # Убираем слова, записанные буквами разного регистра
        text = ' '.join([word for word in text.split() if word.islower() or word.isupper()])

        # Убираем все слова, состоящие из 3-х символов и меньше
        text = ' '.join([word for word in text.split() if len(word) > 3])

        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logging.error(f"Ошибка очистки текста субтитров: {str(e)}")
        return ""

# Функция для фильтрации шума из текста субтитров
# def clean_subtitles_text(text):
#     try:
#         text = re.sub(r'[^A-Za-zА-Яа-я0-9\s.,?!]', '', text)
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()
#     except Exception as e:
#         logging.error(f"Ошибка очистки текста субтитров: {str(e)}")
#         return ""

def append_to_json_file(file_path, new_data):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    if not isinstance(existing_data, dict):
        raise ValueError("Existing data should be a dictionary.")

    if not isinstance(new_data, dict):
        raise ValueError("New data should be a dictionary.")

    existing_data.update(new_data)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

try:
    with open(all_videos_file, 'r', encoding='utf-8') as f:
        all_videos = json.load(f)
except FileNotFoundError as e:
    logging.error(f"Файл {all_videos_file} не найден: {str(e)}")
    raise
except json.JSONDecodeError as e:
    logging.error(f"Ошибка декодирования JSON в файле {all_videos_file}: {str(e)}")
    raise

# Ограничение количества видео для тестирования (обрабатываем только первые 50 видео)
all_videos = dict(list(all_videos.items())[:10])

subtitles_results = {}
none_subtitles_results = {}
subtitles_fail_results = {}

program_start_time = time.time()

for index, (video_id, video_info) in enumerate(all_videos.items(), start=1):
    url = video_info['url']
    logging.info(f'Processing {index}/{len(all_videos)}: ID={video_id}, URL={url}')
    try:
        video_data = requests.get(url).content
        video_path = f'temp_video_{video_id}.mp4'
        with open(video_path, 'wb') as video_file:
            video_file.write(video_data)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")

        start_time = time.time()
        frame_count = 0  # Счетчик обработанных кадров
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

        subtitles_text = []
        for sec in range(0, int(video_duration), 2):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            subtitle = extract_subtitles_from_frame(frame)
            subtitle = clean_subtitles_text(subtitle)
            if subtitle.strip():
                subtitles_text.append(subtitle.strip())
                frame_count += 1  # Увеличиваем счетчик только если субтитры найдены

        subtitles = " ".join(subtitles_text).replace('\n', ' ')
        processing_time = time.time() - start_time

        if subtitles:
            subtitles_results[video_id] = {
                "url": url,
                "processing_time": processing_time,
                "video_duration": video_duration,
                "frame_count": frame_count,
                "subtitles": subtitles
            }
            logging.info(f'Subtitles: {subtitles}')
        else:
            none_subtitles_results[video_id] = {
                "url": url,
                "processing_time": processing_time,
                "video_duration": video_duration,
                "frame_count": frame_count,
                "subtitles": None
            }

        cap.release()
        os.remove(video_path)
        logging.info(
            f'Processed {index}/{len(all_videos)}: ID={video_id}, Duration={video_duration:.2f}s, Frames={frame_count}, Time={processing_time:.2f}s')

    except Exception as e:
        processing_time = time.time() - start_time
        subtitles_fail_results[video_id] = {
            "url": url,
            "video_url": url,
            "error": str(e)
        }
        logging.error(f'Failed to process {index}/{len(all_videos)}: ID={video_id}, Error={str(e)}')

append_to_json_file(subtitles_json_file, subtitles_results)
append_to_json_file(none_subtitles_json_file, none_subtitles_results)
append_to_json_file(subtitles_fail_json_file, subtitles_fail_results)

program_end_time = time.time()
program_execution_time = program_end_time - program_start_time
logging.info(f'Processing completed, program execution time: {program_execution_time:.2f}s')
