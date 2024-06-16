import json
import logging
import os
import cv2
import pytesseract
import requests
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к файлу all_videos.json и создание папки subtitles_extraction
all_videos_file = 'video_description/all_videos.json'
subtitles_extraction_dir = 'subtitles_extraction'
os.makedirs(subtitles_extraction_dir, exist_ok=True)

# Имена файлов для результатов
subtitles_json_file = os.path.join(subtitles_extraction_dir, 'subtitles_1.json')
none_subtitles_json_file = os.path.join(subtitles_extraction_dir, 'none_subtitles_1.json')
subtitles_fail_json_file = os.path.join(subtitles_extraction_dir, 'subtitles_fail.json')


# Функция для извлечения субтитров
def extract_subtitles_from_frame(frame):
    # Настройка на распознавание русского языка
    # custom_config = r'--oem 3 --psm 6 -l rus'
    # Настройка на распознавание русского и английского языков
    custom_config = r'--oem 3 --psm 6 -l rus+eng'
    text = pytesseract.image_to_string(frame, config=custom_config)
    return text


with open(all_videos_file, 'r', encoding='utf-8') as f:
    all_videos = json.load(f)

# Ограничение количества видео для тестирования (если требуется)
all_videos = dict(list(all_videos.items())[100:1000])
'''
МЕНЯТЬ СРЕЗ ИЗ ОБЩЕГО СПИСКА ВИДЕО В ЗАВИСИМОСТИ ОТ УЖЕ ОБРАБОТАННОГО НАБОРА
'''

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
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

        subtitles_text = []
        for sec in range(0, int(video_duration), 2):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            subtitle = extract_subtitles_from_frame(frame)
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


append_to_json_file(subtitles_json_file, subtitles_results)
append_to_json_file(none_subtitles_json_file, none_subtitles_results)
append_to_json_file(subtitles_fail_json_file, subtitles_fail_results)

# with open(subtitles_json_file, 'w', encoding='utf-8') as f:
#     json.dump(subtitles_results, f, ensure_ascii=False, indent=4)
#
# with open(none_subtitles_json_file, 'w', encoding='utf-8') as f:
#     json.dump(none_subtitles_results, f, ensure_ascii=False, indent=4)
#
# with open(subtitles_fail_json_file, 'w', encoding='utf-8') as f:
#     json.dump(subtitles_fail_results, f, ensure_ascii=False, indent=4)

program_end_time = time.time()
program_execution_time = program_end_time - program_start_time
logging.info(f'Processing completed, program execution time: {program_execution_time:.2f}s')
