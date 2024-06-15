import json
import time
import os
import logging
import requests



from download_video_by_url_and_make_frames import create_thumbnails_for_video_message, get_video_duration
from upload_only_VIDEO_vector import process_only_video_data, delete_frames

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

vectors_file_path = 'new_normalized_vectors_separated_frames_4000-5000.json'
statistics_file_path = 'new_normalized_statistics_separated_frames.json'
unprocessed_videos_log = 'unprocessed_videos.log'

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

def load_last_state():
    vectors = load_json(vectors_file_path)
    statistics = load_json(statistics_file_path)
    return vectors, statistics

def log_unprocessed_video(video_id):
    with open(unprocessed_videos_log, 'a', encoding='utf-8') as file:
        file.write(f"{video_id}\n")

def process_only_video_data(video_id):
    url = "http://176.109.106.184:8000/encode"
    json_file_path = 'video_description/all_videos.json'
    frames_dir = "frames"

    with open(json_file_path, 'r', encoding='utf-8') as file:
        all_videos = json.load(file)

    video_url = all_videos.get(video_id, {}).get('url', None)
    text = None

    files = []
    file_handles = []
    try:
        for filename in os.listdir(frames_dir):
            if filename.startswith(f'key_frame_{video_id}_') and filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(frames_dir, filename)
                file_handle = open(file_path, 'rb')
                files.append(('images', (filename, file_handle, 'image/jpeg')))
                file_handles.append(file_handle)
        data = {'texts': [text]}

        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("сейчас получила ответ")

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
    vectors, statistics = load_last_state()
    json_file_path = 'video_description/all_videos.json'

    with open(json_file_path, 'r', encoding='utf-8') as file:
        try:
            content = file.read().strip()
            if content:
                all_videos = json.loads(content)
            else:
                logging.error(f"JSON file {json_file_path} is empty.")
                return
        except json.JSONDecodeError as e:
            logging.error(f"Error loading JSON file {json_file_path}: {str(e)}")
            return

    last_processed_id = max(vectors.keys(), default=None)
    start_index = 0

    if last_processed_id:
        video_ids = list(all_videos.keys())
        start_index = video_ids.index(last_processed_id) + 1
    else:
        video_ids = list(all_videos.keys())[4000:5000]

    for i in range(start_index, len(video_ids)):
        if i >= 5000:  # Проверка на количество обработанных записей
            break

        video_id = video_ids[i]
        if video_id in vectors:
            continue  # Пропустить уже обработанные видео

    for i in range(start_index, len(video_ids)):
        video_id = video_ids[i]
        if video_id in vectors:
            continue  # Пропустить уже обработанные видео

        start_time = time.time()

        output_folder = "frames"
        frames, video_duration, frames_count = create_thumbnails_for_video_message(video_id, all_videos[video_id]['url'], output_folder)

        success, image_vectors, text_vector = process_only_video_data(video_id)
        if success and image_vectors is not None:
            vectors[video_id] = {
                "url": all_videos[video_id]['url'],
                "vectors": image_vectors  # Список тензоров (списков) для каждого изображения
            }
            delete_frames(output_folder, video_id)
            log_message = f"Successfully processed data for {video_id} and frames deleted."
            print(log_message)
            logging.info(log_message)
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
            "frames_count": frames_count
        }

        log_message = f"Total execution time for {video_id}: {total_time} seconds"
        print(log_message)
        logging.info(log_message)

        log_message = f"Video duration for {video_id}: {video_duration} seconds, frames count: {frames_count}"
        print(log_message)
        logging.info(log_message)

        # Сохранение данных после обработки каждого видео
        save_json(vectors, vectors_file_path)
        save_json(statistics, statistics_file_path)

if __name__ == "__main__":
    try:
        main_handle_videos()
    except Exception as e:
        log_message = f"An error occurred: {str(e)}"
        print(log_message)
        logging.error(log_message)
