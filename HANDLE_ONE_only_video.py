import json
import time
import os
import logging
from download_video_by_url_and_make_frames import create_thumbnails_for_video_message, get_video_duration
from upload_only_VIDEO_vector import process_only_video_data, delete_frames

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

vectors_file_paths = ['vectors_1.json', 'vectors_2.json']
statistics_file_path = 'statistics.json'

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
    vectors_1 = load_json(vectors_file_paths[0])
    vectors_2 = load_json(vectors_file_paths[1])
    statistics = load_json(statistics_file_path)
    return vectors_1, vectors_2, statistics


def main_handle_videos():
    vectors_1, vectors_2, statistics = load_last_state()
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

    video_ids = list(all_videos.keys())[:2000]

    for i, video_id in enumerate(video_ids):
        if video_id in vectors_1 or video_id in vectors_2:
            continue  # Skip already processed videos

        start_time = time.time()

        output_folder = "frames"
        frames, video_path = create_thumbnails_for_video_message(video_id, all_videos[video_id]['url'], output_folder)

        success, vector = process_only_video_data(video_id)
        if success and vector is not None:
            if i < 1000:
                vectors_1[video_id] = {
                    "url": all_videos[video_id]['url'],
                    "vector": vector
                }
            else:
                vectors_2[video_id] = {
                    "url": all_videos[video_id]['url'],
                    "vector": vector
                }
            delete_frames(output_folder, video_id)
            log_message = f"Successfully processed data for {video_id} and frames deleted."
            print(log_message)
            logging.info(log_message)
        else:
            log_message = f"Data for {video_id} was not processed, frames remain in the folder."
            print(log_message)
            logging.warning(log_message)

        end_time = time.time()
        total_time = end_time - start_time

        video_duration = get_video_duration(video_path)
        os.unlink(video_path)

        statistics[video_id] = {
            "processing_time": total_time,
            "video_duration": video_duration
        }

        log_message = f"Total execution time for {video_id}: {total_time} seconds"
        print(log_message)
        logging.info(log_message)

        log_message = f"Video duration for {video_id}: {video_duration} seconds"
        print(log_message)
        logging.info(log_message)

        # Сохранение данных после обработки каждого видео
        save_json(vectors_1, vectors_file_paths[0])
        save_json(vectors_2, vectors_file_paths[1])
        save_json(statistics, statistics_file_path)


if __name__ == "__main__":
    try:
        main_handle_videos()
    except Exception as e:
        log_message = f"An error occurred: {str(e)}"
        print(log_message)
        logging.error(log_message)
