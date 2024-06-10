import json
import time
import os
import logging
from download_video_by_url_and_make_frames import create_thumbnails_for_video_message, get_video_duration
from upload_description_and_frames_to_CLIP import process_video_data, delete_frames
from upload_only_VIDEO_vector import process_only_video_data

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def main_handle_videos():
    json_file_path = 'video_description/all_videos.json'
    statistics_file_path = 'statistics.json'

    statistics = load_json(statistics_file_path)  # Загрузка статистики
    vectors = {}
    vector_file_index = 1
    vector_count = 0

    with open(json_file_path, 'r', encoding='utf-8') as file:
        try:
            all_videos = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error loading JSON file {json_file_path}: {str(e)}")
            return

    for video_id, video_info in all_videos.items():
        start_time = time.time()

        output_folder = "frames"
        frames, video_path = create_thumbnails_for_video_message(video_id, video_info['url'], output_folder)

        success, vector = process_only_video_data(video_id)
        if success and vector is not None:
            vectors[video_id] = {'url': video_info['url'], 'vector': vector}
            delete_frames(output_folder, video_id)
            logging.info(f"Successfully processed data for {video_id} and frames deleted.")
        else:
            logging.warning(f"Data for {video_id} was not processed, frames remain in the folder.")

        vector_count += 1
        if vector_count >= 1000:
            save_json(vectors, f'only_video_vectors_{vector_file_index}.json')
            vector_file_index += 1
            vectors = {}  # Reset the dictionary for the next 1000 videos
            vector_count = 0

        end_time = time.time()
        total_time = end_time - start_time

        video_duration = get_video_duration(video_path)
        os.unlink(video_path)

        statistics[video_id] = {
            "processing_time": total_time,
            "video_duration": video_duration
        }

        logging.info(f"Total execution time for {video_id}: {total_time} seconds")
        logging.info(f"Video duration for {video_id}: {video_duration} seconds")

    # Save remaining videos if any
    if vectors:
        save_json(vectors, f'only_video_vectors_{vector_file_index}.json')

    save_json(statistics, statistics_file_path)


if __name__ == "__main__":
    try:
        main_handle_videos()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
