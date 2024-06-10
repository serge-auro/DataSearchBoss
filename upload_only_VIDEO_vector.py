import json
import os
import requests
import logging

def process_only_video_data(video_id):
    url = "http://127.0.0.1:8000/encode"
    json_file_path = 'video_description/all_videos.json'
    frames_dir = "frames"

    with open(json_file_path, 'r', encoding='utf-8') as file:
        all_videos = json.load(file)

    video_url = all_videos.get(video_id, {}).get('url', None)
    #text = all_videos.get(video_id, {}).get('description', None)
    text=None

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
            vector = response.json().get('features', None)
            result = True
        else:
            log_message = f"Failed to get a proper response. Status code: {response.status_code}\nResponse: {response.text}"
            print(log_message)
            logging.error(log_message)
            vector = None
            result = False
    except Exception as e:
        log_message = f"Error during data processing: {str(e)}"
        print(log_message)
        logging.error(log_message)
        vector = None
        result = False
    finally:
        for file_handle in file_handles:
            file_handle.close()

    return result, vector

def delete_frames(folder, video_id):
    for filename in os.listdir(folder):
        if filename.startswith(f'key_frame_{video_id}_') and filename.endswith(('.jpg', '.jpeg', '.png')):
            os.remove(os.path.join(folder, filename))
    log_message = "All frames have been deleted."
    print(log_message)
    logging.info(log_message)

