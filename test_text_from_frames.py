import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector
import pytesseract
from PIL import Image


@dataclass
class VideoFrame:
    video_url: str
    file: BytesIO


# Основная функция для создания миниатюр и извлечения текста из них
def create_thumbnails_for_video_message(
        video_url: str,
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10,
        thumbnail_folder: str = 'test_thumbnails',
        text_file: str = 'text_from_frames.txt'
) -> list[VideoFrame]:
    # Загрузка видео по URL и создание временного файла
    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_url).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name

    # Setup Scene Detection
    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))

    # Gradually reduce number of key frames with a sliding window
    while len(scenes) > num_of_thumbnails:
        scenes.pop()
        scenes.pop(0)

    if not os.path.exists(thumbnail_folder):
        os.makedirs(thumbnail_folder)

    with open(text_file, 'w') as f:
        for i, scene in enumerate(scenes):
            scene_start, _ = scene
            output_path = os.path.join(thumbnail_folder, f'key_frame_{i}.jpg')
            save_frame(video_path, scene_start.get_timecode(), output_path)
            with open(output_path, 'rb') as frame_data:
                frame: VideoFrame = VideoFrame(video_url=video_url, file=BytesIO(frame_data.read()))
                frames.append(frame)

            # Извлечение текста из кадра
            text = extract_text_from_image(output_path)
            f.write(f"Frame {i}:\n{text}\n")
            print(f"Frame {i}: {text}")

        os.unlink(video_path)

    # Sometimes threshold is too high to find at least 1 key frame.
    if not frames and frame_change_threshold > 2.6:
        return create_thumbnails_for_video_message(
            video_url=video_url,
            frame_change_threshold=frame_change_threshold - 2.5,
            num_of_thumbnails=num_of_thumbnails,
            thumbnail_folder=thumbnail_folder,
            text_file=text_file
        )
    return frames


# Функция для сохранения кадра
def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])


# Функция для извлечения текста из изображения
def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


# URL вашего видео
video_url = 'https://cdn-st.rutubelist.ru/media/d0/69/7fbb5296415cae300e31ac480044/fhd.mp4'

# Создание миниатюр и извлечение текста из видео
create_thumbnails_for_video_message(video_url)
