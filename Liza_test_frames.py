import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector
import ffmpeg


@dataclass
class VideoFrame:
    video_url: str
    file: BytesIO


def create_thumbnails_for_video_message(
        video_url: str,
        output_folder: str,  # Новый параметр для указания папки
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
) -> list[VideoFrame]:
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
    for i, scene in enumerate(scenes):
        scene_start, _ = scene
        output_path = os.path.join(output_folder, f'key_frame_{i}.jpg')
        save_frame(video_path, scene_start.get_timecode(), output_path)
        with open(output_path, 'rb') as frame_data:
            frame: VideoFrame = VideoFrame(video_url=video_url, file=BytesIO(frame_data.read()))
            frames.append(frame)
        # os.remove(output_path)  # Уберите или закомментируйте эту строку, чтобы не удалять файлы
    os.unlink(video_path)

    # Sometimes threshold is too high to find at least 1 key frame.
    if not frames and frame_change_threshold > 2.6:
        return create_thumbnails_for_video_message(
            video_url=video_url,
            output_folder=output_folder,  # Передача новой папки
            frame_change_threshold=frame_change_threshold - 2.5,
            num_of_thumbnails=num_of_thumbnails
        )
    return frames


def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])


video_url = 'https://cdn-st.rutubelist.ru/media/b0/e9/ef285e0241139fc611318ed33071/fhd.mp4'
output_folder = 'test_thumbnails'

create_thumbnails_for_video_message(video_url, output_folder)

