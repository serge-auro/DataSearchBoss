import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector, FrameTimecode
import ffmpeg


@dataclass
class VideoFrame:
    video_url: str
    file: BytesIO


def create_thumbnails_for_video_message(
        video_id: str,
        video_url: str,
        output_folder: str,
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
) -> tuple[list[VideoFrame], str]:
    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_url).content)

    # Используем уникальный идентификатор видео в качестве префикса для имени временного файла
    with tempfile.NamedTemporaryFile(delete=False, prefix=f"{video_id}_", suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name
        print(f"Video path: {video_path}")

    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))
    print(f"Scenes: {scenes}")
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration}")
    print(len(scenes))

    selected_scenes = []
    if len(scenes) > num_of_thumbnails:
        # Разделяем сцены по секциям видео
        start_scenes = [scene for scene in scenes if scene[0].get_seconds() <= duration * 0.1]
        middle_scenes = [scene for scene in scenes if duration * 0.1 < scene[0].get_seconds() < duration * 0.9]
        end_scenes = [scene for scene in scenes if scene[0].get_seconds() >= duration * 0.9]

        # Определяем, сколько сцен мы можем взять из начала и конца
        num_start_scenes = min(len(start_scenes), 3)
        num_end_scenes = min(len(end_scenes), 3)

        # Добавляем начальные и конечные сцены
        selected_scenes.extend(start_scenes[:num_start_scenes])
        selected_scenes.extend(end_scenes[:num_end_scenes])

        # Определяем, сколько сцен осталось для средней части
        remaining_scenes = num_of_thumbnails - len(selected_scenes)

        # Добавляем средние сцены, если они есть
        if remaining_scenes > 0 and middle_scenes:
            middle_count = min(len(middle_scenes), remaining_scenes)
            step = len(middle_scenes) // middle_count
            selected_scenes.extend(middle_scenes[i] for i in range(0, len(middle_scenes), step)[:middle_count])
    else:
        selected_scenes = scenes

    # Добавляем первый и последний кадр, если сцен недостаточно
    if len(selected_scenes) < num_of_thumbnails:
        video_fps = get_video_fps(video_path)
        if not any(scene[0].get_seconds() == 0 for scene in selected_scenes if isinstance(scene[0], FrameTimecode)):
            selected_scenes.insert(0, (FrameTimecode(timecode='00:00:00', fps=video_fps), FrameTimecode(timecode='00:00:00', fps=video_fps)))  # Добавляем первый кадр
        if not any(scene[0].get_seconds() == duration for scene in selected_scenes if isinstance(scene[0], FrameTimecode)):
            # Используем временную метку последней полной секунды
            last_timecode = FrameTimecode(timecode=f'{int(duration)}', fps=video_fps)
            selected_scenes.append((last_timecode, last_timecode))  # Добавляем последний кадр

    # Создаем output_folder, если он еще не существует
    os.makedirs(output_folder, exist_ok=True)

    # Сохраняем кадры
    for i, scene in enumerate(selected_scenes):
        scene_start, _ = scene
        output_path = os.path.join(output_folder, f'key_frame_{video_id}_{i:03d}.jpg')
        save_frame(video_path, scene_start.get_timecode(), output_path)
        with open(output_path, 'rb') as frame_data:
            frame = VideoFrame(video_url=video_url, file=BytesIO(frame_data.read()))
            frames.append(frame)

    return frames, video_path


def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


def get_video_fps(video_path: str) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    fps = result.stdout.decode().strip().split('/')
    return float(fps[0]) / float(fps[1]) if len(fps) == 2 else float(fps[0])

# Пример вызова функции
#video_id = 'ef285e0241139fc611318ed33071'
#video_url = 'https://cdn-st.rutubelist.ru/media/b0/e9/ef285e0241139fc611318ed33071/fhd.mp4'
#output_folder = 'frames2'
#create_thumbnails_for_video_message(video_id, video_url, output_folder)
