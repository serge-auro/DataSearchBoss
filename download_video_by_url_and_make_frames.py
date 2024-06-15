import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector, FrameTimecode
import math

@dataclass
class VideoFrame:
    video_url: str
    file: BytesIO

def create_thumbnails_for_video_message(
        video_id: str,
        video_url: str,
        output_folder: str,
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 15
) -> tuple[list[VideoFrame], float, int]:
    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_url).content)
    print(video_data)

    with tempfile.NamedTemporaryFile(delete=False, prefix=f"{video_id}_", suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name
        print(video_path)

    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))
    print(scenes)
    print(len(scenes))
    duration = get_video_duration(video_path)
    print(duration)

    selected_scenes = scenes
    # Если количество обнаруженных сцен больше, чем количество требуемых миниатюр
    if len(scenes) > num_of_thumbnails:
        # Разделяем сцены на начальные, средние и конечные на основе продолжительности видео
        start_scenes, middle_scenes, end_scenes = split_scenes(scenes, duration)
        # Выбираем сцены для миниатюр
        selected_scenes = choose_scenes(start_scenes, end_scenes, middle_scenes, num_of_thumbnails)


    # Добавление первого и последнего кадра, если сцены не были обнаружены
    if not scenes:
        print("Не обнаружено изменений в сценах")
        video_fps = get_video_fps(video_path)
        first_scene = FrameTimecode(timecode='00:00:00', fps=video_fps)
        scenes.append((first_scene, first_scene))
        last_timecode = FrameTimecode(timecode=f'{max(0, int(duration - 0.1)):.1f}', fps=video_fps)

        scenes.append((last_timecode, last_timecode))
        print(scenes)
        selected_scenes = scenes

    os.makedirs(output_folder, exist_ok=True)
    saved_frames_count = 0  # Подсчет успешно сохраненных кадров

    # Обработка и сохранение каждого кадра из selected_scenes
    for i, (scene_start, _) in enumerate(selected_scenes):
        output_path = os.path.join(output_folder, f'key_frame_{video_id}_{i:03d}.jpg')
        if save_frame(video_path, scene_start.get_seconds(), output_path, duration):
            try:
                with open(output_path, 'rb') as frame_data:
                    frames.append(VideoFrame(video_url=video_url, file=BytesIO(frame_data.read())))
                saved_frames_count += 1
                print(f"Сохранен кадр {i + 1}/{len(selected_scenes)}")
            except FileNotFoundError:
                print(f"Не удалось найти файл: {output_path}")
            except Exception as e:
                print(f"Ошибка при сохранении кадра {i + 1}: {e}")
        else:
            print(f"Не удалось сохранить кадр {i + 1}/{len(selected_scenes)}")

    os.unlink(video_path)  # Удаление временного файла
    return frames, duration, saved_frames_count

def save_frame(video_path: str, timecode: float, output_path: str, duration: float) -> bool:
    # Уменьшаем время на 100 миллисекунд, чтобы избежать черного кадра на конце
    safe_duration = duration - 0.1
    if timecode < safe_duration:
        subprocess.call([
            'ffmpeg', '-y', '-i', video_path, '-ss', str(timecode),
            '-vframes', '1', '-update', '1', output_path
        ])
        return True
    return False



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

def split_scenes(scenes, duration):
    start_scenes = [scene for scene in scenes if scene[0].get_seconds() <= duration * 0.1]
    middle_scenes = [scene for scene in scenes if duration * 0.1 < scene[0].get_seconds() < duration * 0.9]
    end_scenes = [scene for scene in scenes if scene[0].get_seconds() >= duration * 0.9]
    return start_scenes, middle_scenes, end_scenes

def choose_scenes(start_scenes, end_scenes, middle_scenes, num_of_thumbnails):
    num_start_scenes = min(len(start_scenes), 2)
    num_end_scenes = min(len(end_scenes), 2)
    selected_scenes = start_scenes[:num_start_scenes] + end_scenes[:num_end_scenes]
    remaining_scenes = num_of_thumbnails - len(selected_scenes)
    if remaining_scenes > 0 and middle_scenes:
        middle_count = min(len(middle_scenes), remaining_scenes)
        step = len(middle_scenes) // middle_count
        selected_scenes.extend(middle_scenes[i] for i in range(0, len(middle_scenes), step)[:middle_count])
    return selected_scenes



# Пример вызова функции
video_id = '2252e44042798abe3c2fe7e64392'
video_url = 'https://cdn-st.rutubelist.ru/media/e7/b5/2252e44042798abe3c2fe7e64392/fhd.mp4'
output_folder = 'frames'
print(create_thumbnails_for_video_message(video_id, video_url, output_folder))
