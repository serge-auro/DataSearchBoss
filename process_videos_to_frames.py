# extract_frames.py
import os
import subprocess

# Директория, где хранятся видеофайлы
video_dir = "video"
# Директория, куда будут сохраняться кадры
frames_dir = "frames"

# Создаем папку для кадров, если она не существует
os.makedirs(frames_dir, exist_ok=True)

def extract_frames(video_path, output_dir):
    # Команда для извлечения кадров каждую секунду
    command = [
        'ffmpeg', '-i', video_path, '-vf', 'fps=1', '-qscale:v', '2',
        os.path.join(output_dir, 'frame_%04d.jpg')
    ]
    subprocess.run(command, check=True)

# Проходим по всем файлам в папке video
for filename in os.listdir(video_dir):
    if filename.endswith(('.mp4', '.mkv', '.avi')):
        video_path = os.path.join(video_dir, filename)
        extract_frames(video_path, frames_dir)

print("Кадры извлечены и сохранены в папку frames")
