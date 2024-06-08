import os
import subprocess
import tempfile
from io import BytesIO

from django.core.files.base import ContentFile
import requests
from scenedetect import detect, ContentDetector
from stores.models import Message, MessageThumbnail

def create_thumbnails_for_video_message(
        message: Message,
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
    ) -> list[MessageThumbnail]:
    thumbnails = []
    video_data = BytesIO(requests.get(message.file.url).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name

    # Setup Scene Detection
    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))

    # Gradully reduce number of key frames with a sliding window
    while len(scenes) > num_of_thumbnails:
        scenes.pop()
        scenes.pop(0)
    for i, scene in enumerate(scenes):
        scene_start, scene_end = scene
        output_path = f'key_frame_{i}.jpg'
        save_frame(video_path, scene_start.get_timecode(), output_path)
        with open(output_path, 'rb') as frame_data:
            file_content = ContentFile(frame_data.read())
            thumbnail: MessageThumbnail = MessageThumbnail.objects.create(message=message, file=file_content)
            thumbnail.file.save(output_path, file_content, save=True)
            thumbnail.save()
            thumbnails.append(thumbnail)
        os.remove(output_path)
    os.unlink(video_path)
    return thumbnails

def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])

# TODO
# TODO
# TODO
# Важно: это фрагмент кода из Django проекта, и вот этот импорт и классы:
# from stores.models import Message, MessageThumbnail
# нужно будет заменить чем-то своим, источником видео (Message) и списком ключевых кадров (MessageThumbnail).
# В моём случае, это были ORM-объекты из таблиц БД.
#
# Из жёстких зависимостей здесь только scenedetect, и ещё
# ОЧЕНЬ ВАЖНО чтобы на машине где работает код был установлен ffmpeg.
# Опять же, видео скачивается прямо в функции по указанной ссылке,
# нужно будет переписать чтобы передавался сразу бинарный объект видоса :)
