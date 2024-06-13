import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector
import pytesseract
from PIL import Image
import re
from collections import Counter


@dataclass
class VideoFrame:
    video_url: str
    file: BytesIO


def create_thumbnails_for_video_message(
        video_url: str,
        frame_interval: int = 3,  # Интервал в секундах между кадрами
        thumbnail_folder: str = 'test_thumbnails',
        text_file: str = 'text_from_frames.txt'
) -> list[VideoFrame]:
    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_url).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name

    if not os.path.exists(thumbnail_folder):
        os.makedirs(thumbnail_folder)

    all_texts = []

    subtitles = extract_subtitles(video_path)
    if subtitles:
        all_texts.extend(subtitles)
    else:
        with open(text_file, 'w') as f:
            duration = get_video_duration(video_path)
            for i in range(0, duration, frame_interval):
                output_path = os.path.join(thumbnail_folder, f'frame_{i}.jpg')
                save_frame(video_path, i, output_path)
                with open(output_path, 'rb') as frame_data:
                    frame: VideoFrame = VideoFrame(video_url=video_url, file=BytesIO(frame_data.read()))
                    frames.append(frame)

                text = extract_text_from_image(output_path)
                all_texts.append(text)
                f.write(f"Frame at {i} seconds:\n{text}\n")
                print(f"Frame at {i} seconds: {text}")

    combined_text = combine_texts(all_texts)
    with open(text_file, 'w') as f:
        f.write("\nCombined Text:\n")
        f.write(combined_text)
        print("\nCombined Text:\n" + combined_text)

    os.unlink(video_path)

    return frames


def extract_subtitles(video_path: str) -> list[str]:
    subtitles = []
    try:
        # Extract subtitles using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as subtitle_file:
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-map', '0:s:0', subtitle_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            with open(subtitle_file.name, 'r', encoding='utf-8') as f:
                subtitles = f.readlines()
            os.unlink(subtitle_file.name)
    except Exception as e:
        print(f"Failed to extract subtitles: {e}")
    return subtitles


def get_video_duration(video_path: str) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return int(float(result.stdout))


def save_frame(video_path: str, time_in_seconds: int, output_path: str):
    subprocess.call(
        ['ffmpeg', '-y', '-i', video_path, '-ss', str(time_in_seconds), '-vframes', '1', output_path])


def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    custom_config = r'--oem 3 --psm 6 -l rus'  # Настройка языка на русский
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


# def generate_ngrams(text: str, n: int) -> list[str]:
#     words = text.split()
#     ngrams = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
#     return ngrams
#
#
# def combine_texts(texts: list[str], ngram_size: int = 6) -> str:
#     seen_texts = set()
#     combined_text = ' '.join([text for text in texts if not (text in seen_texts or seen_texts.add(text))])
#     combined_text = re.sub(r'\s+', ' ', combined_text)  # Удаление лишних пробелов и переносов строк
#     combined_text = re.sub(r'(\w)-\s*(\w)', r'\1\2', combined_text)  # Удаление переносов слов
#
#     # Поиск повторяющихся фрагментов, которые встречаются более 5 раз
#     ngrams = generate_ngrams(combined_text, ngram_size)
#     ngram_counts = Counter(ngrams)
#     repeated_fragments = {ngram for ngram, count in ngram_counts.items() if count > 5}
#
#     # Удаление всех вхождений, кроме первого
#     for fragment in repeated_fragments:
#         combined_text = combined_text.replace(fragment, '', ngram_counts[fragment] - 1)
#
#     # Удаление лишних пробелов после удаления фрагментов
#     combined_text = re.sub(r'\s+', ' ', combined_text).strip()
#
#     return combined_text

def combine_texts(texts: list[str]) -> str:
    seen_texts = set()
    combined_text = ''
    for text in texts:
        if text not in seen_texts:
            seen_texts.add(text)
            combined_text += ' ' + text
    combined_text = re.sub(r'\s+', ' ', combined_text)  # Удаление лишних пробелов и переносов строк
    combined_text = re.sub(r'(\w)-\s*(\w)', r'\1\2', combined_text)  # Удаление переносов слов
    return combined_text.strip()


video_url = 'https://cdn-st.rutubelist.ru/media/d0/69/7fbb5296415cae300e31ac480044/fhd.mp4'

create_thumbnails_for_video_message(video_url)
