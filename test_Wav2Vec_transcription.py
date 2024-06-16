from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
from moviepy.editor import VideoFileClip
import requests
import os

# Функция для загрузки видеофайла по URL
def download_video(url, output_path="video.mp4"):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    return output_path

# Функция для извлечения аудио из видеофайла
def extract_audio_from_video(video_path, output_audio_path="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)
    return output_audio_path

# Загрузите модель и процессор
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# URL видеофайла
video_url = 'https://cdn-st.rutubelist.ru/media/f2/bd/d78d847b4fd88c1e210e4bc48764/fhd.mp4'

# Загрузите видеофайл
video_file_path = download_video(video_url)

# Извлеките аудио из видеофайла
audio_file_path = extract_audio_from_video(video_file_path)

# Загрузите аудио файл и преобразуйте его в нужный формат
speech, rate = librosa.load(audio_file_path, sr=16000)
input_values = processor(speech, return_tensors="pt", sampling_rate=rate).input_values

# Прогоните аудио через модель
with torch.no_grad():
    logits = model(input_values).logits

# Получите предсказанные индексы
predicted_ids = torch.argmax(logits, dim=-1)

# Преобразуйте индексы в текст
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)

# Удалите временные файлы
os.remove(video_file_path)
os.remove(audio_file_path)
