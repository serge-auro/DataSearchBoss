import json
import numpy as np
import faiss
import torch
# from transformers import AutoTokenizer, AutoModel  # CLIPModel  # CLIPProcessor CLIPTokenizer
from faiss_module import FaissIndex
import os
import ruclip

# Установим переменную среды для предотвращения конфликта OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Загрузка данных
# with open('audio_processing/audio_vectors.json', 'r', encoding='utf-8') as f:
with open('audio_processing/audio_vectors_512.json', 'r', encoding='utf-8') as f:
    audio_vectors = json.load(f)

with open('video_description/all_videos.json', 'r', encoding='utf-8') as f:
    video_descriptions = json.load(f)

# Преобразование аудио векторов в нужный формат
audio_ids = list(audio_vectors.keys())
audio_vectors_list = [np.array(audio_vectors[key][0], dtype=np.float32) for key in audio_ids]

# Проверим размерность векторов аудио
d = len(audio_vectors_list[0])
print(f"Audio vectors dimension: {d}")

# Инициализация и заполнение индекса
d = len(audio_vectors_list[0])
index = FaissIndex(d, index_type='FlatL2')
index.add_vectors(np.vstack(audio_vectors_list))

# Загрузка модели CLIP для преобразования текста в векторы
# clip_model = AutoModel.from_pretrained("ai-forever/ruclip-vit-base-patch32-224")
# clip_tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruclip-vit-base-patch32-224")
# clip_model = CLIPModel.from_pretrained("ai-forever/ruclip-vit-base-patch32-384")
# clip_tokenizer = CLIPTokenizer.from_pretrained("ai-forever/ruclip-vit-base-patch32-384")
# clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("ai-forever/ruclip-vit-base-patch32-384")
# clip_model = CLIPModel.from_pretrained("sberbank-ai/ruCLIP-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("sberbank-ai/ruCLIP-vit-base-patch32")
# model_path = 'C:/Users/lizat/ru-clip'  # /ruclip
model_name = 'ruclip-vit-base-patch32-224'
clip_model, clip_processor = ruclip.load(model_name, device="cpu")

# Линейный слой для преобразования размерности
# linear_layer = torch.nn.Linear(512, d)


def text_to_vector(text):
    inputs = clip_processor(text, return_tensors="pt", padding=True)
    # text_features = clip_model.get_text_features(**inputs)
    text_features = clip_model.encode_text(inputs['input_ids'])
    return text_features.detach().numpy()
    # transformed_features = linear_layer(text_features)
    # return transformed_features.detach().numpy()


# Прием текстового запроса от пользователя
query = input("Введите текстовый запрос: ")
query_vector = text_to_vector(query)

# Проверим размерность векторов текста
print(f"Query vector dimension: {query_vector.shape[1]}")

# Поиск ближайших соседей
D, I = index.search_vectors(query_vector, k=10)  # Например, ищем топ-10 ближайших соседей

# Вывод результатов
print("Наиболее подходящие видеофайлы:")
for idx in I[0]:
    video_id = audio_ids[idx]
    video_info = video_descriptions[video_id]
    print(f"URL: {video_info['url']}, Description: {video_info['description']}")

