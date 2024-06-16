import time
import os
import logging
import numpy as np
from pymongo import MongoClient

from translation import translate_text
from upload_search_request_to_CLIP import process_search_request
from faiss_module import FaissIndex  # Ваш класс FaissIndex

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')

# Параметры базы данных
db_name = 'video_database'
collection_name = 'videos'
index_mapping_collection_name = 'index_mapping'

# Путь к файлу для записи и загрузки индекса
index_file_path = 'combined_vectors.faiss'

# Весовые коэффициенты
v_weight = 0.6  # Вес для видео
d_weight = 0.1  # Вес для описания
s_weight = 0.1  # Вес для субтитров
a_weight = 0.2  # Вес для аудио



# Функция для загрузки векторов из MongoDB
def load_vectors_from_db():
    client = MongoClient("mongodb://mongo:27017/")
    db = client[db_name]
    collection = db[collection_name]

    vectors = []
    ids = []
    types = []

    cursor = collection.find({})
    for document in cursor:
        video_id = document['id']
        if document['video_vectors']:
            for vec in document['video_vectors']:
                vectors.append(np.frombuffer(vec, dtype='float32'))
                ids.append(video_id)
                types.append('video')
        if document.get('description_vector') is not None:
            vectors.append(np.frombuffer(document['description_vector'], dtype='float32'))
            ids.append(video_id)
            types.append('description')
        if document.get('subtitle_vector') is not None:
            vectors.append(np.frombuffer(document['subtitle_vector'], dtype='float32'))
            ids.append(video_id)
            types.append('subtitle')
        if document.get('audio_vector') is not None:
            vectors.append(np.frombuffer(document['audio_vector'], dtype='float32'))
            ids.append(video_id)
            types.append('audio')

    return vectors, ids, types

try:
    vectors, ids, types = load_vectors_from_db()
    logging.info("Successfully loaded vectors from MongoDB.")
except Exception as e:
    logging.error(f"Failed to load vectors from MongoDB: {str(e)}")
    raise

# Размерность векторов
d = len(vectors[0])
logging.info(f"Vector dimension: {d}")

# Создание или загрузка Faiss индекса
index = FaissIndex(d, index_type='FlatL2')

if os.path.exists(index_file_path):
    # Загрузка индекса из файла
    index.load_index(index_file_path)
    logging.info("Loaded existing Faiss index.")
else:
    # Добавление векторов и сохранение индекса
    index.add_vectors(np.vstack(vectors))
    index.save_index(index_file_path)
    logging.info("Created and saved new Faiss index.")

def user_search_request():
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    video_collection = db[collection_name]

    # Ввод слова или фразу для поиска
    search_query = input("Введите слово или фразу для поиска: ")
    translated_query = None
    if search_query is not None:
        translated_query = translate_text(search_query)
    logging.debug(f"Search query: {search_query}")

    start_time = time.time()  # Засекаем время начала обработки
    try:
        # Обработка введенного текста
        success, vector = process_search_request(translated_query)
        logging.debug(f"Processing result: success={success}, vector={vector}")
        if not success:
            raise ValueError("Failed to process text data.")

        clip_processing_time = time.time() - start_time  # Вычисляем время обработки
        logging.info(f"CLIP processing time: {clip_processing_time:.2f} seconds.")

        # Поиск по Faiss индексу
        query_vector = np.array(vector).astype('float32').reshape(1, -1)
        logging.debug(f"Query vector: {query_vector}")

        k = 500  # Количество ближайших соседей для поиска

        start_faiss_time = time.time()
        distances, indices = index.search_vectors(query_vector, k)
        faiss_search_time = time.time() - start_faiss_time
        logging.info(f"FAISS search time: {faiss_search_time:.2f} seconds.")

        # Создание карты расстояний и корректировка весов
        id_distance_map = {}
        for i in range(k):
            idx = indices[0][i]
            video_id = ids[idx]
            vector_type = types[idx]
            distance = distances[0][i]

            if vector_type == 'video':
                weight = v_weight
            elif vector_type == 'description':
                weight = d_weight
            elif vector_type == 'subtitle':
                weight = s_weight
            elif vector_type == 'audio':
                weight = a_weight
            else:
                weight = 1  # Default weight if type is unknown

            if video_id in id_distance_map:
                id_distance_map[video_id] += distance * weight
            else:
                id_distance_map[video_id] = distance * weight

        # Корректировка весов для видео
        total_weights = {video_id: 0 for video_id in id_distance_map.keys()}
        for idx, video_id in enumerate(ids):
            if types[idx] == 'video':
                total_weights[video_id] += v_weight
            elif types[idx] == 'description':
                total_weights[video_id] += d_weight
            elif types[idx] == 'subtitle':
                total_weights[video_id] += s_weight
            elif types[idx] == 'audio':
                total_weights[video_id] += a_weight

        for video_id in id_distance_map.keys():
            if total_weights[video_id] != 1:
                id_distance_map[video_id] /= total_weights[video_id]

        # Сортировка по суммарному расстоянию
        sorted_results = sorted(id_distance_map.items(), key=lambda item: item[1])

        # Формирование результатов
        video_results = []
        for video_id, total_distance in sorted_results[:10]:
            video_doc = video_collection.find_one({'id': video_id})
            video_url = video_doc.get('url', '')
            video_results.append({
                "url": video_url,
                "video_distance": float(total_distance)
            })

        formatted_video_results = "\n".join(
            [f"{i + 1}. Video Distance: {result['video_distance']:.2f}\nURL: {result['url']}" for i, result in
             enumerate(video_results)])

        log_message = (f"Successfully processed data for query '{search_query}'.\n"
                       f"Processing time: {clip_processing_time:.2f} seconds,\n"
                       f"FAISS search time: {faiss_search_time:.2f} seconds.\n"
                       f"Top 10 results by video distance:\n{formatted_video_results}")
        print(log_message)
        logging.info(log_message)
    except Exception as e:
        log_message = f"An error occurred: {str(e)}"
        print(log_message)
        logging.error(log_message)

if __name__ == "__main__":
    try:
        user_search_request()
    except Exception as e:
        log_message = f"An error occurred: {str(e)}"
        print(log_message)
        logging.error(log_message)

