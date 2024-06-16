import os
import logging
import numpy as np
from pymongo import MongoClient
from faiss_module import FaissIndex  # Ваш класс FaissIndex

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')

# Параметры базы данных
db_name = 'video_database'
collection_name = 'videos'

# Путь к файлу для записи и загрузки индекса
index_file_path = 'combined_vectors.faiss'

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

# Функция для создания или загрузки Faiss индекса
def create_faiss_index():
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


    # Добавление векторов и сохранение индекса
    index.add_vectors(np.vstack(vectors))
    index.save_index(index_file_path)
    logging.info("Created and saved new Faiss index.")

if __name__ == "__main__":
    create_faiss_index()
