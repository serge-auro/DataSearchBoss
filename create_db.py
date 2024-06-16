from pymongo import MongoClient, ASCENDING, errors
from bson.binary import Binary
import pickle

class VideoIndex:
    def __init__(self, db_name, collection_name, index_mapping_collection_name):
        """
        Инициализация индекса для видео в MongoDB.

        :param db_name: Название базы данных.
        :param collection_name: Название коллекции.
        :param index_mapping_collection_name: Название коллекции для хранения связи индексов и идентификаторов видео.
        """
        try:
            self.client = MongoClient("mongodb://mongo:27017/")
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.index_mapping_collection = self.db[index_mapping_collection_name]
            self.collection.create_index([('id', ASCENDING)], unique=True)
            self.index_mapping_collection.create_index([('index', ASCENDING)], unique=True)
        except errors.ServerSelectionTimeoutError as e:
            print(f"Ошибка подключения к MongoDB: {e}")
            raise
        except errors.CollectionInvalid as e:
            print(f"Ошибка создания коллекции: {e}")
            raise

    def add_video(self, video_id, video_vectors, description_vector, subtitle_vector, audio_vector):
        """
        Добавление видео и его векторов в коллекцию.

        :param video_id: Уникальный идентификатор видео (URL).
        :param video_vectors: Список векторов для видео.
        :param description_vector: Вектор для описания.
        :param subtitle_vector: Вектор для субтитров.
        :param audio_vector: Вектор для аудио.
        """
        document = {
            'id': video_id,
            'video_vectors': [Binary(pickle.dumps(vec, protocol=2)) for vec in video_vectors],
            'description_vector': Binary(pickle.dumps(description_vector, protocol=2)) if description_vector is not None else None,
            'subtitle_vector': Binary(pickle.dumps(subtitle_vector, protocol=2)) if subtitle_vector is not None else None,
            'audio_vector': Binary(pickle.dumps(audio_vector, protocol=2)) if audio_vector is not None else None
        }
        try:
            self.collection.insert_one(document)
            # Сохранение связи индексов и идентификаторов видео
            for idx, vec in enumerate(video_vectors):
                self.index_mapping_collection.insert_one({'index': idx, 'video_id': video_id, 'vector_type': 'video'})
            if description_vector is not None:
                self.index_mapping_collection.insert_one({'index': len(video_vectors), 'video_id': video_id, 'vector_type': 'description'})
            if subtitle_vector is not None:
                self.index_mapping_collection.insert_one({'index': len(video_vectors) + 1, 'video_id': video_id, 'vector_type': 'subtitle'})
            if audio_vector is not None:
                self.index_mapping_collection.insert_one({'index': len(video_vectors) + 2, 'video_id': video_id, 'vector_type': 'audio'})
        except errors.DuplicateKeyError:
            print(f"Дублирующийся ключ: {video_id}")
        except errors.PyMongoError as e:
            print(f"Ошибка при добавлении видео: {e}")
            raise

    def remove_video(self, video_id):
        """
        Удаление видео по идентификатору.

        :param video_id: Идентификатор видео для удаления.
        """
        try:
            self.collection.delete_one({'id': video_id})
            self.index_mapping_collection.delete_many({'video_id': video_id})
        except errors.PyMongoError as e:
            print(f"Ошибка при удалении видео с id {video_id}: {e}")
            raise

    def update_video(self, video_id, new_video_vectors, new_description_vector, new_subtitle_vector, new_audio_vector):
        """
        Обновление векторов для существующего видео.

        :param video_id: Идентификатор видео для обновления.
        :param new_video_vectors: Новые векторы для видео.
        :param new_description_vector: Новый вектор для описания.
        :param new_subtitle_vector: Новый вектор для субтитров.
        :param new_audio_vector: Новый вектор для аудио.
        """
        try:
            self.remove_video(video_id)
            self.add_video(video_id, new_video_vectors, new_description_vector, new_subtitle_vector, new_audio_vector)
        except ValueError as e:
            print(f"Ошибка при обновлении видео: {e}")
            raise

    def close(self):
        """
        Закрытие соединения с базой данных.
        """
        self.client.close()

# Пример использования класса VideoIndex
video_index = VideoIndex(db_name="video_database", collection_name="videos", index_mapping_collection_name="index_mappings")
