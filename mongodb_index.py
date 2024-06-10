from pymongo import MongoClient, ASCENDING, errors
import numpy as np
from bson.binary import Binary
import pickle


class MongoDBIndex:
    def __init__(self, db_name, collection_name, dimension):
        """
        Инициализация индекса MongoDB.

        :param db_name: Название базы данных.
        :param collection_name: Название коллекции.
        :param dimension: Размерность векторов.
        """
        self.dimension = dimension
        try:
            self.client = MongoClient()
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.collection.create_index([('id', ASCENDING)], unique=True)
        except errors.ServerSelectionTimeoutError as e:
            print(f"Ошибка подключения к MongoDB: {e}")
            raise
        except errors.CollectionInvalid as e:
            print(f"Ошибка создания коллекции: {e}")
            raise

    def add_vectors(self, vectors, ids):
        """
        Добавление новых векторов в индекс.

        :param vectors: Вектора для добавления.
        :param ids: Список уникальных идентификаторов (URL видео).
        """
        if len(vectors) != len(ids):
            raise ValueError("Количество векторов должно соответствовать количеству идентификаторов")

        for vec, vec_id in zip(vectors, ids):
            vec_binary = Binary(pickle.dumps(vec, protocol=2))
            try:
                self.collection.insert_one({'id': vec_id, 'vector': vec_binary, 'dimension': self.dimension})
            except errors.DuplicateKeyError:
                print(f"Дублирующийся ключ: {vec_id}")
            except errors.PyMongoError as e:
                print(f"Ошибка при добавлении вектора: {e}")
                raise

    def remove_vectors(self, ids):
        """
        Удаление векторов по их идентификаторам.

        :param ids: Идентификаторы векторов для удаления.
        """
        for vec_id in ids:
            try:
                self.collection.delete_one({'id': vec_id})
            except errors.PyMongoError as e:
                print(f"Ошибка при удалении вектора с id {vec_id}: {e}")
                raise

    def update_vectors(self, ids, new_vectors):
        """
        Обновление существующих векторов.

        :param ids: Идентификаторы векторов для обновления.
        :param new_vectors: Новые значения векторов.
        """
        if len(ids) != len(new_vectors):
            raise ValueError("Количество векторов должно соответствовать количеству идентификаторов")

        try:
            self.remove_vectors(ids)
            self.add_vectors(new_vectors, ids)
        except ValueError as e:
            print(f"Ошибка при обновлении векторов: {e}")
            raise

    def search_vectors(self, query_vectors, k):
        """
        Поиск ближайших соседей для заданных запросных векторов.

        :param query_vectors: Вектора для поиска.
        :param k: Количество ближайших соседей.
        :return: Идентификаторы и расстояния до ближайших соседей.
        """
        neighbors = []
        try:
            for query_vec in query_vectors:
                all_vectors = list(self.collection.find({}))
                distances = [(vec['id'], np.linalg.norm(pickle.loads(vec['vector']) - query_vec)) for vec in
                             all_vectors]
                distances.sort(key=lambda x: x[1])
                neighbors.append(distances[:k])
            return neighbors
        except errors.PyMongoError as e:
            print(f"Ошибка при поиске векторов: {e}")
            raise

    def save_index(self, file_path):
        """
        Сохранение индекса в файл.

        :param file_path: Путь к файлу для сохранения.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(list(self.collection.find({})), f)
        except (OSError, pickle.PickleError) as e:
            print(f"Ошибка при сохранении индекса: {e}")
            raise

    def load_index(self, file_path):
        """
        Загрузка индекса из файла.

        :param file_path: Путь к файлу.
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                for record in data:
                    try:
                        self.collection.insert_one(record)
                    except errors.DuplicateKeyError:
                        print(f"Дублирующийся ключ: {record['id']}")
                    except errors.PyMongoError as e:
                        print(f"Ошибка при вставке данных: {e}")
                        raise
        except (OSError, pickle.PickleError) as e:
            print(f"Ошибка при загрузке индекса: {e}")
            raise
        except errors.BulkWriteError as e:
            print(f"Ошибка при массовой вставке данных: {e}")
            raise

    def get_total_vectors(self):
        """
        Получение общего количества векторов в индексе.

        :return: Количество векторов.
        """
        try:
            return self.collection.count_documents({})
        except errors.PyMongoError as e:
            print(f"Ошибка при получении количества документов: {e}")
            raise
