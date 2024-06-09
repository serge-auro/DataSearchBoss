import faiss
import numpy as np

class FaissIndex:
    def __init__(self, d, index_type='FlatL2'):
        """
        Инициализация индекса.

        :param d: Размерность векторов.
        :param index_type: Тип индекса ('FlatL2' или 'IVFFlat').
        """
        self.d = d
        self.index_type = index_type
        if index_type == 'FlatL2':
            self.index = faiss.IndexFlatL2(d)
        elif index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, 100)
            self.index.train(np.random.random((1000, d)).astype('float32'))  # Тренировка индекса
        else:
            raise ValueError("Неподдерживаемый тип индекса")

    def add_vectors(self, vectors):
        """
        Добавление новых векторов в индекс.

        :param vectors: Вектора для добавления.
        """
        self.index.add(vectors)

    def remove_vectors(self, ids):
        """
        Удаление векторов по их идентификаторам.

        :param ids: Идентификаторы векторов для удаления.
        """
        if self.index_type == 'FlatL2':
            raise NotImplementedError("Этот индекс не поддерживает удаление векторов")
        elif hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(faiss.IDSelectorBatch(ids))
        else:
            raise NotImplementedError("Этот индекс не поддерживает удаление векторов")

    def update_vectors(self, ids, new_vectors):
        """
        Обновление существующих векторов.

        :param ids: Идентификаторы векторов для обновления.
        :param new_vectors: Новые значения векторов.
        """
        self.remove_vectors(ids)
        self.add_vectors(new_vectors)

    def search_vectors(self, query_vectors, k):
        """
        Поиск ближайших соседей для заданных запросных векторов.

        :param query_vectors: Вектора для поиска.
        :param k: Количество ближайших соседей.
        :return: Индексы и расстояния до ближайших соседей.
        """
        D, I = self.index.search(query_vectors, k)
        return D, I

    def save_index(self, file_path):
        """
        Сохранение индекса в файл.

        :param file_path: Путь к файлу для сохранения.
        """
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path):
        """
        Загрузка индекса из файла.

        :param file_path: Путь к файлу.
        """
        self.index = faiss.read_index(file_path)

    def get_total_vectors(self):
        """
        Получение общего количества векторов в индексе.

        :return: Количество векторов.
        """
        return self.index.ntotal
