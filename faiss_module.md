# Документация модуля FAISS

## Обзор

Модуль `faiss_module.py` предоставляет простой интерфейс для работы с библиотекой FAISS. Он позволяет создавать, управлять и выполнять поиск по индексам FAISS с использованием векторов заданной размерности.

## Класс: FaissIndex

### Методы

---
#### `__init__(self, d, index_type='FlatL2')`

Инициализирует индекс FAISS.

- **Параметры:**
  - `d` (int): Размерность векторов.
  - `index_type` (str): Тип индекса. Поддерживаемые типы: 'FlatL2' и 'IVFFlat'.

- **Исключения:**
  - `ValueError`: Если указан неподдерживаемый тип индекса.

##### Пояснение:
Этот метод инициализирует индекс FAISS заданного типа и размерности. Для `FlatL2` создается индекс `faiss.IndexFlatL2`, а для `IVFFlat` создается индекс `faiss.IndexIVFFlat` с квантователем `faiss.IndexFlatL2` и выполняется его тренировка.

---
#### `add_vectors(self, vectors)`

Добавляет новые вектора в индекс.

- **Параметры:**
  - `vectors` (numpy.ndarray): Вектора для добавления в индекс. Размерность должна быть (n, d), где n - количество векторов, d - размерность.

##### Пояснение:
Этот метод добавляет вектора в индекс. Вектора должны быть в формате `numpy.ndarray` с размерностью (n, d), где n - количество векторов, а d - размерность.

---
#### `remove_vectors(self, ids)`

Удаляет вектора из индекса по их идентификаторам.

- **Параметры:**
  - `ids` (numpy.ndarray): Идентификаторы векторов для удаления.

- **Исключения:**
  - `NotImplementedError`: Если индекс не поддерживает удаление векторов.

##### Пояснение:
Этот метод удаляет вектора из индекса по их идентификаторам. Для индексов типа `FlatL2` явно выбрасывает исключение `NotImplementedError`, так как они не поддерживают удаление. Для индексов, которые поддерживают удаление, выполняется проверка наличия метода `remove_ids`.

---
#### `update_vectors(self, ids, new_vectors)`

Обновляет существующие вектора.

- **Параметры:**
  - `ids` (numpy.ndarray): Идентификаторы векторов для обновления.
  - `new_vectors` (numpy.ndarray): Новые значения векторов.

##### Пояснение:
Этот метод обновляет существующие вектора новыми значениями. Сначала удаляются старые вектора по указанным идентификаторам, затем добавляются новые вектора.

---
#### `search_vectors(self, query_vectors, k)`

Ищет ближайшие соседи для заданных запросных векторов.

- **Параметры:**
  - `query_vectors` (numpy.ndarray): Вектора для поиска. Размерность должна быть (m, d), где m - количество запросных векторов, d - размерность.
  - `k` (int): Количество ближайших соседей.

- **Возвращает:**
  - `D` (numpy.ndarray): Матрица расстояний до ближайших соседей. Размерность (m, k).
  - `I` (numpy.ndarray): Матрица индексов ближайших соседей. Размерность (m, k).

##### Пояснение:
Этот метод выполняет поиск ближайших соседей для заданных запросных векторов. Возвращает матрицу расстояний `D` и матрицу индексов `I` ближайших соседей.

---
#### `save_index(self, file_path)`

Сохраняет индекс в файл.

- **Параметры:**
  - `file_path` (str): Путь к файлу для сохранения индекса.

##### Пояснение:
Этот метод сохраняет текущий индекс в указанный файл. Это позволяет сохранить состояние индекса для последующего использования.

---
#### `load_index(self, file_path)`

Загружает индекс из файла.

- **Параметры:**
  - `file_path` (str): Путь к файлу.

##### Пояснение:
Этот метод загружает индекс из указанного файла. Это позволяет восстановить состояние индекса из ранее сохраненного файла.

---
#### `get_total_vectors(self)`

Возвращает общее количество векторов в индексе.

- **Возвращает:**
  - `int`: Количество векторов в индексе.

##### Пояснение:
Этот метод возвращает общее количество векторов, хранящихся в индексе.