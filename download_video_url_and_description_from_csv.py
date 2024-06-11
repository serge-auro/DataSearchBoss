import pandas as pd
import json
import os

def extract_video_id(url: str) -> str:
    # Разбиваем URL на части по слешу, берем предпоследнюю часть
    parts = url.strip('/').split('/')
    unique_id = parts[-2]
    return unique_id

def process_video_csv(csv_path):
    # Создаем папку для JSON, если она не существует
    if not os.path.exists('video_description'):
        os.makedirs('video_description')

    # Загрузка данных из CSV
    data = pd.read_csv(csv_path, header=None, skiprows=1, encoding='utf-8')

    # Замена NaN значений на None в описаниях
    data[1] = data[1].apply(lambda x: None if pd.isna(x) else x)

    # Словарь для хранения данных
    video_data = {}

    # Обработка каждой строки данных
    for index, row in data.iterrows():
        url = row[0]  # первый столбец - ссылка
        description = row[1]  # второй столбец - описание, теперь с None вместо NaN

        # Извлечение уникального идентификатора из URL
        unique_id = extract_video_id(url)

        # Сохранение URL и описания в словарь
        video_data[unique_id] = {
            'url': url,
            'description': description
        }

    # Сохранение словаря в JSON файл
    with open('video_description/all_videos.json', 'w', encoding='utf-8') as f:
        json.dump(video_data, f, ensure_ascii=False)

    print("Сохранение описаний и ссылок для всех видео завершено.")

# Вызов функции с указанием пути к CSV файлу
csv_path = 'db_links/yappy_hackaton_2024_400k.csv'
process_video_csv(csv_path)