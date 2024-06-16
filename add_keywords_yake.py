import json
import yake
import logging
import time
from langdetect import detect, LangDetectException

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Настройка параметров YAKE для английского и русского языков
language_en = "en"
language_ru = "ru"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfKeywords = 20
kw_extractor_en = yake.KeywordExtractor(lan=language_en, n=max_ngram_size, dedupLim=deduplication_threshold,
                                        top=numOfKeywords, features=None)
kw_extractor_ru = yake.KeywordExtractor(lan=language_ru, n=max_ngram_size, dedupLim=deduplication_threshold,
                                        top=numOfKeywords, features=None)


def extract_keywords(text, lang):
    if lang == "en":
        keywords = kw_extractor_en.extract_keywords(text)
    elif lang == "ru":
        keywords = kw_extractor_ru.extract_keywords(text)
    else:
        return None
    return [keyword for keyword, score in keywords] if keywords else None


# Загрузка JSON-файла
with open('subtitles_extraction/subtitles_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Начало отсчета времени выполнения программы
start_time = time.time()

# Обработка каждого видео в JSON
for index, (key, video_info) in enumerate(data.items(), start=1):
    subtitles = video_info.get("subtitles", "")
    video_url = video_info.get("url", "")

    if subtitles:
        start_extraction_time = time.time()

        # Определение языка субтитров
        try:
            lang = detect(subtitles)
        except LangDetectException:
            logging.warning(f"Не удалось определить язык для видео {key}. Пропускаем...")
            video_info["keywords"] = None
            continue

        key_words = extract_keywords(subtitles, lang)
        extraction_time = time.time() - start_extraction_time
        video_info["keywords"] = key_words

        logging.info(f"Видео №{index}")
        logging.info(f"ID видео: {key}")
        logging.info(f"Ссылка на видео: {video_url}")
        logging.info(f"Субтитры: {subtitles}")
        logging.info(f"Ключевые слова: {key_words}")
        logging.info(f"Время извлечения ключевых слов: {extraction_time:.4f} секунд")
    else:
        video_info["keywords"] = None

# Сохранение обновленного JSON-файла
with open('subtitles_extraction/subtitles_1_updated_yake.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# Подсчет количества записей с ключевыми словами и без
with_keywords = sum(1 for video_info in data.values() if video_info["keywords"])
without_keywords = len(data) - with_keywords

# Конец отсчета времени выполнения программы
end_time = time.time()
total_time = end_time - start_time

logging.info(f"Обработка завершена. Обновленный файл сохранен как 'subtitles_extraction/subtitles_1_updated_yake.json'.")
logging.info(f"Общее время выполнения программы: {total_time:.4f} секунд")
logging.info(f"Количество записей с ключевыми словами: {with_keywords}")
logging.info(f"Количество записей без ключевых слов: {without_keywords}")
