import json
import spacy
import logging
import time
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_sm")

# vowels = '[AEIOUYaeiouyАЕЁИОУЫЭЮЯаеёиоуыэюя]'
# consonants = '[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxzБВГДЖЗЙКЛМНПрстфхцчшщбвгджзйклмнпрстфхцчшщ]'

# Регулярные выражения для проверки последовательностей из двух и более согласных и гласных
vowels = '[AIUaiuАИОУЯаиоуя]'
consonants_rus = '[ВКСвкс]'

# Регулярные выражения для проверки последовательностей из одной и более согласных и гласных
vowels_once = '[EOYeoyЕЁЫЭЮеёыэю]'
consonants_eng_once = '[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz]'
consonants_rus_once = '[БГДЖЗЙЛМНПРТФХЦЧШЩбгджзйлмнпртфхцчшщ]'

# Исключения для буквенно-цифровых последовательностей
exceptions = {"3Д", "3д", "2Д", "2д", "2D", "3D", "4G", "5G", "H2O", "CO2", "R2D2", "C3PO",
              "B2B", "B2C", "G8", "G20", "2d", "3d", "4g", "5g", "h2o", "co2",
              "r2d2", "c3po", "b2b", "b2c", "g8", "g20"}


def contains_mixed_languages(word):
    """
    Проверяет, содержит ли слово одновременно английские и русские буквы.
    """
    contains_russian = bool(re.search('[А-Яа-я]', word))
    contains_english = bool(re.search('[A-Za-z]', word))
    return contains_russian and contains_english


def is_valid_keyword(keyword):
    """
    Проверяет, является ли ключевое слово валидным.
    Убирает ключевые слова, содержащие некорректные символы или состоящие из отдельных букв и знаков.
    Также убирает ключевые слова, содержащие последовательности из двух и более согласных или гласных.
    """
    # Условие для проверки валидности ключевого слова
    if not bool(re.match(r'^[A-Za-zА-Яа-я0-9\s\-]+$', keyword)):
        return False

    # Проверка, чтобы ключевое слово содержало хотя бы одно слово длиной больше 2 символов
    if not any(len(word) > 2 for word in keyword.split()):
        return False

    # Проверка, чтобы ключевое слово не содержало более 3 пробелов
    if keyword.count(' ') > 3:
        return False

    if contains_mixed_languages(keyword):
        return False

    # Проверка на наличие отдельных слов, состоящих из двух и более согласных или гласных
    for word in keyword.split():
        if (
                re.fullmatch(f'{vowels}{{2,}}', word) or
                re.fullmatch(f'{consonants_rus}{{2,}}', word) or
                re.fullmatch(f'{vowels_once}{{+}}', word) or
                re.fullmatch(f'{consonants_rus_once}{{+}}', word) or
                re.fullmatch(f'{consonants_eng_once}{{+}}', word)
        ):
            return False

        if (word not in exceptions and re.search(r'\d', word) and
                re.search(r'[A-Za-zА-Яа-я]', word)):
            return False

    return True


def clean_keyword(keyword):
    """
    Удаляет ненужные знаки из ключевого слова.
    """
    # Удаление знаков кроме букв и цифр, если они не являются частью слова с дефисом
    cleaned_keyword = re.sub(r'(^[\W_]+|[\W_]+$)', '', keyword)
    return cleaned_keyword


def extract_keywords(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keyword = chunk.text.strip()
        if len(keyword) > 2:
            keyword = clean_keyword(keyword)
            if is_valid_keyword(keyword):
                keywords.add(keyword)
    return list(keywords) if keywords else None


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
        key_words = extract_keywords(subtitles)
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
with open('subtitles_extraction/subtitles_1_updated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# Подсчет количества записей с ключевыми словами и без
with_keywords = sum(1 for video_info in data.values() if video_info["keywords"])
without_keywords = len(data) - with_keywords

# Конец отсчета времени выполнения программы
end_time = time.time()
total_time = end_time - start_time

logging.info(f"Обработка завершена. Обновленный файл сохранен как 'subtitles_extraction/subtitles_1_updated.json'.")
logging.info(f"Общее время выполнения программы: {total_time:.4f} секунд")
logging.info(f"Количество записей с ключевыми словами: {with_keywords}")
logging.info(f"Количество записей без ключевых слов: {without_keywords}")
