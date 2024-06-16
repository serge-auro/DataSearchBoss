import json
import spacy
import logging
import time
import re
from spacy.lang.ru.stop_words import STOP_WORDS as RUSSIAN_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_sm")

# Регулярные выражения для проверки последовательностей гласных и согласных
vowels = '[AIUaiuАИОУЯаиоуя]'
consonants_rus = '[ВКСвкс]'
vowels_once = '[EOYeoyЕЁЫЭЮеёыэю]'
consonants_eng_once = '[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz]'
consonants_rus_once = '[БГДЖЗЙЛМНПРТФХЦЧШЩбгджзйлмнпртфхцчшщ]'

# Исключения для буквенно-цифровых последовательностей
exceptions = {
    "H2O", "CO2", "R2D2", "C3PO", "B2B", "B2C", "G8", "G20", "ООН", "НАТО", "ЕС", "ВОЗ", "ВТО",
    "4G", "5G", "AI", "ML", "IT", "CPU", "GPU", "HTML", "CSS", "XML", "IBM", "NASA", "BMW", "GM",
    "Google", "Apple", "Microsoft", "Facebook", "Tesla", "Samsung", "iPhone", "Android", "Windows",
    "Linux", "MacOS", "PlayStation", "Xbox", "FYI", "ASAP", "FAQ", "МГУ", "ФСБ", "ФНС", "МВД",
    "ВДВ", "ВВС", "ВМС", "ГУМ", "ЦУМ", "РАН", "МЧС", "ГИБДД", "МКАД", "ВКонтакте", "Одноклассники",
    "Instagram", "Twitter", "TikTok", "Snapchat", "Reddit", "LinkedIn", "Pinterest", "WhatsApp",
    "Viber", "Telegram", "3Д", "3д", "2Д", "2д", "2D", "3D", "2d", "3d", "4g", "5g", "h2o", "co2",
    "r2d2", "c3po", "b2b", "b2c", "g8", "g20",
}


# Функция для проверки, содержит ли слово одновременно английские и русские буквы
def contains_mixed_languages(word):
    contains_russian = bool(re.search('[А-Яа-я]', word))
    contains_english = bool(re.search('[A-Za-z]', word))
    return contains_russian and contains_english


# Функция для проверки валидности ключевого слова
def is_valid_keyword(keyword):
    if not bool(re.match(r'^[A-Za-zА-Яа-я0-9\s\-]+$', keyword)):
        logging.debug(f"Удалено ключевое слово '{keyword}': содержит недопустимые символы.")
        return False
    if not any(len(word) > 2 for word in keyword.split()):
        logging.debug(f"Удалено ключевое слово '{keyword}': нет слов длиннее двух символов.")
        return False
    if keyword.count(' ') > 3:
        logging.debug(f"Удалено ключевое слово '{keyword}': содержит более трех пробелов.")
        return False
    if contains_mixed_languages(keyword):
        logging.debug(f"Удалено ключевое слово '{keyword}': содержит смешанные английские и русские буквы.")
        return False
    if ("лайк" in keyword.lower() or "like" in keyword.lower() or
            ("субт" in keyword.lower() and not any(sub in keyword.lower() for sub in
                                                   ["субтр", "субте", "субто", "субту", "субта"]))):
        return False

    for word in keyword.split():
        if re.fullmatch(f'{vowels}{{2,}}', word) or re.fullmatch(f'{consonants_rus}{{2,}}', word) or re.fullmatch(
                f'{vowels_once}{{+}}', word) or re.fullmatch(f'{consonants_rus_once}{{+}}', word) or re.fullmatch(f'{consonants_eng_once}{{+}}', word):
            logging.debug(f"Удалено ключевое слово '{keyword}': содержит непрерывные последовательности "
                          f"гласных или согласных.")
            return False
        if word not in exceptions and re.search(r'\d', word) and re.search(r'[A-Za-zА-Яа-я]', word):
            logging.debug(f"Удалено ключевое слово '{keyword}': содержит буквенно-цифровые последовательности.")
            return False
        if len(re.findall(r'(\w)\1{2,}', word)) > 0:
            logging.debug(f"Удалено ключевое слово '{keyword}': содержит повторяющиеся символы.")
            return False
        if len(word) < 3 and word not in exceptions:
            logging.debug(f"Удалено ключевое слово '{keyword}': содержит слово длиной менее трех символов, "
                          f"не являющееся исключением.")
            return False

    if re.search(r'[^\w\s]', keyword) and not len(keyword) > 2:
        return False

    return True


# Функция для удаления ненужных знаков из ключевого слова
def clean_keyword(keyword):
    cleaned_keyword = re.sub(r'(^[\W_]+|[\W_]+$)', '', keyword)  # Удаление знаков в начале и конце слова
    cleaned_keyword = re.sub(r'(\w)\1{2,}', r'\1', cleaned_keyword)  # Удаление повторяющихся символов
    return cleaned_keyword.lower().strip()


# Функция для выделения ключевых слов
def extract_keywords(text):
    doc = nlp(text)
    keywords = set()

    # Извлечение ключевых слов из именных групп
    for chunk in doc.noun_chunks:
        keyword = clean_keyword(chunk.text)
        if is_valid_keyword(keyword):
            keywords.add(keyword)

    # Извлечение ключевых слов из отдельных токенов
    for token in doc:
        if token.is_alpha and not (
                token.is_stop or token.lower_ in RUSSIAN_STOP_WORDS or token.lower_ in ENGLISH_STOP_WORDS):
            word = clean_keyword(token.text)
            if is_valid_keyword(word):
                keywords.add(word)

    return list(keywords) if keywords else None


# Загрузка JSON-файла
try:
    with open('subtitles_extraction/subtitles_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    logging.error("Файл не найден.")
    raise
except json.JSONDecodeError:
    logging.error("Ошибка декодирования JSON.")
    raise

# Начало отсчета времени выполнения программы
start_time = time.time()

# Обработка каждого видео в JSON
for index, (key, video_info) in enumerate(data.items(), start=1):
    subtitles = video_info.get("subtitles", "")
    video_url = video_info.get("url", "")
    key_words = video_info.get("keywords", None)

    if subtitles and not key_words:
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
    elif key_words:
        logging.info(f"Видео №{index} уже обработано. Пропуск.")
    else:
        video_info["keywords"] = None

# Сохранение обновленного JSON-файла
try:
    with open('subtitles_extraction/new_subtitles_1.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
except IOError:
    logging.error("Ошибка записи JSON-файла.")
    raise

# Подсчет количества записей с ключевыми словами и без
with_keywords = sum(1 for video_info in data.values() if video_info["keywords"])
without_keywords = len(data) - with_keywords

# Конец отсчета времени выполнения программы
end_time = time.time()
total_time = end_time - start_time

logging.info(f"Обработка завершена. Обновленный файл сохранен как 'new_subtitles_1.json'.")
logging.info(f"Общее время выполнения программы: {total_time:.4f} секунд")
logging.info(f"Количество записей с ключевыми словами: {with_keywords}")
logging.info(f"Количество записей без ключевых слов: {without_keywords}")