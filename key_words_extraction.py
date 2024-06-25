import spacy
import logging
import re
from spacy.lang.ru.stop_words import STOP_WORDS as RUSSIAN_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

# Дополнительные стоп-слова
additional_stop_words = {
    "you", "или", "нет", "да", "ты", "подпишись", "подписывайся", "подписаться", "подписывайтесь", "подпишитесь",
    "вау", "subscribe", "затем", "тот", "тому", "этот", "этому", "этой", "той", "I", "I'm", "still", "here", "there"
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
                f'{vowels_once}+', word) or re.fullmatch(f'{consonants_rus_once}+', word) or re.fullmatch(f'{consonants_eng_once}+', word):
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

    if keyword.lower() in additional_stop_words:
        logging.debug(f"Удалено ключевое слово '{keyword}': находится в списке дополнительных стоп-слов.")
        return False

    return True

# Функция для удаления ненужных знаков из ключевого слова
def clean_keyword(keyword):
    cleaned_keyword = re.sub(r'(^[\W_]+|[\W_]+$)', '', keyword)  # Удаление знаков в начале и конце слова
    cleaned_keyword = re.sub(r'(\w)\1{2,}', r'\1', cleaned_keyword)  # Удаление повторяющихся символов
    return cleaned_keyword.lower().strip()

# Функция для выделения ключевых слов
def extract_keywords(text, nlp):

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
                token.is_stop or token.lower_ in RUSSIAN_STOP_WORDS or token.lower_ in ENGLISH_STOP_WORDS or token.lower_ in additional_stop_words):
            word = clean_keyword(token.text)
            if is_valid_keyword(word):
                keywords.add(word)

    return ", ".join(keywords) if keywords else None


