import json
import logging
import cv2
import time
import re
import easyocr

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Исключения для буквенно-цифровых последовательностей
exceptions = {"3Д", "3д", "2Д", "2д", "2D", "3D", "4G", "5G", "H2O", "CO2", "R2D2", "C3PO",
              "B2B", "B2C", "G8", "G20", "2d", "3d", "4g", "5g", "h2o", "co2",
              "r2d2", "c3po", "b2b", "b2c", "g8", "g20"}

# Инициализация EasyOCR
reader = easyocr.Reader(['ru', 'en'])

# Функция для предобработки кадра
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return gray

# Функция для извлечения субтитров
def extract_subtitles_from_frame(frame):
    try:
        # Предобработка кадра
        processed_frame = preprocess_frame(frame)
        # Извлечение текста
        result = reader.readtext(processed_frame, detail=0)
        text = ' '.join(result)
        return text
    except Exception as e:
        logging.error(f"Ошибка извлечения субтитров: {str(e)}")
        return ""

# Функция для очистки текста субтитров
def clean_subtitles_text(text):
    try:
        # Убираем все знаки препинания и другие знаки кроме букв
        text = re.sub(r'[^A-Za-zА-Яа-я0-9\s]', '', text)

        # Убираем цифры или буквенно-цифровые записи, если они не в списке исключений
        words = text.split()
        cleaned_words = []
        for word in words:
            if word in exceptions:
                cleaned_words.append(word)
            elif not re.search(r'\d', word):
                cleaned_words.append(word)

        text = ' '.join(cleaned_words)

        # Убираем слова, состоящие из русских и английских букв одновременно
        text = re.sub(r'\b(?=[A-Za-zА-Яа-я]*[A-Za-z])(?=[A-Za-zА-Яа-я]*[А-Яа-я])[A-Za-zА-Яа-я]+\b', '', text)

        # Убираем слова, состоящие только из согласных
        vowels = 'аеёиоуыэюяaeiouy'
        text = ' '.join([word for word in text.split() if any(char in vowels for char in word.lower())])

        # Убираем слова, в которых есть более 3 согласных подряд
        text = re.sub(r'\b\w*[бвгджзйклмнпрстфхцчшщ]{4,}\w*\b', '', text, flags=re.IGNORECASE)

        # Убираем слова, записанные буквами разного регистра
        text = ' '.join([word for word in text.split() if word.islower() or word.isupper()])

        # Убираем все слова, состоящие из 3-х символов и меньше
        text = ' '.join([word for word in text.split() if len(word) > 3])

        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logging.error(f"Ошибка очистки текста субтитров: {str(e)}")
        return ""

def get_subtitles(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Не удается открыть видео")

        start_time = time.time()
        frame_count = 0  # Счетчик обработанных кадров
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

        subtitles_text = []
        for sec in range(0, int(video_duration), 2):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            subtitle = extract_subtitles_from_frame(frame)
            subtitle = clean_subtitles_text(subtitle)
            if subtitle.strip():
                subtitles_text.append(subtitle.strip())
                frame_count += 1  # Увеличиваем счетчик только если субтитры найдены

        subtitles = " ".join(subtitles_text).replace('\n', ' ')
        processing_time = time.time() - start_time

        cap.release()

        return subtitles if subtitles else None, processing_time  # Возвращаем None, если субтитров нет
    except Exception as e:
        logging.error(f'Ошибка обработки видео: {str(e)}')
        return None, None

# Пример вызова функции
#video_path = 'path_to_your_video.mp4'  # Замените на путь к вашему видео
#subtitles, processing_time = process_video(video_path)
#if subtitles:
#    print(f"Субтитры: {subtitles}")
#    print(f"Время обработки: {processing_time:.2f} секунд")
#else:
#    print("Субтитры не найдены или произошла ошибка.")
