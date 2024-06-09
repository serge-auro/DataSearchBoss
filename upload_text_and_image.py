import os
import requests

# URL вашего FastAPI приложения
url = "http://127.0.0.1:8000/encode"

# Директория, где хранятся изображения
frames_dir = "frames"

# Список текстов для отправки
texts = ["Пример текста", "Другой пример текста"]
#texts = []

# Подготовка списка файлов изображений для отправки
files = []
for filename in os.listdir(frames_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Проверяем формат файлов
        file_path = os.path.join(frames_dir, filename)
        files.append(('images', (filename, open(file_path, 'rb'), 'image/jpeg')))

# Добавляем тексты к запросу
data = {'texts': texts}

# Отправка POST-запроса с изображениями и текстом
response = requests.post(url, files=files, data=data)

# Закрытие всех открытых файлов
for _, file_spec in files:
    file_spec[1].close()  # Закрываем файл после отправки

# Проверяем ответ сервера
if response.status_code == 200:
    try:
        print(response.json())
    except ValueError:
        print("Response is not in JSON format.")
else:
    print(f"Failed to get a proper response. Status code: {response.status_code}")
    print("Response:", response.text)
