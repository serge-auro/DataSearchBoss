import json
import requests
import logging

# Настройка логирования
logging.basicConfig(filename='search_processing.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_search_request(query_text):
    url = "http://127.0.0.1:8000/encode"

    if not query_text:
        log_message = "Empty query text provided."
        print(log_message)
        logging.error(log_message)
        return False, None

    data = {'texts': [query_text]}
    files = []  # Пустой список файлов

    try:
        logging.debug(f"Sending request to {url} with data: {data} and files: {files}")
        response = requests.post(url, files=files, data=data)
        logging.debug(f"Received response with status code: {response.status_code}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                logging.debug(f"Response JSON: {response_json}")
                text_features = response_json.get('text_features', None)
                if text_features is None:
                    log_message = "No text_features found in the response."
                    print(log_message)
                    logging.error(log_message)
                    result = False
                    vector = None
                else:
                    result = True
                    vector = text_features[0]  # Предполагается, что нужен первый вектор из списка
            except json.JSONDecodeError as e:
                log_message = f"Error decoding JSON response: {str(e)}"
                print(log_message)
                logging.error(log_message)
                vector = None
                result = False
        else:
            log_message = f"Failed to get a proper response. Status code: {response.status_code}\nResponse: {response.text}"
            print(log_message)
            logging.error(log_message)
            vector = None
            result = False
    except Exception as e:
        log_message = f"Error during data processing: {str(e)}"
        print(log_message)
        logging.error(log_message)
        vector = None
        result = False

    return result, vector


# Вызов функции для обработки данных
#print(process_search_request('спорт'))