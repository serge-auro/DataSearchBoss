from fastapi import FastAPI, HTTPException
import paramiko
import logging

# Настройка логирования
logging.basicConfig(filename='api_requests.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Параметры удаленного сервера
server_ip = '176.109.106.184'
server_port = 8080

# Пути к скриптам на удаленном сервере
remote_script_path = '/home/user1/projects/DataSearchBoss/create_FAISS_index.py'
remote_handle_script_path = '/home/user1/projects/DataSearchBoss/HANDLE_ONE_with_MONGO.py'
remote_search_script_path = '/home/user1/projects/DataSearchBoss/HANDLE_TWO_search_with_FAISS.py'


# Функция для запуска удаленной команды через SSH
def run_remote_script(script_path, args=None):
    try:
        # Создание SSH клиента
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Подключение к серверу без использования ключа или пароля
        ssh_client.connect(server_ip, port=server_port)

        # Подготовка команды для запуска скрипта с аргументами
        command = f'python3 {script_path}'
        if args:
            command += f' {args}'

        # Запуск удаленного скрипта
        stdin, stdout, stderr = ssh_client.exec_command(command)

        # Чтение вывода и ошибок
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()

        # Закрытие подключения
        ssh_client.close()

        return stdout_result, stderr_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/recreate_faiss/")
def run_create_faiss_index():
    stdout, stderr = run_remote_script(remote_script_path)
    if stderr:
        raise HTTPException(status_code=500, detail=f"Error: {stderr}")
    logging.info(f"Encoded response: {stdout}")
    return {"stdout": stdout}


@app.get("/process_video/")
def handle_videos(video_name: str, description_name: str):
    args = f"{video_name} {description_name}"
    stdout, stderr = run_remote_script(remote_handle_script_path, args=args)
    if stderr:
        raise HTTPException(status_code=500, detail=f"Error: {stderr}")
    logging.info(f"Encoded response: {stdout}")
    return {"stdout": stdout}


@app.get("/get_videos/")
def user_search_request(word: str):
    stdout, stderr = run_remote_script(remote_search_script_path, args=word)
    if stderr:
        raise HTTPException(status_code=500, detail=f"Error: {stderr}")
    logging.info(f"Encoded response: {stdout}")
    return {"stdout": stdout}


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the FAISS Index and Video Handling API. Use /run-create-faiss-index to create the FAISS index, /handle-videos to handle videos, or /user-search-request to search for videos."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
