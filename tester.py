import pandas as pd
import requests

# URL вашего сервера
BASE_URL = "http://127.0.0.1:8000"


# 1. Запрос для одного объекта
def predict_single_item():
    df = pd.read_csv('cars_train.csv')
    # Преобразуем первую строку в словарь
    item = df.iloc[0].to_dict()

    # Отправляем запрос
    response = requests.post(f"{BASE_URL}/predict_item", json=item)
    if response.status_code == 200:
        print(f"Предсказание для одного объекта: {response.json()}")
    else:
        print(f"Ошибка: {response.status_code}, {response.text}")


# 2. Запрос для коллекции объектов
def predict_multiple_items():
    df = pd.read_csv('cars_train.csv')
    # Преобразуем первые 10 строк в список словарей
    items = df.iloc[15:20].to_dict(orient="records")

    # Формируем запрос
    data = {"objects": items}
    response = requests.post(f"{BASE_URL}/predict_items", json=data)
    if response.status_code == 200:
        print(f"Предсказания для коллекции объектов: {response.json()}")
    else:
        print(f"Ошибка: {response.status_code}, {response.text}")


# 3. Запрос для загрузки .csv файла
def predict_from_csv():
    file_path = "cars_train.csv"
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/csv")}
        response = requests.post(f"{BASE_URL}/predict_csv", files=files)
        if response.status_code == 200:
            print(f"Ответ сервера: {response.json()}")
        else:
            print(f"Ошибка: {response.status_code}, {response.text}")


if __name__ == "__main__":
    predict_single_item()  # Запрос для одного объекта
    predict_multiple_items()  # Запрос для коллекции объектов
    predict_from_csv()  # Запрос для загрузки CSV файла
