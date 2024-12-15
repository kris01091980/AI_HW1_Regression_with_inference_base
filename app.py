from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
from io import StringIO
import pickle
import warnings

from starlette.responses import JSONResponse

warnings.filterwarnings('ignore')

# Создаем приложение FastAPI
app = FastAPI()

# Загружаем данные из одного .pickle файла
with open("models.pickle", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["linear_model"]  # Используем модель Ridge
scaler = saved_data["scaler"]  # Загружаем StandardScaler
feature_names = saved_data["feature_names"]  # Список признаков
label_encoder = saved_data["label_encoder"]  # LabelEncoder для 'name'
one_hot_columns = saved_data["one_hot_columns"]  # OneHot-кодированные столбцы


# Описание базового объекта с помощью Pydantic
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Описание коллекции объектов
class Items(BaseModel):
    objects: List[Item]


# Метод для обработки одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Преобразование объекта в DataFrame
    data = preprocess(pd.DataFrame([item.dict()]))

    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                       'seats']
    df_train_numeric = data[numeric_columns]

    # Предсказание
    prediction = model.predict(df_train_numeric)
    return float(prediction[0])


# Метод для обработки коллекции объектов
@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    # Преобразование объектов в DataFrame
    data = pd.DataFrame([item.dict() for item in items.objects])

    # Предобработка данных
    data = preprocess(data)

    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                       'seats']
    df_train_numeric = data[numeric_columns]

    # Предсказание
    predictions = model.predict(df_train_numeric)
    return predictions.tolist()


# Метод для обработки .csv файла
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> JSONResponse:
    # Чтение содержимого файла
    contents = await file.read()

    # Создание DataFrame из CSV
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Предобработка данных
    df = preprocess(df)

    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power',
                       'seats']
    df_train_numeric = df[numeric_columns]

    # Предсказание
    predictions = model.predict(df_train_numeric)
    df["predicted_price"] = predictions

    # Сохранение результата в новый CSV файл
    output_file = "predicted_" + file.filename
    df.to_csv(output_file, index=False)
    return {"filename": output_file}


def preprocess(data):
    data = data.dropna()

    # Убедимся, что столбцы являются строковыми перед заменой
    data['mileage'] = data['mileage'].astype(str).str.replace(r' kmpl| km/kg', '', regex=True)
    data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')

    data['engine'] = data['engine'].astype(str).str.replace(' CC', '')
    data['engine'] = pd.to_numeric(data['engine'], errors='coerce')

    data['max_power'] = data['max_power'].astype(str).str.replace(' bhp', '')
    data['max_power'] = pd.to_numeric(data['max_power'], errors='coerce')

    # Удаляем столбец 'torque'
    data = data.drop(columns=['torque'])

    # Преобразуем столбец engine в числовой тип, заменяя все недопустимые значения на NaN
    data['engine'] = pd.to_numeric(data['engine'], errors='coerce')

    # Если нужно, можно заполнить NaN значениями (например, средним или медианным значением)
    data['engine'].fillna(data['engine'].mean(), inplace=True)

    # Преобразуем в целочисленный тип
    data['engine'] = data['engine'].astype(int)

    # Преобразуем столбец seats в целочисленный тип
    data['seats'] = data['seats'].fillna(0).astype(int)

    return data
