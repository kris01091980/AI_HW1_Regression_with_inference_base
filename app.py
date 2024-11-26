from io import StringIO
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return float(prediction[0])


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    data = pd.DataFrame([item.dict() for item in items.objects])
    data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    return predictions.tolist()


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> str:
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    df['predicted_price'] = predictions
    output_file = "predicted_" + file.filename
    df.to_csv(output_file, index=False)
    return {"filename": output_file}
