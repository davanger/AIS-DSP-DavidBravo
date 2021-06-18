import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json
from typing import List

model = joblib.load("../../models/diabetes_model.pkl")
app = FastAPI()


class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/random_message")
async def random_message():
    return 4


@app.post("/predict")
async def predict_diabetes_progresss(age: float, sex: float, bmi: float,
                                     bp: float, s1: float, s2: float, s3:
                                     float, s4: float, s5: float, s6: float):
    print(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
    model_input_data = np.array(
        [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]).reshape(1, -1)
    progression = model.predict(model_input_data)
    # print(type(progression))
    print(f"model input {model_input_data}")
    return progression[0]

@app.post("/predict_patients")
async def predict_diabetes_progresss(patients: List[DiabetesInfo]):
    progressions = {}
    for pno, patient in enumerate(patients):
        model_input_data = np.array([patient.age, patient.sex, patient.bmi, patient.bp, patient.s1, patient.s2, patient.s3, patient.s4, patient.s5, patient.s6]).reshape(1, -1)
        progressions[str(pno)] = float(model.predict(model_input_data))
    # print(type(progression))
    print(progressions)
    progression = json.dumps(progressions, indent = 4) 
    print(f"model input {model_input_data}")
    return progression

# @app.post("/predict_obj")
# async def predict_diabetes_progress_1(diabetes_info: DiabetesInfo):
#     print(diabetes_info.dict())
#     model_input_data = pd.DataFrame([diabetes_info.dict()])
#     print(f"model input {model_input_data.to_numpy().reshape(1, -1)}")
#     # progression = model.predict(model_input_data.to_numpy().reshape(1, -1))
#     # return progression
