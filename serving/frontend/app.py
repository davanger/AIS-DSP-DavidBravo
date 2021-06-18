import joblib
from requests.api import request
import streamlit as st
from PIL import Image
import pandas as pd
import os
import requests
import json

print(os.getcwd())

ICON_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/72/twitter/282/robot_1f916.png"
st.set_page_config(
    page_title="Diabetes Predictions", page_icon=ICON_URL,
)

SERVER_URL = "http://127.0.0.1:8000"


# model = joblib.load("./models/diabetes_model.pkl")
# model = joblib.load("../../models/diabetes_model.pkl")
# dataframe

# Page header
st.header('Diabetes progress predictions')

st.subheader('Multiple patients:')
st.text("Please upload a CSV file with the patients' information")

# Upload CSV file
csv_file = st.file_uploader("Choose a CSV file")
if csv_file:
    st.write("filename : ", csv_file.name)
    dataframe = pd.read_csv(csv_file, sep='\t')
    st.write(dataframe)

st.subheader('Single patient:')
st.text("Please input the patient's information")
# Form columns
col1, col2, col3 = st.beta_columns(3)

AGE = col1.number_input(
    "AGE", 1, None, 50,
)
BP = col1.number_input(
    "BP", 1, None, 80,
)
S3 = col1.number_input(
    "S3", 1, None, 50,
)
S6 = col1.number_input(
    "S6", 1, None, 65,
)

SEX = col2.number_input(
    "SEX", 1, None, 1,
)
S1 = col2.number_input(
    "S1", 1, None, 130,
)
S4 = col2.number_input(
    "S4", 1, None, 5,
)
Y = col2.number_input(
    "Y", 1, None, 202,
)

BMI = col3.number_input(
    "BMI", 1.0, None, value=301.2, step=0.5
)
S2 = col3.number_input(
    "S2", 1.0, None, value=90.2, step=0.5
)
S5 = col3.number_input(
    "S5", 1.0, None, value=5.0626, step=0.5
)


# Model prediction

if st.button("Predict diabetes progression"):  # and csv file
    if csv_file:
        # predictions = model.predict(dataframe)
        package = dataframe.to_json(orient="records")
        url = f"{SERVER_URL}/predict_patients"
        # print(type(package))
        # package = "{" + package + "}"
        package = package.lower()
        print(package)
        print(json.dumps(package))
        predictions =  requests.post(url,headers={'accept': 'application/json','Content-Type': 'application/json'},data=package)
        print(predictions.text)
        predictions = json.loads(json.loads(predictions.text))
        print(type(predictions))

        for pno in predictions.keys():
            st.info(f"Patient {pno} results: {predictions[pno]}")
    else:
        st.info("Predicting from form data.")
        url = f"{SERVER_URL}/predict?age={AGE}&sex={SEX}&bmi={BMI}&bp={BP}&s1={S1}&s2={S2}&s3={S3}&s4={S4}&s5={S5}&s6={S6}"
        predictions =  requests.post(url)
        # st.warning("You need to upload a CSV file")
        for i in range(10):
            st.info(f"Results: {predictions.text}")

