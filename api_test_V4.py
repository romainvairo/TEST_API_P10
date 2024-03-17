from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from pydantic import BaseModel
from tempfile import TemporaryFile
from google.cloud import storage
from google.oauth2 import service_account
import joblib
from dotenv import load_dotenv
import os
import io



app = FastAPI()
# filename = 'logistic_regression.sav'

# # load the model from disk
# loaded_model = joblib.load(filename)

class User_input(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int 



@app.post("/predict/")
def predict_diabetes(input : User_input):

    cole = {
        'pregnancies': input.pregnancies, 
        'glucose' : input.glucose,
        'blood_pressure' : input.blood_pressure,
        'skin_thickness' :  input.skin_thickness,
        'insulin' : input.insulin,
        'bmi' : input.bmi,
        'diabetes_pedigree' : input.diabetes_pedigree,
        'age' :  input.age,
    }

    # print('Accessing model in cloud...')
    bucket_name = 'api_test_p10'
    model_bucket = 'tabpfn.pkl'
    json_key_path = 'credentials.json'
    # load_dotenv()

    # json_key_path = os.getenv("CREDENTIALS")

    credentials = service_account.Credentials.from_service_account_file(json_key_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_bucket)

    with TemporaryFile() as temp_file:
        blob.download_to_file(temp_file)
        temp_file.seek(0) 
        loaded_model = joblib.load(temp_file)


    input_data = list(cole.values())

    scaler = StandardScaler()

    # Adapter le scaler aux données d'entrée
    scaler.fit(np.array(input_data).reshape(-1, 1))

    # Normaliser chaque chiffre un par un
    normalized_input = scaler.transform(np.array(input_data).reshape(-1, 1))

    input_data_as_numpy_array = np.asarray(normalized_input)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # prediction = loaded_model.predict(input_data_reshaped)
    prediction = loaded_model.predict(input_data_reshaped)

    return {"prediction" : float(prediction[0])}


    # if (prediction[0] == 0):
    #     return 'The person is not diabetic'
    # else:
    #     return'The person is diabetic'


