# Code for local API.
# Put the code for your API here.
from fastapi import FastAPI
from starter import inference_model_local
from pydantic import BaseModel, Field
from typing import Union
import pandas as pd

# Creating input data model structure using Pydantic
class DataModel(BaseModel):
    age: Union[int, list]
    workclass: Union[str, list]
    fnlgt: Union[int, list]
    education: Union[str, list]
    education_num: Union[int, list] = Field(alias="education-num")
    marital_status: Union[str, list] = Field(alias="marital-status")
    occupation: Union[str, list]
    relationship: Union[str, list]
    race: Union[str, list]
    sex: Union[str, list]
    capital_gain: Union[int, list] = Field(alias="capital-gain")
    capital_loss: Union[int, list] = Field(alias="capital-loss")
    hours_per_week: Union[int, list] = Field(alias="hours-per-week")
    native_country: Union[str, list] = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": [21],
                "workclass": ["Private"],
                "fnlgt": [338400],
                "education": ["Masters"],
                "education-num": [10],
                "marital-status": ["Married-AF-spouse"],
                "occupation": ["Sales"],
                "relationship": ["Other-relative"],
                "race": ["White"],
                "sex": ["Male"],
                "capital-gain": [2100],
                "capital-loss": [0],
                "hours-per-week": [32],
                "native-country": ["Scotland"]
            }
        }


# Instantiating the app
app = FastAPI()

# Defining a GET on the specified endpoint
@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome!"}

# Defining a POST request performing model inference
@app.post("/")
async def perform_inference(input_data: DataModel):
    '''
    POST that does model inference using Pydantic model to ingest the body from POST
    :param input_data: input data structure defined in DataModel class
    :return: classifier prediction
    '''
    input_dict = input_data.dict()
    input_df = pd.DataFrame(input_dict)
    input_df.columns = input_df.columns.str.replace("[_]", "-")
    prediction = inference_model_local.execute_inference(in_data=input_df)
    return prediction
