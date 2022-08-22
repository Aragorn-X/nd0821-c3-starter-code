import json
import requests


data = {
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


url = "https://app-prj3.herokuapp.com/"
resp = requests.post(url, json=data)
print("Response code: ", resp.status_code)
print('\nModel classification result: ', resp.json())
