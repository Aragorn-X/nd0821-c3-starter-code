import json
from main import app
from fastapi.testclient import TestClient


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


client = TestClient(app)
data = json.dumps(data)
resp = client.post("http://127.0.0.1:8000/", data=data)   # response
print('\nModel classification result: ', resp.json())
