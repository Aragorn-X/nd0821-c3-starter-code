import json
from main import app
from fastapi.testclient import TestClient


client = TestClient(app)

# Testing GET
def test_say_welcome():
    r = client.get("http://127.0.0.1:8000/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!"}

# Testing POST for prediction <= 50k
def test_perform_inference():
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

    data = json.dumps(data)
    resp = client.post("http://127.0.0.1:8000/", data=data)
    resp_str = json.loads(resp.text)
    assert resp.status_code == 200
    assert resp_str == "<=50K"


# Testing POST for prediction > 50k
def test_perform_inference_2():
    data_2 = {
        "age": [40],
        "workclass": ["Private"],
        "fnlgt": [193524],
        "education": ["Doctorate"],
        "education-num": [16],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Prof-specialty"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [60],
        "native-country": ["United-States"]
    }

    data_2 = json.dumps(data_2)
    resp = client.post("http://127.0.0.1:8000/", data=data_2)
    resp_str = json.loads(resp.text)
    assert resp.status_code == 200
    assert resp_str == ">50K"
