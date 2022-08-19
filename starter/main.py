# Put the code for your API here.
from fastapi import FastAPI
from input_data_definition import InputStructure

# instantiating the app
app = FastAPI()

# defining a GET on the specified endpoint
@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome!"}

@app.post("/")
async def perform_inference(input_data: InputStructure):
    return