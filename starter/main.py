# Put the code for your API here.
from fastapi import FastAPI

# instantiating the app
app = FastAPI()

# defining a GET on the specified endpoint
@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome!"}
