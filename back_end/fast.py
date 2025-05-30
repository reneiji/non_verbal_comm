from fastapi import FastAPI

app = FastAPI()

# Define a root '/' endpoint

@app.get("/")
def index():
    return {"message": "Welcome to non verbal communication project!"}
