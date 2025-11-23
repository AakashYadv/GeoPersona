from fastapi import FastAPI
import json
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "GeoPersona API is running"}

@app.get("/results")
def get_results():
    file_path = os.path.join("results", "results.json")
    if not os.path.exists(file_path):
        return {"error": "results.json not found. Run the pipeline first."}
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}
