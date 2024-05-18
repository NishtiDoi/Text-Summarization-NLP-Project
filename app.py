from fastapi import FastAPI, Form
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.textSummarizer.pipeline.prediction import PredictionPipeline
import subprocess
from src.textSummarizer.logging import logger


app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
        if result.returncode == 0:
            return Response("Training successful !!")
        else:
            return Response(f"Error Occurred! {result.stderr}")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        prediction = obj.predict(text)
        return {"summary": prediction}
    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
