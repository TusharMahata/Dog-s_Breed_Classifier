from fastapi import FastAPI, UploadFile
from model import model_pipeline
from PIL import Image
import os
from io import BytesIO
import numpy as np
import cv2


app = FastAPI()


@app.get("/")
async def root():
    return {"message":"Welcome to Dog's Breed Classifier"}


def load_image_into_numpy_array(data):
    f = np.array(Image.open(BytesIO(data)))
    image_resized = cv2.resize(f, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return image_rgb


@app.post('/predict')
async def predict(file: UploadFile):
    image = load_image_into_numpy_array(await file.read())
    result = model_pipeline(image)
    return result