from fastapi import FastAPI, UploadFile, File
import shutil
import os
from deepfake_detection import preprocess_video  # Import your function

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f"video_path = {video_path}")
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video (deepfake detection function)
    deepfake_probability = preprocess_video(video_path)  

    print(f"deepfake_probability = {deepfake_probability}")

    return {"deepfake_probability": deepfake_probability}
