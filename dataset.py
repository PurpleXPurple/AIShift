import csv
import os
import torch
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel

CSV_FILE = "interactive_dataset.csv"
MODEL_DIR = "./finetuned-model"
HTML_FILE = "server.html"

# Host IP - 127.0.0.1 is local, but if you want your friend to access it,
# change this to your machine's local IPv4 (e.g., "192.168.1.50")
HOST_IP = "127.0.0.1" 
PORT = 8000

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "text", "model_guess", "actual_label"])
            
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    yield

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str

class SaveRequest(BaseModel):
    text: str
    model_guess: int
    actual_label: int

@app.get("/")
async def serve_html():
    return FileResponse(HTML_FILE, media_type="text/html")

@app.post("/api/predict")
async def predict(req: PredictRequest):
    def _predict():
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        return probs[1].item()
        
    ai_prob = await run_in_threadpool(_predict)
    model_guess = 1 if ai_prob > 0.5 else 0
    
    return {
        "ai_probability": round(ai_prob * 100, 2),
        "model_guess": model_guess
    }

@app.post("/api/save")
async def save_data(req: SaveRequest):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), req.text, req.model_guess, req.actual_label])
    return {"status": "success"}

if __name__ == "__main__":
    print(f"🚀 Server starting!")
    print(f"Local access: http://127.0.0.1:{PORT}")
    if HOST_IP != "127.0.0.1":
        print(f"Network access: http://{HOST_IP}:{PORT}")
    
    # Changed host from "0.0.0.0" to HOST_IP to prevent the getaddrinfo error
    uvicorn.run(app, host=HOST_IP, port=PORT)