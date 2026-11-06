import json
import os
import hashlib
import threading
from datetime import datetime

REVIEWS_FILE = "reviews.jsonl"
DATASET_FILE = "dataset.jsonl"
_lock = threading.Lock()

def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:12]

def get_reviews():
    if not os.path.exists(REVIEWS_FILE):
        return []
    
    reviews = []
    with _lock:
        # Fixed encoding error by explicitly using utf-8
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        reviews.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    
    reviews.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return reviews

def add_review(name: str, rating: int, text: str, ip: str) -> dict:
    review = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "name": name.strip()[:50] or "Anonymous",
        "rating": max(1, min(5, int(rating))),
        "text": text.strip()[:1000],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ip_hash": _hash_ip(ip)
    }
    
    with _lock:
        with open(REVIEWS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(review) + "\n")
            
    return review

def log_to_dataset(text: str, ai_prob: float, label: str, ip: str):
    """Saves the raw text and the model's guess for future training."""
    entry = {
        "text": text,
        "ai_probability": ai_prob,
        "model_label": label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ip_hash": _hash_ip(ip) # Hash so we can't tie it to the person easily, but can deduplicate
    }
    with _lock:
        with open(DATASET_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")