import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline():
    global models
    if models: return

    orig_id = "akshayvkt/detect-ai-text"
    models['original'] = {
        'tokenizer': AutoTokenizer.from_pretrained(orig_id),
        'model': AutoModelForSequenceClassification.from_pretrained(orig_id).to(device).eval()
    }

    ft_path = "./finetuned-model"
    if not os.path.exists(ft_path):
        ft_path = "ViolentlyPurple/Detect-AI" 
        
    models['finetuned'] = {
        'tokenizer': AutoTokenizer.from_pretrained(ft_path),
        'model': AutoModelForSequenceClassification.from_pretrained(ft_path).to(device).eval()
    }

def classify(text: str, model_choice="finetuned") -> dict:
    if not models: raise RuntimeError("Models not loaded.")
    if model_choice not in models: model_choice = "finetuned"
        
    data = models[model_choice]
    inputs = data['tokenizer'](text, return_tensors="pt", truncation=True, max_length=64, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = data['model'](**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    best_idx = torch.argmax(probs).item()
    score = probs[best_idx].item()
    label = "AI" if best_idx == 1 else "HUMAN"

    return {"label": label.upper(), "score": score}