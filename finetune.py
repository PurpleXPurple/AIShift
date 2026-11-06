import torch, pandas as pd, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

CSV_FILE = "interactive_dataset.csv"
MODEL_DIR = "./finetuned-model"
BATCH_SIZE = 8
STEPS_PER_EPOCH = 30
EPOCHS = 9

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading data from {CSV_FILE}...")
    
    df = pd.read_csv(CSV_FILE).dropna()
    # FIX: Rename 'actual_label' to 'label' so HuggingFace recognizes it
    df = df.rename(columns={"actual_label": "label"})
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True)
    # Now the columns are correct
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    labels = np.array(ds["label"])
    class_weights = 1.0 / np.bincount(labels, minlength=2)
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=STEPS_PER_EPOCH * BATCH_SIZE, 
        replacement=True
    )
    
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)

    print(f"Starting FULL fine-tune on {device} ({EPOCHS} epochs, {STEPS_PER_EPOCH} steps each)...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2).to(device)
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", total=STEPS_PER_EPOCH)
        
        for step, batch in enumerate(loop):
            if step >= STEPS_PER_EPOCH:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / STEPS_PER_EPOCH:.4f}")

    print(f"Overwriting model in {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()