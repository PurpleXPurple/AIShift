import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ORIGINAL_MODEL_ID = "akshayvkt/detect-ai-text"
FINETUNED_MODEL_DIR = "./finetuned-model"

TEST_SET = [
    # --- HUMAN (17 cases) ---
    ("lol i cant believe i stayed up till 4am watching paint dry on youtube. my sleep schedule is ruined", 0),
    ("The Declaration of Independence was adopted by the Continental Congress on July 4, 1776. It announced that the thirteen American colonies were at war with Great Britain.", 0),
    ("I swear my washing machine eats socks. I put in ten pairs and only get back seven. Where do they go?", 0),
    ("Listen, I know it sounds crazy, but I think my goldfish actually recognizes me. He swims to the glass whenever I walk in the room.", 0),
    ("The mitochondria is the powerhouse of the cell. I remember learning that in 7th grade biology and it's the only thing I retained.", 0),
    ("Ugh, the weather today is so gross. It's that sticky, humid heat where you just sweat through your shirt the second you step outside.", 0),
    ("So my grandma called me today and asked how to reset her router. I spent 40 minutes trying to explain what an IP address is.", 0),
    ("I'm reading this book about the history of salt. I know that sounds boring, but it's actually fascinating how wars were fought over it.", 0),
    ("Just got back from the gym. My legs feel like jelly and I'm pretty sure I'm going to collapse on the way to the shower.", 0),
    ("The meeting was supposed to start at 9, but the boss was late. We just sat there in awkward silence for fifteen minutes.", 0),
    ("I can't believe I failed my driving test again. I parallel parked perfectly but then I forgot to check my blind spot when merging.", 0),
    ("Does anyone else remember the smell of those old mimeograph machines they used to have in schools? That purple ink smell?", 0),
    ("My neighbor's dog won't stop barking. It's been three hours. I'm about to lose my mind.", 0),
    ("I tried to make sourdough starter from scratch. It just grew mold. I am not a baker.", 0),
    ("The plot of that movie made absolutely no sense. Why did the villain just leave the hero alone in the room with the key?", 0),
    ("Okay so my code keeps throwing a segmentation fault and I have no idea why. The debugger just says 'core dumped'. Thanks for nothing.", 0),
    ("I'm so tired of people driving slow in the left lane. It's called the fast lane for a reason, people!", 0),

    # --- AI (17 cases) ---
    ("Sure, here is a revised version of your paragraph that improves the flow and clarity while maintaining your original meaning.", 1),
    ("As an AI, I am incapable of experiencing emotions, but I can understand how this situation might be frustrating for you.", 1),
    ("It is crucial to approach this problem with a multifaceted strategy that addresses both the immediate symptoms and the underlying systemic issues.", 1),
    ("In the rapidly evolving landscape of artificial intelligence, it is essential to remain adaptable and continuously learn new skills.", 1),
    ("First, preheat the oven to 350 degrees. Next, mix the dry ingredients in a large bowl. Finally, fold in the wet ingredients until just combined.", 1),
    ("I cannot fulfill this request. My safety guidelines strictly prohibit the generation of harmful, violent, or explicit content.", 1),
    ("The interplay between macroeconomic policy and microeconomic behavior creates a complex web of incentives that must be carefully navigated.", 1),
    ("Here is a list of the top 10 most influential technological innovations of the 21st century, along with a brief explanation of their impact.", 1),
    ("Delve into the rich tapestry of human history, where every epoch offers profound lessons for the future of our society.", 1),
    ("It is important to note that correlation does not imply causation. Further empirical research is required to establish a definitive link.", 1),
    ("Let's explore the nuances of this topic. On one hand, there are significant benefits. On the other hand, there are potential drawbacks.", 1),
    ("The novel's protagonist, a quintessential anti-hero, embarks on a journey of self-discovery that challenges the reader's moral compass.", 1),
    ("To optimize your website for search engines, you must ensure that your meta tags, headers, and alt text are all properly configured.", 1),
    ("I understand that you are feeling overwhelmed. It might be helpful to break the task down into smaller, more manageable steps.", 1),
    ("The data indicates a statistically significant increase in user engagement following the implementation of the new UI.", 1),
    ("As we look towards the future, it is evident that sustainable practices will become paramount in the global business environment.", 1),
    ("I am sorry, but I cannot browse the internet or access real-time information. My knowledge is limited to my training data.", 1),
]

def predict(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        ai_prob = probs[0][1].item()
    return 1 if ai_prob > 0.5 else 0, ai_prob

def evaluate_model(model_id, model_name):
    print(f"\n--- Evaluating: {model_name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()

    true_labels, pred_labels, latencies = [], [], []

    for i, (text, label) in enumerate(TEST_SET):
        start = time.perf_counter()
        pred, prob = predict(model, tokenizer, text, device)
        end = time.perf_counter()
        
        latencies.append(end - start)
        true_labels.append(label)
        pred_labels.append(pred)
        
        status = "✅" if pred == label else "❌"
        print(f"{status} True: {'AI' if label==1 else 'HUMAN'} | Pred: {'AI' if pred==1 else 'HUMAN'} ({prob*100:.1f}%) | {text[:50]}...")

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary", zero_division=0)
    avg_latency = sum(latencies) / len(latencies) * 1000

    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Avg Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    evaluate_model(ORIGINAL_MODEL_ID, "Original Baseline")
    if os.path.exists(FINETUNED_MODEL_DIR):
        evaluate_model(FINETUNED_MODEL_DIR, "Fine-tuned Model")