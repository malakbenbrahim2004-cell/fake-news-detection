from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, time

app = FastAPI(title="Fake News Detector API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Chargement unique au démarrage
MODEL_PATH = "./bert_final"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ArticleInput(BaseModel):
    title: str
    text:  str

class PredictionOutput(BaseModel):
    label:            str
    fake_probability: float
    real_probability: float
    confidence:       float
    inference_ms:     float

@app.get("/")
def root():
    return {"status": "ok", "model": "bert-base-uncased fine-tuned"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(article: ArticleInput):
    if not article.text.strip():
        raise HTTPException(422, "Le texte ne peut pas être vide")

    t0     = time.time()
    full   = article.title + " " + article.text[:400]
    inputs = tokenizer(full, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    fake_p = probs[0].item()
    real_p = probs[1].item()

    return PredictionOutput(
        label            = "FAKE" if fake_p > real_p else "REAL",
        fake_probability = round(fake_p, 4),
        real_probability = round(real_p, 4),
        confidence       = round(max(fake_p, real_p), 4),
        inference_ms     = round((time.time()-t0)*1000, 1)
    )
