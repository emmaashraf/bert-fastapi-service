from fastapi import FastAPI, Query
from transformers import BertTokenizer, BertForSequenceClassification
from pydantic import BaseModel
import torch

class Item(BaseModel):
    text: str

app = FastAPI()

# Load your pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load('bert_sequence_classification_model.pth'))

@app.get("/predict")
async def predict(text: str = Query(None, min_length=1)):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs
    return {"prediction": str(prediction)}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the BERT NLP Model API"}
