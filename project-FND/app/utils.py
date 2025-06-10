import torch
import numpy as np
import shap
import joblib
from transformers import BertTokenizer, BertModel

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Load classifier
classifier = joblib.load("models/logreg_bert_model.pkl")

# Get BERT CLS embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[0][0]
    return cls_embedding.numpy().reshape(1, -1)

# Get SHAP explanation object for the prediction
'''def get_shap_explanation(text):
    emb = get_embedding(text)
    explainer = shap.LinearExplainer(classifier, emb)
    shap_values = explainer.shap_values(emb)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=emb[0],
        feature_names=[f"bert_dim_{i}" for i in range(emb.shape[1])]
    )
    return explanation'''

