#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import pandas as pd

from transformers import (
    AutoProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration
)

from PIL import Image
import matplotlib.pyplot as plt
import evaluate
from nltk.translate.bleu_score import sentence_bleu


# In[2]:


df = pd.read_csv("../results/colonoscopy_metadata.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))


# In[3]:


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Example test
preds = ["the mucosa looks inflamed"]
refs = [["the mucosa is inflamed"]]

print("BLEU:", bleu.compute(predictions=preds, references=refs))
print("ROUGE:", rouge.compute(predictions=preds, references=refs))


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")


# In[5]:


# Path where images are stored
IMG_DIR = "../images"

# Take one example row
sample = df.iloc[0]
image_path = os.path.join(IMG_DIR, sample["img_id"] + ".jpg")

if os.path.exists(image_path):
    image = Image.open(image_path).convert("RGB")
    question = sample["question"]

    inputs = blip_processor(image, question, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    prediction = blip_processor.decode(out[0], skip_special_tokens=True)

    print("Question:", question)
    print("Ground Truth Answer:", sample["answer"])
    print("BLIP Prediction:", prediction)
else:
    print("⚠️ Image not found:", image_path)


# In[6]:


subset = df.sample(20, random_state=42)

preds, refs = [], []

for _, row in subset.iterrows():
    image_path = os.path.join(IMG_DIR, row["img_id"] + ".jpg")
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    question = row["question"]
    answer = row["answer"]

    inputs = blip_processor(image, question, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    prediction = blip_processor.decode(out[0], skip_special_tokens=True)

    preds.append(prediction)
    refs.append([answer])  # wrapped in list for evaluate

# Compute metrics
print("Subset size:", len(preds))
print("BLEU:", bleu.compute(predictions=preds, references=refs))
print("ROUGE:", rouge.compute(predictions=preds, references=refs))


# In[ ]:




