#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from transformers import AutoImageProcessor, ViTModel
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image


# In[3]:


CSV_PATH = "../results/colonoscopy_metadata.csv" 
IMG_DIR  = "../images"                             

df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))


# In[5]:


IMG_DIR = Path(IMG_DIR)
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def resolve_img_path(img_id: str):
    for ext in EXTS:
        p = IMG_DIR / f"{img_id}{ext}"
        if p.exists():
            return str(p)
    return None

df["image_path"] = df["img_id"].apply(resolve_img_path)
df = df.dropna(subset=["image_path"]).reset_index(drop=True)
print("After attaching paths:", df.shape)


# In[7]:


df["answer_norm"] = df["answer"].astype(str).str.strip().str.lower()
yn_df = df[df["answer_norm"].isin(["yes", "no"])].copy()

print("Yes/No subset:", yn_df.shape, Counter(yn_df["answer_norm"]))


# In[8]:


MAX_PER_CLASS = 1200  # bump later if you want more data
balanced = []
for lab, grp in yn_df.groupby("answer_norm"):
    balanced.append(grp.sample(min(MAX_PER_CLASS, len(grp)), random_state=42))
yn_df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced subset:", yn_df.shape, Counter(yn_df["answer_norm"]))


# In[9]:


label_map = {"no": 0, "yes": 1}
yn_df["label"] = yn_df["answer_norm"].map(label_map)

train_df, test_df = train_test_split(
    yn_df[["image_path", "label", "question", "answer"]],
    test_size=0.2,
    random_state=42,
    stratify=yn_df["label"]
)
print("Train/Test sizes:", train_df.shape, test_df.shape)


# In[10]:


############
# CNN Baseline 


# In[11]:


TARGET_SIZE = (128, 128)
BATCH_SIZE  = 16
AUTOTUNE    = tf.data.AUTOTUNE

def load_img_tf(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)
    img = img / 255.0
    return img, label

def make_ds(frame, shuffle=False):
    paths = frame["image_path"].values
    labels = frame["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=42)
    ds = ds.map(load_img_tf, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_df, shuffle=True)
test_ds  = make_ds(test_df, shuffle=False)

cnn = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(2, activation="softmax")
])

cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("\n[ CNN ] training…")
cnn.fit(train_ds, validation_data=test_ds, epochs=2, verbose=1)


# In[12]:


y_pred_cnn = np.argmax(cnn.predict(test_ds, verbose=0), axis=1)
y_true     = test_df["label"].values
print("\n[ CNN ] classification report")
print(classification_report(y_true, y_pred_cnn, target_names=["no","yes"]))
cm = confusion_matrix(y_true, y_pred_cnn)
ConfusionMatrixDisplay(cm, display_labels=["no","yes"]).plot(cmap="Blues"); plt.title("CNN Confusion Matrix"); plt.show()

acc_cnn = accuracy_score(y_true, y_pred_cnn)
f1_cnn  = f1_score(y_true, y_pred_cnn, average="macro")


# In[14]:


##################
# ViT + Logistic Regression


# In[15]:


print("\n[ ViT ] extracting frozen features… (sampling to keep it quick)")
# To keep it fast on CPU, subsample
VIT_MAX = 800  # increase later if you want
train_v = train_df.sample(min(VIT_MAX, len(train_df)), random_state=42).reset_index(drop=True)
test_v  = test_df.sample(min(VIT_MAX//4, len(test_df)), random_state=42).reset_index(drop=True)

vit_name = "google/vit-base-patch16-224-in21k"
vit_proc = AutoImageProcessor.from_pretrained(vit_name)
vit      = ViTModel.from_pretrained(vit_name)

def vit_embed(paths):
    # returns CLS embeddings
    batch = []
    feats = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        batch.append(vit_proc(images=im, return_tensors="pt"))
        if len(batch) == 8:
            inp = {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0]}
            with torch.no_grad():
                out = vit(**inp).last_hidden_state[:,0,:].numpy()
            feats.append(out)
            batch = []
    if batch:
        inp = {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0]}
        with torch.no_grad():
            out = vit(**inp).last_hidden_state[:,0,:].numpy()
        feats.append(out)
    return np.vstack(feats)

import torch
torch.set_grad_enabled(False)

X_train_vit = vit_embed(train_v["image_path"].tolist())
y_train_vit = train_v["label"].values
X_test_vit  = vit_embed(test_v["image_path"].tolist())
y_test_vit  = test_v["label"].values

clf_vit = LogisticRegression(max_iter=200, n_jobs=1)
clf_vit.fit(X_train_vit, y_train_vit)
y_pred_vit = clf_vit.predict(X_test_vit)

print("\n[ ViT ] classification report")
print(classification_report(y_test_vit, y_pred_vit, target_names=["no","yes"]))
cm = confusion_matrix(y_test_vit, y_pred_vit)
ConfusionMatrixDisplay(cm, display_labels=["no","yes"]).plot(cmap="Blues"); plt.title("ViT (frozen)+LR Confusion Matrix"); plt.show()

acc_vit = accuracy_score(y_test_vit, y_pred_vit)
f1_vit  = f1_score(y_test_vit, y_pred_vit, average="macro")


# In[16]:


################
# BLIP VQA (zero/few-shot) mapped to yes/no


# In[17]:


print("\n[ BLIP VQA ] evaluating a small sample (maps to yes/no)")
blip_name = "Salesforce/blip-vqa-base"
blip_proc = BlipProcessor.from_pretrained(blip_name)
blip      = BlipForQuestionAnswering.from_pretrained(blip_name)

BLIP_N = 120
sample_blip = test_df.sample(min(BLIP_N, len(test_df)), random_state=42).reset_index(drop=True)

def to_yesno(text):
    t = str(text).lower()
    if "yes" in t:
        return 1
    if "no" in t:
        return 0
    # fallback: default to 'no' if unclear
    return 0

preds_blip, true_blip = [], []
for _, row in sample_blip.iterrows():
    # Use the dataset question (works well for VQA)
    q = str(row["question"]) if isinstance(row["question"], str) and len(str(row["question"]).strip())>0 else "Is the answer yes or no?"
    im = Image.open(row["image_path"]).convert("RGB")
    inputs = blip_proc(images=im, text=q, return_tensors="pt")
    out = blip.generate(**inputs, max_new_tokens=10)
    txt = blip_proc.decode(out[0], skip_special_tokens=True)
    preds_blip.append(to_yesno(txt))
    true_blip.append(row["label"])

y_true_blip = np.array(true_blip)
y_pred_blip = np.array(preds_blip)

print("\n[ BLIP ] classification report")
print(classification_report(y_true_blip, y_pred_blip, target_names=["no","yes"]))
cm = confusion_matrix(y_true_blip, y_pred_blip)
ConfusionMatrixDisplay(cm, display_labels=["no","yes"]).plot(cmap="Blues"); plt.title("BLIP (VQA) Confusion Matrix"); plt.show()

acc_blip = accuracy_score(y_true_blip, y_pred_blip)
f1_blip  = f1_score(y_true_blip, y_pred_blip, average="macro")


# In[19]:


#####################
# Comparison


# In[20]:


results = pd.DataFrame([
    {"Model": "Tiny CNN (from scratch)", "Accuracy": acc_cnn, "Macro-F1": f1_cnn, "Notes": f"{len(train_df)} train / {len(test_df)} test"},
    {"Model": "ViT (frozen) + LR",       "Accuracy": acc_vit, "Macro-F1": f1_vit, "Notes": f"{len(train_v)} train / {len(test_v)} test (sampled)"},
    {"Model": "BLIP (VQA, zero-shot)",   "Accuracy": acc_blip,"Macro-F1": f1_blip,"Notes": f"{len(sample_blip)} VQA eval (sampled)"}
])
print("\n=== Comparison ===")
print(results.to_string(index=False))


# In[ ]:




