#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input, Embedding, GRU, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# In[2]:


df = pd.read_csv("../results/colonoscopy_metadata.csv")

print("Full dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))


# In[3]:


uc_df = df[df['source'].str.contains("colitis", case=False, na=False)].copy()

uc_df['qa_text'] = uc_df['question'].astype(str) + " " + uc_df['answer'].astype(str)

y = uc_df['answer_type']
labels, y = np.unique(y, return_inverse=True)

print("\nUC subset shape:", uc_df.shape)
print("Classes:", labels, "Counts:", np.bincount(y))
print(uc_df[['qa_text','answer_type']].head(3))


# In[4]:


vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(uc_df['qa_text'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)


# In[5]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("UC Classification (Decision Tree, answer_type)")
plt.show()

