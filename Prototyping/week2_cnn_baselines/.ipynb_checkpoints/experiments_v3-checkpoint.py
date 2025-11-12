#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input, Embedding, GRU, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[ ]:


df = pd.read_csv("/Users/jarvis/PycharmProjects/rag-vqa-medical/Prototyping/results/colonoscopy_metadata.csv")

print("Full dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))

uc_df = df[df['source'].str.contains("colitis", case=False, na=False)].copy()

y = uc_df['severity']
labels, y = np.unique(y, return_inverse=True)

print("\nUC subset shape:", uc_df.shape)
print("Classes:", labels, "Counts:", np.bincount(y))
uc_df.head(3)


# In[ ]:





# In[ ]:




