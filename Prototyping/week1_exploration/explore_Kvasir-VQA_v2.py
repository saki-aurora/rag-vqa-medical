#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.translate.bleu_score import sentence_bleu

import os


# In[2]:


data_path = "../results/colonoscopy_metadata.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
df.head()


# In[3]:


uc_df = df[df['source'].str.contains("colitis", case=False, na=False)].copy()

print("Ulcerative Colitis subset:", uc_df.shape)
uc_df.head()


# In[4]:


def map_severity(row):
    text = (str(row['answer']) + " " + str(row['question'])).lower()

    # Severe
    if any(word in text for word in [
        "severe", "bleeding", "hemorrhage", "ulcer", "deep ulcer", "spontaneous bleeding"
    ]):
        return 3

    # Moderate
    elif any(word in text for word in [
        "moderate", "friability", "erosion", "marked erythema", "loss of vascular pattern"
    ]):
        return 2

    # Mild
    elif any(word in text for word in [
        "mild", "erythema", "mild inflammation", "slight vascular pattern loss"
    ]):
        return 1

    # Normal
    else:
        return 0

uc_df['severity'] = uc_df.apply(map_severity, axis=1)
print("Severity distribution:\n", uc_df['severity'].value_counts())


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

uc_df['qa_text'] = uc_df['question'].astype(str) + " " + uc_df['answer'].astype(str)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(uc_df['qa_text'])
y = uc_df['severity']

print("Feature matrix shape:", X.shape, " | Labels distribution:", np.bincount(y))


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)


# In[7]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[8]:


import numpy as np

cm = confusion_matrix(y_test, y_pred)
labels = np.unique(np.concatenate((y_test, y_pred)))  # only classes that exist
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("UC Severity Classification (Decision Tree)")
plt.show()

