#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')


# In[14]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# In[15]:


IMG_DIR = "../images"  
IMG_SIZE = (224, 224)  

files = os.listdir(IMG_DIR)
print("Total files:", len(files))
print("First 5:", files[:5])

all_images = []
all_labels = []

for i, f in enumerate(files[:1000]):
    try:
        img = load_img(os.path.join(IMG_DIR, f), target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        all_images.append(arr)
        all_labels.append(i % 2)  # fake binary labels
    except:
        continue

X = np.array(all_images)
y = np.array(all_labels)

print("X shape:", X.shape, "y shape:", y.shape)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train:", X_train.shape, "Test:", X_test.shape)


# In[17]:


base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False  # freeze ResNet

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[18]:


history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32
)


# In[19]:


y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()


# In[ ]:




