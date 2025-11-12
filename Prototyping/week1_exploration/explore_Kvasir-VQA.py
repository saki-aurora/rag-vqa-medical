#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install datasets')


# In[2]:


from datasets import load_dataset
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
ds


# ## Preview a row

# In[3]:


idx= 42 # random index of a row
ds['raw'][idx]


# In[4]:


ds['raw'][idx]['image']


# # Downloading Dataset as an Image foler and CSV Metadata

# In[5]:


d_path = "../"  #existing folder where you want to save images and metadata.csv
df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
df.to_csv(f"{d_path}/metadata.csv", index=False)
df


# In[6]:


import os
os.makedirs(f"{d_path}/images", exist_ok=True)

for i, row in df.groupby('img_id').nth(0).iterrows(): # for images
  image = ds['raw'][i]['image'].save(f"{d_path}/images/{row['img_id']}.jpg")


#   The total image size is around 1.5 GB. The CSV file will have 58,849 rows.
