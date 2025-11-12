#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')


# In[ ]:


from datasets import load_dataset
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
ds


# ## Preview a row

# In[ ]:


idx= 42 # random index of a row
ds['raw'][idx]


# In[ ]:


ds['raw'][idx]['image']


# # Downloading Dataset as an Image foler and CSV Metadata

# In[ ]:


d_path = "../"  #existing folder where you want to save images and metadata.csv
df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
df.to_csv(f"{d_path}/metadata.csv", index=False)
df


# In[ ]:


import os
os.makedirs(f"{d_path}/images", exist_ok=True)

for i, row in df.groupby('img_id').nth(0).iterrows(): # for images
  image = ds['raw'][i]['image'].save(f"{d_path}/images/{row['img_id']}.jpg")


#   The total image size is around 1.5 GB. The CSV file will have 58,849 rows.
