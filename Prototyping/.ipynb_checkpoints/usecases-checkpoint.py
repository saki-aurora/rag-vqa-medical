#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ========================== #
# USMLE-Style Scenario Runner
# ========================== #

# ---- installs (first run only) ----
get_ipython().run_line_magic('pip', 'install pyyaml pandas numpy matplotlib scikit-learn sacrebleu rouge-score pillow transformers accelerate torch --quiet')

import os, json, math, re, yaml, difflib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Optional metrics for free-text
import sacrebleu
from rouge_score import rouge_scorer

import torch


# In[ ]:




