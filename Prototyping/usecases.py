#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install statsmodels


# In[2]:


pip install scipy


# In[4]:


import os, re, json, math, yaml, difflib, warnings, random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np, pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_fscore_support)
from scipy.stats import mstats 
from statsmodels.stats.contingency_tables import mcnemar

import sacrebleu
from rouge_score import rouge_scorer
import nltk
try:
    nltk.download('wordnet', quiet=True)
except:
    pass
from nltk.translate.meteor_score import single_meteor_score

import torch
from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    BlipForQuestionAnswering, BlipProcessor,
    AutoProcessor, Blip2ForConditionalGeneration
)


# In[5]:


# ---------------------------- #
#           CONFIG             #
# ---------------------------- #
SCENARIOS_YAML = "scenarios.yaml"
IMAGE_MANIFEST_CSV = "image_manifest.csv"
OUT_DIR = Path("scenario_outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)

# Canonical vocab (edit if you use different normalization)
YES_NO = ["yes", "no"]
PARIS = ["0–Is","0–IIa","0–IIb","0–IIc","Ip","Isp"]     # for plotting if you include a Paris question
DEVICE = ["snare","forceps","none"]

RANDOM_SEED = 13
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)


# In[7]:


# ---------------------------- #
#   LOAD PRETRAINED MODELS     #
# ---------------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model A: ViLT VQA (classification over large answer vocab)
vilt_name = "dandelin/vilt-b32-finetuned-vqa"
vilt_processor = ViltProcessor.from_pretrained(vilt_name)
vilt_model = ViltForQuestionAnswering.from_pretrained(vilt_name).to(device)

# Model B: BLIP VQA (Q->answer classification)
blip_vqa_name = "Salesforce/blip-vqa-base"
blip_vqa_processor = BlipProcessor.from_pretrained(blip_vqa_name)
blip_vqa_model = BlipForQuestionAnswering.from_pretrained(blip_vqa_name).to(device)

# Model C: BLIP-2 FLAN-T5 (generative; zero-shot)
blip2_name = "Salesforce/instructblip-flan-t5-xl"
blip2_processor = AutoProcessor.from_pretrained(blip2_name)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    blip2_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)


# In[8]:


# ---------------------------- #
#         PREDICTORS           #
# ---------------------------- #
@torch.inference_mode()
def predict_vilt(img: Image.Image, question: str) -> Dict[str, Any]:
    inputs = vilt_processor(images=img, text=question, return_tensors="pt").to(device)
    outputs = vilt_model(**inputs)
    logits = outputs.logits[0].softmax(-1).detach().cpu().numpy()
    # The model has an internal answer vocab; get top-1 string
    idx = logits.argmax()
    answer = vilt_model.config.id2label[idx]
    return {"text": answer, "probs": logits, "labels": vilt_model.config.id2label}

@torch.inference_mode()
def predict_blip_vqa(img: Image.Image, question: str) -> Dict[str, Any]:
    inputs = blip_vqa_processor(images=img, text=question, return_tensors="pt").to(device)
    out = blip_vqa_model(**inputs)
    logits = out.logits[0].softmax(-1).detach().cpu().numpy()
    idx = logits.argmax()
    # BLIP VQA uses a fixed answer vocab too:
    label = blip_vqa_model.config.text_config.vocab_size  # not label list
    # Try to get id2label if exposed; otherwise use processor tokenizer to decode argmax token
    # Easiest robust way: generate directly (BLIPVQA also supports generate via QA head)
    # But logit->token mapping is not straightforwardly exposed. We'll return text via generate().
    gen = blip_vqa_model.generate(**inputs, max_new_tokens=10)
    text = blip_vqa_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return {"text": text, "probs": logits, "labels": None}

@torch.inference_mode()
def predict_blip2(img: Image.Image, question: str, max_new_tokens=16) -> Dict[str, Any]:
    prompt = f"Question: {question}\nAnswer:"
    inputs = blip2_processor(images=img, text=prompt, return_tensors="pt").to(device, dtype=blip2_model.dtype)
    out = blip2_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = blip2_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    # BLIP-2 is generative; we don't get calibrated class probs
    return {"text": text, "probs": None, "labels": None}

MODELS = [
    ("ViLT (vqa)", predict_vilt),
    ("BLIP-VQA", predict_blip_vqa),
    ("BLIP-2 (ZS)", predict_blip2),
]


# In[9]:


# ---------------------------- #
#      NORMALIZATION UTILS     #
# ---------------------------- #
def normalize(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = s.replace("–","-")
    s = re.sub(r"\s+", " ", s)
    mapping = {
        "y":"yes","yeah":"yes","yep":"yes","true":"yes",
        "n":"no","nope":"no","false":"no",
        "snare present":"snare","forceps present":"forceps",
        "biopsy forceps":"forceps","polyp snare":"snare",
        "iis":"ii s","iia":"ii a","0-iia":"0-ii a","0-iis":"0-is"
    }
    return mapping.get(s, s)

def fuzzy_map(pred: str, label_set: List[str]) -> str:
    if not pred: return ""
    pn = normalize(pred)
    # exact
    for lab in label_set:
        if pn == normalize(lab): return lab
    # numeric extraction for count
    m = re.search(r"(-?\d+)", pn)
    if m and all(x.isdigit() or x in {"+","-"} for x in m.group(0)):
        # let counting path handle integer
        pass
    # fuzzy
    return max(label_set, key=lambda lab: difflib.SequenceMatcher(None, pn, normalize(lab)).ratio())

def parse_int(s: str):
    if s is None: return None
    m = re.search(r"-?\d+", str(s))
    return int(m.group(0)) if m else None


# In[11]:


# ---------------------------- #
#     LOAD SCENARIOS & DATA    #
# ---------------------------- #
manifest = pd.read_csv(IMAGE_MANIFEST_CSV)
id2path = dict(zip(manifest.image_id, manifest.image_path))
with open(SCENARIOS_YAML, "r") as f:
    scenarios = yaml.safe_load(f)

rows = []
for sc in scenarios:
    sid = sc["id"]
    for it in sc["items"]:
        img_id = it["image_id"]
        img_path = id2path.get(img_id, None)
        assert img_path and os.path.exists(img_path), f"Image not found for {img_id} -> {img_path}"
        img = Image.open(img_path).convert("RGB")
        for q in it["questions"]:
            qtext = q["text"]; qtype = q["type"]
            labels = q.get("labels", [])
            gt = q.get("gt", q.get("gt_free",""))
            for mname, mpred in MODELS:
                try:
                    out = mpred(img, qtext)
                    raw = out["text"]
                except Exception as e:
                    raw = f"ERR:{e.__class__.__name__}"
                    out = {"probs": None, "labels": None}
                # normalization / mapping
                if qtype == "binary":
                    pred = fuzzy_map(raw, YES_NO)
                    gt_n = fuzzy_map(gt, YES_NO)
                    prob = None
                elif qtype == "closed_set":
                    pred = fuzzy_map(raw, labels)
                    gt_n = fuzzy_map(gt, labels)
                    prob = None
                elif qtype == "count":
                    pred = parse_int(raw)
                    gt_n = parse_int(gt)
                    prob = None
                elif qtype == "free_text":
                    pred = (raw or "").strip()
                    gt_n = (q.get("gt_free","") or "").strip()
                    prob = None
                else:
                    pred, gt_n, prob = raw, gt, None
                rows.append({
                    "scenario_id": sid,
                    "image_id": img_id,
                    "question": qtext,
                    "type": qtype,
                    "labels": "|".join(labels) if labels else "",
                    "gt_label": gt_n if qtype!="count" else ("" if gt_n is None else str(gt_n)),
                    "model": mname,
                    "pred": pred if qtype!="count" else ("" if pred is None else str(pred)),
                    "pred_raw": raw
                })

pred_df = pd.DataFrame(rows)
pred_path = OUT_DIR / "scenario_predictions.csv"
pred_df.to_csv(pred_path, index=False)
print("Saved:", pred_path)
display(pred_df.head())


# In[12]:


# ---------------------------- #
#     METRICS (RICH SET)       #
# ---------------------------- #
def bootstrap_ci(values, iters=1000, alpha=0.05, seed=13):
    if len(values)==0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(iters):
        samp = rng.choice(values, size=n, replace=True)
        boots.append(np.mean(samp))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return float(lo), float(hi)

def acc_macro_metrics(y_true, y_pred, labels):
    # accuracy
    acc = np.mean([a==b for a,b in zip(y_true, y_pred)]) if y_true else 0.0
    # precision/recall/f1 macro
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    # per-class
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class_df = pd.DataFrame({
        "label": labels,
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3]
    })
    # Cohen's kappa & MCC
    try:
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    except:
        kappa = np.nan
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = np.nan
    return float(acc), float(p), float(r), float(f1), float(kappa), float(mcc), per_class_df

def summarize_closed_or_binary(pred_df, question_text, labels, out_prefix):
    out_rows = []
    for mname, g in pred_df.groupby("model"):
        y_true = g["gt_label"].tolist()
        y_pred = g["pred"].tolist()
        acc, p, r, f1, kappa, mcc, per_class_df = acc_macro_metrics(y_true, y_pred, labels)
        out_rows.append({
            "question": question_text,
            "model": mname,
            "accuracy": round(acc,4),
            "precision_macro": round(p,4),
            "recall_macro": round(r,4),
            "f1_macro": round(f1,4),
            "cohen_kappa": round(kappa,4),
            "mcc": round(mcc,4),
            "labels": "|".join(labels)
        })
        # Confusion matrix plot
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(4.5,4.5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{question_text} — {mname}")
        plt.tight_layout()
        fig_path = OUT_DIR / f"{out_prefix}_confusion_{mname.replace(' ','_').replace('(','').replace(')','')}.png"
        plt.savefig(fig_path, dpi=220); plt.close(fig)
        # Per-class CSV
        per_class_df.to_csv(OUT_DIR / f"{out_prefix}_perclass_{mname.replace(' ','_')}.csv", index=False)
    return out_rows

# Aggregate metrics
agg_rows = []

# Closed-set + Binary metrics (with confusion, per-class)
for qtext, gq in pred_df[pred_df["type"].isin(["binary","closed_set"])].groupby("question"):
    # derive label set
    if "Is there active bleeding" in qtext:
        labels = YES_NO
        out_prefix = "binary_bleeding"
    elif "Is a snare or forceps visible" in qtext:
        labels = DEVICE
        out_prefix = "device"
    elif "morphology" in qtext.lower():
        labels = PARIS if "0–" in " ".join(gq["labels"].tolist()) or any("0–" in s for s in PARIS) else sorted(set(sum([x.split("|") for x in gq["labels"] if x], [])))
        out_prefix = "paris"
    else:
        # fallback to labels encoded in rows
        labs = sorted(set(sum([x.split("|") for x in gq["labels"] if x], [])))
        labels = labs if labs else YES_NO
        out_prefix = re.sub(r"[^a-z0-9]+","_", qtext.lower())[:30]
    agg_rows += summarize_closed_or_binary(gq, qtext, labels, out_prefix)

# Counting metrics
count_rows = []
csub = pred_df[pred_df["type"]=="count"]
def bucket(x):
    try:
        v = int(x); 
        if v in (0,1,2,3): return str(v)
        return "3+"
    except:
        return "NA"
for qtext, gq in csub.groupby("question"):
    for mname, gm in gq.groupby("model"):
        y_true = [int(x) for x in gm["gt_label"].tolist() if x not in ("",None)]
        y_pred = []
        for x in gm["pred"].tolist():
            try: y_pred.append(int(x))
            except: y_pred.append(None)
        pairs = [(a,b) for a,b in zip(y_true,y_pred) if b is not None]
        if pairs:
            em = np.mean([a==b for a,b in pairs])
            off1 = np.mean([abs(a-b)==1 for a,b in pairs])
            mae = np.mean([abs(a-b) for a,b in pairs])
            rmse = math.sqrt(np.mean([(a-b)**2 for a,b in pairs]))
        else:
            em = off1 = mae = rmse = np.nan
        # bucket confusion
        gb = pd.DataFrame({
            "gt_bucket": [bucket(x) for x in gm["gt_label"]],
            "pred_bucket": [bucket(x) for x in gm["pred"]],
        })
        ct = pd.crosstab(gb["gt_bucket"], gb["pred_bucket"]).reindex(index=["0","1","2","3","3+","NA"], columns=["0","1","2","3","3+","NA"]).fillna(0).astype(int)
        ct_path = OUT_DIR / f"count_confusion_buckets_{mname.replace(' ','_')}.csv"
        ct.to_csv(ct_path)
        count_rows.append({
            "question": qtext, "model": mname,
            "EM": round(float(em),4) if not np.isnan(em) else np.nan,
            "off_by_1": round(float(off1),4) if not np.isnan(off1) else np.nan,
            "MAE": round(float(mae),3) if not np.isnan(mae) else np.nan,
            "RMSE": round(float(rmse),3) if not np.isnan(rmse) else np.nan
        })


# In[13]:


# Free-text: BLEU-1..4, ROUGE-L, METEOR, BERTScore
ft = pred_df[pred_df["type"]=="free_text"]
ft_rows = []
if not ft.empty:
    from bert_score import score as bertscore
    for qtext, gq in ft.groupby("question"):
        refs = gq["gt_label"].tolist()
        for mname, gm in gq.groupby("model"):
            hyps = gm["pred"].tolist()
            # SacreBLEU corpus BLEU (gives BLEU-1..4 via precisions)
            bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, lowercase=True, tokenize="13a")
            p1,p2,p3,p4 = bleu.precisions
            # ROUGE-L (avg)
            rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rougeL = np.mean([rs.score(r, h)["rougeL"].fmeasure for h,r in zip(hyps, refs)])
            # METEOR (avg)
            meteor = np.mean([single_meteor_score(r, h) for h,r in zip(hyps, refs)])
            # BERTScore (F1)
            P,R,F = bertscore(hyps, refs, lang="en", rescale_with_baseline=True)
            bert_f1 = float(F.mean().item())
            ft_rows.append({
                "question": qtext, "model": mname,
                "BLEU": round(bleu.score,2),
                "BLEU-1": round(p1,2), "BLEU-2": round(p2,2), "BLEU-3": round(p3,2), "BLEU-4": round(p4,2),
                "ROUGE-L": round(rougeL,3),
                "METEOR": round(float(meteor),3),
                "BERTScore_F1": round(bert_f1,3)
            })

# McNemar’s test (binary only) for top two models (ViLT vs BLIP-VQA)
mcnemar_rows = []
bsub = pred_df[(pred_df["type"]=="binary")]
if not bsub.empty:
    for qtext, gq in bsub.groupby("question"):
        # pivot: image-question rows aligned
        pivot = gq.pivot_table(index=["scenario_id","image_id","question"],
                               columns="model", values=["gt_label","pred"], aggfunc="first")
        # Ensure both models exist
        if ("pred","ViLT (vqa)") in pivot.columns and ("pred","BLIP-VQA") in pivot.columns:
            y = pivot[("gt_label","ViLT (vqa)")].tolist()
            a = pivot[("pred","ViLT (vqa)")].tolist()
            b = pivot[("pred","BLIP-VQA")].tolist()
            agree_a = [ai==yi for ai,yi in zip(a,y)]
            agree_b = [bi==yi for bi,yi in zip(b,y)]
            # contingency
            b01 = sum((not x) and yb for x,yb in zip(agree_a,agree_b))  # A wrong, B right
            b10 = sum(x and (not yb) for x,yb in zip(agree_a,agree_b))  # A right, B wrong
            table = [[0, b01],[b10, 0]]
            res = mcnemar(table, exact=False, correction=True)
            mcnemar_rows.append({"question": qtext, "A":"ViLT (vqa)", "B":"BLIP-VQA",
                                 "b01": b01, "b10": b10, "statistic": round(res.statistic,4), "p_value": float(res.pvalue)})

# Save aggregates
agg_df = pd.DataFrame(agg_rows)
agg_path = OUT_DIR / "scenario_closed_or_binary_metrics.csv"
agg_df.to_csv(agg_path, index=False); print("Saved:", agg_path)

count_df = pd.DataFrame(count_rows)
if not count_df.empty:
    count_path = OUT_DIR / "scenario_count_metrics.csv"
    count_df.to_csv(count_path, index=False); print("Saved:", count_path)

if ft_rows:
    ft_df = pd.DataFrame(ft_rows)
    ft_path = OUT_DIR / "free_text_metrics.csv"
    ft_df.to_csv(ft_path, index=False); print("Saved:", ft_path)

if mcnemar_rows:
    mc_df = pd.DataFrame(mcnemar_rows)
    mc_path = OUT_DIR / "mcnemar_binary_viLT_vs_blipvqa.csv"
    mc_df.to_csv(mc_path, index=False); print("Saved:", mc_path)

print("Artifacts written to:", OUT_DIR.resolve())


# In[ ]:




