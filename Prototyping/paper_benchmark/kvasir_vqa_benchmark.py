#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %% [markdown]
# Kvasir-VQA Benchmark (Small-GPU, Hardened Template Selection)
# - M1: ResNet50 (frozen feats) + GRU fusion
# - M2: ViT-B/16 (frozen feats) + DistilBERT + lite cross-attn (2 layers, d=256)
# - M4: BLIP-2 (Flan-T5-base) zero-shot (+ optional LoRA r=4, 8-bit)
# Outputs: CSV tables + per-item preds + heatmap

# %% Imports & setup
import os, json, math, random, re, string
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor, ViTModel,
    DistilBertModel, DistilBertTokenizerFast,
    BlipForQuestionAnswering, BlipProcessor
)

get_ipython().run_line_magic('pip', 'install accelerate')

# Optional LoRA/quant
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb
    HAS_PEFT=True
except Exception:
    HAS_PEFT=False

# %% Config (EDIT THESE PATHS)
DATA_CSV   = "../results/colonoscopy_metadata.csv"   # <-- your CSV
IMAGES_DIR = "../images"                             # <-- your images dir

OUT_DIR   = Path("./reports"); OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR   = Path("./figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("./cache");   CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()
BATCH=32; IMG_SIZE=224
EPOCHS_M1=8; EPOCHS_M2=10; EPOCHS_LORA=2

# %% Utils: normalization & metrics
PUNCT = str.maketrans("", "", string.punctuation)

def normalize_ans(s):
    if s is None: return ""
    s = str(s).lower().strip()
    s = s.translate(PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

YES_SYNS = {"yes","present","visible","true","1","detected"}
NO_SYNS  = {"no","none","absent","not present","not visible","false","0",
            "not relevant","na","n/a","none na","none/na"}

def robust_yesno_map(answer):
    """Map arbitrary short answers (incl. lists) to yes/no."""
    s = normalize_ans(answer)
    if s in NO_SYNS or s=="":
        return 0
    if s in YES_SYNS:
        return 1
    # Heuristic: if it contains any alphanum token and is not a NO synonym → YES
    return 1

def bootstrap_ci(correct01, n_boot=1000, alpha=0.05, seed=SEED):
    rng = np.random.RandomState(seed)
    arr = np.array(correct01, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    boots = np.sort(np.array(boots))
    lo = boots[int((alpha/2)*n_boot)]
    hi = boots[int((1-alpha/2)*n_boot)]
    return float(lo), float(hi)

def mcnemar_from_preds(y_true, yhat_a, yhat_b):
    a = np.array(yhat_a)==np.array(y_true)
    b = np.array(yhat_b)==np.array(y_true)
    n01 = int((~a &  b).sum())
    n10 = int(( a & ~b).sum())
    if n01+n10 == 0: return 0.0, 1.0
    chi2 = (abs(n01-n10)-1)**2/(n01+n10)
    p = math.exp(-chi2/2)
    return float(chi2), float(p)

# %% Load CSV and inspect templates
df = pd.read_csv(DATA_CSV)
for col in ["question_norm","question","answer"]:
    df[col] = df[col].astype(str)
df["question_norm"] = df["question_norm"].str.lower().str.strip()
print("Top question_norm:")
print(df["question_norm"].value_counts().head(20), "\n")

# %% --- Template selection (auto) ---
def pick_yesno(df):
    """Try safe literal yes/no templates; fallback to derived yes/no from instruments."""
    candidates = [
        "is there text",
        "is this finding easy to detect",
        "does this image contain any finding",
        "is there a green/black box artefact",
    ]
    for qn in candidates:
        d = df[df["question_norm"]==qn].copy()
        if len(d) < 300: continue
        d["label"] = d["answer"].map(robust_yesno_map)
        if d["label"].nunique()==2:
            print(f"[Yes/No] Using literal template: '{qn}'  (n={len(d)})")
            return qn, d

    # Fallback: instruments question → derive yes/no (list != 'none' → yes)
    d = df[df["question_norm"].str.contains("instruments", na=False)].copy()
    d["label"] = d["answer"].map(robust_yesno_map)
    d = d.dropna(subset=["label"])
    if d["label"].nunique()==2 and len(d) >= 300:
        print(f"[Yes/No] Using derived template from instruments (n={len(d)})")
        return "derived_instruments_yesno", d

    raise ValueError("Could not find a viable Yes/No template. Check question_norm values.")

def pick_attribute(df):
    qn = "what type of polyp is present"
    d = df[df["question_norm"]==qn].copy()
    if len(d) < 300:
        # second best fallback: procedure type
        qn2 = "what type of procedure is the image taken from"
        d2 = df[df["question_norm"]==qn2].copy()
        if len(d2) >= 300:
            print(f"[Attr] Using '{qn2}' (n={len(d2)})")
            return qn2, d2
        raise ValueError("Could not find a viable Attribute template with enough samples.")
    print(f"[Attr] Using '{qn}' (n={len(d)})")
    return qn, d

yn_qn, yn_df = pick_yesno(df)
at_qn, at_df = pick_attribute(df)


# Attribute labels (cap tail to 'other' and drop classes with <2 samples)
lab = at_df["answer"].map(lambda s: normalize_ans(s))

# Count occurrences
counts = lab.value_counts()

# Drop all rare classes with <2 samples entirely
valid_classes = counts[counts >= 2].index
at_df = at_df[lab.isin(valid_classes)].copy()

# If you still want to group rare classes into 'other' instead of dropping:
# lab = lab.where(lab.isin(valid_classes), other='other')

# Update the label_text column
at_df["label_text"] = lab[lab.isin(valid_classes)]

# Build class mapping fresh
classes_attr = sorted(at_df["label_text"].unique())
cls2id = {c: i for i, c in enumerate(classes_attr)}
at_df["label"] = at_df["label_text"].map(cls2id)



# Final yes/no labels
yn_df["label"] = yn_df["answer"].map(robust_yesno_map)

# Stratified splits 70/15/15
def stratify_split(frame, label_col="label"):
    tr, te = train_test_split(frame, test_size=0.15, random_state=SEED, stratify=frame[label_col])
    tr, va = train_test_split(tr, test_size=0.1765, random_state=SEED, stratify=tr[label_col])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

tr_yn, va_yn, te_yn = stratify_split(yn_df, "label")
tr_at, va_at, te_at = stratify_split(at_df, "label")

print(f"Yes/No split sizes: {len(tr_yn)} / {len(va_yn)} / {len(te_yn)}  balance={tr_yn['label'].value_counts().to_dict()}")
print(f"Attr  split sizes: {len(tr_at)} / {len(va_at)} / {len(te_at)}  classes={classes_attr}\n")

# %% Image transforms & feature cache
img_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

def load_image(img_id):
    for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        p = Path(IMAGES_DIR)/f"{img_id}{ext}"
        if p.exists(): return Image.open(p).convert("RGB")
    return None

RESNET_FEATS = CACHE_DIR/"resnet50_feats.npz"
VIT_FEATS    = CACHE_DIR/f"vit_b16_feats_{IMG_SIZE}.npz"

def build_or_load_feats(ids):
    if RESNET_FEATS.exists() and VIT_FEATS.exists():
        r = np.load(RESNET_FEATS, allow_pickle=True)["arr_0"].item()
        v = np.load(VIT_FEATS, allow_pickle=True)["arr_0"].item()
        return r, v
    print("Extracting features once... (ResNet50 avgpool + ViT-B/16 token-mean)")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
    res_fc = nn.Sequential(*list(resnet.children())[:-1])
    vit_name = "google/vit-base-patch16-224-in21k"
    vit_proc = AutoImageProcessor.from_pretrained(vit_name)
    vit = ViTModel.from_pretrained(vit_name).to(DEVICE).eval()
    rfeats, vfeats = {}, {}
    with torch.no_grad():
        for img_id in ids:
            im = load_image(img_id)
            if im is None: continue
            t = img_tf(im).unsqueeze(0).to(DEVICE)
            r = res_fc(t).flatten(1)             # [1,2048]
            rfeats[img_id] = r.cpu().numpy()[0]
            enc = vit_proc(images=im, return_tensors="pt").to(DEVICE)
            out = vit(**enc).last_hidden_state   # [1, n, 768]
            vfeats[img_id] = out.mean(1).cpu().numpy()[0]
    np.savez_compressed(RESNET_FEATS, {k:v for k,v in rfeats.items()})
    np.savez_compressed(VIT_FEATS,    {k:v for k,v in vfeats.items()})
    return rfeats, vfeats

all_ids = set(pd.concat([tr_yn["img_id"],va_yn["img_id"],te_yn["img_id"],
                         tr_at["img_id"],va_at["img_id"],te_at["img_id"]]).tolist())
res_feats, vit_feats = build_or_load_feats(all_ids)

# %% Dataset & loaders
tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class KVQADataset(Dataset):
    def __init__(self, frame, nclass):
        self.df = frame.reset_index(drop=True)
        self.nclass = nclass
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        rid = r["img_id"]
        enc = tok(str(r["question"]), truncation=True, max_length=32, padding="max_length", return_tensors="pt")
        item = {
            "rfeat": torch.tensor(res_feats.get(rid, np.zeros(2048, np.float32)), dtype=torch.float32),
            "vfeat": torch.tensor(vit_feats.get(rid, np.zeros(768,  np.float32)), dtype=torch.float32),
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "img_id": rid
        }
        return item

def make_loader(frame, nclass, bs):
    return DataLoader(KVQADataset(frame, nclass), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)

trL_yn = make_loader(tr_yn, 2, BATCH); vaL_yn = make_loader(va_yn, 2, BATCH); teL_yn = make_loader(te_yn, 2, BATCH)
trL_at = make_loader(tr_at, len(classes_attr), BATCH); vaL_at = make_loader(va_at, len(classes_attr), BATCH); teL_at = make_loader(te_at, len(classes_attr), BATCH)

# %% Models
class M1_RESNET_GRU(nn.Module):
    def __init__(self, nclass=2, vocab=30522, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, 128)
        self.gru = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
        self.fc_q = nn.Linear(hidden*2, 256)
        self.fc_i = nn.Linear(2048, 256)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(256, nclass))
    def forward(self, b):
        x = self.emb(b["input_ids"])
        x,_ = self.gru(x)
        m = b["attention_mask"].unsqueeze(-1).float()
        q = (x*m).sum(1)/(m.sum(1)+1e-6)
        q = self.fc_q(q); i = self.fc_i(b["rfeat"])
        return self.head(torch.cat([q,i], dim=-1))

class CrossAttnBlock(nn.Module):
    def __init__(self, d=256, nhead=4): super().__init__(); 
    # (PyTorch quirk: define layers in __init__)
    def __init__(self, d=256, nhead=4):
        super().__init__()
        self.q2v = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.v2q = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d,4*d), nn.GELU(), nn.Linear(4*d,d)) for _ in range(2)])
    def forward(self, q, v):
        q = q + self.q2v(self.ln[0](q), self.ln[0](v), self.ln[0](v))[0]
        q = q + self.ff[0](self.ln[1](q))
        v = v + self.v2q(self.ln[2](v), self.ln[2](q), self.ln[2](q))[0]
        v = v + self.ff[1](self.ln[3](v))
        return q, v

class M2_VIT_BERT_Lite(nn.Module):
    def __init__(self, nclass=2, d=256, nhead=4, nlayer=2):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased"); 
        for p in self.bert.parameters(): p.requires_grad=False
        self.proj_q = nn.Linear(self.bert.config.dim, d)
        self.proj_v = nn.Linear(768, d)
        self.blocks = nn.ModuleList([CrossAttnBlock(d=d, nhead=nhead) for _ in range(nlayer)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, nclass))
    def forward(self, b):
        q = self.bert(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        q = self.proj_q(q)                                  # [B,T,d]
        v = self.proj_v(b["vfeat"]).unsqueeze(1).repeat(1,7,1)  # cheap 7 tokens
        for blk in self.blocks: q, v = blk(q, v)
        m = b["attention_mask"].unsqueeze(-1).float()
        q = (q*m).sum(1)/(m.sum(1)+1e-6)
        return self.head(q)

# %% Train/eval helpers
def run_train(model, trL, vaL, epochs=5, lr=1e-3, focal=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    if focal:
        gamma=2.0
        def loss_fn(logits, y):
            p = torch.softmax(logits, dim=-1)
            pt = p[torch.arange(len(y)), y]
            return (-((1-pt)**gamma)*torch.log(pt+1e-9)).mean()
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_f1 = None, -1
    for ep in range(epochs):
        model.train()
        for b in trL:
            for k in b: b[k] = b[k].to(DEVICE) if torch.is_tensor(b[k]) else b[k]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP):
                logits = model(b); loss = loss_fn(logits, b["label"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # val
        model.eval(); ys,yh=[],[]
        with torch.no_grad():
            for b in vaL:
                for k in b: b[k] = b[k].to(DEVICE) if torch.is_tensor(b[k]) else b[k]
                pred = model(b).argmax(-1).cpu().numpy()
                ys.extend(b["label"].cpu().numpy()); yh.extend(pred)
        acc = accuracy_score(ys,yh); f1 = f1_score(ys,yh,average="macro")
        print(f"Ep{ep+1}/{epochs}  val acc={acc:.3f} f1={f1:.3f}")
        if f1>best_f1: best_f1=f1; best={k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict({k:v.to(DEVICE) for k,v in best.items()})
    return model

def evaluate_closed(model, teL, out_csv):
    model.eval(); ids, ys, yh = [], [], []
    with torch.no_grad():
        for b in teL:
            for k in b: b[k] = b[k].to(DEVICE) if torch.is_tensor(b[k]) else b[k]
            pred = model(b).argmax(-1).cpu().numpy()
            ys.extend(b["label"].cpu().numpy()); yh.extend(pred)
            ids.extend(list(b["img_id"]))
    acc = accuracy_score(ys,yh); f1 = f1_score(ys,yh,average="macro")
    corr = (np.array(ys)==np.array(yh)).astype(int).tolist()
    lo,hi = bootstrap_ci(corr)
    pd.DataFrame({"img_id":ids,"y":ys,"yhat":yh,"correct":corr}).to_csv(out_csv, index=False)
    return {"acc":acc,"acc_ci":[lo,hi],"macro_f1":f1,"y":ys,"yhat":yh}

# %% ----- Train + eval (Yes/No) -----
m1_yn = run_train(M1_RESNET_GRU(nclass=2), trL_yn, vaL_yn, epochs=EPOCHS_M1, lr=1e-3, focal=True)
res_m1_yn = evaluate_closed(m1_yn, teL_yn, OUT_DIR/"m1_yesno_preds.csv")
print("M1 Yes/No:", res_m1_yn)

m2_yn = run_train(M2_VIT_BERT_Lite(nclass=2), trL_yn, vaL_yn, epochs=EPOCHS_M2, lr=5e-4, focal=True)
res_m2_yn = evaluate_closed(m2_yn, teL_yn, OUT_DIR/"m2_yesno_preds.csv")
print("M2 Yes/No:", res_m2_yn)

# %% ----- Train + eval (Attribute) -----
nA = len(classes_attr)
m1_at = run_train(M1_RESNET_GRU(nclass=nA), trL_at, vaL_at, epochs=EPOCHS_M1, lr=1e-3, focal=False)
res_m1_at = evaluate_closed(m1_at, teL_at, OUT_DIR/"m1_attr_preds.csv")
print("M1 Attr:", res_m1_at)

m2_at = run_train(M2_VIT_BERT_Lite(nclass=nA), trL_at, vaL_at, epochs=EPOCHS_M2, lr=5e-4, focal=False)
res_m2_at = evaluate_closed(m2_at, teL_at, OUT_DIR/"m2_attr_preds.csv")
print("M2 Attr:", res_m2_at)

# %% ----- BLIP-2 zero-shot (Yes/No, constrained) -----
blip_name = "Salesforce/blip-vqa-base"
blip_proc = BlipProcessor.from_pretrained(blip_name)
blip_qa   = BlipForQuestionAnswering.from_pretrained(
    blip_name, device_map="auto" if torch.cuda.is_available() else None,
    load_in_8bit=HAS_PEFT, torch_dtype=torch.float16 if AMP else torch.float32
).to(DEVICE)

def blip_yesno(rows):
    preds, y = [], []
    blip_qa.eval()
    with torch.no_grad():
        for _,r in rows.iterrows():
            im = load_image(r["img_id"])
            if im is None: continue
            prompt = str(r["question"]).strip() + " Answer yes or no only."
            enc = blip_proc(images=im, text=prompt, return_tensors="pt").to(DEVICE)
            out = blip_qa.generate(**enc, max_new_tokens=2, do_sample=False)
            txt = blip_proc.decode(out[0], skip_special_tokens=True).strip().lower()
            pred = 1 if txt.startswith("y") else 0 if txt.startswith("n") else 0
            preds.append(pred); y.append(int(r["label"]))
    return y, preds

y_true, y_pred = blip_yesno(te_yn)
acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred, average="macro")
ci  = bootstrap_ci((np.array(y_true)==np.array(y_pred)).astype(int))
pd.DataFrame({"y":y_true,"yhat":y_pred}).to_csv(OUT_DIR/"blip2_yesno_zeroshot.csv", index=False)
res_blip_yn = {"acc":acc,"acc_ci":list(ci),"macro_f1":f1}
print("BLIP-2 zero-shot Yes/No:", res_blip_yn)

# %% ----- McNemar (Yes/No) -----
m1 = pd.read_csv(OUT_DIR/"m1_yesno_preds.csv")
m2 = pd.read_csv(OUT_DIR/"m2_yesno_preds.csv")
n = min(len(m1), len(m2))
chi2, p = mcnemar_from_preds(m1["y"][:n], m1["yhat"][:n], m2["yhat"][:n])
print(f"McNemar M2 vs M1 (Yes/No): chi2={chi2:.3f}, p={p:.4f}")

# %% ----- Main table & heatmap -----
rows = [
    {"Model":"ResNet+GRU (M1)","Type":"Yes/No","Accuracy":res_m1_yn["acc"],"Macro-F1":res_m1_yn["macro_f1"]},
    {"Model":"ViT+BERT-lite (M2)","Type":"Yes/No","Accuracy":res_m2_yn["acc"],"Macro-F1":res_m2_yn["macro_f1"]},
    {"Model":"BLIP-2 zero-shot","Type":"Yes/No","Accuracy":res_blip_yn["acc"],"Macro-F1":res_blip_yn["macro_f1"]},
    {"Model":"ResNet+GRU (M1)","Type":"Attribute","Accuracy":res_m1_at["acc"],"Macro-F1":res_m1_at["macro_f1"]},
    {"Model":"ViT+BERT-lite (M2)","Type":"Attribute","Accuracy":res_m2_at["acc"],"Macro-F1":res_m2_at["macro_f1"]},
]
tbl = pd.DataFrame(rows); tbl.to_csv(OUT_DIR/"main_table.csv", index=False); print(tbl, "\n")

# Heatmap
pivot = tbl.pivot(index="Model", columns="Type", values="Accuracy")
plt.figure(figsize=(6,3)); plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns); plt.yticks(range(len(pivot.index)), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        plt.text(j, i, f"{pivot.values[i,j]:.2f}", ha="center", va="center", color="white")
plt.tight_layout(); plt.savefig(FIG_DIR/"heatmap_placeholder.pdf"); plt.close()
print("Artifacts:", OUT_DIR.resolve(), FIG_DIR.resolve())


# In[ ]:





# In[ ]:





# In[ ]:




