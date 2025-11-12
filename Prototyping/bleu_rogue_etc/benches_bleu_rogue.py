#!/usr/bin/env python
# coding: utf-8

# # BLEU & ROUGE Prototype (Kvasir-VQA)
# 
# This notebook prototypes corpus-level BLEU and ROUGE metrics for the Kvasir-VQA predictions. It loads previously saved model outputs and reports comparable scores that complement the accuracy-style benchmarks in the IBD analysis workflow.

# In[3]:


from pathlib import Path
import os
import pandas as pd
import re
import unicodedata
from typing import Dict

NOTEBOOK_DIR = Path.cwd().resolve()
default_cache = NOTEBOOK_DIR / '_hf_metrics_cache'
os.environ.setdefault('HF_METRICS_CACHE', str(default_cache.resolve()))
default_cache.mkdir(parents=True, exist_ok=True)

try:
    import evaluate
except ImportError as exc:
    raise ImportError(
        "Install evaluate>=0.4.0 and rouge-score to run this notebook (e.g. `pip install evaluate rouge-score`)."
    ) from exc


# In[4]:


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace for fair text matching."""
    if text is None:
        text = ""
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_predictions_dir(base_dir: Path) -> Path:
    """Locate the folder that holds model prediction CSV files."""
    candidates = [
        base_dir / 'phase3_results',
        base_dir.parent / 'phase3_results',
        base_dir.parent / 'benchmark_outputs' / 'reports',
        base_dir.parents[1] / 'phase3_results' if len(base_dir.parents) > 1 else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and any(candidate.glob('*_predictions.csv')):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find *_predictions.csv files near {base_dir}. Update `resolve_predictions_dir` with the correct path."
    )


def load_prediction_frames(predictions_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read each prediction CSV and attach normalized text columns."""
    frames: Dict[str, pd.DataFrame] = {}
    csv_paths = sorted(predictions_dir.glob('*_predictions.csv'))
    if not csv_paths:
        raise FileNotFoundError(f"No *_predictions.csv files were found in {predictions_dir}")
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if 'model' in df.columns:
            model_id = df['model'].iloc[0]
        else:
            stem = csv_path.stem
            model_id = stem.replace('_uc_predictions', '')
        df = df.copy()
        df['reference'] = df.get('ground_truth', '').fillna('').astype(str)
        df['prediction'] = df.get('predicted', '').fillna('').astype(str)
        df['reference_norm'] = df['reference'].map(normalize_answer)
        df['prediction_norm'] = df['prediction'].map(normalize_answer)
        frames[model_id] = df
    return frames


# In[5]:


predictions_dir = resolve_predictions_dir(NOTEBOOK_DIR)
model_frames = load_prediction_frames(predictions_dir)
print(f"Resolved {len(model_frames)} model outputs from {predictions_dir}")
for model_name, frame in model_frames.items():
    print(f"- {model_name}: {len(frame)} QA pairs")


# In[6]:


bleu_metric = evaluate.load('bleu')
rouge_metric = evaluate.load('rouge')

metric_rows = []
for model_name, frame in model_frames.items():
    references = frame['reference_norm'].tolist()
    predictions = frame['prediction_norm'].tolist()
    bleu_scores = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        max_order=4,
        smooth=True,
    )
    rouge_scores = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    metric_rows.append(
        {
            'model': model_name,
            'num_samples': len(frame),
            'bleu': bleu_scores['bleu'],
            'bleu1': bleu_scores['precisions'][0],
            'bleu2': bleu_scores['precisions'][1],
            'bleu3': bleu_scores['precisions'][2],
            'bleu4': bleu_scores['precisions'][3],
            'brevity_penalty': bleu_scores['brevity_penalty'],
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'rougeLsum': rouge_scores.get('rougeLsum'),
        }
    )

metrics_df = pd.DataFrame(metric_rows).sort_values('bleu', ascending=False).reset_index(drop=True)
metrics_df


# In[7]:


metric_cols = [col for col in metrics_df.columns if col not in {'model', 'num_samples'}]
print('Corpus-level BLEU and ROUGE scores (normalized text):')
try:
    print(metrics_df.to_markdown(index=False, floatfmt='.4f'))
except ImportError:
    print('Install `tabulate` to enable Markdown export; showing raw DataFrame instead.')
metrics_df


# Next steps: extend this notebook with METEOR, CIDEr, and Exact Match to round out the text-generation metrics recommended in the benchmark plan.
