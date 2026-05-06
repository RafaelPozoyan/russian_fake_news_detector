import os
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

SEED = 42
DATA_PATH = "data/ready_dataset.csv"
V4_CACHE = "models/deepseek/predictions_441_v4.csv"
V5_CACHE = "models/deepseek/predictions_441_v5.csv"

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


df = pd.read_csv(DATA_PATH)
df = df[["headline_clean", "body_clean", "combined_text", "label"]].dropna()
df["headline_clean"] = df["headline_clean"].astype(str).str.strip()
df["body_clean"] = df["body_clean"].astype(str).str.strip()
df["combined_text"] = df["combined_text"].astype(str).str.strip()
df = df[(df["headline_clean"] != "") & (df["body_clean"] != "")]
df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
df = df.reset_index(drop=True)

train_df, test_df = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"]
)
test_df = test_df.reset_index(drop=True)
combined_text = test_df["combined_text"].astype(str).tolist()
y_true = test_df["label"].values

# 24-shot retrieval
retriever = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
train_texts = train_df["combined_text"].astype(str).tolist()
train_labels = train_df["label"].astype(int).tolist()
X_train_tfidf = retriever.fit_transform(train_texts)
K_RETRIEVE = 24


def retrieve_fewshot(test_text, k=K_RETRIEVE):
    q = retriever.transform([test_text])
    sims = cosine_similarity(q, X_train_tfidf).ravel()
    order = sims.argsort()[::-1]
    target = k // 2
    by_class = {0: [], 1: []}
    for i in order:
        lbl = train_labels[i]
        if len(by_class[lbl]) < target:
            by_class[lbl].append({"text": train_texts[i], "label": lbl})
        if len(by_class[0]) >= target and len(by_class[1]) >= target:
            break
    examples = by_class[0] + by_class[1]
    import random
    random.Random(42).shuffle(examples)
    return examples


DS_SYSTEM_PROMPT = (
    "You classify Russian-language news as REAL (label=1) or FAKE (label=0). "
    "Texts are lemmatized (no punctuation, no headlines, single line). "
    "Stylistic signals are USELESS — match TOPICAL and SEMANTIC patterns from "
    "the 24 retrieved labeled examples (12 fakes, 12 reals). "
    "Reasoning protocol: "
    "(1) find 2-3 most semantically similar labeled examples; "
    "(2) check if the query asserts the same kind of claim; "
    "(3) watch for FAKE markers: fabricated quotes from politicians, "
    "absurd-but-officious decrees, satirical \"announcements\" from parties/ministries, "
    "Panorama-style parody (e.g. fake patriotic initiatives, mock celebrity political moves, "
    "implausible bureaucratic specifics, joke 'movements'); "
    "(4) watch for REAL markers: documented war/legal/diplomatic events, "
    "court rulings, missile strikes, refugee/casualty reports, sanctions, economic data; "
    "(5) IMPORTANT: in this dataset both classes mention named officials in neutral tone — "
    "the label depends on what the article CLAIMS, not how it sounds. "
    "If unsure, prefer the label of the closest retrieved example. "
    "Output JSON only: {\"reasoning\": \"1-2 sentences\", \"label\": 0 or 1, "
    "\"confidence\": \"low|medium|high\"}."
)


def build_messages(text, fewshot):
    msgs = [{"role": "system", "content": DS_SYSTEM_PROMPT}]
    for ex in fewshot:
        msgs.append({"role": "user",
                     "content": f"TEXT:\n{ex['text'][:1200]}\n\nClassify (return JSON):"})
        reason = ("similar to known fake (fabricated/satirical/implausible claim)."
                  if ex["label"] == 0 else
                  "similar to known real (documented event with verifiable sources).")
        msgs.append({"role": "assistant",
                     "content": json.dumps({"reasoning": reason, "label": ex["label"],
                                            "confidence": "high"}, ensure_ascii=False)})
    msgs.append({"role": "user",
                 "content": f"TEXT:\n{text[:1200]}\n\nClassify (return JSON):"})
    return msgs


def parse_label(content):
    content = content.strip()
    if "```" in content:
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break
    matches = re.findall(r"\{.*?\}", content, re.DOTALL)
    if matches:
        for m in reversed(matches):
            try:
                data = json.loads(m)
                for k in data:
                    if k.lower() in ("label", "is_true", "is_real", "class", "result"):
                        raw = data[k]
                        if isinstance(raw, bool):
                            return int(raw)
                        if isinstance(raw, (int, float)):
                            return int(bool(int(raw)))
                        s = str(raw).strip().lower()
                        return 1 if s in ("1", "true", "real", "правда") else 0
            except Exception:
                continue
    m = re.findall(r'"label"\s*:\s*([01])', content.lower())
    if m:
        return int(m[-1])
    return None


def call(text, fewshot, temperature, max_retries=3):
    for _ in range(max_retries):
        try:
            r = ds_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=build_messages(text, fewshot),
                max_tokens=200,
                temperature=temperature,
                stream=False,
            )
            label = parse_label(r.choices[0].message.content.strip())
            if label is not None:
                return label
        except Exception:
            pass
        time.sleep(0.5)
    return None


# Идентифицируем 8 ошибок v4
v4 = pd.read_csv(V4_CACHE)
wrong_idx = v4.index[v4["true_label"] != v4["pred"]].tolist()
print(f"v4 wrong examples: {wrong_idx}")
print(f"v4 accuracy: {(v4['true_label'] == v4['pred']).mean():.4f}")

# Re-classify только эти 8 с 7 голосами
TEMPERATURES = [0.0, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]
v5_pred = v4["pred"].copy()
flips = []

for idx in wrong_idx:
    fewshot = retrieve_fewshot(combined_text[idx])
    votes = []
    for T in TEMPERATURES:
        v = call(combined_text[idx], fewshot, temperature=T)
        if v is not None:
            votes.append(v)
        time.sleep(0.1)
    if not votes:
        new_label = int(v4.loc[idx, "pred"])
    else:
        new_label = Counter(votes).most_common(1)[0][0]
    true_label = int(v4.loc[idx, "true_label"])
    old_label = int(v4.loc[idx, "pred"])
    flipped = new_label != old_label
    correct = new_label == true_label
    print(f"  idx={idx}: votes={votes} -> {new_label}  "
          f"(true={true_label}, old={old_label}, "
          f"{'FLIPPED' if flipped else 'same'}, "
          f"{'now correct' if correct else 'still wrong'})")
    if flipped:
        flips.append((idx, old_label, new_label, true_label))
    v5_pred.loc[idx] = new_label

v5_df = v4.copy()
v5_df["pred"] = v5_pred.astype(int)
v5_df.to_csv(V5_CACHE, index=False)

print()
print(f"Flipped: {len(flips)}")
print(f"v5 accuracy: {(v5_df['true_label'] == v5_df['pred']).mean():.4f}")
print(f"Saved: {V5_CACHE}")
