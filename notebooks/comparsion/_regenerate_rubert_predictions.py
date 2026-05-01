import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUBERT_DIR = "models/rubert/best_model"
OUT_PATH = "models/rubert/test_predictions.csv"

print(f"Device: {DEVICE}")

df = pd.read_csv("data/ready_dataset.csv")
df = df[["headline_clean", "body_clean", "label"]].dropna()
df["headline_clean"] = df["headline_clean"].astype(str).str.strip()
df["body_clean"] = df["body_clean"].astype(str).str.strip()
df = df[(df["headline_clean"] != "") & (df["body_clean"] != "")]
df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
df = df.reset_index(drop=True)

_, test_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])
test_df = test_df.reset_index(drop=True)
print(f"Test rows: {len(test_df)}")

tok = AutoTokenizer.from_pretrained(RUBERT_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(RUBERT_DIR).to(DEVICE)
mdl.eval()

y_true = []
y_pred = []
prob_real = []

with torch.no_grad():
    for i, row in test_df.iterrows():
        text = f"{row['headline_clean']} {row['body_clean']}"
        enc = tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        out = mdl(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE))
        p = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        y_true.append(int(row["label"]))
        y_pred.append(int(np.argmax(p)))
        prob_real.append(float(p[1]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_df)}")

acc = (np.array(y_pred) == np.array(y_true)).mean()
print(f"Accuracy: {acc:.4f}")

out_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "prob_real": prob_real})
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"Saved: {OUT_PATH}")
