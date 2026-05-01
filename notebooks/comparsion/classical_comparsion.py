import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc,
)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")

SEED = 42
DATA_PATH = "data/ready_dataset.csv"
ASSETS = "assets"

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("russian"))


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"\S+@\S+", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^а-яёa-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)


print("Загрузка датасета")
df = pd.read_csv(DATA_PATH)
df = df[["headline_clean", "body_clean", "label"]].dropna()
df["headline_clean"] = df["headline_clean"].astype(str).str.strip()
df["body_clean"] = df["body_clean"].astype(str).str.strip()
df = df[(df["headline_clean"] != "") & (df["body_clean"] != "")]
df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
df = df.reset_index(drop=True)

_, test_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])
test_df = test_df.reset_index(drop=True)

y_true = test_df["label"].values
headlines = test_df["headline_clean"].astype(str).tolist()
bodies = test_df["body_clean"].astype(str).tolist()
print(f"Тестовая выборка: {len(test_df)} примеров")


print("TF-IDF инференс")
vec = pickle.load(open("models/tfidf_vectorizer_tf.pkl", "rb"))
lr_tf = pickle.load(open("models/logistic_regression_model_tf.pkl", "rb"))
nb_tf = pickle.load(open("models/naive_bayes_model_tf.pkl", "rb"))
rf_tf = pickle.load(open("models/random_forest_model_tf.pkl", "rb"))

h_clean = [preprocess_text(h) for h in headlines]
b_clean = [preprocess_text(b) for b in bodies]
X_tf = vec.transform([f"{h} {b}" for h, b in zip(h_clean, b_clean)])

tfidf_models = {
    "Логистическая\nрегрессия": (lr_tf.predict(X_tf), lr_tf.predict_proba(X_tf)[:, 1]),
    "Наивный\nБайес": (nb_tf.predict(X_tf), nb_tf.predict_proba(X_tf)[:, 1]),
    "Случайный лес": (rf_tf.predict(X_tf), rf_tf.predict_proba(X_tf)[:, 1]),
}


print("Word2Vec инференс")
kv = KeyedVectors.load("models/w2v_vectors.kv")
lr_w2v = pickle.load(open("models/logisticregression_model.pkl", "rb"))
rf_w2v = pickle.load(open("models/randomforest_model.pkl", "rb"))


def doc_vector(tokens, kv_model):
    vecs = [kv_model[w] for w in tokens if w in kv_model]
    if not vecs:
        return np.zeros(kv_model.vector_size, dtype=np.float32)
    return np.vstack(vecs).mean(axis=0)


feats = []
for h, b in zip(h_clean, b_clean):
    htoks = h.split()[:150]
    btoks = b.split()[:150]
    h_vec = doc_vector(htoks, kv)
    b_vec = doc_vector(btoks, kv)
    cos = float(np.dot(h_vec, b_vec) / max(np.linalg.norm(h_vec) * np.linalg.norm(b_vec), 1e-8))
    jacc = len(set(htoks) & set(btoks)) / max(1, len(set(htoks) | set(btoks)))
    ovr = len(set(htoks) & set(btoks)) / max(1, len(set(htoks)))
    l2 = float(np.linalg.norm(h_vec - b_vec))
    feats.append(np.hstack([h_vec, b_vec, np.abs(h_vec - b_vec), h_vec * b_vec, [cos, jacc, ovr, l2]]))

X_w2v = np.array(feats)

w2v_models = {
    "Логистическая\nрегрессия": (lr_w2v.predict(X_w2v), lr_w2v.predict_proba(X_w2v)[:, 1]),
    "Случайный лес": (rf_w2v.predict(X_w2v), rf_w2v.predict_proba(X_w2v)[:, 1]),
}


def plot_group(models, title, out_path):
    n_models = len(models)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, max(n_models, 3), hspace=0.45, wspace=0.35)

    ax_metrics = fig.add_subplot(gs[0, :2])
    rows = []
    for name, (preds, _) in models.items():
        rows.append({
            "Модель": name.replace("\n", " "),
            "Accuracy": accuracy_score(y_true, preds),
            "F1": f1_score(y_true, preds, average="weighted"),
            "Precision": precision_score(y_true, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, preds, average="weighted", zero_division=0),
        })
    metrics_df = pd.DataFrame(rows).set_index("Модель")

    metric_cols = ["Accuracy", "F1", "Precision", "Recall"]
    rus_labels = {"Accuracy": "Точность", "F1": "F1-мера", "Precision": "Precision", "Recall": "Recall"}
    x = np.arange(len(metrics_df))
    width = 0.2
    palette = ["#2E86AB", "#A23B72", "#F18F01", "#3FA34D"]

    for i, col in enumerate(metric_cols):
        bars = ax_metrics.bar(x + i * width, metrics_df[col], width, label=rus_labels[col], color=palette[i])
        for bar in bars:
            v = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

    ax_metrics.set_xticks(x + width * 1.5)
    ax_metrics.set_xticklabels(metrics_df.index)
    ax_metrics.set_ylim(0.0, 1.05)
    ax_metrics.set_ylabel("Значение метрики")
    ax_metrics.set_title("Метрики качества (взвешенные средние)")
    ax_metrics.legend(loc="lower right", ncol=4, fontsize=9)
    ax_metrics.grid(axis="y", alpha=0.3)

    ax_roc = fig.add_subplot(gs[0, 2:])
    for (name, (_, probs)), color in zip(models.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name.replace(chr(10), ' ')} (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax_roc.set_xlabel("Доля ложных срабатываний (FPR)")
    ax_roc.set_ylabel("Доля верных срабатываний (TPR)")
    ax_roc.set_title("ROC-кривые")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    for i, (name, (preds, _)) in enumerate(models.items()):
        ax_cm = fig.add_subplot(gs[1, i])
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Фейк", "Реальная"],
                    yticklabels=["Фейк", "Реальная"],
                    cbar=False, annot_kws={"size": 12})
        acc = accuracy_score(y_true, preds)
        ax_cm.set_title(f"{name.replace(chr(10), ' ')}\n(Точность {acc:.3f})", fontsize=10)
        ax_cm.set_xlabel("Предсказание")
        ax_cm.set_ylabel("Истина")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


print("Отрисовка TF-IDF")
plot_group(tfidf_models, "Сравнение классических моделей на TF-IDF признаках",
           os.path.join(ASSETS, "classical_comparsion_tfidf.png"))

print("Отрисовка Word2Vec")
plot_group(w2v_models, "Сравнение классических моделей на Word2Vec признаках",
           os.path.join(ASSETS, "classical_comparsion_w2v.png"))

print("Готово")
