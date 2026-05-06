import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ASSETS = "assets"

RB_COLOR = "#1F77B4"
RG_COLOR = "#FF7F0E"


rb_df = pd.read_csv("models/rubert/test_predictions.csv")
rg_df = pd.read_csv("models/llm_v3_tuned/test_predictions.csv")


y_true = rb_df["y_true"].values.astype(int)
rb_pred = rb_df["y_pred"].values.astype(int)
rb_prob = rb_df["prob_real"].values.astype(float)
rg_pred = rg_df["pred"].values.astype(int)
rg_prob = rg_df["prob_real"].values.astype(float)


N = len(y_true)
print(f"Test data: {N} samples")


def metrics(y_t, y_p):
    return {
        "Accuracy": accuracy_score(y_t, y_p),
        "F1": f1_score(y_t, y_p, average="weighted"),
        "Precision": precision_score(y_t, y_p, average="weighted", zero_division=0),
        "Recall": recall_score(y_t, y_p, average="weighted", zero_division=0),
    }


rb_m = metrics(y_true, rb_pred)
rg_m = metrics(y_true, rg_pred)

cmp_df = pd.DataFrame(
    [
        {"Модель": "RuBERT", **rb_m},
        {"Модель": "ruGPT-3 + LoRA", **rg_m},
    ]
)
csv_path = os.path.join(ASSETS, "rubert_vs_rugpt_comparsion.csv")
cmp_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Saved: {csv_path}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, y_pred, m) in zip(
    axes,
    [
        ("RuBERT", rb_pred, rb_m),
        ("ruGPT-3 + LoRA", rg_pred, rg_m),
    ],
):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Фейк", "Реальная"],
        yticklabels=["Фейк", "Реальная"],
        annot_kws={"size": 14},
        cbar=False,
    )
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f"{name}\n"
        f"Точность {m['Accuracy']:.4f}, F1 {m['F1']:.4f}\n"
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        fontsize=11,
    )
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинная метка")
fig.suptitle(
    "Матрицы ошибок: RuBERT и ruGPT-3 + LoRA", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_rugpt_comparsion_confusion.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


fig, ax = plt.subplots(figsize=(8, 8))
fpr_rb, tpr_rb, _ = roc_curve(y_true, rb_prob)
roc_auc_rb = auc(fpr_rb, tpr_rb)
ax.plot(
    fpr_rb,
    tpr_rb,
    color=RB_COLOR,
    linewidth=2.2,
    label=f"RuBERT (AUC = {roc_auc_rb:.4f})",
)

fpr_rg, tpr_rg, _ = roc_curve(y_true, rg_prob)
roc_auc_rg = auc(fpr_rg, tpr_rg)
ax.plot(
    fpr_rg,
    tpr_rg,
    color=RG_COLOR,
    linewidth=2.2,
    label=f"ruGPT-3 + LoRA (AUC = {roc_auc_rg:.4f})",
)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Доля ложных срабатываний (FPR)")
ax.set_ylabel("Доля верных срабатываний (TPR)")
ax.set_title("ROC-кривые: RuBERT и ruGPT-3 + LoRA", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
out = os.path.join(ASSETS, "rubert_vs_rugpt_comparsion_roc.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

correct_rb = rb_pred == y_true
axes[0].hist(
    [rb_prob[correct_rb], rb_prob[~correct_rb]],
    bins=25,
    stacked=True,
    color=["#3FA34D", "#D7263D"],
    alpha=0.85,
    label=[f"Верные ({correct_rb.sum()})", f"Ошибочные ({(~correct_rb).sum()})"],
)
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
axes[0].set_xlabel("P(реальная) по RuBERT")
axes[0].set_ylabel("Количество примеров")
axes[0].set_title("Распределение уверенности RuBERT")
axes[0].legend(loc="upper center")
axes[0].grid(alpha=0.3)


correct_rg = rg_pred == y_true
axes[1].hist(
    [rg_prob[correct_rg], rg_prob[~correct_rg]],
    bins=25,
    stacked=True,
    color=["#3FA34D", "#D7263D"],
    alpha=0.85,
    label=[f"Верные ({correct_rg.sum()})", f"Ошибочные ({(~correct_rg).sum()})"],
)
axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
axes[1].set_xlabel("P(реальная) по ruGPT-3 + LoRA")
axes[1].set_ylabel("Количество примеров")
axes[1].set_title("Распределение уверенности ruGPT-3 + LoRA")
axes[1].legend(loc="upper center")
axes[1].grid(alpha=0.3)

fig.suptitle(
    "Уверенность и калибровка предсказаний", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_rugpt_comparsion_confidence.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


both_right = ((rb_pred == y_true) & (rg_pred == y_true)).sum()
both_wrong = ((rb_pred != y_true) & (rg_pred != y_true)).sum()
only_rb = ((rb_pred == y_true) & (rg_pred != y_true)).sum()
only_rg = ((rb_pred != y_true) & (rg_pred == y_true)).sum()
agree = (rb_pred == rg_pred).sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

matrix = np.array(
    [
        [both_right, only_rb],
        [only_rg, both_wrong],
    ]
)
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[0],
    xticklabels=["ruGPT-3 прав", "ruGPT-3 ошибся"],
    yticklabels=["RuBERT прав", "RuBERT ошибся"],
    annot_kws={"size": 14},
    cbar=False,
)
axes[0].set_title(f"Согласие моделей: {agree}/{N} ({agree/N:.1%})", fontsize=11)

cats = [
    ("Обе модели правы", both_right, "#3FA34D"),
    ("Прав только RuBERT", only_rb, RB_COLOR),
    ("Прав только ruGPT-3", only_rg, RG_COLOR),
    ("Обе ошиблись", both_wrong, "#D7263D"),
]
labels = [c[0] for c in cats]
values = [c[1] for c in cats]
colors = [c[2] for c in cats]
bars = axes[1].bar(labels, values, color=colors)
for bar, v in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        v + 1,
        f"{v}\n({v/N:.1%})",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axes[1].set_ylim(0, max(values) * 1.18)
axes[1].set_ylabel("Количество примеров")
axes[1].set_title("Распределение исходов по категориям")
axes[1].grid(axis="y", alpha=0.3)
plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right", fontsize=9)

fig.suptitle(
    "Анализ согласованности: RuBERT и ruGPT-3 + LoRA",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_rugpt_comparsion_agreement.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
