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
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")

ASSETS = "assets"
RB_COLOR = "#1F77B4"
DS_COLOR = "#9467BD"


rb_df = pd.read_csv("models/rubert/test_predictions.csv")
ds_path = "models/deepseek/predictions_441_v5.csv"
if not os.path.exists(ds_path):
    ds_path = "models/deepseek/predictions_441_v4.csv"
ds_df = pd.read_csv(ds_path)

y_true = rb_df["y_true"].values.astype(int)
rb_pred = rb_df["y_pred"].values.astype(int)
rb_prob = rb_df["prob_real"].values.astype(float)
ds_pred = ds_df["pred"].values.astype(int)

ds_y_true = ds_df["true_label"].values.astype(int)

N = len(y_true)
print(f"Test dataset: {N} samples")


def metrics(y_t, y_p):
    return {
        "Accuracy": accuracy_score(y_t, y_p),
        "F1": f1_score(y_t, y_p, average="weighted"),
        "Precision": precision_score(y_t, y_p, average="weighted", zero_division=0),
        "Recall": recall_score(y_t, y_p, average="weighted", zero_division=0),
    }


rb_m = metrics(y_true, rb_pred)
ds_m = metrics(ds_y_true, ds_pred)

cmp_df = pd.DataFrame(
    [
        {"Модель": "RuBERT", **rb_m},
        {"Модель": "DeepSeek (API)", **ds_m},
    ]
)
csv_path = os.path.join(ASSETS, "rubert_vs_deepseek_comparsion.csv")
cmp_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Saved: {csv_path}")


fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, y_t, y_p, m) in zip(
    axes,
    [
        ("RuBERT", y_true, rb_pred, rb_m),
        ("DeepSeek (API)", ds_y_true, ds_pred, ds_m),
    ],
):
    cm = confusion_matrix(y_t, y_p)
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
    "Матрицы ошибок: RuBERT и DeepSeek", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_deepseek_comparsion_confusion.png")
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


cm_ds = confusion_matrix(ds_y_true, ds_pred)
tn, fp, fn, tp = cm_ds.ravel()
fpr_pt, tpr_pt = fp / (fp + tn), tp / (tp + fn)
ax.scatter(
    [fpr_pt],
    [tpr_pt],
    color=DS_COLOR,
    s=240,
    marker="*",
    edgecolors="black",
    linewidths=1.2,
    zorder=5,
    label=f"DeepSeek (TPR={tpr_pt:.3f}, FPR={fpr_pt:.3f})",
)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Доля ложных срабатываний (FPR)")
ax.set_ylabel("Доля верных срабатываний (TPR)")
ax.set_title(
    "ROC-кривые: RuBERT и DeepSeek\n"
    "(DeepSeek API возвращает только дискретный класс)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
out = os.path.join(ASSETS, "rubert_vs_deepseek_comparsion_roc.png")
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

ds_acc_fake = (ds_pred[ds_y_true == 0] == 0).mean()
ds_acc_real = (ds_pred[ds_y_true == 1] == 1).mean()
rb_acc_fake = (rb_pred[y_true == 0] == 0).mean()
rb_acc_real = (rb_pred[y_true == 1] == 1).mean()

x = np.arange(2)
w = 0.35
labels = ["Класс «Фейк»", "Класс «Реальная»"]
axes[1].bar(x - w / 2, [rb_acc_fake, rb_acc_real], w, color=RB_COLOR, label="RuBERT")
axes[1].bar(x + w / 2, [ds_acc_fake, ds_acc_real], w, color=DS_COLOR, label="DeepSeek")
for i, (rb_v, ds_v) in enumerate(
    zip([rb_acc_fake, rb_acc_real], [ds_acc_fake, ds_acc_real])
):
    axes[1].text(i - w / 2, rb_v + 0.01, f"{rb_v:.3f}", ha="center", fontsize=9)
    axes[1].text(i + w / 2, ds_v + 0.01, f"{ds_v:.3f}", ha="center", fontsize=9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 1.08)
axes[1].set_ylabel("Точность на классе (recall)")
axes[1].set_title("Точность по каждому классу")
axes[1].legend(loc="lower center")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Уверенность RuBERT и пер-классовая точность DeepSeek",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_deepseek_comparsion_confidence.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


both_right = ((rb_pred == y_true) & (ds_pred == y_true)).sum()
both_wrong = ((rb_pred != y_true) & (ds_pred != y_true)).sum()
only_rb = ((rb_pred == y_true) & (ds_pred != y_true)).sum()
only_ds = ((rb_pred != y_true) & (ds_pred == y_true)).sum()
agree = (rb_pred == ds_pred).sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

matrix = np.array(
    [
        [both_right, only_rb],
        [only_ds, both_wrong],
    ]
)
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[0],
    xticklabels=["DeepSeek прав", "DeepSeek ошибся"],
    yticklabels=["RuBERT прав", "RuBERT ошибся"],
    annot_kws={"size": 14},
    cbar=False,
)
axes[0].set_title(f"Согласие моделей: {agree}/{N} ({agree/N:.1%})", fontsize=11)

cats = [
    ("Обе модели правы", both_right, "#3FA34D"),
    ("Прав только RuBERT", only_rb, RB_COLOR),
    ("Прав только DeepSeek", only_ds, DS_COLOR),
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
    "Анализ согласованности: RuBERT и DeepSeek", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
out = os.path.join(ASSETS, "rubert_vs_deepseek_comparsion_agreement.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
