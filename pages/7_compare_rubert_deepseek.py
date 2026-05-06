import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from utils.style import inject_css, sidebar_nav, load_json, back_to_main

st.set_page_config(page_title="RuBERT vs DeepSeek", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

st.markdown("# RuBERT vs DeepSeek")
st.markdown(
    "Это сопоставление двух принципиально разных стратегий: RuBERT обучается "
    "на нашем датасете «с нуля» (full fine-tuning), а DeepSeek — внешняя большая "
    "языковая модель, к которой обращаются через API в режиме few-shot. "
    "Цель сравнения — понять, какой ценой достигается высокое качество: "
    "за счёт собственного дообучения или за счёт масштаба исходной модели."
)

st.markdown(
    "RuBERT — это специализированный под русский язык вариант BERT (≈178 млн параметров), "
    "который мы дообучаем с полным обновлением весов. Он работает локально и "
    "не зависит от сторонних сервисов. DeepSeek-V3, напротив, — модель с архитектурой "
    "Mixture-of-Experts на 671 млрд параметров; её обучают авторы, а наша задача "
    "сводится к подбору информативных примеров для контекста."
)


rows = []

m_rb = load_json("models/rubert/metrics.json")
rb = m_rb.get("rubert", {})
if rb:
    rows.append(
        {
            "Модель": "RuBERT",
            "Стратегия": "Полный fine-tuning на нашем train",
            "Accuracy": rb.get("test_acc", 0),
            "F1": rb.get("test_f1", 0),
            "Precision": rb.get("test_precision", 0),
            "Recall": rb.get("test_recall", 0),
        }
    )

deepseek_metrics = None
deepseek_cm = None
deepseek_n = 0
for path in (
    "models/deepseek/predictions_441_v5.csv",
    "models/deepseek/predictions_441_v4.csv",
    "models/deepseek/predictions_441_v3.csv",
):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if {"true_label", "pred"}.issubset(df.columns):
            y_t, y_p = df["true_label"].values, df["pred"].values
            deepseek_metrics = {
                "Accuracy": accuracy_score(y_t, y_p),
                "F1": f1_score(y_t, y_p, average="weighted"),
                "Precision": precision_score(y_t, y_p, average="weighted"),
                "Recall": recall_score(y_t, y_p, average="weighted"),
            }
            deepseek_cm = confusion_matrix(y_t, y_p)
            deepseek_n = len(df)
            break

if deepseek_metrics:
    rows.append(
        {
            "Модель": "DeepSeek (API)",
            "Стратегия": "24-shot retrieval, без обучения",
            **deepseek_metrics,
        }
    )

if rows:
    st.markdown("---")
    st.markdown("## Метрики на тестовой выборке")
    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ["Accuracy", "F1", "Precision", "Recall"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if x > 0 else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Метрики не найдены.")


rb_preds_path = "models/rubert/test_predictions.csv"
ds_preds_path = "models/deepseek/predictions_441_v5.csv"
if not os.path.exists(ds_preds_path):
    ds_preds_path = "models/deepseek/predictions_441_v4.csv"
if not os.path.exists(ds_preds_path):
    ds_preds_path = "models/deepseek/predictions_441_v3.csv"

if os.path.exists(rb_preds_path) and os.path.exists(ds_preds_path):
    rb_df = pd.read_csv(rb_preds_path)
    ds_df = pd.read_csv(ds_preds_path)

    if len(rb_df) == len(ds_df):
        true_rb = next(
            (c for c in rb_df.columns if "true" in c.lower()), rb_df.columns[0]
        )
        pred_rb = next(
            (c for c in rb_df.columns if "pred" in c.lower()), rb_df.columns[1]
        )
        pred_ds = next(
            (c for c in ds_df.columns if "pred" in c.lower()), ds_df.columns[1]
        )

        y_true = rb_df[true_rb].values
        rb_p = rb_df[pred_rb].values
        ds_p = ds_df[pred_ds].values

        rb_wrong = set(np.where(rb_p != y_true)[0])
        ds_wrong = set(np.where(ds_p != y_true)[0])

        st.markdown("---")
        st.markdown("## Где модели ошибаются")
        st.markdown(
            f"RuBERT ошибается на **{len(rb_wrong)}** примерах из {len(y_true)}, "
            f"DeepSeek — на **{len(ds_wrong)}**. Любопытная деталь: "
            f"пересечение их ошибок составляет всего {len(rb_wrong & ds_wrong)} "
            f"примеров. Это значит, что модели ошибаются на **разных** новостях, "
            f"и теоретически их предсказания можно было бы объединить в ансамбль "
            f"для улучшения общего результата."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Категория": [
                        "Только RuBERT ошибается",
                        "Только DeepSeek ошибается",
                        "Обе модели ошибаются",
                        "Обе правы",
                    ],
                    "Количество": [
                        len(rb_wrong - ds_wrong),
                        len(ds_wrong - rb_wrong),
                        len(rb_wrong & ds_wrong),
                        len(y_true) - len(rb_wrong | ds_wrong),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


img_paths = [
    (
        "assets/rubert_vs_deepseek_comparsion_confusion.png",
        "Матрицы ошибок: RuBERT и DeepSeek",
    ),
    ("assets/rubert_vs_deepseek_comparsion_roc.png", "ROC-кривые"),
    (
        "assets/rubert_vs_deepseek_comparsion_confidence.png",
        "Распределение уверенности",
    ),
    (
        "assets/rubert_vs_deepseek_comparsion_agreement.png",
        "Согласованность предсказаний",
    ),
    ("assets/final_deepseek_vs_top.png", "Сравнение топовых моделей"),
]
images_shown = False
for path, caption in img_paths:
    if os.path.exists(path):
        if not images_shown:
            st.markdown("---")
            st.markdown("## Визуализация")
            images_shown = True
        st.image(path, caption=caption)


st.markdown("---")
st.markdown("## Выводы")
st.markdown(
    "Сравнение показывает, что DeepSeek в режиме few-shot выходит на качество, "
    "сопоставимое с дообученным RuBERT, не видя нашей обучающей выборки на этапе "
    "тренировки модели. Это свидетельствует прежде всего о том, что задача "
    "детекции фейков на используемом датасете в значительной мере решаема за счёт "
    "масштаба и обобщающих способностей крупной языковой модели. С другой стороны, "
    "RuBERT работает локально и предсказуемо: его поведение не зависит от внешнего "
    "API, доступности интернета и тарифной политики провайдера. Выбор между "
    "подходами в реальных задачах должен учитывать оба фактора — качество и "
    "эксплуатационную надёжность."
)
