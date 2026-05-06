import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.style import inject_css, sidebar_nav, load_json, back_to_main

st.set_page_config(page_title="Сравнение всех моделей", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

st.markdown("# Сравнение всех моделей")
st.markdown(
    "Финальная сводная таблица по всем восьми моделям, использованным в работе. "
    "В сравнение включены пять классических классификаторов "
    "(LR/NB/RF на TF-IDF и LR/RF на Word2Vec), трансформер RuBERT с полным fine-tuning, "
    "локальная LLM ruGPT-3 с LoRA-адаптацией и большая API-модель DeepSeek в режиме "
    "retrieval-augmented few-shot."
)

st.markdown(
    "Метрики получены на одном и том же hold-out, что обеспечивает прямую "
    "сопоставимость. Подробный разбор каждого подхода доступен в соответствующих "
    "разделах сайта."
)


rows = []

# TF-IDF
m_tfidf = load_json("results/metrics/metrics_tfidf_tuned.json")
for name, key in [
    ("LR + TF-IDF", "logistic_regression"),
    ("NB + TF-IDF", "naive_bayes"),
    ("RF + TF-IDF", "random_forest"),
]:
    if key in m_tfidf:
        d = m_tfidf[key]
        rows.append(
            {
                "Модель": name,
                "Категория": "Классическая",
                "Accuracy": d.get("val_acc", 0),
                "F1": d.get("val_f1", 0),
                "Precision": d.get("precision", d.get("val_precision", 0)),
                "Recall": d.get("recall", d.get("val_recall", 0)),
            }
        )

# Word2Vec
m_w2v = load_json("results/metrics/metrics_w2v.json")
for name, key in [
    ("LR + Word2Vec", "logisticregression"),
    ("RF + Word2Vec", "randomforest"),
]:
    if key in m_w2v:
        d = m_w2v[key]
        rows.append(
            {
                "Модель": name,
                "Категория": "Классическая",
                "Accuracy": d.get("val_accuracy", 0),
                "F1": d.get("val_f1", d.get("f1", 0)),
                "Precision": d.get("precision", 0),
                "Recall": d.get("recall", 0),
            }
        )

# RuBERT
m_rb = load_json("models/rubert/metrics.json")
rb = m_rb.get("rubert", {})
if rb:
    rows.append(
        {
            "Модель": "RuBERT",
            "Категория": "Трансформер",
            "Accuracy": rb.get("test_acc", 0),
            "F1": rb.get("test_f1", 0),
            "Precision": rb.get("test_precision", 0),
            "Recall": rb.get("test_recall", 0),
        }
    )

# ruGPT-3 + LoRA
m_rugpt = load_json("models/llm_v3_tuned/metrics.json")
rugpt = m_rugpt.get("test", {})
if rugpt:
    rows.append(
        {
            "Модель": "ruGPT-3 + LoRA",
            "Категория": "LLM (локальная)",
            "Accuracy": rugpt.get("accuracy", 0),
            "F1": rugpt.get("f1", 0),
            "Precision": rugpt.get("precision", 0),
            "Recall": rugpt.get("recall", 0),
        }
    )

# DeepSeek
for path in (
    "models/deepseek/predictions_441_v5.csv",
    "models/deepseek/predictions_441_v4.csv",
    "models/deepseek/predictions_441_v3.csv",
):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if {"true_label", "pred"}.issubset(df.columns):
            y_t, y_p = df["true_label"].values, df["pred"].values
            rows.append(
                {
                    "Модель": "DeepSeek (API)",
                    "Категория": "LLM (API)",
                    "Accuracy": accuracy_score(y_t, y_p),
                    "F1": f1_score(y_t, y_p, average="weighted"),
                    "Precision": precision_score(y_t, y_p, average="weighted"),
                    "Recall": recall_score(y_t, y_p, average="weighted"),
                }
            )
            break


if rows:
    st.markdown("---")
    st.markdown("## Сводная таблица")

    df = (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    df.index = range(1, len(df) + 1)

    best_model = df.iloc[0]["Модель"]
    best_acc = df.iloc[0]["Accuracy"]
    st.markdown(f"Лучший результат: **{best_model}** с точностью **{best_acc:.4f}**.")

    display_df = df.copy()
    for col in ["Accuracy", "F1", "Precision", "Recall"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if x > 0 else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=False)
else:
    st.info("Метрики не найдены. Запустите соответствующие ноутбуки.")


st.markdown("---")
st.markdown("## Визуализация")

img_paths = [
    ("assets/final_comparsion_accuracy_f1.png", "Сравнение моделей по Accuracy и F1"),
    (
        "assets/final_comparsion_4metrics.png",
        "Все четыре метрики в одном представлении",
    ),
    ("assets/final_comparsion_cm.png", "Матрицы ошибок всех моделей"),
    ("assets/final_comparsion_roc.png", "ROC-кривые"),
]
for path, caption in img_paths:
    if os.path.exists(path):
        st.image(path, caption=caption)


st.markdown("---")
st.markdown("## Итоги")
st.markdown(
    "Эксперимент демонстрирует, что задача детекции фейков на собранном датасете "
    "решается достаточно хорошо самыми разными методами. Логистическая регрессия "
    "на TF-IDF уже обеспечивает приемлемое качество и остаётся наилучшим выбором, "
    "когда важна скорость и интерпретируемость. RuBERT и ruGPT-3 + LoRA дают "
    "примерно одинаковую точность, превосходящую классические методы за счёт умения "
    "учитывать контекст. DeepSeek в режиме few-shot выходит на тот же уровень "
    "вообще без обучения на нашем датасете — это аргумент в пользу того, что "
    "крупные API-модели могут быть конкурентоспособной альтернативой "
    "собственному дообучению, если допустимы зависимость от внешнего сервиса и "
    "затраты на запросы."
)
st.markdown(
    "Из практических соображений выбор конкретного подхода зависит от "
    "ограничений: при жёстких требованиях к скорости и автономности — TF-IDF + "
    "Logistic Regression; при наличии собственной GPU и потребности в высоком "
    "качестве — RuBERT или ruGPT-3 + LoRA; при готовности использовать внешний "
    "API и работать с короткими сериями запросов — DeepSeek."
)
