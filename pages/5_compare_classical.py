import os
import json
import pandas as pd
import streamlit as st
from utils.style import inject_css, sidebar_nav, load_json, back_to_main

st.set_page_config(page_title="Сравнение классических моделей", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

st.markdown("# Сравнение классических моделей")
st.markdown(
    "На этой странице собраны результаты пяти классических моделей машинного обучения, "
    "обученных на двух способах векторизации текста: TF-IDF и Word2Vec. "
    "Все модели проверялись на одной и той же тестовой выборке."
)

st.markdown(
    "TF-IDF опирается на статистику встречаемости слов и хорошо улавливает "
    "лексические маркеры, характерные для конкретных классов. Word2Vec работает "
    "иначе — он представляет слова в виде плотных векторов, отражающих их "
    "семантическую близость, а итоговый вектор документа получается усреднением. "
    "Сопоставление этих двух подходов позволяет понять, что важнее в задаче "
    "детекции фейков: точное совпадение характерной лексики или общая семантическая "
    "согласованность заголовка и тела статьи."
)

# ── Сбор метрик ──────────────────────────────────────────────────────────────

rows = []

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
                "Векторизация": "TF-IDF",
                "Accuracy": d.get("val_acc", 0),
                "F1": d.get("val_f1", 0),
            }
        )

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
                "Векторизация": "Word2Vec",
                "Accuracy": d.get("val_accuracy", 0),
                "F1": d.get("val_f1", d.get("f1", 0)),
            }
        )

if rows:
    st.markdown("---")
    st.markdown("## Метрики на валидационной выборке")

    df = (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    df.index = range(1, len(df) + 1)

    best = df.iloc[0]
    st.markdown(
        f"Наилучший результат среди классических моделей показывает **{best['Модель']}** "
        f"с точностью **{best['Accuracy']:.4f}**."
    )

    display_df = df.copy()
    for col in ["Accuracy", "F1"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if x > 0 else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=False)
else:
    st.info("Метрики классических моделей не найдены.")

# ── Графики ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Визуализация результатов")

img_pairs = [
    ("assets/classical_comparsion_tfidf.png", "Сравнение моделей на TF-IDF признаках"),
    ("assets/classical_comparsion_w2v.png", "Сравнение моделей на Word2Vec признаках"),
]
for path, caption in img_pairs:
    if os.path.exists(path):
        st.image(path, caption=caption)

# ── Выводы ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Выводы")
st.markdown(
    "Логистическая регрессия на TF-IDF показывает лучший баланс между точностью и "
    "интерпретируемостью среди всех классических моделей. Random Forest на тех же "
    "признаках близок по качеству, но обучается дольше. Naive Bayes даёт чуть более "
    "слабый результат, что ожидаемо: модель опирается на предположение о независимости "
    "признаков, которое плохо выполняется на естественном языке. Подходы на Word2Vec "
    "уступают TF-IDF на этом датасете — для коротких новостных текстов лексические "
    "признаки оказываются информативнее усреднённых семантических векторов."
)
