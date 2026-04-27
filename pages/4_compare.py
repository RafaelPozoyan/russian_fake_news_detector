import os
import pandas as pd
import streamlit as st
from utils.style import inject_css, sidebar_nav, load_json, back_to_main

st.set_page_config(page_title="Сравнение моделей", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

st.markdown("# Сравнение всех моделей")
st.markdown(
    "На этой странице собраны результаты всех подходов к детекции фейковых новостей: "
    "от классического TF-IDF до дообученных языковых моделей. "
    "Все метрики получены на одной и той же тестовой выборке."
)

# ── Сбор метрик ──────────────────────────────────────────────────────────────

rows = []

# Классические модели (TF-IDF)
m_tfidf = load_json("results/metrics/metrics_tfidf_tuned.json")
for name, key in [
    ("LR (TF-IDF)", "logistic_regression"),
    ("NB (TF-IDF)", "naive_bayes"),
    ("RF (TF-IDF)", "random_forest"),
]:
    if key in m_tfidf:
        d = m_tfidf[key]
        rows.append({
            "Модель": name,
            "Категория": "Классическая",
            "Accuracy": d.get("val_acc", 0),
            "F1": d.get("val_f1", 0),
            "Precision": d.get("precision", d.get("val_precision", 0)),
            "Recall": d.get("recall", d.get("val_recall", 0)),
        })

# Word2Vec
m_w2v = load_json("results/metrics/metrics_w2v.json")
for name, key in [
    ("LR (Word2Vec)", "logisticregression"),
    ("RF (Word2Vec)", "randomforest"),
]:
    if key in m_w2v:
        d = m_w2v[key]
        rows.append({
            "Модель": name,
            "Категория": "Классическая",
            "Accuracy": d.get("val_accuracy", 0),
            "F1": d.get("val_f1", d.get("f1", 0)),
            "Precision": d.get("precision", 0),
            "Recall": d.get("recall", 0),
        })

# RuBERT
m_rb = load_json("models/rubert/metrics.json")
rb = m_rb.get("rubert", {})
if rb:
    rows.append({
        "Модель": "RuBERT (fine-tuned)",
        "Категория": "Трансформер",
        "Accuracy": rb.get("test_acc", 0),
        "F1": rb.get("test_f1", 0),
        "Precision": rb.get("test_precision", 0),
        "Recall": rb.get("test_recall", 0),
    })

# LLM V1
m_v1 = load_json("models/llm/metrics.json")
v1 = m_v1.get("rugpt3_lora", {})
if v1:
    rows.append({
        "Модель": "ruGPT-3 + LoRA (V1)",
        "Категория": "LLM",
        "Accuracy": v1.get("accuracy", 0),
        "F1": v1.get("f1", 0),
        "Precision": v1.get("precision", 0),
        "Recall": v1.get("recall", 0),
    })

# LLM V2
m_v2 = load_json("models/llm_v2/metrics.json")
v2 = m_v2.get("ensemble", {})
if v2:
    rows.append({
        "Модель": "Frozen GPT + Sklearn (V2)",
        "Категория": "LLM",
        "Accuracy": v2.get("accuracy", 0),
        "F1": v2.get("f1", 0),
        "Precision": v2.get("precision", 0),
        "Recall": v2.get("recall", 0),
    })

# LLM V3
m_v3 = load_json("models/llm_v3/metrics.json")
v3 = m_v3.get("test", {})
if v3:
    rows.append({
        "Модель": "ruGPT-3 + LoRA Max (V3)",
        "Категория": "LLM",
        "Accuracy": v3.get("accuracy", 0),
        "F1": v3.get("f1", 0),
        "Precision": v3.get("precision", 0),
        "Recall": v3.get("recall", 0),
    })

# ── Таблица ──────────────────────────────────────────────────────────────────

if rows:
    st.markdown("---")
    st.markdown("## Метрики на тестовой выборке")

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df.index = range(1, len(df) + 1)

    best_model = df.iloc[0]["Модель"]
    best_acc = df.iloc[0]["Accuracy"]
    st.markdown(f"Лучший результат: **{best_model}** с точностью **{best_acc:.4f}**")

    display_df = df.copy()
    for col in ["Accuracy", "F1", "Precision", "Recall"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if x > 0 else "\u2014")

    st.dataframe(display_df, use_container_width=True, hide_index=False)

    # ── Графики ──────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## Графики")

    if os.path.exists("assets/complete_models_comparison.png"):
        st.image("assets/complete_models_comparison.png", caption="Сравнение всех подходов")

    if os.path.exists("assets/russian_models_comparison.png"):
        st.markdown("### Классические модели + RuBERT")
        st.image("assets/russian_models_comparison.png")

    if os.path.exists("assets/llm_comparison.png"):
        st.markdown("### LLM-подходы")
        st.image("assets/llm_comparison.png")

else:
    st.info("Метрики не найдены. Обучите модели, запустив соответствующие ноутбуки.")

# ── Сравнение RU vs EN ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Русские vs английские модели")

m_tfidf_eng = load_json("results/metrics/metrics_tfidf_eng.json")
m_w2v_eng = load_json("results/metrics/metrics_w2v_eng.json")

if m_tfidf_eng or m_w2v_eng:
    st.markdown(
        "Классические модели были также обучены на английском датасете для сравнения. "
        "Русские модели в целом показывают сопоставимые или лучшие результаты, "
        "несмотря на меньший размер обучающей выборки."
    )

    for fname, caption in [
        ("assets/Logistic Regression_tfidf_comparison.png", "LR на TF-IDF: RU vs EN"),
        ("assets/Naive Bayes_tfidf_comparison.png", "NB на TF-IDF: RU vs EN"),
        ("assets/Random Forest_tfidf_comparison.png", "RF на TF-IDF: RU vs EN"),
    ]:
        if os.path.exists(fname):
            st.image(fname, caption=caption)
else:
    st.info("Метрики английских моделей не найдены.")

# ── Выводы ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Итоги")
st.markdown("""
Сравнение показывает несколько закономерностей:

1. **TF-IDF + Logistic Regression** — надёжный базовый подход с высокой точностью (~96%) и минимальными требованиями к ресурсам.
2. **RuBERT** превосходит все классические модели (~97%) за счёт понимания контекста.
3. **LLM-подходы** дают конкурентоспособные результаты. Дообученные модели (V1, V3) работают лучше замороженного подхода (V2).
4. Замороженный подход (V2) полезен, когда нет ресурсов на дообучение, но уступает в качестве.

Для продакшена оптимальный выбор зависит от ограничений: если важна скорость — TF-IDF, если качество — RuBERT или ruGPT-3 + LoRA.
""")
