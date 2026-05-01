import os
import json
import streamlit as st
from utils.style import inject_css, sidebar_nav, render_metric_row, back_to_main, render_comparsion_section

st.set_page_config(page_title="Классические модели", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

# ── Загрузка метрик ──────────────────────────────────────────────────────────

metrics_tfidf = {}
metrics_w2v = {}
if os.path.exists("results/metrics/metrics_tfidf_tuned.json"):
    with open("results/metrics/metrics_tfidf_tuned.json", encoding="utf-8") as f:
        metrics_tfidf = json.load(f)
if os.path.exists("results/metrics/metrics_w2v.json"):
    with open("results/metrics/metrics_w2v.json", encoding="utf-8") as f:
        metrics_w2v = json.load(f)

# ── Заголовок ────────────────────────────────────────────────────────────────

st.markdown("# Классические модели машинного обучения")
st.markdown(
    "В проекте используются два подхода к векторизации текста: **TF-IDF** и **Word2Vec**. "
    "На их основе обучены три классификатора: логистическая регрессия, наивный Байес и случайный лес."
)

# ── TF-IDF ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## TF-IDF")

st.markdown(
    "TF-IDF (Term Frequency — Inverse Document Frequency) преобразует текст в числовой вектор, "
    "где каждое слово получает вес, зависящий от его частоты в документе и редкости в корпусе. "
    "Частые слова вроде предлогов и союзов получают низкий вес, а характерные для конкретного "
    "документа — высокий."
)

st.markdown(
    "Заголовок и тело статьи объединяются в один текст, проходят предобработку "
    "(удаление стоп-слов, спецсимволов, приведение к нижнему регистру), "
    "после чего TF-IDF-векторизатор формирует признаковое пространство."
)

st.markdown("### Результаты на валидации")

if metrics_tfidf:
    c1, c2, c3 = st.columns(3)

    models_info = [
        ("Logistic Regression", "logistic_regression", c1),
        ("Naive Bayes", "naive_bayes", c2),
        ("Random Forest", "random_forest", c3),
    ]

    for name, key, col in models_info:
        if key in metrics_tfidf:
            m = metrics_tfidf[key]
            with col:
                st.markdown(f"**{name}**")
                render_metric_row({
                    "Accuracy": m.get("val_acc", 0),
                    "F1": m.get("val_f1", 0),
                })
else:
    st.info("Метрики TF-IDF не найдены.")

if os.path.exists("assets/classical_comparsion_tfidf.png"):
    st.image("assets/classical_comparsion_tfidf.png", caption="Сравнение моделей на TF-IDF признаках")

# ── Word2Vec ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Word2Vec")

st.markdown(
    "Word2Vec учитывает семантику слов: слова с похожим значением получают близкие "
    "вектора в многомерном пространстве. Вектор документа — это среднее арифметическое "
    "векторов всех его слов."
)

st.markdown(
    "Для пары \"заголовок — тело\" вычисляются дополнительные признаки:"
)

st.markdown("""
- **Косинусная близость** — насколько схожи вектора заголовка и тела
- **Jaccard-коэффициент** — доля общих слов
- **Overlap** — какая часть слов заголовка встречается в теле
- **L2-расстояние** — евклидово расстояние между векторами
""")

st.markdown(
    "Эти признаки также используются в бизнес-правилах: если косинусная "
    "близость слишком низкая или заголовок не пересекается с текстом, "
    "новость автоматически помечается как фейковая."
)

st.markdown("### Результаты на валидации")

if metrics_w2v:
    c1, c2 = st.columns(2)
    for name, key, col in [
        ("Logistic Regression", "logisticregression", c1),
        ("Random Forest", "randomforest", c2),
    ]:
        if key in metrics_w2v:
            m = metrics_w2v[key]
            with col:
                st.markdown(f"**{name}**")
                render_metric_row({
                    "Accuracy": m.get("val_accuracy", 0),
                    "F1": m.get("val_f1", m.get("f1", 0)),
                })
else:
    st.info("Метрики Word2Vec не найдены.")

if os.path.exists("assets/classical_comparsion_w2v.png"):
    st.image("assets/classical_comparsion_w2v.png", caption="Сравнение моделей на Word2Vec признаках")

# ── Визуализации ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Анализ данных")

c1, c2 = st.columns(2)
with c1:
    if os.path.exists("assets/text_lenght.png"):
        st.image("assets/text_lenght.png", caption="Распределение длин текстов")
with c2:
    if os.path.exists("assets/wordcloud_vectorized.png"):
        st.image("assets/wordcloud_vectorized.png", caption="Облако слов")

# ── Выводы ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Выводы")
st.markdown(
    "Лучший результат среди классических моделей показывает **Logistic Regression на TF-IDF** "
    "с точностью около 96%. Word2Vec-подход даёт чуть ниже результат (~92%), "
    "но лучше справляется с перефразированными заголовками благодаря учёту семантики. "
    "Бизнес-правила поверх Word2Vec помогают отсеивать явные несоответствия."
)

# ── Сравнение с другими подходами ────────────────────────────────────────────

render_comparsion_section()
