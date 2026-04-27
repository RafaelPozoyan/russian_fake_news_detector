import os
import json
import streamlit as st
from utils.style import inject_css, sidebar_nav, render_metric_row, back_to_main, render_comparison_section

st.set_page_config(page_title="RuBERT", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

# ── Загрузка метрик ──────────────────────────────────────────────────────────

rubert_metrics = {}
if os.path.exists("models/rubert/metrics.json"):
    with open("models/rubert/metrics.json", encoding="utf-8") as f:
        data = json.load(f)
        rubert_metrics = data.get("rubert", data)

# ── Заголовок ────────────────────────────────────────────────────────────────

st.markdown("# RuBERT")
st.markdown(
    "RuBERT — это адаптация модели BERT для русского языка, предобученная на большом "
    "русскоязычном корпусе. В отличие от классических подходов, трансформер работает "
    "с контекстом на уровне предложений: одно и то же слово получает разное представление "
    "в зависимости от окружающих слов."
)

# ── Архитектура ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Архитектура")

st.markdown(
    "Для классификации используется полное дообучение (fine-tuning) RuBERT: все "
    "параметры модели обновляются в процессе обучения на нашем датасете."
)

st.markdown("""
**Как работает:**

1. Заголовок и тело статьи объединяются в один текст
2. Токенизатор разбивает текст на подслова (до 256 токенов)
3. Энкодер BERT обрабатывает последовательность, учитывая контекст каждого токена
4. Специальный токен `[CLS]` агрегирует информацию о всём тексте
5. Линейный классификатор поверх `[CLS]` выдаёт вердикт: фейк или нет
""")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **Параметры модели:**
    - Архитектура: BERT-base
    - Количество параметров: ~178M
    - Метод: full fine-tuning
    - Длина контекста: 256 токенов
    """)
with c2:
    st.markdown("""
    **Обучение:**
    - Оптимизатор: AdamW
    - Learning rate: 2e-5
    - Батч: 16
    - Эпохи: 3-5
    """)

# ── Результаты ───────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Результаты")

if rubert_metrics:
    render_metric_row({
        "Accuracy": rubert_metrics.get("test_acc", rubert_metrics.get("accuracy", 0)),
        "F1": rubert_metrics.get("test_f1", rubert_metrics.get("f1", 0)),
        "Precision": rubert_metrics.get("test_precision", rubert_metrics.get("precision", 0)),
        "Recall": rubert_metrics.get("test_recall", rubert_metrics.get("recall", 0)),
    })

    st.markdown(
        "RuBERT достигает точности **~97%** на тестовой выборке — это лучший результат "
        "среди всех классических подходов. Трансформер лучше улавливает тонкие несоответствия "
        "между заголовком и текстом, которые пропускают модели на основе TF-IDF."
    )
else:
    st.info("Метрики RuBERT не найдены.")

# ── Матрица ошибок ───────────────────────────────────────────────────────────

if os.path.exists("assets/russian_models_confusion_matrices.png"):
    st.markdown("---")
    st.markdown("## Матрицы ошибок")
    st.image("assets/russian_models_confusion_matrices.png")

# ── Преимущества и ограничения ───────────────────────────────────────────────

st.markdown("---")
st.markdown("## Преимущества и ограничения")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **Преимущества:**
    - Лучшая точность среди не-LLM подходов
    - Понимание контекста на уровне предложений
    - Устойчивость к перефразированию
    - Работа с морфологией русского языка
    """)
with c2:
    st.markdown("""
    **Ограничения:**
    - Большой размер модели (~700 МБ)
    - Медленнее классических моделей при инференсе
    - Требует GPU для комфортного обучения
    - Ограничен 256 токенами (часть длинных статей обрезается)
    """)

# ── Сравнение с другими подходами ────────────────────────────────────────────

render_comparison_section()
