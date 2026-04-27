import os
import pandas as pd
import streamlit as st
from utils.style import inject_css, sidebar_nav, render_metric_row, load_json, back_to_main, render_comparison_section

st.set_page_config(page_title="LLM-подходы", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

# ── Загрузка метрик ──────────────────────────────────────────────────────────

llm_v1_metrics = load_json("models/llm/metrics.json")
llm_v2_metrics = load_json("models/llm_v2/metrics.json")
llm_v3_metrics = load_json("models/llm_v3/metrics.json")

# ── Заголовок ────────────────────────────────────────────────────────────────

st.markdown("# LLM-подходы к детекции фейков")
st.markdown(
    "Помимо классических моделей, в проекте исследованы три подхода "
    "на основе больших языковых моделей. Все они используют ruGPT-3 small (125M параметров) — "
    "русскоязычную авторегрессионную модель от AI Forever."
)

# ── Обзор подходов ───────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Обзор подходов")

approaches = [
    {
        "title": "V1: ruGPT-3 + LoRA",
        "desc": (
            "Базовый подход: дообучение ruGPT-3 методом LoRA (r=16). "
            "Адаптируется только модуль `c_attn` (Query/Key/Value), "
            "что позволяет обучить менее 1% параметров модели."
        ),
        "params": "LoRA r=16, 1 модуль, max_length=256",
        "metrics_key": "rugpt3_lora",
        "metrics_src": llm_v1_metrics,
    },
    {
        "title": "V2: Frozen ruGPT-3 + Sklearn",
        "desc": (
            "ruGPT-3 работает как замороженный экстрактор признаков. "
            "Заголовок и тело кодируются отдельно, из их эмбеддингов формируются "
            "признаки взаимодействия (конкатенация, разность, произведение). "
            "Классификацию выполняет ансамбль из логистической регрессии, "
            "линейного SVM и градиентного бустинга."
        ),
        "params": "4x768=3072 признака, StandardScaler, soft voting",
        "metrics_key": "ensemble",
        "metrics_src": llm_v2_metrics,
    },
    {
        "title": "V3: ruGPT-3 + LoRA (max quality)",
        "desc": (
            "Агрессивное дообучение ruGPT-3 с максимальными настройками LoRA. "
            "Адаптируются два модуля: `c_attn` (на что модель обращает внимание) "
            "и `c_proj` (как комбинирует информацию). Высокий ранг, label smoothing, "
            "gradient accumulation и cosine schedule обеспечивают лучшее качество."
        ),
        "params": "LoRA r=32, 2 модуля, effective batch=32, 10 эпох",
        "metrics_key": "test",
        "metrics_src": llm_v3_metrics,
    },
]

for i, a in enumerate(approaches):
    st.markdown(f"### {a['title']}")
    st.markdown(a["desc"])
    st.markdown(f"*Параметры:* `{a['params']}`")

    m = a["metrics_src"].get(a["metrics_key"], {})
    if m:
        render_metric_row({
            "Accuracy": m.get("accuracy", 0),
            "F1": m.get("f1", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
        })
    else:
        st.info("Модель ещё не обучена — метрики появятся после запуска ноутбука.")

    if i < len(approaches) - 1:
        st.markdown("")

# ── Сводная таблица ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Сводная таблица")

rows = []
for a in approaches:
    m = a["metrics_src"].get(a["metrics_key"], {})
    if m:
        rows.append({
            "Подход": a["title"],
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
        })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Графики ──────────────────────────────────────────────────────────────────

if os.path.exists("assets/llm_comparison.png"):
    st.markdown("---")
    st.markdown("## Визуализация")
    st.image("assets/llm_comparison.png", caption="Сравнение LLM-подходов")

if os.path.exists("assets/llm_confusion_matrices.png"):
    st.image("assets/llm_confusion_matrices.png", caption="Матрицы ошибок")

# ── Выводы ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Выводы")
st.markdown(
    "Подходы на базе LLM показывают конкурентоспособные результаты. "
    "V1 (LoRA r=16) и V3 (LoRA r=32) напрямую дообучают языковую модель — "
    "они лучше понимают семантические связи между заголовком и телом статьи. "
    "V2 использует замороженный энкодер как экстрактор признаков, что "
    "быстрее в обучении, но ограничивает адаптацию к задаче."
)

# ── Сравнение с другими подходами ────────────────────────────────────────────

render_comparison_section()
