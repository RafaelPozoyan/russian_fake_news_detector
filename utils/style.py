"""Общие стили и утилиты для всех страниц."""
import streamlit as st
import json
import os

ACCENT = "#4A9EF5"
SUCCESS = "#2ECC71"
DANGER = "#E74C3C"


def inject_css():
    """Минимальный общий CSS для всего сайта."""
    st.markdown("""
    <style>
    /* Карточки */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .card h4 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #ccc; }
    .card .value { font-size: 2rem; font-weight: 700; }

    /* Метрики в ряд */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        flex: 1;
        text-align: center;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.8rem 0.5rem;
    }
    .metric-box .label { font-size: 0.8rem; color: #999; margin-bottom: 0.3rem; }
    .metric-box .val { font-size: 1.5rem; font-weight: 700; }

    /* Результат предсказания */
    .result-fake {
        background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(231,76,60,0.05));
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-real {
        background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(46,204,113,0.05));
        border: 1px solid rgba(46,204,113,0.3);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-conf { font-size: 0.95rem; color: #aaa; }

    /* Скрытие дефолтного заголовка страницы */
    header[data-testid="stHeader"] { background: transparent; }

    /* Таблицы */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .styled-table th {
        background: rgba(74,158,245,0.15);
        padding: 0.6rem 0.8rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid rgba(74,158,245,0.3);
    }
    .styled-table td {
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .styled-table tr:hover td {
        background: rgba(255,255,255,0.03);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
    }

    /* Кнопки */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: dict):
    """Рендерит ряд метрик в карточках."""
    html = '<div class="metric-row">'
    for label, value in metrics.items():
        if isinstance(value, float):
            val_str = f"{value:.4f}"
        else:
            val_str = str(value)
        html += f'<div class="metric-box"><div class="label">{label}</div><div class="val">{val_str}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_prediction(label: int, confidence: float, model_name: str = ""):
    """Рендерит результат предсказания."""
    cls = "result-fake" if label == 0 else "result-real"
    text = "Фейк" if label == 0 else "Достоверная"
    color = DANGER if label == 0 else SUCCESS
    prefix = f"<div style='font-size:0.8rem;color:#888;margin-bottom:0.2rem;'>{model_name}</div>" if model_name else ""
    st.markdown(
        f'<div class="{cls}">{prefix}'
        f'<div class="result-label" style="color:{color};">{text}</div>'
        f'<div class="result-conf">Уверенность: {confidence:.1%}</div></div>',
        unsafe_allow_html=True,
    )


def load_json(path: str) -> dict:
    """Загружает JSON, возвращает {} если файл не найден."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sidebar_nav():
    """Навигация в сайдбаре."""
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", use_column_width=True)

        st.markdown("---")
        st.page_link("app.py", label="Главная")
        st.page_link("pages/1_classical.py", label="Классические модели")
        st.page_link("pages/2_rubert.py", label="RuBERT")
        st.page_link("pages/3_llm.py", label="LLM-подходы")
        st.markdown("---")
        st.page_link("pages/4_compare.py", label="Сравнение моделей")
        st.page_link("pages/5_english.py", label="Английский детектор")
        st.markdown("---")
        st.caption("ВКР | Позоян Р.О. | БПМ-22-ПО-3")


def back_to_main():
    """Кнопка возврата на главную страницу."""
    if st.button("← На главную", key="_back_to_main"):
        st.switch_page("app.py")


def render_comparison_section():
    """Секция сравнения с другими подходами (для страниц методов)."""
    st.markdown("---")
    st.markdown("## Сравнение с другими подходами")
    if os.path.exists("assets/complete_models_comparison.png"):
        st.image("assets/complete_models_comparison.png", caption="Сравнение всех моделей по Accuracy и F1")
    if st.button("Подробное сравнение всех моделей →", key="_compare_link"):
        st.switch_page("pages/4_compare.py")
