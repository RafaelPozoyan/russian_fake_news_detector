import streamlit as st
from PIL import Image
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="LLM-детектор фейковых новостей", page_icon="🤖", layout="wide"
)


def set_styles():
    st.markdown(
        """
    <style>

    .stApp {
        background: rgb(20, 19, 28);
    }

    h1 {
        color: #FFFFE0; font-weight: 700; text-align: center; font-size: 3rem; margin-bottom: 0.5rem;
    }

    h2, h3 {
        color: #FFFFE0; font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFE0; border-radius: 7px; padding: 0.75rem 3rem; font-weight: 600;
        font-size: 1.1rem; border: none; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease; width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 7px; border: 2px solid #e5e7eb; font-size: 1rem; padding: 0.75rem; transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .stTextInput>label, .stTextArea>label {
        font-weight: 1000; color: #FFFFE0; font-size: 1.1rem;
    }

    [data-testid="stSidebar"] {
        background: #0e1117;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }

    .metric-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; padding: 0.1rem; text-align: center; margin: 0.1rem;
    }

    .metric-value {
        font-size: 2.5rem; font-weight: 700; color: #0369a1; margin-top: 0.1rem;
    }

    .metric-label {
        font-size: 1rem; color: #64748b;
    }

    .stImage.round-logo img {
        border-radius: 50%;
    }

    .streamlit-expanderHeader {
        color: #FFFFE0; font-weight: 600;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


set_styles()

# ── Конфиг модели ────────────────────────────────────────────────────────────

LLM_MODEL_DIR = "models/llm/rugpt3_lora"
BASE_MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Загрузка модели ──────────────────────────────────────────────────────────


@st.cache_resource
def load_llm_model(model_dir=LLM_MODEL_DIR, base_name=BASE_MODEL_NAME):
    """Загружаем базовую GPT-модель + LoRA адаптер и сливаем для быстрого инференса."""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=2
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_llm(text, tokenizer, model, max_length=256):
    text = str(text).strip()

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

    return {
        "label": pred,
        "label_name": "Правдивая" if pred == 1 else "Фейковая",
        "fake_proba": float(probs[0]),
        "real_proba": float(probs[1]),
    }


# ── Загрузка метрик ──────────────────────────────────────────────────────────

llm_metrics = None
METRICS_PATH = "models/llm/metrics.json"
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        llm_metrics = json.load(f)

# ── Загрузка тестовых данных ─────────────────────────────────────────────────

test_bodies = pd.read_csv("./data/test_bodies.csv")
test_stances = pd.read_csv("./data/test_stances_unlebeledb.csv")
test_data = test_stances.merge(test_bodies, on="Body ID", how="left")

# ── Сайдбар ──────────────────────────────────────────────────────────────────

with st.sidebar:
    logo = Image.open("assets/logo.png")
    st.image(logo, output_format="round-logo")

    st.markdown("---")
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h3 style='color: #FFFFE0;'>О подходе</h3>
            <p>На этой странице используется большая языковая модель
            <b>ruGPT-3</b>, дообученная методом <b>LoRA</b> (Low-Rank Adaptation)
            для детекции фейковых новостей.</p>
            <p>LoRA позволяет адаптировать модель, обучая менее 3% параметров.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    llm_status = os.path.exists(LLM_MODEL_DIR)
    if llm_status:
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>Статус LLM:</h4><p>ruGPT-3 + LoRA загружена</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>Статус LLM:</h4><p>Модель не найдена. Запусти llm_detection.ipynb</p></div>",
            unsafe_allow_html=True,
        )

    if llm_metrics:
        st.markdown("---")
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>Метрики на тесте:</h4></div>",
            unsafe_allow_html=True,
        )
        lora_m = llm_metrics.get("rugpt3_lora", {})
        for metric_name, metric_key in [
            ("Accuracy", "accuracy"),
            ("F1", "f1"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ]:
            val = lora_m.get(metric_key)
            if val is not None:
                st.markdown(
                    f"<div style='color:#FFFFE0;'><b>{metric_name}:</b> {val:.4f}</div>",
                    unsafe_allow_html=True,
                )


# ── Навигация ────────────────────────────────────────────────────────────────

if st.button("←", help="Вернуться на главную страницу", type="secondary"):
    st.switch_page("app.py")


# ── Главный контент ──────────────────────────────────────────────────────────

st.title("Детектор фейков на основе LLM")
st.info("ruGPT-3 + LoRA — параметрически эффективное дообучение большой языковой модели")

with st.expander("Как это работает?", expanded=False):
    st.markdown(
        """<div style='color:#FFFFE0;'>
    <b>LoRA (Low-Rank Adaptation)</b> — метод дообучения, при котором к замороженным
    слоям внимания модели добавляются обучаемые низкоранговые матрицы.
    Это позволяет:
    <ul>
    <li>Обучать менее 3% параметров вместо всех 125M</li>
    <li>Сохранять общие знания предобученной модели</li>
    <li>Хранить адаптер в несколько МБ</li>
    </ul>
    <br>
    <b>ruGPT-3 Small</b> — декодерная модель архитектуры GPT-2 (125M параметров),
    предобученная на русскоязычном корпусе (ai-forever).
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if "headline_llm" not in st.session_state:
    st.session_state.headline_llm = ""
if "body_llm" not in st.session_state:
    st.session_state.body_llm = ""


def pick_news():
    idx = random.randint(0, len(test_data) - 1)
    st.session_state.headline_llm = test_data.loc[idx, "Headline1"]
    st.session_state.body_llm = test_data.loc[idx, "articleBody"]


def clear_fields():
    st.session_state.headline_llm = ""
    st.session_state.body_llm = ""


with st.container():
    headline = st.text_input(
        "Заголовок новости:",
        placeholder="Вставьте заголовок новости",
        key="headline_llm",
    )
    body = st.text_area(
        "Текст новости:",
        height=250,
        placeholder="Вставьте основной текст новости",
        key="body_llm",
    )
    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        but1, but2 = st.columns(2)
        with but1:
            st.button(
                "Подобрать новость",
                help="Взять рандомную новость из тестового датасета",
                on_click=pick_news,
                use_container_width=True,
            )
        with but2:
            st.button(
                "Очистить поля",
                help="Привести поля в исходный вид",
                on_click=clear_fields,
                use_container_width=True,
            )

        check_button = st.button(
            "Проверить новость", type="primary", use_container_width=True
        )


# ── Предсказание ─────────────────────────────────────────────────────────────

if check_button:
    st.markdown("---")
    if not headline or not body:
        st.warning("Пожалуйста, заполните заголовок и текст новости")
    elif not os.path.exists(LLM_MODEL_DIR):
        st.error(
            "Модель не найдена. Сначала обучите модель, запустив ноутбук llm_detection.ipynb"
        )
    else:
        with st.spinner("Анализирую новость с помощью LLM..."):
            try:
                tokenizer_llm, model_llm = load_llm_model()
                combined_text = f"{headline} {body}"
                result = predict_llm(combined_text, tokenizer_llm, model_llm)

                st.markdown("### ruGPT-3 + LoRA")

                if result["label"] == 1:
                    st.success("**РЕАЛЬНАЯ НОВОСТЬ**")
                    st.markdown(
                        f"""
                        <div class='metric-container'>
                            <div class='metric-label'>ruGPT-3 + LoRA</div>
                            <div class='metric-label'>Уверенность</div>
                            <div class='metric-value'>{result['real_proba']*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("**ФЕЙКОВАЯ НОВОСТЬ**")
                    st.markdown(
                        f"""
                        <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                            <div class='metric-label'>ruGPT-3 + LoRA</div>
                            <div class='metric-label'>Уверенность</div>
                            <div class='metric-value' style='color: #b91c1c;'>{result['fake_proba']*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Детали
                with st.expander("Детали предсказания"):
                    st.markdown(
                        f"""<div style='color:#FFFFE0;'>
                        <b>Модель:</b> ai-forever/rugpt3small_based_on_gpt2 + LoRA<br>
                        <b>Метод:</b> Parameter-Efficient Fine-Tuning (PEFT)<br>
                        <b>P(реальная):</b> {result['real_proba']:.4f}<br>
                        <b>P(фейковая):</b> {result['fake_proba']:.4f}<br>
                        <b>Устройство:</b> {DEVICE}
                        </div>""",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

# ── Сравнение ────────────────────────────────────────────────────────────────

st.markdown("---")

if os.path.exists("assets/llm_comparison.png"):
    with st.expander("Сравнение всех моделей", expanded=False):
        st.image("assets/llm_comparison.png")

if os.path.exists("assets/llm_confusion_matrices.png"):
    with st.expander("Матрицы ошибок LLM-подходов", expanded=False):
        st.image("assets/llm_confusion_matrices.png")

# ── Навигация внизу ──────────────────────────────────────────────────────────

st.markdown("---")

but1, but2 = st.columns(2)
with but1:
    if st.button(
        "Перейти на главную страницу", type="secondary", use_container_width=True
    ):
        st.switch_page("app.py")
with but2:
    if st.button(
        "Сравнение классических моделей с RuBERT",
        type="secondary",
        use_container_width=True,
    ):
        st.switch_page("pages/russian_comparison.py")
