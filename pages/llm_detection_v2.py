import streamlit as st
from PIL import Image
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

st.set_page_config(
    page_title="Advanced LLM-детектор", page_icon="⚡", layout="wide"
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
        background: linear-gradient(135deg, #27ae60 0%, #16a085 100%);
        color: #FFFFE0; border-radius: 7px; padding: 0.75rem 3rem; font-weight: 600;
        font-size: 1.1rem; border: none; box-shadow: 0 4px 12px rgba(39, 174, 96, 0.4);
        transition: all 0.3s ease; width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(39, 174, 96, 0.6);
        background: linear-gradient(135deg, #16a085 0%, #27ae60 100%);
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 7px; border: 2px solid #e5e7eb; font-size: 1rem; padding: 0.75rem; transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #27ae60; box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.1);
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
        background: linear-gradient(135deg, #d1f2eb 0%, #abebc6 100%); border-radius: 12px; padding: 0.1rem; text-align: center; margin: 0.1rem;
    }

    .metric-value {
        font-size: 2.5rem; font-weight: 700; color: #0b5345; margin-top: 0.1rem;
    }

    .metric-label {
        font-size: 1rem; color: #186a3b;
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

LLM_V2_DIR = "models/llm_v2"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Загрузка модели ──────────────────────────────────────────────────────────


@st.cache_resource
def load_qwen_model(model_dir=LLM_V2_DIR, base_name=BASE_MODEL_NAME, seed_idx=0):
    """Загружаем одну модель из ensemble."""
    from peft import PeftModel, prepare_model_for_kbit_training

    seed_path = os.path.join(model_dir, f"seed_{seed_idx}")
    if not os.path.exists(seed_path):
        st.warning(f"Модель seed_{seed_idx} не найдена")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(seed_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=2, quantization_config=bnb_config, device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(base_model, seed_path)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_ensemble_qwen(text, seed_indices=[0, 1, 2], max_length=512):
    """Ensemble предсказание из 3 моделей."""
    all_probs = []

    for seed_idx in seed_indices:
        tokenizer, model = load_qwen_model(seed_idx=seed_idx)
        if model is None:
            continue

        text = str(text).strip()

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(DEVICE if DEVICE == "cuda" else "cpu")
        attention_mask = encoded["attention_mask"].to(DEVICE if DEVICE == "cuda" else "cpu")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    if not all_probs:
        return None

    # Усредняем вероятности
    ensemble_probs = np.mean(all_probs, axis=0)
    pred = int(np.argmax(ensemble_probs))

    return {
        "label": pred,
        "label_name": "Правдивая" if pred == 1 else "Фейковая",
        "fake_proba": float(ensemble_probs[0]),
        "real_proba": float(ensemble_probs[1]),
    }


# ── Загрузка метрик ──────────────────────────────────────────────────────────

llm_v2_metrics = None
METRICS_PATH = "models/llm_v2/metrics.json"
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        llm_v2_metrics = json.load(f)

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
            <h3 style='color: #FFFFE0;'>💎 Advanced LLM v2</h3>
            <p>На этой странице используется <b>Qwen2.5</b> —
            современная, мультиязычная, instruction-tuned модель (0.5B).</p>
            <p><b>QLoRA</b> (4-bit) + <b>3x Ensemble</b> = высокая точность.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    llm_v2_status = all(
        os.path.exists(os.path.join(LLM_V2_DIR, f"seed_{i}")) for i in range(3)
    )
    if llm_v2_status:
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>✅ Статус LLM V2:</h4><p>Qwen2.5 + QLoRA (3x) готовы</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>⚠️ Статус LLM V2:</h4><p>Запусти llm_detection_2.ipynb</p></div>",
            unsafe_allow_html=True,
        )

    if llm_v2_metrics:
        st.markdown("---")
        st.markdown(
            "<div style='color:#FFFFE0;'><h4>📊 Результаты Ensemble на тесте:</h4></div>",
            unsafe_allow_html=True,
        )
        m = llm_v2_metrics.get("ensemble", {})
        metrics_list = [
            ("Accuracy", "accuracy"),
            ("F1", "f1"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ]
        for metric_name, metric_key in metrics_list:
            val = m.get(metric_key)
            if val is not None:
                color = "#27ae60" if val >= 0.97 else "#f39c12"
                st.markdown(
                    f"<div style='color:{color};'><b>{metric_name}:</b> {val:.4f}</div>",
                    unsafe_allow_html=True,
                )


# ── Навигация ────────────────────────────────────────────────────────────────

if st.button("←", help="Вернуться на главную", type="secondary"):
    st.switch_page("app.py")


# ── Главный контент ──────────────────────────────────────────────────────────

st.title("⚡ Advanced LLM Detector v2")
st.info(
    "Qwen2.5-0.5B-Instruct + QLoRA + 3x Ensemble — современный подход с **максимальной точностью**"
)

with st.expander("О технологиях", expanded=False):
    st.markdown(
        """<div style='color:#FFFFE0;'>
    <h4>🔹 Qwen2.5-0.5B-Instruct</h4>
    <ul>
    <li>Современная архитектура (2024+)</li>
    <li>Мультиязычная поддержка</li>
    <li>Instruction-following для лучшего понимания</li>
    <li>500M параметров = баланс качество/скорость</li>
    </ul>

    <h4>🔹 QLoRA (Quantized LoRA)</h4>
    <ul>
    <li><b>4-bit квантизация</b> обучаемых весов</li>
    <li>50% экономия GPU памяти</li>
    <li>Без потери качества обучения</li>
    <li>Адаптер = только несколько МБ</li>
    </ul>

    <h4>🔹 3x Ensemble</h4>
    <ul>
    <li>Обучение 3 моделей с разными seeds</li>
    <li>Усреднение предсказаний</li>
    <li>Снижение дисперсии ошибок</li>
    <li>+1-2% точности за счет ансамбля</li>
    </ul>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if "headline_v2" not in st.session_state:
    st.session_state.headline_v2 = ""
if "body_v2" not in st.session_state:
    st.session_state.body_v2 = ""


def pick_news():
    idx = random.randint(0, len(test_data) - 1)
    st.session_state.headline_v2 = test_data.loc[idx, "Headline1"]
    st.session_state.body_v2 = test_data.loc[idx, "articleBody"]


def clear_fields():
    st.session_state.headline_v2 = ""
    st.session_state.body_v2 = ""


with st.container():
    headline = st.text_input(
        "Заголовок новости:",
        placeholder="Вставьте заголовок новости",
        key="headline_v2",
    )
    body = st.text_area(
        "Текст новости:",
        height=250,
        placeholder="Вставьте основной текст новости",
        key="body_v2",
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
    elif not all(
        os.path.exists(os.path.join(LLM_V2_DIR, f"seed_{i}")) for i in range(3)
    ):
        st.error(
            "Модель не найдена. Сначала обучите модель, запустив ноутбук llm_detection_2.ipynb"
        )
    else:
        with st.spinner("Анализирую новость с помощью Advanced LLM Ensemble..."):
            try:
                combined_text = f"{headline} {body}"
                result = predict_ensemble_qwen(combined_text)

                if result is None:
                    st.error("Ошибка при загрузке модели")
                else:
                    st.markdown("### Qwen2.5 + QLoRA + 3x Ensemble")

                    if result["label"] == 1:
                        st.success("**✅ РЕАЛЬНАЯ НОВОСТЬ**")
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Qwen2.5 (3x Ensemble)</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-value'>{result['real_proba']*100:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("**❌ ФЕЙКОВАЯ НОВОСТЬ**")
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fadbd8 0%, #f5b7b1 100%);'>
                                <div class='metric-label'>Qwen2.5 (3x Ensemble)</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-value' style='color: #78281f;'>{result['fake_proba']*100:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Детали
                    with st.expander("Детали предсказания"):
                        st.markdown(
                            f"""<div style='color:#FFFFE0;'>
                            <b>Модель:</b> Qwen/Qwen2.5-0.5B-Instruct<br>
                            <b>Метод:</b> QLoRA (4-bit quantization) + 3x Ensemble<br>
                            <b>P(реальная):</b> {result['real_proba']:.4f}<br>
                            <b>P(фейковая):</b> {result['fake_proba']:.4f}<br>
                            <b>Ensemble:</b> Усреднение 3 моделей<br>
                            <b>Устройство:</b> {DEVICE}
                            </div>""",
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

# ── Сравнение ────────────────────────────────────────────────────────────────

st.markdown("---")

if os.path.exists("assets/complete_models_comparison.png"):
    with st.expander(
        "📊 ПОЛНОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ (Classical vs RuBERT vs LLM v1 vs LLM v2)",
        expanded=False,
    ):
        st.image("assets/complete_models_comparison.png")

if os.path.exists(os.path.join(LLM_V2_DIR, "confusion_matrix.png")):
    with st.expander("📈 Матрица ошибок Qwen2.5 Ensemble", expanded=False):
        st.image(os.path.join(LLM_V2_DIR, "confusion_matrix.png"))

# ── Навигация внизу ──────────────────────────────────────────────────────────

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("На главную", type="secondary", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button(
        "LLM v1 (ruGPT-3)", type="secondary", use_container_width=True
    ):
        st.switch_page("pages/llm_detection.py")
with col3:
    if st.button(
        "RuBERT сравнение", type="secondary", use_container_width=True
    ):
        st.switch_page("pages/russian_comparison.py")
