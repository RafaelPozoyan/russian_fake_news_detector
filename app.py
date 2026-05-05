import os
import re
import json
import pickle
import random
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

from utils.style import inject_css, sidebar_nav, render_prediction

load_dotenv()

st.set_page_config(page_title="Детектор фейковых новостей", layout="wide")
inject_css()
sidebar_nav()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ═════════════════════════════════════════════════════════════════════════════

test_bodies = pd.read_csv("./data/test_bodies.csv")
test_stances = pd.read_csv("./data/test_stances_unlebeledb.csv")
test_df = test_stances.merge(test_bodies, on="Body ID", how="left")


@st.cache_data
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    return set(stopwords.words("russian"))


STOPWORDS = load_stopwords()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"\S+@\S+", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^а-яёa-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)


# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ: Word2Vec
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_w2v():
    kv = (
        KeyedVectors.load("models/w2v_vectors.kv")
        if os.path.exists("models/w2v_vectors.kv")
        else None
    )
    lr = (
        pickle.load(open("models/logisticregression_model.pkl", "rb"))
        if os.path.exists("models/logisticregression_model.pkl")
        else None
    )
    rf = (
        pickle.load(open("models/randomforest_model.pkl", "rb"))
        if os.path.exists("models/randomforest_model.pkl")
        else None
    )
    return kv, lr, rf


w2v_kv, w2v_lr, w2v_rf = load_w2v()


def doc_vector(tokens, kv_model):
    vecs = [kv_model[w] for w in tokens if w in kv_model]
    return (
        np.vstack(vecs).mean(axis=0)
        if vecs
        else np.zeros(kv_model.vector_size, dtype=np.float32)
    )


def predict_w2v(headline, body, model):
    h_clean = preprocess_text(headline)
    b_clean = preprocess_text(body)
    if len(h_clean) < 2 or len(b_clean) < 5:
        return None, None
    htoks, btoks = h_clean.split()[:150], b_clean.split()[:150]
    h_vec, b_vec = doc_vector(htoks, w2v_kv), doc_vector(btoks, w2v_kv)
    cos_sim = float(
        np.dot(h_vec, b_vec) / max(np.linalg.norm(h_vec) * np.linalg.norm(b_vec), 1e-8)
    )
    jacc = len(set(htoks) & set(btoks)) / max(1, len(set(htoks) | set(btoks)))
    ovr = len(set(htoks) & set(btoks)) / max(1, len(set(htoks)))
    l2 = float(np.linalg.norm(h_vec - b_vec))
    feats = np.hstack(
        [h_vec, b_vec, np.abs(h_vec - b_vec), h_vec * b_vec, [cos_sim, jacc, ovr, l2]]
    ).reshape(1, -1)
    prob = model.predict_proba(feats)[0]
    if cos_sim < 0.10 or ovr <= 0.00:
        return 0, max(prob[0], 0.7)
    if (
        prob[1] >= 0.55
        and cos_sim >= 0.20
        and jacc >= 0.005
        and ovr >= 0.10
        and l2 <= 14.0
    ):
        return 1, prob[1]
    return 0, max(prob[0], 0.5)


# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ: TF-IDF
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_tfidf():
    try:
        rf = pickle.load(open("models/random_forest_model_tf.pkl", "rb"))
        nb = pickle.load(open("models/naive_bayes_model_tf.pkl", "rb"))
        lr = pickle.load(open("models/logistic_regression_model_tf.pkl", "rb"))
        vec = pickle.load(open("models/tfidf_vectorizer_tf.pkl", "rb"))
        return rf, nb, lr, vec
    except FileNotFoundError:
        return None, None, None, None


tfidf_rf, tfidf_nb, tfidf_lr, tfidf_vec = load_tfidf()


def predict_tfidf(headline, body, model):
    h_clean = preprocess_text(headline)
    b_clean = preprocess_text(body)
    X = tfidf_vec.transform([f"{h_clean} {b_clean}"])
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]
    return pred, float(prob[pred])


# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ: RuBERT
# ═════════════════════════════════════════════════════════════════════════════

RUBERT_DIR = "models/rubert/best_model"


@st.cache_resource
def load_rubert():
    if not os.path.exists(RUBERT_DIR):
        return None, None
    tok = AutoTokenizer.from_pretrained(RUBERT_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(RUBERT_DIR).to(DEVICE)
    mdl.eval()
    return tok, mdl


rubert_tok, rubert_mdl = load_rubert()


@torch.no_grad()
def predict_rubert(headline, body):
    enc = rubert_tok(
        f"{headline} {body}",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    out = rubert_mdl(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
    )
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ: ruGPT-3 + LoRA (тонко настроенная конфигурация)
# ═════════════════════════════════════════════════════════════════════════════

RUGPT_LORA_DIR = "models/llm_v3_tuned/lora_adapter"
GPT_BASE = "ai-forever/rugpt3small_based_on_gpt2"


@st.cache_resource
def load_rugpt_lora():
    if not os.path.exists(RUGPT_LORA_DIR):
        return None, None
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(RUGPT_LORA_DIR)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(GPT_BASE, num_labels=2)
    base.config.pad_token_id = tok.pad_token_id
    model = (
        PeftModel.from_pretrained(base, RUGPT_LORA_DIR).merge_and_unload().to(DEVICE)
    )
    model.eval()
    return tok, model


@torch.no_grad()
def predict_rugpt_lora(headline, body):
    tok, mdl = load_rugpt_lora()
    if tok is None:
        return None, None
    enc = tok(
        f"{headline} | {body}",
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    out = mdl(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
    )
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


# ═════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ: DeepSeek API
# ═════════════════════════════════════════════════════════════════════════════

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DS_SYSTEM_PROMPT = (
    "You classify Russian-language news as REAL (label=1) or FAKE (label=0). "
    "Read the headline and body, identify the topical claim, and decide whether it is "
    "consistent with documented events (real) or has signs of fabrication, conspiracy framing, "
    "or implausible specifics (fake). "
    'Output JSON only: {"reasoning": "1-2 sentences", "label": 0 or 1, "confidence": "low|medium|high"}.'
)


@st.cache_resource
def load_deepseek_client():
    if not DEEPSEEK_API_KEY:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    except Exception:
        return None


def parse_deepseek_response(content: str):
    content = content.strip()
    matches = re.findall(r"\{.*?\}", content, re.DOTALL)
    if matches:
        for m in reversed(matches):
            try:
                data = json.loads(m)
                for k in data:
                    if k.lower() in ("label", "is_true", "is_real", "class", "result"):
                        raw = data[k]
                        if isinstance(raw, bool):
                            return int(raw), float(
                                _conf_to_num(data.get("confidence", "medium"))
                            )
                        if isinstance(raw, (int, float)):
                            return int(bool(int(raw))), float(
                                _conf_to_num(data.get("confidence", "medium"))
                            )
                        s = str(raw).strip().lower()
                        label = 1 if s in ("1", "true", "real", "правда") else 0
                        return label, float(
                            _conf_to_num(data.get("confidence", "medium"))
                        )
            except Exception:
                continue
    return None, None


def _conf_to_num(conf):
    if isinstance(conf, (int, float)):
        return float(conf)
    s = str(conf).strip().lower()
    return {"high": 0.95, "medium": 0.8, "low": 0.6}.get(s, 0.8)


def predict_deepseek(headline, body):
    client = load_deepseek_client()
    if client is None:
        return None, None
    text = f"{headline}\n\n{body}"[:1500]
    r = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": DS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"TEXT:\n{text}\n\nClassify (return JSON):",
            },
        ],
        max_tokens=200,
        temperature=0.0,
        stream=False,
    )
    return parse_deepseek_response(r.choices[0].message.content)


# ═════════════════════════════════════════════════════════════════════════════
# ИНТЕРФЕЙС
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("# Детектор фейковых новостей")
st.markdown(
    "Введите заголовок и текст новости — система проверит их через **все доступные модели** и покажет результат."
)

if "headline" not in st.session_state:
    st.session_state.headline = ""
if "body" not in st.session_state:
    st.session_state.body = ""


def pick_random():
    idx = random.randint(0, len(test_df) - 1)
    st.session_state.headline = str(test_df.loc[idx, "Headline1"])
    st.session_state.body = str(test_df.loc[idx, "articleBody"])


def clear_fields():
    st.session_state.headline = ""
    st.session_state.body = ""


headline = st.text_input(
    "Заголовок", placeholder="Введите заголовок новости", key="headline"
)
body = st.text_area(
    "Текст статьи", height=180, placeholder="Введите текст статьи", key="body"
)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.button("Случайная новость", on_click=pick_random, use_container_width=True)
with c2:
    st.button("Очистить", on_click=clear_fields, use_container_width=True)
with c3:
    check = st.button("Проверить", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТЫ
# ═════════════════════════════════════════════════════════════════════════════

if check:
    if not headline.strip() or not body.strip():
        st.warning("Заполните оба поля.")
        st.stop()

    st.markdown("---")
    all_results = []

    # Word2Vec
    if w2v_kv and w2v_lr and w2v_rf:
        for model, name in [(w2v_lr, "LR + Word2Vec"), (w2v_rf, "RF + Word2Vec")]:
            pred, conf = predict_w2v(headline, body, model)
            if pred is not None:
                all_results.append((name, "Классические", pred, conf))

    # TF-IDF
    if tfidf_vec is not None:
        for model, name in [
            (tfidf_lr, "LR + TF-IDF"),
            (tfidf_nb, "NB + TF-IDF"),
            (tfidf_rf, "RF + TF-IDF"),
        ]:
            pred, conf = predict_tfidf(headline, body, model)
            all_results.append((name, "Классические", pred, conf))

    # RuBERT
    if rubert_tok is not None:
        pred, conf = predict_rubert(headline, body)
        all_results.append(("RuBERT", "Трансформер", pred, conf))

    # ruGPT-3 + LoRA
    pred, conf = predict_rugpt_lora(headline, body)
    if pred is not None:
        all_results.append(("ruGPT-3 + LoRA", "LLM (локальная)", pred, conf))

    # DeepSeek API
    with st.spinner("Запрос к DeepSeek API..."):
        pred, conf = predict_deepseek(headline, body)
    if pred is not None:
        all_results.append(("DeepSeek (API)", "LLM (API)", pred, conf))

    # Вывод результатов
    if not all_results:
        st.error("Ни одна модель не доступна.")
        st.stop()

    categories = {}
    for name, cat, pred, conf in all_results:
        categories.setdefault(cat, []).append((name, pred, conf))

    for cat_name, items in categories.items():
        st.markdown(f"#### {cat_name}")
        if len(items) == 1:
            render_prediction(items[0][1], items[0][2], items[0][0])
        elif len(items) <= 3:
            cols = st.columns(len(items))
            for i, (name, pred, conf) in enumerate(items):
                with cols[i]:
                    render_prediction(pred, conf, name)
        else:
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    render_prediction(items[i][1], items[i][2], items[i][0])
            for name, pred, conf in items[3:]:
                render_prediction(pred, conf, name)

    # Итоговый вердикт
    st.markdown("---")
    total = len(all_results)
    real_count = sum(1 for _, _, p, _ in all_results if p == 1)
    fake_count = total - real_count

    st.markdown("#### Итоговый вердикт")
    if real_count > fake_count:
        st.success(
            f"**Достоверная новость** — {real_count} из {total} моделей считают новость настоящей"
        )
    elif fake_count > real_count:
        st.error(f"**Фейк** — {fake_count} из {total} моделей считают новость фейковой")
    else:
        st.warning(
            f"**Мнения разделились** — {real_count} за, {fake_count} против из {total}"
        )

    with st.expander("Подробная сводка"):
        df = pd.DataFrame(
            [
                {
                    "Модель": name,
                    "Категория": cat,
                    "Вердикт": "Реальная" if pred == 1 else "Фейк",
                    "Уверенность": f"{conf:.1%}",
                }
                for name, cat, pred, conf in all_results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# ССЫЛКИ НА ПОДХОДЫ
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("#### Подробнее о подходах")
st.markdown(
    "На страницах ниже описана теоретическая часть: как работает каждый метод, "
    "на чём обучались модели и какие результаты они показали."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Классические модели", use_container_width=True):
        st.switch_page("pages/1_classical.py")
with c2:
    if st.button("RuBERT", use_container_width=True):
        st.switch_page("pages/2_rubert.py")
with c3:
    if st.button("ruGPT-3 + LoRA", use_container_width=True):
        st.switch_page("pages/3_rugpt_lora.py")
with c4:
    if st.button("DeepSeek API", use_container_width=True):
        st.switch_page("pages/4_deepseek.py")
