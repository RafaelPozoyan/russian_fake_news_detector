import os
import re
import pickle
import random
import json
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors
import nltk
from utils.style import (
    inject_css,
    sidebar_nav,
    render_prediction,
    render_metric_row,
    back_to_main,
)

st.set_page_config(page_title="Английский детектор", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

# Данные и стоп-слова

eng_df = pd.read_csv("./data/eng_data")


@st.cache_data
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


STOPWORDS = load_stopwords()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"\S+@\S+", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)


# Модели


@st.cache_resource
def load_w2v_eng():
    kv = (
        KeyedVectors.load("models/w2v_vectors_eng.kv")
        if os.path.exists("models/w2v_vectors_eng.kv")
        else None
    )
    lr = (
        pickle.load(open("models/logisticregression_model_w2v_eng.pkl", "rb"))
        if os.path.exists("models/logisticregression_model_w2v_eng.pkl")
        else None
    )
    rf = (
        pickle.load(open("models/randomforest_model_w2v_eng.pkl", "rb"))
        if os.path.exists("models/randomforest_model_w2v_eng.pkl")
        else None
    )
    return kv, lr, rf


@st.cache_resource
def load_tfidf_eng():
    try:
        rf = pickle.load(open("models/random_forest_model_tf_eng.pkl", "rb"))
        nb = pickle.load(open("models/naive_bayes_model_tf_eng.pkl", "rb"))
        lr = pickle.load(open("models/logistic_regression_model_tf_eng.pkl", "rb"))
        vec = pickle.load(open("models/tfidf_vectorizer_tf_eng.pkl", "rb"))
        return rf, nb, lr, vec
    except FileNotFoundError:
        return None, None, None, None


w2v_kv, w2v_lr, w2v_rf = load_w2v_eng()
tfidf_rf, tfidf_nb, tfidf_lr, tfidf_vec = load_tfidf_eng()

metrics_w2v = {}
metrics_tfidf = {}
if os.path.exists("results/metrics/metrics_w2v_eng.json"):
    with open("results/metrics/metrics_w2v_eng.json", encoding="utf-8") as f:
        metrics_w2v = json.load(f)
if os.path.exists("results/metrics/metrics_tfidf_eng.json"):
    with open("results/metrics/metrics_tfidf_eng.json", encoding="utf-8") as f:
        metrics_tfidf = json.load(f)


def doc_vector(tokens, kv_model):
    vecs = [kv_model[w] for w in tokens if w in kv_model]
    return (
        np.vstack(vecs).mean(axis=0)
        if vecs
        else np.zeros(kv_model.vector_size, dtype=np.float32)
    )


def predict_w2v_eng(headline, body, model):
    h_clean = preprocess_text(headline)
    b_clean = preprocess_text(body)
    if len(h_clean) < 2 or len(b_clean) < 5:
        return None, None
    htoks, btoks = h_clean.split()[:150], b_clean.split()[:150]
    h_vec, b_vec = doc_vector(htoks, w2v_kv), doc_vector(btoks, w2v_kv)
    cos_sim = float(
        np.dot(h_vec, b_vec) / max(np.linalg.norm(h_vec) * np.linalg.norm(b_vec), 1e-8)
    )
    feats = np.hstack([h_vec, b_vec, [cos_sim]]).reshape(1, -1)
    prob = model.predict_proba(feats)[0]
    pred = int(model.predict(feats)[0])
    return pred, float(prob[pred])


def predict_tfidf_eng(headline, body, model):
    h_clean = preprocess_text(headline)
    b_clean = preprocess_text(body)
    combined = f"{h_clean} {b_clean}"
    X = tfidf_vec.transform([combined])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return int(pred), float(prob[pred])


# ── Интерфейс ────────────────────────────────────────────────────────────────

st.markdown("# Детектор фейковых новостей (английский)")
st.markdown(
    "Те же классические модели, обученные на английском датасете. "
    "Введите заголовок и текст статьи на английском языке для проверки."
)

use_translate = False
try:
    from deep_translator import GoogleTranslator

    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

if HAS_TRANSLATOR:
    use_translate = st.checkbox(
        "Перевести с русского на английский перед проверкой", value=False
    )

if "eng_headline" not in st.session_state:
    st.session_state.eng_headline = ""
if "eng_body" not in st.session_state:
    st.session_state.eng_body = ""


def pick_eng():
    idx = random.randint(0, len(eng_df) - 1)
    st.session_state.eng_headline = str(eng_df.loc[idx, "title"])
    st.session_state.eng_body = str(eng_df.loc[idx, "text"])


def clear_eng():
    st.session_state.eng_headline = ""
    st.session_state.eng_body = ""


headline = st.text_input(
    "Заголовок", key="eng_headline", placeholder="Введите заголовок на английском"
)
body_text = st.text_area(
    "Текст статьи",
    key="eng_body",
    height=200,
    placeholder="Введите текст статьи на английском",
)

cb1, cb2, cb3 = st.columns([1, 1, 2])
with cb1:
    st.button(
        "Случайная новость", on_click=pick_eng, key="eng_pick", use_container_width=True
    )
with cb2:
    st.button("Очистить", on_click=clear_eng, key="eng_clear", use_container_width=True)
with cb3:
    check = st.button(
        "Проверить", type="primary", key="eng_check", use_container_width=True
    )

if check:
    if not headline.strip() or not body_text.strip():
        st.warning("Заполните оба поля.")
        st.stop()

    h_input = headline
    b_input = body_text

    if use_translate and HAS_TRANSLATOR:
        with st.spinner("Перевод..."):
            h_input = GoogleTranslator(source="auto", target="en").translate(h_input)
            b_input = GoogleTranslator(source="auto", target="en").translate(b_input)
            st.caption(f"Переведённый заголовок: {h_input[:100]}...")

    all_votes = []

    # Word2Vec
    if w2v_kv and w2v_lr and w2v_rf:
        st.markdown("#### Word2Vec")
        c1, c2 = st.columns(2)
        for col, model, name in [
            (c1, w2v_lr, "Logistic Regression"),
            (c2, w2v_rf, "Random Forest"),
        ]:
            with col:
                pred, conf = predict_w2v_eng(h_input, b_input, model)
                if pred is not None:
                    render_prediction(pred, conf, name)
                    all_votes.append(pred)

    # TF-IDF
    if tfidf_vec is not None:
        st.markdown("#### TF-IDF")
        c1, c2, c3 = st.columns(3)
        for col, model, name in [
            (c1, tfidf_lr, "Logistic Regression"),
            (c2, tfidf_nb, "Naive Bayes"),
            (c3, tfidf_rf, "Random Forest"),
        ]:
            with col:
                pred, conf = predict_tfidf_eng(h_input, b_input, model)
                render_prediction(pred, conf, name)
                all_votes.append(pred)

    # Итог
    if all_votes:
        st.markdown("---")
        real_count = sum(all_votes)
        total = len(all_votes)
        fake_count = total - real_count
        if real_count > fake_count:
            st.success(
                f"**Достоверная новость** — {real_count} из {total} моделей считают новость настоящей"
            )
        elif fake_count > real_count:
            st.error(
                f"**Фейк** — {fake_count} из {total} моделей считают новость фейковой"
            )
        else:
            st.warning(
                f"**Мнения разделились** — {real_count} за, {fake_count} против из {total}"
            )
