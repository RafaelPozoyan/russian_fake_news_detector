import streamlit as st
from PIL import Image
from gensim.models import KeyedVectors
from deep_translator import GoogleTranslator

import os
import re
import pickle
import numpy as np
import nltk
import json
import random
import pandas as pd


st.set_page_config(page_title="Детектор с английским датасетом", page_icon="🔍", layout="wide")

def set_styles():
    st.markdown("""
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
    
    .stImage.default-img img {
        border-radius: 0;
    }
    
    .streamlit-expanderHeader {
        color: #FFFFE0; font-weight: 600;
    }
    
    </style>
    """, unsafe_allow_html=True)

set_styles()

if st.button("←", help="Вернуться на главную страницу", type="secondary"):
    st.switch_page("app.py")

@st.cache_data
def load_stopwords():
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    return set(stopwords.words('russian'))

def preprocess_text(text, stopwords_list):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'http\S+|www\S+|https\S+', '', t)
    t = re.sub(r'\S+@\S+', '', t)
    t = re.sub(r'<.*?>', '', t)
    t = re.sub(r'[^а-яёa-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    toks = [w for w in t.split() if w not in stopwords_list and len(w) > 2]
    return " ".join(toks)

STOPWORDS = load_stopwords()

## Перевод текста ##
def translate_to_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text 


# @st.cache_resource
# def load_artifacts():
#     clf = None; kv = None; metrics = None

#     if os.path.exists("models/fake_news_w2v_lr.pkl"):
#         with open("models/fake_news_w2v_lr.pkl", "rb") as f:
#             clf = pickle.load(f)

#     if os.path.exists("models/w2v_vectors.kv"):
#         kv = KeyedVectors.load("models/w2v_vectors.kv")

#     if os.path.exists("models/metrics.pkl"):
#         with open("models/metrics.pkl", "rb") as f:
#             metrics = pickle.load(f)
    
#     return clf, kv, metrics


def load_model():
    """Загрузка обученных моделей и векторизатора"""
    try:
        model_randfor_tf = pickle.load(open('models/random_forest_model_tf_eng.pkl', 'rb'))
        model_naibayes_tf = pickle.load(open('models/naive_bayes_model_tf_eng.pkl', 'rb'))
        model_logreg_tf = pickle.load(open('models/logistic_regression_model_tf_eng.pkl', 'rb'))
        vectorizer_tf = pickle.load(open('models/tfidf_vectorizer_tf_eng.pkl', 'rb'))
        return model_randfor_tf, model_naibayes_tf, model_logreg_tf, vectorizer_tf, True
    except FileNotFoundError:
        return None, None, None, None, False

with st.sidebar:
    logo = Image.open('assets/logo.png')
    st.image(logo, output_format="round-logo")

    st.markdown("---")
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h3 style='color: #FFFFE0;'>О проекте</h3>
            <p>Система автоматической детекции фейковых новостей. При введении заголовка и основного текста статьи, сайт проверит,
            является ли инфоповод подлинным.</p>
            </p>При решении задачи используются TF-IDF и Word2Vec. Материал проверяется на согласованность заголовка и основного текста.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h3 style='color: #FFFFE0;'>Способ решения проблемы</h3>
            <p>На данной странице модели обучены на датасете на английском языке.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.title("Предсказание с использованием английского датасета", anchor=None)

with st.expander("Как пользоваться?", expanded=False):
    st.markdown("""<div style='color:#FFFFE0;'>
    1. Введите заголовок новости в первое поле
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style='color:#FFFFE0;'>
    2. Вставьте текст новости во второе поле
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style='color:#FFFFE0;'>
    3. Нажмите кнопку "Проверить новость"
    </div>""", unsafe_allow_html=True)

test_bodies = pd.read_csv('./data/test_bodies.csv')
test_stances = pd.read_csv('./data/test_stances_unlebeledb.csv') 
test_df = test_stances.merge(test_bodies, on='Body ID', how='left')
with open("results/metrics/metrics_tfidf_eng.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

def pick_news():
    idx = random.randint(0, len(test_df)-1)
    st.session_state.headline = test_df.loc[idx, "Headline1"]
    st.session_state.body = test_df.loc[idx, "articleBody"]

def clear_fields():
    st.session_state.headline = ""
    st.session_state.body = ""

def clear_fields():
    st.session_state.headline = ""
    st.session_state.body = ""

with st.container():
    headline = st.text_input(
        'Заголовок новости:',
        placeholder='Вставьте заголовок новости',
        key='headline'
    )
    body = st.text_area(
        'Текст новости:',
        height=250,
        placeholder='Вставьте основной текст новости',
        key='body'
    )
    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:

        but1, but2 = st.columns(2)

        with but1:
            st.button('Подобрать новость', help="Взять рандомную новость из тестового датасета", on_click=pick_news, use_container_width=True)
        with but2:
            st.button('Очистить поля', help="Привести поля в исходный вид" ,on_click=clear_fields, use_container_width=True)

        check_button = st.button('Проверить новость', type="primary", use_container_width=True)

model_randfor_tf, model_naibayes_tf, model_logreg_tf, vectorizer_tf, model_loaded = load_model()
stopwords_list = load_stopwords()


if check_button:
    st.markdown("---")
    if not headline or not body:
        st.warning('⚠️ Пожалуйста, заполните заголовок и текст новости')
    else:
        with st.spinner('🔄 Перевожу новость на нужный язык...'):

                # =================================
                # Вывод результатов TF-IDF
                # =================================

                headline = translate_to_en(headline)
                body = translate_to_en(body)

                headline_clean = preprocess_text(headline, stopwords_list)
                body_clean = preprocess_text(body, stopwords_list)
                
                # Комбинируем
                body_words = body_clean.split()
                combined_text = f"{headline_clean} {' '.join(body_words)}"

                text_vec = vectorizer_tf.transform([combined_text])
                
                # предикты на random forest
                prediction_rf = model_randfor_tf.predict(text_vec)[0]
                probabilities_rf = model_randfor_tf.predict_proba(text_vec)[0]

                # предикты на naive bayes
                prediction_nb = model_naibayes_tf.predict(text_vec)[0]
                probabilities_nb = model_naibayes_tf.predict_proba(text_vec)[0]  

                # предикты на logistic regression
                prediction_lr = model_logreg_tf.predict(text_vec)[0]
                probabilities_lr = model_logreg_tf.predict_proba(text_vec)[0]  


                with st.expander("Как выглядит переведенный текст?", expanded=False):
                    st.markdown("""<div style='color:#FFFFE0;'>
                    Переведенный заголовок
                    </div>""", unsafe_allow_html=True)
                    headline

                    st.markdown("""<div style='color:#FFFFE0;'>
                    Переведенный текст новости
                    </div>""", unsafe_allow_html=True)
                    body


                st.markdown('---')

                st.markdown("### TF-IDF")

                col1, col2, col3 = st.columns(3)

                results = [
                    {
                        "col": col3,
                        "name": "Random Forest",
                        "prediction": prediction_rf,
                        "probabilities": probabilities_rf,
                        "metric_key": "random_forest",
                        "metrics": metrics,
                    },
                    {
                        "col": col2,
                        "name": "Naive Bayes",
                        "prediction": prediction_nb,
                        "probabilities": probabilities_nb,
                        "metric_key": "naive_bayes",
                        "metrics": metrics,
                    },
                    {
                        "col": col1,
                        "name": "Logistic Regression",
                        "prediction": prediction_lr,
                        "probabilities": probabilities_lr,
                        "metric_key": "logistic_regression",
                        "metrics": metrics,
                    },
                ]

                for res in results:
                    with res["col"]:
                        if res["prediction"] == 1:
                            confidence = res["probabilities"][1] * 100
                            st.success('✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                            st.markdown(
                                f"""
                                <div class='metric-container'>
                                    <div class='metric-label'>{res['name']}</div>
                                    <div class='metric-label'>Уверенность</div>
                                    <div class='metric-value'>{confidence:.1f}%</div>
                                    <div class='metric-label'><strong>Accuracy:</strong> {res['metrics'][res["metric_key"]]["val_acc"]:.3f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )         
                        else:
                            confidence = res["probabilities"][0] * 100
                            st.error('❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                            st.markdown(
                                f"""
                                <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                    <div class='metric-label'>{res['name']}</div>
                                    <div class='metric-label'>Уверенность</div>
                                    <div class='metric-value' style='color: #b91c1c;'>{confidence:.1f}%</div>
                                    <div class='metric-label'><strong>Accuracy:</strong> {res['metrics'][res["metric_key"]]["val_acc"]:.3f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                # st.markdown("---")
                # st.markdown("### Усредненный ответ")
                # sum_1 = final_label_rf + final_label_rf + prediction_lr + prediction_nb + prediction_rf
                # if sum_1 >= 3:
                #     st.success(f'✅ **РЕАЛЬНАЯ НОВОСТЬ**')

                #     st.markdown("Количество предиктов на то, что новость реальная больше.")
                # else:
                #     st.error(f'❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                #     st.markdown("Количество предиктов на то, что новость фейковая больше.")
                


            # except Exception as e:
            #     st.error(f'❌ Ошибка: {str(e)}')

st.markdown("---")

but1, but2 = st.columns(2)

with but1:
    if st.button("Обзор подходов", help="Открыть страницу с обзором подходов: использованных моделей и векторизаторов", 
                 type="secondary", use_container_width=True):
        st.switch_page("pages/info.py")
with but2:
    if st.button("Вернуться на главную страницу", help="Вернуться на страницу с использованием русского датасета", 
                 type="secondary", use_container_width=True):
        st.switch_page("app.py")
