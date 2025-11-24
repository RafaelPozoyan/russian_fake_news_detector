import streamlit as st
import json

with open("results/metrics/metrics_w2v_eng.json", "r", encoding="utf-8") as f:
    metrics_w2v_eng = json.load(f)
    
with open("results/metrics/metrics_tfidf_eng.json", "r", encoding="utf-8") as f:
    metrics_tfidf_eng = json.load(f)

st.set_page_config(page_title="Обзор подходов (английский датасет)", page_icon="🔍", layout="wide")

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

st.title("Обзор подходов (английский датасет)", anchor=None)

st.markdown("### Основные отличия от русского датасета")

st.markdown("""
При работе с английским датасетом использовались следующие подходы и отличия от метода с русским датасетом:

#### 1. **Обработка входных данных**
- **Русский датасет**: Текст обрабатывается напрямую на русском языке
- **Английский датасет**: Входящий текст (независимо от языка) автоматически переводится на английский язык с помощью Google Translator перед обработкой. Это позволяет унифицировать обработку и использовать модели, обученные на англоязычных данных.

#### 2. **TF-IDF векторизация**
- **Русский датасет**: Параметры оптимизированы для русской морфологии
- **Английский датасет**: 
  - Используются расширенные параметры: `max_features=8000`, `ngram_range=(1, 2)` (униграммы и биграммы)
  - `min_df=3`, `max_df=0.85`, `sublinear_tf=True`
  - Обучение происходит на английском тексте, что дает более точные результаты для англоязычных данных

#### 3. **Word2Vec обучение**
- **Русский датасет**: Модель обучена на русских текстах
- **Английский датасет**: 
  - Модель обучена непосредственно на английском датасете из заголовков и текстов статей
  - Параметры: `vector_size=300`, `window=3`, `min_count=3`, `sg=1` (Skip-gram), `epochs=10`
  - Это позволяет лучше учитывать семантическую связь между словами в английском языке

#### 4. **Модели классификации**
- Используются те же типы моделей (Logistic Regression, Naive Bayes, Random Forest)
- Но обучены на английском датасете, что дает разные результаты по сравнению с русским
- Метрики показывают очень высокую точность (accuracy > 0.98) для всех моделей
""")

st.markdown("---")

st.info("Векторизация через TF-IDF (английский датасет)")

st.markdown("""
Для английского датасета использовалась векторизация через ***TF-IDF*** с расширенными параметрами. 
При переводе русского текста на английский, модель обрабатывает текст на языке, на котором была обучена, 
что позволяет получить более точные результаты. Метрики показывают высокую точность моделей на валидационном наборе данных.
""")

st.markdown("---")
st.markdown("При написании кода сравнил ***3 модели***:")
st.markdown("* Logistic Regression (val_acc: **{:.4f}**)".format(metrics_tfidf_eng["logistic_regression"]["val_acc"]))
st.markdown("* Naive Bayes (val_acc: **{:.4f}**)".format(metrics_tfidf_eng["naive_bayes"]["val_acc"]))
st.markdown("* Random Forest (val_acc: **{:.4f}**)".format(metrics_tfidf_eng["random_forest"]["val_acc"]))

st.markdown("""
Наибольший скор и на валидационном, и на тренировочном датасетах выдал ***Random Forest***, 
за ним следует Logistic Regression. Все модели показывают отличные результаты с accuracy выше 0.98.
""")

# График может быть создан аналогично, показывая сравнение моделей
st.image("assets/models_scores.png")
st.caption("График сравнения моделей TF-IDF. Для английского датасета Random Forest показывает наилучшие результаты.")

st.markdown("---")

st.info("Векторизация через Word2Vec (английский датасет)")

st.markdown("""
При использовании Word2Vec для английского датасета модель обучена непосредственно на английских текстах. 
Это позволяет лучше учитывать семантическую связь между заголовком новости и основным текстом на английском языке. 
Как и в случае с русским датасетом, каждый запрос проходит проверку на соответствие бизнес-правилам. 
При изменении заголовка новости на ложный, модель это понимает и выводится объяснение, почему при высокой вероятности 
правдивости новости, она все равно фейковая.
""")

st.markdown("---")
st.markdown("При написании кода сравнил ***2 модели***:")
st.markdown("* Logistic Regression (val_acc: **{:.4f}**)".format(metrics_w2v_eng["logisticregression"]["val_accuracy"]))
st.markdown("* Random Forest (val_acc: **{:.4f}**)".format(metrics_w2v_eng["randomforest"]["val_accuracy"]))
st.markdown("Наибольший скор и на валидационном, и на тренировочном датасетах выдала ***Logistic Regression***.")

# Используем существующий график Word2Vec
st.image("assets/models_scores_w2v.png")
st.caption("График сравнения моделей Word2Vec")

st.markdown("---")

st.markdown("### Сравнение метрик")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### TF-IDF метрики")
    st.markdown(f"**Logistic Regression**: {metrics_tfidf_eng['logistic_regression']['val_acc']:.4f}")
    st.markdown(f"**Naive Bayes**: {metrics_tfidf_eng['naive_bayes']['val_acc']:.4f}")
    st.markdown(f"**Random Forest**: {metrics_tfidf_eng['random_forest']['val_acc']:.4f}")

with col2:
    st.markdown("#### Word2Vec метрики")
    st.markdown(f"**Logistic Regression**: {metrics_w2v_eng['logisticregression']['val_accuracy']:.4f}")
    st.markdown(f"**Random Forest**: {metrics_w2v_eng['randomforest']['val_accuracy']:.4f}")

st.markdown("---")

st.markdown("### Отличия в параметрах обучения")

with st.expander("Параметры TF-IDF векторизатора"):
    st.markdown("""
    - **max_features**: 8000 (максимальное количество признаков)
    - **ngram_range**: (1, 2) (униграммы и биграммы)
    - **min_df**: 3 (минимальная частота документа)
    - **max_df**: 0.85 (максимальная частота документа)
    - **sublinear_tf**: True (логарифмическое масштабирование)
    """)

with st.expander("Параметры Word2Vec модели"):
    st.markdown("""
    - **vector_size**: 300 (размерность векторов)
    - **window**: 3 (размер окна контекста)
    - **min_count**: 3 (минимальная частота слова)
    - **sg**: 1 (Skip-gram алгоритм)
    - **epochs**: 10 (количество эпох обучения)
    - **workers**: 4 (количество потоков)
    """)

st.markdown("---")

but1, but2 = st.columns(2)

with but1:
    if st.button("Перейти на страницу с использованием английского датасета", help="Перейти на предыдущую страницу",
    type="secondary", use_container_width=True):
        st.switch_page("pages/eng.py")
with but2:
    if st.button("Перейти на страницу с использованием русского датасета", help="Вернуться на главную страницу",
    type="secondary", use_container_width=True):
        st.switch_page("app.py")

