import streamlit as st
import json

with open("results/metrics/metrics_w2v_eng.json", "r", encoding="utf-8") as f:
    metrics_w2v_eng = json.load(f)
    
with open("results/metrics/metrics_tfidf_eng.json", "r", encoding="utf-8") as f:
    metrics_tfidf_eng = json.load(f)

st.set_page_config(page_title="Сравнение работы моделей", page_icon="🔍", layout="wide")

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

st.title("Сравнение работы моделей, обученных на разных языках", anchor=None)

st.markdown(
    """
### Как проводилось сравнение моделей?
- Из-за нехватки датасетов с фейковыми новостями, на которых можно было бы обучить модели, было принято решение использовать датасет только с реальными новостями. 
  При разработке использовался <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">датасет</a> с русскими новостями.
- При сравнении делался предикт с использованием всех разработанных моделей, после чего проверялось количество новостей, которые модель посчитала правдивыми. 
  Количество таких предиктов мы и будем считать определителем качества модели.
### Принцип сравнения
- Для сравнения использовались по 2 датасета на каждую модель. Причем, в одном датасете было 50 строк, а в другом - 10.
### Результаты
""",
    unsafe_allow_html=True
)
st.image('assets/Logistic Regression_tfidf_comparison.png')
st.image('assets/Naive Bayes_tfidf_comparison.png')
st.image('assets/Random Forest_tfidf_comparison.png')

st.markdown("---")

st.markdown(
    """
- Можем сделать вывод, что использование моделей, обученных на английском языке, дает менее точные результаты, несмотря на то, что тренировочный датасет
в 3 раза больше, чем у моделей на русском языке.
- Наилучший результат среди англоязычных моделей показал **Naive Bayes**
- Русскозычные модели показали большую точность, в некоторых случаях угадывая 100% предложенных новостей.
""",
    unsafe_allow_html=True
)

but1, but2 = st.columns(2)
with but1:
    if st.button("Перейти на страницу с использованием английского датасета", help="Перейти на предыдущую страницу",
    type="secondary", use_container_width=True):
        st.switch_page("pages/eng.py")

with but2:
    if st.button("Перейти на страницу с использованием русского датасета", help="Перейти на предыдущую страницу",
    type="secondary", use_container_width=True):
        st.switch_page("app.py")

