import streamlit as st
import pandas as pd

st.set_page_config(page_title="Сравнение русскоязычных моделей", page_icon="🔍", layout="wide")

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

    .streamlit-expanderHeader {
        color: #FFFFE0; font-weight: 600;
    }

    </style>
    """, unsafe_allow_html=True)

set_styles()

if st.button("←", help="Вернуться на главную страницу", type="secondary"):
    st.switch_page("app.py")

st.title("Сравнение русскоязычных моделей детекции фейков", anchor=None)

st.markdown(
    """
### Как проводилось сравнение?
- Все 6 моделей оценивались на одной и той же тестовой выборке из датасета `data/ready_dataset.csv`
  (10% данных, `random_state=42`, стратификация по метке — **441 пример**: 222 фейковых и 219 реальных новостей).
- Метрики: **Accuracy**, **F1-score**, **Precision**, **Recall** (класс 1 = реальная новость).

### Сравниваемые модели

| Модель | Векторизация |
|--------|-------------|
| Naive Bayes | TF-IDF |
| Random Forest | TF-IDF |
| Logistic Regression | TF-IDF |
| Random Forest | Word2Vec |
| Logistic Regression | Word2Vec |
| **RuBERT** | Transformer (DeepPavlov/rubert-base-cased) |

### Результаты
""",
    unsafe_allow_html=True
)

# Таблица метрик
try:
    df = pd.read_csv("assets/russian_models_comparison.csv")
    df.index = range(1, len(df) + 1)
    st.dataframe(
        df.style.highlight_max(subset=["Accuracy", "F1", "Precision", "Recall"], color="#2ecc7155"),
        use_container_width=True
    )
except Exception:
    st.warning("Файл assets/russian_models_comparison.csv не найден. Запусти russian_models_comparison.ipynb.")

st.markdown("---")

st.markdown("### График метрик")
st.image("assets/russian_models_comparison.png")

st.markdown("---")

st.markdown("### Матрицы ошибок")
st.image("assets/russian_models_confusion_matrices.png")

st.markdown("---")

st.markdown(
    """
### Выводы
- **RuBERT** показывает наилучший результат среди всех моделей.
- Среди классических моделей лидирует **Logistic Regression (TF-IDF)**.
- Модели на основе **Word2Vec** уступают TF-IDF-моделям: у Logistic Regression (W2V) самый низкий результат — Accuracy 89.80%.
- Transformer-архитектура (RuBERT) превосходит все классические подходы.
""",
    unsafe_allow_html=True
)

st.markdown("---")

but1, but2 = st.columns(2)
with but1:
    if st.button("Перейти на главную страницу", type="secondary", use_container_width=True):
        st.switch_page("app.py")

with but2:
    if st.button("Сравнение русских и английских моделей", type="secondary", use_container_width=True):
        st.switch_page("pages/comparsion.py")
