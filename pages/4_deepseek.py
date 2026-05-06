import os
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from utils.style import (
    inject_css,
    sidebar_nav,
    render_metric_row,
    back_to_main,
    render_comparsion_section,
)

st.set_page_config(page_title="DeepSeek API", layout="wide")
inject_css()
sidebar_nav()

back_to_main()


cache_path = "models/deepseek/predictions_441_v5.csv"
fallback_path = "models/deepseek/predictions_441_v4.csv"
metrics = None
cm = None
n_total = 0

if os.path.exists(cache_path):
    df = pd.read_csv(cache_path)
elif os.path.exists(fallback_path):
    df = pd.read_csv(fallback_path)
else:
    df = None

if df is not None and {"true_label", "pred"}.issubset(df.columns):
    y_true, y_pred = df["true_label"].values, df["pred"].values
    n_total = len(df)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }
    cm = confusion_matrix(y_true, y_pred)


st.markdown("# DeepSeek API")
st.markdown(
    "DeepSeek — это семейство больших языковых моделей китайской компании DeepSeek AI. "
    "В работе использована модель `deepseek-chat` (DeepSeek-V3, MoE-архитектура с "
    "≈671 млрд параметров, из которых активируется около 37 млрд на токен). "
    "В отличие от остальных подходов проекта, эта модель **не обучалась на нашем датасете**: "
    "взаимодействие с ней происходит через REST-API в режиме few-shot."
)


st.markdown("---")
st.markdown("## Идея метода")

st.markdown(
    "Поскольку дообучить модель такого размера напрямую невозможно, мы используем "
    "альтернативную стратегию — **retrieval-augmented few-shot**. Для каждого "
    "проверяемого текста из обучающей выборки выбираются 24 наиболее похожих "
    "размеченных примера (по 12 фейков и 12 настоящих новостей), отбор ведётся по "
    "косинусной близости в TF-IDF-представлении. Эти примеры подставляются в "
    "контекст модели как пары «текст → метка», после чего DeepSeek классифицирует "
    "новый текст на основе наблюдаемых закономерностей."
)

st.markdown(
    "Такой режим работы фактически превращает API-классификацию в гибрид k-ближайших "
    "соседей и языковой модели: ретривер отвечает за поиск релевантного контекста, "
    "а LLM — за обобщение и принятие решения."
)


st.markdown("---")
st.markdown("## Промпт и протокол рассуждения")

st.markdown(
    "В системном промпте модели задаётся явная инструкция: ориентироваться не на "
    "стилистические особенности (тексты прошли лемматизацию и потеряли исходное "
    "оформление), а на **содержательные признаки** — характер заявленных событий, "
    "наличие документальных источников, признаки фабрикации или конспирологического "
    "обрамления."
)

st.markdown(
    "Дополнительно модель просят оформлять ответ как JSON с тремя полями: "
    "краткое рассуждение, метку (0 — фейк, 1 — правда) и уровень уверенности. "
    "Структурированный вывод упрощает парсинг ответа и повышает воспроизводимость "
    "результата."
)


st.markdown("---")
st.markdown("## Self-consistency")

st.markdown(
    "Чтобы снизить чувствительность к случайностям самплинга, для каждого текста "
    "выполняется несколько запросов: один с температурой 0 (детерминированный) и "
    "ещё два с температурой 0,4 (с разнообразием). Итоговая метка определяется "
    "большинством голосов. Этот приём, известный как **self-consistency**, "
    "систематически улучшает результаты LLM на задачах с дискретным ответом и "
    "обходится примерно в три раза дороже одиночного запроса."
)


st.markdown("---")
st.markdown("## Результаты на тестовой выборке")

if metrics:
    render_metric_row(
        {
            "Accuracy": metrics["accuracy"],
            "F1": metrics["f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
        }
    )
    st.caption(f"Оценка проведена на hold-out из {n_total} примеров.")

    if cm is not None and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        errors = fp + fn
        st.markdown(
            f"Из {n_total} примеров модель ошиблась в **{errors}** случаях: "
            f"{fp} ложно-положительных (фейк отнесён к достоверным) и "
            f"{fn} ложно-отрицательных (настоящая новость отмечена как фейк). "
            "Полученное качество примечательно тем, что модель ни разу не видела "
            "обучающую выборку при тренировке: всё работает за счёт few-shot-контекста."
        )
else:
    st.info(
        "Кэш предсказаний DeepSeek не найден. Запустите ноутбук "
        "`notebooks/comparsion/final_comparsion.ipynb`, чтобы получить метрики."
    )


st.markdown("---")
st.markdown("## Преимущества и ограничения")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**Преимущества:**
- Не требует обучения и собственной инфраструктуры с GPU
- Хорошо обобщается за счёт большого размера базовой модели
- Few-shot подход легко перенастроить под другие домены
- Возвращает не только метку, но и пояснение к решению
""")
with c2:
    st.markdown("""
**Ограничения:**
- Зависимость от внешнего API: задержки сети, лимиты, оплата по токенам
- Невозможность запустить офлайн или провести воспроизводимый аудит весов
- Потенциальная утечка данных за пределы инфраструктуры
- При увеличении числа голосов растёт стоимость инференса
""")


render_comparsion_section()
