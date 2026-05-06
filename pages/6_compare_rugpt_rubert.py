import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.style import inject_css, sidebar_nav, load_json, back_to_main

st.set_page_config(page_title="ruGPT-3 + LoRA vs RuBERT", layout="wide")
inject_css()
sidebar_nav()

back_to_main()

st.markdown("# ruGPT-3 + LoRA vs RuBERT")
st.markdown(
    "Здесь сопоставлены два подхода к дообучению нейросетевых моделей под задачу "
    "детекции фейков: полное fine-tuning RuBERT и параметрически-эффективная "
    "LoRA-адаптация ruGPT-3. Оба метода используют один и тот же обучающий "
    "сплит, но различаются как архитектурой базовой модели "
    "(BERT-encoder против GPT-decoder), так и стратегией дообучения."
)

st.markdown(
    "RuBERT — двунаправленный энкодер, специализированный под понимание текста. "
    "При полном fine-tuning все его параметры обновляются, что даёт максимальную "
    "пластичность, но требует значительных ресурсов и осторожности с регуляризацией. "
    "ruGPT-3 — однонаправленный декодер, чья сильная сторона — генерация. "
    "В нашей реализации мы навешиваем на него классификационную голову и обучаем "
    "только LoRA-адаптеры; такой подход быстрее, экономичнее по памяти и устойчивее к "
    "переобучению на небольшом датасете."
)


rows = []

m_rb = load_json("models/rubert/metrics.json")
rb = m_rb.get("rubert", {})
if rb:
    rows.append(
        {
            "Модель": "RuBERT (full fine-tuning)",
            "Параметры": "~178 млн (все обучаются)",
            "Контекст": "256 токенов",
            "Accuracy": rb.get("test_acc", 0),
            "F1": rb.get("test_f1", 0),
            "Precision": rb.get("test_precision", 0),
            "Recall": rb.get("test_recall", 0),
        }
    )

m_rugpt = load_json("models/llm_v3_tuned/metrics.json")
rugpt = m_rugpt.get("test", {})
config = m_rugpt.get("config", {})
if rugpt:
    rows.append(
        {
            "Модель": "ruGPT-3 + LoRA",
            "Параметры": f"~125 млн (обучается ~1% через LoRA r={config.get('lora_r', 64)})",
            "Контекст": f"{config.get('max_length', 512)} токенов",
            "Accuracy": rugpt.get("accuracy", 0),
            "F1": rugpt.get("f1", 0),
            "Precision": rugpt.get("precision", 0),
            "Recall": rugpt.get("recall", 0),
        }
    )

if rows:
    st.markdown("---")
    st.markdown("## Метрики на тестовой выборке")
    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ["Accuracy", "F1", "Precision", "Recall"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if x > 0 else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Метрики не найдены — обучите модели через соответствующие ноутбуки.")


rb_preds_path = "models/rubert/test_predictions.csv"
rugpt_preds_path = "models/llm_v3_tuned/test_predictions.csv"

if os.path.exists(rb_preds_path) and os.path.exists(rugpt_preds_path):
    rb_df = pd.read_csv(rb_preds_path)
    rugpt_df = pd.read_csv(rugpt_preds_path)

    if len(rb_df) == len(rugpt_df):
        # Унифицируем имена колонок
        def get_pred_cols(d):
            cols = d.columns.tolist()
            true_col = next(
                (
                    c
                    for c in cols
                    if "true" in c.lower()
                    or "label" in c.lower()
                    and "pred" not in c.lower()
                ),
                cols[0],
            )
            pred_col = next(
                (c for c in cols if "pred" in c.lower()),
                cols[1] if len(cols) > 1 else cols[0],
            )
            return true_col, pred_col

        rb_t, rb_p = get_pred_cols(rb_df)
        rg_t, rg_p = get_pred_cols(rugpt_df)

        y_true = rb_df[rb_t].values
        rb_pred = rb_df[rb_p].values
        rugpt_pred = rugpt_df[rg_p].values

        if len(y_true) == len(rugpt_pred):
            agree = (rb_pred == rugpt_pred).sum()
            both_right = ((rb_pred == y_true) & (rugpt_pred == y_true)).sum()
            both_wrong = ((rb_pred != y_true) & (rugpt_pred != y_true)).sum()
            only_rb = ((rb_pred == y_true) & (rugpt_pred != y_true)).sum()
            only_rugpt = ((rb_pred != y_true) & (rugpt_pred == y_true)).sum()

            st.markdown("---")
            st.markdown("## Согласованность предсказаний")
            st.markdown(
                f"Из {len(y_true)} тестовых примеров модели согласились между собой в "
                f"**{agree}** случаях ({agree/len(y_true):.1%}). Обе ошиблись одновременно "
                f"на {both_wrong} примерах — это, по-видимому, особенно сложные случаи в "
                f"датасете. RuBERT оказался прав там, где ruGPT-3 ошибся, в {only_rb} случаях; "
                f"обратная ситуация наблюдалась в {only_rugpt} случаях."
            )

            agreement_data = {
                "Категория": [
                    "Обе модели правы",
                    "Обе ошиблись",
                    "Прав только RuBERT",
                    "Прав только ruGPT-3 + LoRA",
                ],
                "Количество": [
                    int(both_right),
                    int(both_wrong),
                    int(only_rb),
                    int(only_rugpt),
                ],
            }
            st.dataframe(
                pd.DataFrame(agreement_data), use_container_width=True, hide_index=True
            )


img_paths = [
    ("assets/rubert_vs_rugpt_comparsion.csv", None),
    (
        "assets/rubert_vs_rugpt_comparsion_confusion.png",
        "Матрицы ошибок: RuBERT и ruGPT-3 + LoRA",
    ),
    ("assets/rubert_vs_rugpt_comparsion_roc.png", "ROC-кривые двух моделей"),
    (
        "assets/rubert_vs_rugpt_comparsion_confidence.png",
        "Распределение уверенности предсказаний",
    ),
    ("assets/rubert_vs_rugpt_comparsion_agreement.png", "Анализ согласованности"),
]
images_shown = False
for path, caption in img_paths:
    if path.endswith(".png") and os.path.exists(path):
        if not images_shown:
            st.markdown("---")
            st.markdown("## Визуализация")
            images_shown = True
        st.image(path, caption=caption)


st.markdown("---")
st.markdown("## Выводы")
st.markdown(
    "Обе модели работают на сопоставимом уровне точности, но достигают его "
    "разными средствами. RuBERT, как двунаправленный энкодер, опирается на "
    "одновременное чтение всего входа и хорошо улавливает локальную семантическую "
    "согласованность между заголовком и телом. ruGPT-3 + LoRA получает преимущество "
    "за счёт удвоенной длины контекста: длинные статьи он обрабатывает целиком, тогда "
    "как RuBERT вынужден их обрезать. С практической точки зрения LoRA-вариант "
    "выигрывает в стоимости обучения — отдельный адаптер занимает считанные "
    "мегабайты, а не сотни, как полная копия дообученного RuBERT."
)
