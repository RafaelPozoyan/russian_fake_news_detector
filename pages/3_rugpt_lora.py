import os
import json
import streamlit as st
from utils.style import (
    inject_css,
    sidebar_nav,
    render_metric_row,
    back_to_main,
    render_comparsion_section,
)

st.set_page_config(page_title="ruGPT-3 + LoRA", layout="wide")
inject_css()
sidebar_nav()

back_to_main()


metrics = {}
metrics_path = "models/llm_v3_tuned/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

test_m = metrics.get("test", {})
config = metrics.get("config", {})


st.markdown("# ruGPT-3 + LoRA")
st.markdown(
    "В качестве локальной языковой модели использовалась **ruGPT-3 small** — "
    "русскоязычная декодерная модель GPT-семейства от AI Forever (≈125 млн параметров). "
    "Чтобы адаптировать её под задачу классификации, применен **LoRA** "
    "(Low-Rank Adaptation) — метод параметрически-эффективного дообучения, при котором "
    "веса базовой модели остаются замороженными, а обучаются только небольшие "
    "низкоранговые матрицы, добавляемые поверх исходных слоёв."
)


st.markdown("---")
st.markdown("## Почему именно LoRA")

st.markdown(
    "Полное дообучение всех параметров языковой модели требует значительной памяти "
    "и времени, а главное — приводит к высокому риску переобучения на сравнительно "
    "небольшом датасете. LoRA решает эту проблему: вместо того чтобы менять исходные "
    "матрицы весов W, мы аппроксимируем их обновление произведением двух "
    "низкоранговых матриц BA, где r значительно меньше исходной размерности. "
    "В результате обучается порядка 0,5–1% от общего числа параметров, при этом "
    "качество близко к полному fine-tuning."
)

st.markdown(
    "Поверх замороженной ruGPT-3 добавлена линейная классификационная голова "
    "(`AutoModelForSequenceClassification`), которая принимает на вход скрытое "
    "состояние последнего токена и предсказывает один из двух классов: фейк или "
    "достоверная новость."
)


st.markdown("---")
st.markdown("## Конфигурация обучения")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Модель и контекст**")
    st.markdown(f"""
- Базовая модель: `{config.get('model', 'ai-forever/rugpt3small_based_on_gpt2')}`
- Длина контекста: **{config.get('max_length', 512)} токенов**
- Заголовок и тело подаются разделителем `|`
""")
with c2:
    st.markdown("**LoRA-адаптация**")
    st.markdown(f"""
- Ранг адаптации: **r = {config.get('lora_r', 64)}**
- Масштабный коэффициент: α = {config.get('lora_alpha', 128)}
- Dropout: {config.get('lora_dropout', 0.1)}
- Адаптируемые модули: `{', '.join(config.get('target_modules', ['c_attn', 'c_proj', 'c_fc']))}`
""")

st.markdown("**Параметры тренировки**")
st.markdown(f"""
- Эффективный batch size: {config.get('effective_batch', 64)} (с накоплением градиентов)
- Learning rate: {config.get('lr', 5e-5)}
- Эпох: {config.get('epochs', 15)}
- Label smoothing: {config.get('label_smoothing', 0.05)}
- Веса классов: {config.get('class_weights', [1.3, 1.0])} (компенсация дисбаланса)
- Время обучения: {metrics.get('training_time_min', 0):.0f} минут
""")

st.markdown(
    "Адаптация затрагивает не только слои внимания (`c_attn`, `c_proj`), как в "
    "стандартной LoRA-настройке, но и feed-forward блок (`c_fc`). Это даёт модели "
    "больше степеней свободы для подстройки под специфику задачи, сохраняя при этом "
    "общие языковые знания, полученные на этапе предобучения."
)


st.markdown("---")
st.markdown("## Результаты на тестовой выборке")

if test_m:
    render_metric_row(
        {
            "Accuracy": test_m.get("accuracy", 0),
            "F1": test_m.get("f1", 0),
            "Precision": test_m.get("precision", 0),
            "Recall": test_m.get("recall", 0),
        }
    )

    st.markdown(
        "Дообученная ruGPT-3 показывает точность **{:.4f}** и F1 **{:.4f}**. "
        "Этот результат сопоставим с полным fine-tuning RuBERT, при том что у LoRA "
        "обучается лишь малая доля параметров. Сильная сторона подхода — способность "
        "учитывать длинный контекст (до 512 токенов), что важно для статей, в которых "
        "ключевые расхождения между заголовком и телом возникают за пределами первых "
        "нескольких предложений.".format(test_m.get("accuracy", 0), test_m.get("f1", 0))
    )
else:
    st.info("Метрики не найдены — запустите ноутбук обучения.")


img_paths = [
    (
        "models/llm_v3_tuned/training_history.png",
        "История обучения: loss и accuracy по эпохам",
    ),
    ("models/llm_v3_tuned/confusion_matrix.png", "Матрица ошибок на тесте"),
]
for path, caption in img_paths:
    if os.path.exists(path):
        st.markdown("---")
        st.image(path, caption=caption)


st.markdown("---")
st.markdown("## Преимущества и ограничения")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**Преимущества:**
- Эффективная адаптация: обучается ~1% параметров
- Длинный контекст (512 токенов) — больше информации из тела статьи
- Базовая модель остаётся неизменной, адаптеры легко отделяются
- Локальный инференс без внешних API
""")
with c2:
    st.markdown("""
**Ограничения:**
- Базовая ruGPT-3 small по современным меркам — компактная модель (125M)
- Длительное обучение (около часа на одной GPU)
- Чувствительность к подбору ранга r и набора адаптируемых модулей
- На очень коротких или нетипичных текстах результат менее стабилен
""")


render_comparsion_section()
