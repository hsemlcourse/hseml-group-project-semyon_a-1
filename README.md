[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Название проекта]

**Студент:** [Алцыбеев Семен Владимирович / Student АБ-123456]

**Группа:** [232]


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи
Предсказание факта явки пациента на прием

Предскажем придет ли пациент на прием на основе его демографических данных, медицинских, соц-экономических и взаимодействии с ним. Это может быть полезно в том, что мы сможем как-то дополнительно взаимодействовать с такими пациентами: напоминать, уточнять

**Задача:** Классификация

**Датасет:** [Medical Appointment No Shows, https://www.kaggle.com/datasets/joniarroba/noshowappointments?resource=download]

**Целевая метрика:** [Accuracy / F1 / RMSE / ...]


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-semyon_a-1
cd ./hseml-group-project-semyon_a-1

# 2. Создать виртуальное окружение
python -m venv .venv
# source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — очищенные данные после:
  - удаления аномалий (Age < 0, некорректные даты)
  - feature engineering (waiting_days, weekday, is_weekend и др.)
  - подготовки к обучению моделей

## Результаты
Здесь коротко выпишите результаты.
| Модель            | F1-score | ROC-AUC  | Примечание                                          |
| ----------------- | -------- | -------- | --------------------------------------------------- |
| Baseline (LogReg) | 0.41     | 0.57     | Pipeline + scaling + class_weight                   |
| Лучшая (Voting)   | **0.44** | **0.61** | Ансамбль (LogReg + RandomForest), небольшой прирост |

- Основные факторы No-show: waiting_days, возраст, SMS-напоминания


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
