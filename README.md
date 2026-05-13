[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project -- [Название проекта]

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

Датасет искался на Kaggle по тематике здравоохранения и поведения пациентов: интересовала практическая задача бинарной классификации, где предсказание может приносить реальную пользу клинике (снижение потерь от незаполненных слотов, адресные напоминания). Из нескольких кандидатов выбран **Medical Appointment No Shows** по следующим причинам:
- Содержит реальные данные из бразильской системы здравоохранения (~110 тыс. записей о приёмах) -- достаточный объём для устойчивого обучения и валидации.
- В нем отражен богатый мир признаков: демография (возраст, пол), социально-экономический статус (Scholarship), медицинская история (Hipertension, Diabetes, Alcoholism, Handcap), поведенческий фактор (SMS_received) и временны́е метки (ScheduledDay, AppointmentDay) -- это позволяет делать осмысленный feature engineering (`waiting_days`, день недели и т.д.).
- Классы в нем умеренно несбалансированы, что делает задачу нетривиальной, но не вырожденной.
- Данные из Kaggle доступны без ограничений

- **Объём:** 110 527 строк × 14 столбцов
- **Целевая переменная:** `No-show` (Yes / No)
- **Период:** записи о приёмах за апрель–июнь 2016 года
- **Столбцы:**
  - `PatientId`, `AppointmentID` -- идентификаторы
  - `Gender` -- пол (M/F)
  - `ScheduledDay` -- дата и время записи на приём
  - `AppointmentDay` -- дата самого приёма
  - `Age` -- возраст пациента
  - `Neighbourhood` -- район проживания
  - `Scholarship` -- участие в программе соц. поддержки Bolsa Família (0/1)
  - `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` -- медицинские флаги
  - `SMS_received` -- было ли отправлено SMS-напоминание (0/1)
  - `No-show` -- целевая метка (Yes -- не пришёл, No -- пришёл)


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
│   ├── baseline_logreg.joblib  # Baseline LogisticRegression (создаётся modeling.py)
│   └── best_voting.joblib      # VotingClassifier LogReg + RandomForest (создаётся modeling.py)
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

### Обучение моделей

Из корня репозитория:
```bash
python -m src.modeling
```

Скрипт читает `data/raw/KaggleV2-May-2016.csv`, сохраняет очищенные данные в `data/processed/processed.csv` и обучает две модели:

- `models/baseline_logreg.joblib` -- baseline LogisticRegression
- `models/best_voting.joblib` -- VotingClassifier (LogReg + RandomForest)

Файлы `.joblib` игнорируются git -- артефакты генерируются локально.

## Данные

### Обработка и подготовка
- `data/raw/` -- исходные файлы
- `data/processed/` -- очищенные данные после:
  - **Очистки:** удаления аномалий (Age < 0 -- 1 строка, ScheduledDay > AppointmentDay -- 38,567 строк, дубликаты)
  - **Feature engineering:** исходных 14 признаков + 4 новых:
    - `waiting_days` = дни между записью и приёмом (ключевой фактор)
    - `appointment_weekday` = день недели приёма, где 0 -- понедельник, 6 -- воскресенье
    - `scheduled_weekday` = день недели записи
    - `is_weekend` = флаг выходного дня для приёма
  - подготовки к обучению моделей
  
Итого после очистки: **71,959 строк** (70% явились, 30% не явились). После one-hot encoding Neighbourhood: **~96 признаков**.

### Защита от утечки данных (data leakage)
Удалены потенциально опасные для обучения столбцы: `PatientId`, `AppointmentID`, `ScheduledDay`, `AppointmentDay` -- они содержат информацию, недоступную на момент решения о профилактике.

**Train/Test split:** 80/20, группировка по PatientId (один пациент целиком в train или test). **Cross-validation:** 5-fold StratifiedKFold для контроля переобучения.

## Результаты

### Выбор метрик
- **F1-score** (приоритет): баланс между Precision и Recall при дисбалансе классов (70/30). Отвергнута Accuracy -- она даёт ложное впечатление качества (67% точность при 70% базового уровня).
- **Recall**: способность поймать максимум случаев no-show для адресной профилактики.
- **Precision**: минимизация ложных тревог (лишние напоминания).
- **ROC-AUC**: independent от порога классификации, для вероятностной оценки.

### Качество моделей

| Модель                      | F1-score  | ROC-AUC   | Примечание                                            |
| --------------------------- | --------- | --------- | ----------------------------------------------------- |
| Наивный (LogReg default)    | 0.000     | 0.609     | Без class_weight — предсказывает только класс 0       |
| Baseline (LogReg + FE)      | 0.413     | 0.572     | class_weight='balanced' + feature engineering         |
| LightGBM (tuned)            | 0.443     | 0.614     | Лучшая одиночная модель                               |
| **Voting (LR+RF+LightGBM)** | **0.445** | **0.617** | Финальная; test F1=0.464 при threshold=0.445          |

Прирост финальной модели над расширенным baseline: +7.6% F1, +7.8% ROC-AUC.

### Ключевые факторы No-show

По важности признаков (Correlation + RF feature importance):
1. **waiting_days** (время ожидания) -- сильнейший фактор, ведь пациенты, ждущие дольше 2 недель, пропускают приём в 2 раза чаще
2. **Age** (возраст) -- молодые пациенты, кому по 18–35 лет, no-show в 3 раза чаще, чем у пожилых
3. **Neighbourhood** (район) -- в силу социально-экономических различий в бедных районах выше риск неприхода
4. **SMS_received** -- парадокс: SMS связана с выше no-show (с другой стороны, организация уже могла отправлять SMS пациентам высокого риска)


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
