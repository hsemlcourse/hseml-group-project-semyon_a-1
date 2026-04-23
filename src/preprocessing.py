import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Даты
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Фильтрация
    df = df[df['Age'] >= 0]
    df = df[df['ScheduledDay'] <= df['AppointmentDay']]

    # Target
    df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    # Feature engineering
    df['waiting_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['appointment_weekday'] = df['AppointmentDay'].dt.weekday
    df['scheduled_weekday'] = df['ScheduledDay'].dt.weekday
    df['is_weekend'] = df['appointment_weekday'].isin([5, 6]).astype(int)

    return df


def save_processed_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_raw_data('../data/raw/KaggleV2-May-2016.csv')
    df = preprocess_data(df)
    save_processed_data(df, '../data/processed/processed.csv')