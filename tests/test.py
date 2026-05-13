"""Smoke tests for the preprocessing and modeling pipeline."""

from pathlib import Path

import pandas as pd
import pytest

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "KaggleV2-May-2016.csv"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df_raw():
    return pd.read_csv(RAW_PATH)


@pytest.fixture(scope="module")
def df_processed(df_raw):
    from src.preprocessing import preprocess_data

    return preprocess_data(df_raw)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def test_load_raw_data_shape(df_raw):
    assert df_raw.shape == (110527, 14), f"Expected (110527, 14), got {df_raw.shape}"


def test_preprocess_output_columns(df_processed):
    expected = {
        "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
        "Handcap", "SMS_received", "waiting_days", "appointment_weekday",
        "scheduled_weekday", "is_weekend", "No-show",
    }
    assert expected.issubset(set(df_processed.columns))


def test_no_negative_age(df_processed):
    assert (df_processed["Age"] >= 0).all(), "Found rows with Age < 0"


def test_waiting_days_clip(df_processed):
    assert df_processed["waiting_days"].min() >= 0
    assert df_processed["waiting_days"].max() <= 100


def test_target_encoding(df_processed):
    unique_vals = set(df_processed["No-show"].unique())
    assert unique_vals == {0, 1}, f"Unexpected target values: {unique_vals}"


def test_no_duplicates(df_processed):
    assert df_processed.duplicated().sum() == 0


def test_scheduled_before_appointment(df_processed):
    diff = (df_processed["AppointmentDay"] - df_processed["ScheduledDay"]).dt.days
    assert (diff >= 0).all(), "Found rows where ScheduledDay > AppointmentDay"


# ---------------------------------------------------------------------------
# Modeling pipeline tests
# ---------------------------------------------------------------------------


def test_add_interaction_features(df_processed):
    from src.modeling import add_interaction_features, split_features_target

    X, _, _ = split_features_target(df_processed)
    X_fe = add_interaction_features(X)

    new_cols = {"waiting_x_sms", "age_x_diabetes", "is_risky_age", "log_waiting_days"}
    assert new_cols.issubset(set(X_fe.columns)), f"Missing interaction features: {new_cols - set(X_fe.columns)}"
    assert X_fe.shape[1] == X.shape[1] + 4


def test_build_preprocessor_fit_transform(df_processed):
    from src.modeling import build_preprocessor, get_column_types, split_features_target

    X, _, _ = split_features_target(df_processed)
    cat_cols, num_cols = get_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > X.shape[1]


def test_group_split_no_patient_leakage(df_processed):
    from src.modeling import group_train_test_split, split_features_target

    X, y, groups = split_features_target(df_processed)
    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups)

    train_patients = set(groups.loc[X_train.index])
    test_patients = set(groups.loc[X_test.index])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Patient leakage: {len(overlap)} patients in both train and test"


def test_baseline_model_trains(df_processed):
    from sklearn.metrics import f1_score

    from src.modeling import (
        build_baseline_model,
        build_preprocessor,
        get_column_types,
        group_train_test_split,
        split_features_target,
    )

    X, y, groups = split_features_target(df_processed)
    cat_cols, num_cols = get_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = build_baseline_model(preprocessor)

    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    assert f1 > 0.30, f"Baseline F1 too low: {f1:.4f}"
