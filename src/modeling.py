"""Пайплайны, метрики и обучение моделей.

Запуск обучения из корня репозитория:
    python -m src.modeling
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CATEGORICAL_COLS = ['Gender', 'Neighbourhood']
LEAK_COLS = ['No-show', 'PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']

RAW_DATA_PATH = Path('data/raw/KaggleV2-May-2016.csv')
PROCESSED_DATA_PATH = Path('data/processed/processed.csv')
MODELS_DIR = Path('models')


def split_features_target(df: pd.DataFrame):
    groups = df['PatientId']
    X = df.drop(columns=LEAK_COLS)
    y = df['No-show']
    return X, y, groups


def get_column_types(X: pd.DataFrame):
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return categorical_cols, numeric_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ]
    )


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['waiting_x_sms'] = X['waiting_days'] * X['SMS_received']
    X['age_x_diabetes'] = X['Age'] * X['Diabetes']
    X['is_risky_age'] = ((X['Age'] >= 18) & (X['Age'] <= 35)).astype(int)
    X['log_waiting_days'] = np.log1p(X['waiting_days'].fillna(0))
    return X


def group_train_test_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE):
    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state, n_splits=1)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def stratified_train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_baseline_model(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            C=0.5,
        )),
    ])


def build_model_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator),
    ])


def cross_validate_model(model, X, y, cv=None, scoring=None, return_train_score=False):
    if scoring is None:
        scoring = ['f1', 'recall', 'precision', 'roc_auc']
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=-1,
    )
    return pd.DataFrame(results)


def tune_logistic_regression(preprocessor, X, y, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipeline = build_model_pipeline(
        preprocessor,
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    )
    param_grid = {
        'model__C': [0.01, 0.1, 0.5, 1.0],
        'model__class_weight': ['balanced', None],
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    search.fit(X, y)
    return search


def tune_random_forest(preprocessor, X, y, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipeline = build_model_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    )
    param_grid = {
        'model__n_estimators': [200, 300],
        'model__max_depth': [10, 15, 20],
        'model__min_samples_split': [5, 10],
        'model__class_weight': ['balanced', 'balanced_subsample'],
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    search.fit(X, y)
    return search


def tune_gradient_boosting(preprocessor, X, y, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    pipeline = build_model_pipeline(
        preprocessor,
        GradientBoostingClassifier(random_state=RANDOM_STATE),
    )
    param_grid = {
        'model__n_estimators': [100, 150],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__subsample': [0.8],
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    search.fit(X, y)
    return search


def tune_hist_gradient_boosting(preprocessor, X, y, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipeline = build_model_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    )
    param_grid = {
        'model__max_iter': [200, 300],
        'model__learning_rate': [0.05, 0.1, 0.15],
        'model__max_depth': [7, 10, 12],
        'model__l2_regularization': [0.0, 0.01],
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    search.fit(X, y)
    return search


def build_voting_classifier(estimators, voting='soft') -> VotingClassifier:
    return VotingClassifier(estimators=estimators, voting=voting)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def find_best_threshold(y_true, y_proba, low=0.2, high=0.6, n=50):
    thresholds = np.linspace(low, high, n)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def youden_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return thresholds[idx], j_scores[idx], fpr, tpr


def print_classification_report(y_true, y_pred, target_names=('Show', 'No-show')):
    print(classification_report(y_true, y_pred, target_names=list(target_names)))


def get_feature_importance(fitted_pipeline, top_n=20) -> pd.DataFrame:
    feature_names = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()
    model = fitted_pipeline.named_steps.get('model') or fitted_pipeline.named_steps.get('classifier')

    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
        col = 'importance'
    elif hasattr(model, 'coef_'):
        values = model.coef_[0]
        col = 'coef'
    else:
        raise AttributeError('Model has neither feature_importances_ nor coef_')

    df = pd.DataFrame({'feature': feature_names, col: values})
    sort_key = (lambda s: s.abs()) if col == 'coef' else None
    return df.sort_values(col, key=sort_key, ascending=False).head(top_n)


def save_model(model, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def summarize_cv_results(cv_df: pd.DataFrame, model_name: str) -> dict:
    means = cv_df.mean()
    return {
        'Model': model_name,
        'F1': means['test_f1'],
        'Recall': means['test_recall'],
        'Precision': means['test_precision'],
        'ROC-AUC': means['test_roc_auc'],
    }


def train_baseline(df):
    print('[baseline] LogisticRegression (GroupShuffleSplit по PatientId)')

    X, y, groups = split_features_target(df)
    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups)

    categorical_cols, numeric_cols = get_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    model = build_model_pipeline(
        preprocessor,
        LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            C=0.5,
        ),
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print(
        f"  F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}  "
        f"Precision={metrics['precision']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}"
    )

    save_model(model, MODELS_DIR / 'baseline_logreg.joblib')
    print(f"  saved -> {MODELS_DIR / 'baseline_logreg.joblib'}")


def train_best(df):
    print('[best] VotingClassifier (LogReg + RandomForest) + feature engineering')

    X, y, _ = split_features_target(df)
    X = add_interaction_features(X)

    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    categorical_cols, numeric_cols = get_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    lr_pipeline = build_model_pipeline(
        preprocessor,
        LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            C=0.01,
            class_weight='balanced',
        ),
    )
    rf_pipeline = build_model_pipeline(
        preprocessor,
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced_subsample',
        ),
    )

    voting = build_voting_classifier(
        estimators=[('lr', lr_pipeline), ('rf', rf_pipeline)],
        voting='soft',
    )
    voting.fit(X_train, y_train)

    metrics = evaluate_model(voting, X_test, y_test)
    print(
        f"  F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}  "
        f"Precision={metrics['precision']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}"
    )

    best_thresh, best_f1 = find_best_threshold(y_test, metrics['y_proba'])
    print(f'  optimal threshold={best_thresh:.3f}  F1@thresh={best_f1:.4f}')

    save_model(voting, MODELS_DIR / 'best_voting.joblib')
    print(f"  saved -> {MODELS_DIR / 'best_voting.joblib'}")


def main():
    from src.preprocessing import load_raw_data, preprocess_data, save_processed_data

    df = load_raw_data(str(RAW_DATA_PATH))
    df = preprocess_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_processed_data(df, str(PROCESSED_DATA_PATH))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_baseline(df)
    train_best(df)


if __name__ == '__main__':
    main()
