"""Classical STL baselines for Tox21."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .evaluation import safe_auprc, safe_auroc


def get_valid_xy(df: pd.DataFrame, endpoint: str, fp_col: str = "MorganFP"):
    sub = df.loc[df[endpoint].notna()].copy()
    x = np.stack(sub[fp_col].values)
    y = sub[endpoint].astype(int).values
    return x, y


def fit_logistic_regression(x_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def fit_random_forest(x_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_binary_classifier(model, x_test, y_test):
    y_score = model.predict_proba(x_test)[:, 1]
    return {
        "AUPRC": safe_auprc(y_test, y_score),
        "AUROC": safe_auroc(y_test, y_score),
    }
