import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from utils import evaluate_model, recall_at_top_k, precision_at_top_k

TARGET = "Exited"

df = pd.read_csv("data/Bank_Churn.csv")

# Drop obvious identifiers if present.
drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
df = df.drop(columns=drop_cols)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

preprocess_for_linear_ann = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

preprocess_for_trees = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("prep", preprocess_for_linear_ann),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("prep", preprocess_for_trees),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "xgboost": Pipeline(
        steps=[
            ("prep", preprocess_for_trees),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=5.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate_model(name, model, X_test, y_test))

metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print(metrics_df)

# if not open("reports/baseline_metrics.json", "w").writeable():
#     raise Exception("Cannot write to reports/baseline_metrics.json. Please check permissions.")
with open("reports/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)