import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
def recall_at_top_k(y_true, y_prob, k=0.20):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_top = int(np.ceil(len(y_prob) * k))
    top_idx = np.argsort(y_prob)[-n_top:]

    true_positives_in_top = y_true[top_idx].sum()
    total_positives = y_true.sum()

    return true_positives_in_top / total_positives


def precision_at_top_k(y_true, y_prob, k=0.20):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_top = int(np.ceil(len(y_prob) * k))
    top_idx = np.argsort(y_prob)[-n_top:]

    return y_true[top_idx].mean()

def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "recall_at_top_20_pct": recall_at_top_k(y_test, y_prob, k=0.20),
        "precision_at_top_20_pct": precision_at_top_k(y_test, y_prob, k=0.20),
    }