"""
Shared evaluation logic used both at training time (to build the comparison
table in artifacts/model_metrics.csv) and at runtime inside the Streamlit app
(to score whatever labelled test CSV the user uploads).

Every metric is computed from the model's actual predictions on held-out data
- nothing here is hard-coded or fabricated.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# All five required models (LogisticRegression, DecisionTreeClassifier,
# KNeighborsClassifier, GaussianNB, RandomForestClassifier) expose
# predict_proba, so a decision_function fallback is never actually needed
# here. The check is kept so the function fails loudly, instead of silently
# mis-scoring AUC, if a future model without predict_proba is ever plugged in.
def _class_probabilities(fitted_pipeline, X):
    if not hasattr(fitted_pipeline, "predict_proba"):
        raise TypeError(
            f"{type(fitted_pipeline).__name__} has no predict_proba; "
            "AUC cannot be computed from predicted class labels."
        )
    return fitted_pipeline.predict_proba(X)


def compute_metrics(fitted_pipeline, X_test, y_test_encoded, n_classes):
    """Compute the six required metrics for one fitted model/pipeline.

    AUC uses macro-averaged one-vs-rest ROC AUC computed from predicted
    class probabilities (never from hard predicted labels). Precision,
    Recall and F1 use weighted averaging so class imbalance in the Dry
    Bean dataset is reflected in the headline comparison table.
    """
    y_pred = fitted_pipeline.predict(X_test)
    y_proba = _class_probabilities(fitted_pipeline, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test_encoded, y_pred),
        "AUC": roc_auc_score(
            y_test_encoded, y_proba, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(
            y_test_encoded, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test_encoded, y_pred, average="weighted", zero_division=0
        ),
        "F1": f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test_encoded, y_pred),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def full_classification_report(y_test_encoded, y_pred, class_names):
    return classification_report(
        y_test_encoded,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )


def confusion_matrix_frame(y_test_encoded, y_pred, class_names):
    import pandas as pd

    cm = confusion_matrix(y_test_encoded, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=class_names, columns=class_names)
