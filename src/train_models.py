"""
Trains all five required classification models on the Dry Bean dataset,
evaluates each on the same held-out stratified test split, and persists:

  - model/<name>.joblib          fitted scikit-learn pipeline per model
  - model/label_encoder.joblib   fitted LabelEncoder for the Class column
  - model/model_metadata.json    feature order, class names, split/config info
  - artifacts/model_metrics.csv  the comparison table (Accuracy/AUC/.../MCC)
  - test_data.csv                labelled 20% holdout, used by the Streamlit app

Run from the ml-assignment-2/ project root:
    python src/train_models.py
"""

import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
import data_utils  # noqa: E402
import evaluate_models  # noqa: E402

MODEL_DIR = "model"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = data_utils.RANDOM_STATE


def build_knn_pipeline(X_train, y_train):
    """Select k for kNN via 5-fold stratified CV on the training data only
    (test set is never touched during tuning). The candidate grid is small
    on purpose to stay cheap enough for Streamlit Community Cloud rebuilds."""
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
    )
    param_grid = {"clf__n_neighbors": [3, 5, 7, 9, 11]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(
        f"  kNN CV selected n_neighbors={search.best_params_['clf__n_neighbors']} "
        f"(cv weighted-F1={search.best_score_:.4f})"
    )
    return search.best_estimator_


def build_models(X_train, y_train):
    """Returns an ordered dict of {name: fitted pipeline}.

    Scaling is applied only where the algorithm is distance/gradient based
    (Logistic Regression, kNN). Tree-based and Naive-Bayes models are scale
    invariant, so they are trained directly on the raw numeric features.
    class_weight='balanced' is used for Logistic Regression, Decision Tree
    and Random Forest because the dataset is imbalanced (Bombay has 522
    samples vs. 3546 for Dermason); GaussianNB and kNN do not support this
    parameter.
    """
    models = {}

    print("Training Logistic Regression ...")
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    print("Training Decision Tree ...")
    dt = Pipeline(
        [
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    dt.fit(X_train, y_train)
    models["decision_tree"] = dt

    print("Training kNN (with CV-based k selection) ...")
    models["knn"] = build_knn_pipeline(X_train, y_train)

    print("Training Gaussian Naive Bayes ...")
    nb = Pipeline([("clf", GaussianNB())])
    nb.fit(X_train, y_train)
    models["naive_bayes"] = nb

    print("Training Random Forest (ensemble) ...")
    rf = Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        ]
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    return models


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading and cleaning dataset ...")
    df, duplicates_removed = data_utils.load_and_clean_dataset()
    print(f"  Rows after cleaning: {len(df)} (duplicates removed: {duplicates_removed})")

    X, y = data_utils.split_features_target(df)
    y_encoded, label_encoder = data_utils.encode_target(y)
    class_names = list(label_encoder.classes_)
    print(f"  Classes ({len(class_names)}): {class_names}")

    X_train, X_test, y_train, y_test = data_utils.stratified_split(X, y_encoded)
    print(f"  Train size: {len(X_train)}  Test size: {len(X_test)}")

    start = time.time()
    models = build_models(X_train, y_train)
    print(f"All models trained in {time.time() - start:.1f}s")

    print("\nEvaluating on held-out test split ...")
    rows = []
    for key, pipeline in models.items():
        metrics = evaluate_models.compute_metrics(
            pipeline, X_test, y_test, n_classes=len(class_names)
        )
        metrics["ML Model Name"] = DISPLAY_NAMES[key]
        rows.append(metrics)
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{key}.joblib"), compress=3)
        print(f"  {DISPLAY_NAMES[key]:<26} {metrics}")

    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"), compress=3)

    metrics_df = pd.DataFrame(rows)[
        ["ML Model Name"] + evaluate_models.METRIC_COLUMNS
    ]
    metrics_path = os.path.join(ARTIFACTS_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved comparison table to {metrics_path}")

    metadata = {
        "feature_columns": data_utils.FEATURE_COLUMNS,
        "target_column": data_utils.TARGET_COLUMN,
        "class_names": class_names,
        "class_counts": y.value_counts().to_dict(),
        "test_size": data_utils.TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "duplicates_removed": duplicates_removed,
        "model_display_names": DISPLAY_NAMES,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model_metadata.json")

    test_csv_path = data_utils.save_test_data_csv(X_test, y_test, label_encoder)
    print(f"Saved labelled test split to {test_csv_path} ({len(X_test)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
