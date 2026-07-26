"""
Data loading, cleaning and splitting utilities for the Dry Bean classification project.

Dataset: UCI Machine Learning Repository - Dry Bean Dataset
Source: Koklu, M. and Ozkan, I.A. (2020), "Multiclass Classification of Dry Beans
Using Computer Vision and Machine Learning Techniques." Computers and Electronics
in Agriculture, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_XLSX_PATH = os.path.join("data", "raw", "Dry_Bean_Dataset.xlsx")
CLEAN_CSV_PATH = os.path.join("data", "dry_bean_dataset.csv")
TEST_DATA_CSV_PATH = "test_data.csv"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_clean_dataset():
    """Load the raw dataset, remove exact duplicate rows, and cache a clean CSV.

    Returns the cleaned DataFrame and the number of duplicate rows removed.
    """
    df = pd.read_excel(RAW_XLSX_PATH)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)
    os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)
    df.to_csv(CLEAN_CSV_PATH, index=False)

    missing_values = int(df.isnull().sum().sum())
    if missing_values > 0:
        # The published dataset has no missing values; guard against
        # unexpected upstream changes rather than silently imputing.
        raise ValueError(f"Unexpected missing values found: {missing_values}")

    return df, duplicates_removed


def split_features_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def encode_target(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def stratified_split(X, y_encoded):
    return train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )


def save_test_data_csv(X_test, y_test, label_encoder, path=TEST_DATA_CSV_PATH):
    """Write a human-readable test CSV (features + true Class label) for the
    Streamlit app to use for on-demand evaluation."""
    out = X_test.copy()
    out[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    out.to_csv(path, index=False)
    return path


def validate_feature_frame(df, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN):
    """Validate an arbitrary uploaded DataFrame against the trained feature
    schema. Used both by the Streamlit app and by standalone tests, kept
    Streamlit-free so it can be exercised with plain pytest/python.

    Returns (feature_df, target_series_or_None, extra_columns, error_messages).
    On any error, feature_df and target_series are None and error_messages
    is non-empty.
    """
    errors = []

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        errors.append("Missing required feature column(s): " + ", ".join(missing))
        return None, None, [], errors

    extra_cols = [c for c in df.columns if c not in feature_columns + [target_column]]

    feature_df = df[feature_columns].copy()
    non_numeric_cols = []
    for col in feature_columns:
        original_na = feature_df[col].isnull().sum()
        coerced = pd.to_numeric(feature_df[col], errors="coerce")
        if coerced.isnull().sum() > original_na:
            non_numeric_cols.append(col)
        feature_df[col] = coerced

    if non_numeric_cols:
        errors.append("Non-numeric value(s) found in column(s): " + ", ".join(non_numeric_cols))
        return None, None, extra_cols, errors

    if feature_df.isnull().any().any():
        bad_cols = feature_df.columns[feature_df.isnull().any()].tolist()
        errors.append("Missing (blank/NaN) feature value(s) found in column(s): " + ", ".join(bad_cols))
        return None, None, extra_cols, errors

    target_series = df[target_column] if target_column in df.columns else None
    return feature_df, target_series, extra_cols, errors
