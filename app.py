"""
Dry Bean Variety Classifier - Streamlit demo app for ML Assignment 2.

Loads five pre-trained scikit-learn pipelines (no retraining happens here)
and lets a user upload a CSV of dry-bean grain measurements to get
predictions, and - if a Class column is present - live evaluation metrics,
a confusion matrix and a classification report computed on that upload.
"""

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
SRC_DIR = os.path.join(APP_DIR, "src")
sys.path.insert(0, SRC_DIR)

import data_utils  # noqa: E402
import evaluate_models  # noqa: E402

st.set_page_config(
    page_title="Dry Bean Variety Classifier",
    page_icon="\U0001FAD8",
    layout="wide",
)

st.markdown(
    """
    <style>
    .bean-hero {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #3f2b1c 0%, #6b4423 60%, #8a5a2e 100%);
        color: #f5ead9;
        margin-bottom: 1.2rem;
    }
    .bean-hero h1 { margin: 0; font-size: 1.9rem; }
    .bean-hero p { margin: 0.35rem 0 0 0; opacity: 0.9; }
    .bean-tile {
        border: 1px solid rgba(139, 90, 46, 0.35);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        background: rgba(139, 90, 46, 0.06);
        text-align: center;
    }
    .bean-tile .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75; }
    .bean-tile .value { font-size: 1.6rem; font-weight: 700; color: #6b4423; }

    .bean-nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #6b4423;
        padding: 0.1rem 0 1rem 0;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(139, 90, 46, 0.25);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.3rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label div p {
        font-size: 0.98rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(139, 90, 46, 0.12);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #6b4423;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div p {
        color: #f5ead9 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_metadata():
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        return json.load(f)


@st.cache_resource
def load_models_and_encoder(_meta):
    models = {key: joblib.load(os.path.join(MODEL_DIR, f"{key}.joblib")) for key in _meta["model_display_names"]}
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return models, label_encoder


@st.cache_data
def load_comparison_table():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "model_metrics.csv"))


@st.cache_data
def load_observations_text():
    with open(os.path.join(ARTIFACTS_DIR, "model_observations.md"), encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_sample_csv_bytes():
    sample = pd.read_csv(os.path.join(APP_DIR, "test_data.csv")).head(25)
    return sample.to_csv(index=False).encode("utf-8")


meta = load_metadata()
models, label_encoder = load_models_and_encoder(meta)
FEATURE_COLUMNS = meta["feature_columns"]
TARGET_COLUMN = meta["target_column"]
CLASS_NAMES = meta["class_names"]
DISPLAY_NAMES = meta["model_display_names"]
KEY_BY_DISPLAY_NAME = {v: k for k, v in DISPLAY_NAMES.items()}


st.markdown(
    """
    <div class="bean-hero">
        <h1>&#129752; Dry Bean Variety Classifier</h1>
        <p>Predicting one of seven registered dry-bean varieties from 16 camera-derived
        shape measurements, compared across five classical ML models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_PAGES = [
    ("Overview", "\U0001F3E0"),
    ("Predict & Evaluate", "\U0001F52E"),
    ("Model Comparison", "\U0001F4CA"),
    ("About This Project", "ℹ️"),
]

st.sidebar.markdown(
    '<div class="bean-nav-brand">&#129752; <span>DryBean ML</span></div>',
    unsafe_allow_html=True,
)
nav_choice = st.sidebar.radio(
    "Go to",
    [f"{icon}  {name}" for name, icon in NAV_PAGES],
    label_visibility="collapsed",
)
page = nav_choice.split("  ", 1)[1]

# ----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Problem Statement")
    st.write(
        "Dry bean exporters and seed-certification labs need to sort grains into "
        "registered varieties for quality control and market pricing. This app "
        "demonstrates five classical machine-learning classifiers trained to tell "
        "apart seven dry-bean varieties purely from geometric measurements of "
        "each grain's image - no manual inspection required."
    )

    st.subheader("Dataset")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="bean-tile"><div class="label">Instances (cleaned)</div><div class="value">{meta["n_total"]:,}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="bean-tile"><div class="label">Features</div><div class="value">{len(FEATURE_COLUMNS)}</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="bean-tile"><div class="label">Classes</div><div class="value">{len(CLASS_NAMES)}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(
        "**Source:** UCI Machine Learning Repository - Dry Bean Dataset "
        "([dataset page](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)). "
        "Koklu, M. and Ozkan, I.A. (2020), *Multiclass Classification of Dry Beans "
        "Using Computer Vision and Machine Learning Techniques*, Computers and "
        "Electronics in Agriculture, 174, 105507. "
        "[https://doi.org/10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)"
    )
    st.caption(
        f"{meta['duplicates_removed']} exact duplicate rows were removed during cleaning; "
        f"the model was trained on an 80/20 stratified split "
        f"({meta['n_train']:,} train / {meta['n_test']:,} test rows, random_state={meta['random_state']})."
    )

    st.subheader("Class Distribution")
    counts = pd.Series(meta["class_counts"]).sort_values(ascending=False)
    st.bar_chart(counts)
    st.caption(
        "The dataset is imbalanced: Bombay has only 522 samples versus 3,546 for "
        "Dermason. All tree/linear models below use class_weight='balanced' where "
        "supported to reduce the impact of this."
    )

# ----------------------------------------------------------------------------
elif page == "Predict & Evaluate":
    st.subheader("1. Choose a model")
    display_choice = st.selectbox("Model", list(DISPLAY_NAMES.values()))
    model_key = KEY_BY_DISPLAY_NAME[display_choice]
    pipeline = models[model_key]

    st.subheader("2. Upload test data (CSV)")
    st.caption(
        "Upload a CSV containing the 16 feature columns listed below. Include a "
        "'Class' column if you want evaluation metrics computed; omit it for "
        "predictions only."
    )
    with st.expander("Required feature columns"):
        st.code(", ".join(FEATURE_COLUMNS))

    st.download_button(
        "Download a correctly formatted sample CSV",
        data=load_sample_csv_bytes(),
        file_name="sample_test_data.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file as CSV: {exc}")
            st.stop()

        if raw_df.empty:
            st.error("The uploaded CSV has no rows.")
            st.stop()

        feature_df, target_series, extra_cols, errors = data_utils.validate_feature_frame(
            raw_df, FEATURE_COLUMNS, TARGET_COLUMN
        )

        if errors:
            for msg in errors:
                st.error(msg)
            st.stop()

        if extra_cols:
            st.info(f"Ignoring unexpected column(s) not used by the model: {', '.join(extra_cols)}")

        y_pred_encoded = pipeline.predict(feature_df)
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

        results_df = raw_df.copy()
        results_df["Predicted_Class"] = y_pred_labels

        st.subheader("3. Predictions")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{model_key}.csv",
            mime="text/csv",
        )

        st.subheader("4. Evaluation")
        if target_series is None:
            st.warning(
                "No 'Class' column was found in the upload, so evaluation metrics, "
                "the confusion matrix and the classification report cannot be "
                "computed - they require ground-truth labels. Predictions above are "
                "still valid."
            )
        else:
            unseen_labels = set(target_series.astype(str)) - set(CLASS_NAMES)
            if unseen_labels:
                st.error(
                    "The Class column contains label(s) not seen during training: "
                    + ", ".join(sorted(unseen_labels))
                    + ". Metrics cannot be computed against an unknown class."
                )
            else:
                y_true_encoded = label_encoder.transform(target_series.astype(str))

                try:
                    metrics = evaluate_models.compute_metrics(
                        pipeline, feature_df, y_true_encoded, n_classes=len(CLASS_NAMES)
                    )
                    metric_tiles = st.columns(6)
                    for tile, name in zip(metric_tiles, evaluate_models.METRIC_COLUMNS):
                        tile.markdown(
                            f'<div class="bean-tile"><div class="label">{name}</div>'
                            f'<div class="value">{metrics[name]:.4f}</div></div>',
                            unsafe_allow_html=True,
                        )
                except ValueError as exc:
                    st.warning(
                        f"AUC/metric computation was skipped: {exc}. This usually "
                        "happens when the uploaded sample does not contain examples "
                        "of every class."
                    )

                st.write("")
                col_cm, col_report = st.columns([1, 1.3])

                with col_cm:
                    st.markdown("**Confusion Matrix**")
                    cm_df = evaluate_models.confusion_matrix_frame(
                        y_true_encoded, y_pred_encoded, CLASS_NAMES
                    )
                    fig, ax = plt.subplots(figsize=(5.2, 4.4))
                    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlOrBr", ax=ax, cbar=False)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                with col_report:
                    st.markdown("**Classification Report**")
                    report_dict = evaluate_models.full_classification_report(
                        y_true_encoded, y_pred_encoded, CLASS_NAMES
                    )
                    report_df = pd.DataFrame(report_dict).transpose().round(4)
                    st.dataframe(report_df, use_container_width=True)

# ----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.subheader("Comparison Table (held-out 20% test split)")
    comparison_df = load_comparison_table()
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=evaluate_models.METRIC_COLUMNS, color="#f3d9b1"
        ),
        use_container_width=True,
    )

    st.subheader("Accuracy by Model")
    st.bar_chart(comparison_df.set_index("ML Model Name")["Accuracy"])

    st.subheader("Observations")
    st.markdown(load_observations_text())

# ----------------------------------------------------------------------------
elif page == "About This Project":
    st.subheader("About")
    st.write(
        "Built for BITS Pilani WILP M.Tech (AIML/DSE) - Machine Learning, "
        "Assignment 2. Five classical classifiers (Logistic Regression, "
        "Decision Tree, kNN, Gaussian Naive Bayes, Random Forest) were trained "
        "once, offline, on the UCI Dry Bean Dataset and saved with joblib; this "
        "app only loads those saved models and never retrains them."
    )
    st.markdown(
        "- **Student:** Niraj Ghetiya (2025AC05976)\n"
        "- **Course:** Machine Learning, M.Tech (AIML/DSE)\n"
        "- **Repository README:** see `README.md` for the full comparison table, "
        "per-model observations, and preprocessing details."
    )
    st.subheader("Academic Integrity")
    st.write(
        "All code, UI text and written observations in this project are original "
        "and were authored specifically for this assignment. Metrics shown "
        "throughout this app are computed from actual model predictions - none "
        "are hard-coded or fabricated."
    )
