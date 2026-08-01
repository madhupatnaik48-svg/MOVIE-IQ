"""
MovieIQ.py
----------
Stage 6 (Dashboard) — self-contained version.

This file does NOT depend on utils.py or statistical_tests.py. All
preprocessing, encoding, and statistical-test logic is defined inline so
that it matches exactly what MovieIQ_Project.ipynb does (MultiLabelBinarizer
genre encoding, raw model file, separate encoder/feature-column files).

Run with (from a terminal, NOT a notebook cell):
    streamlit run MovieIQ.py
"""

import os
import ast
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_ind, chi2_contingency

# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ — Predictive Analytics on Film Success",
    page_icon="🎬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Metric boxes: force a dark background AND force the label/value text
       to a light color explicitly, so they stay readable regardless of
       whether Streamlit is rendering in light or dark theme. Relying on
       inherited/default text color is what caused the washed-out numbers. */
    div[data-testid="stMetric"] {
        background-color: #1c1f26;
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid #2c2f38;
    }
    div[data-testid="stMetricLabel"] {
        color: #b8bcc4 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .success-banner {
        background: linear-gradient(90deg, #1f8a4c, #27ae60);
        padding: 1rem; border-radius: 10px; color: white; font-weight: 600;
    }
    .failure-banner {
        background: linear-gradient(90deg, #a83232, #e74c3c);
        padding: 1rem; border-radius: 10px; color: white; font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Preprocessing (mirrors Stage 1 of MovieIQ_Project.ipynb exactly)
# ---------------------------------------------------------------------------
def parse_genres(raw_genres):
    """Parse the stringified list-of-dicts genres column into genre names."""
    try:
        items = ast.literal_eval(raw_genres)
        return [item["name"] for item in items]
    except (ValueError, SyntaxError, TypeError):
        return []


@st.cache_data
def load_and_prepare(path: str = "movies.csv") -> pd.DataFrame:
    """Load, clean, label, and genre-process the dataset (cached)."""
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[df["budget"] > 0]
    df = df[df["revenue"] > 0]
    df["success"] = np.where(df["revenue"] > df["budget"], 1, 0)
    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    return df


# ---------------------------------------------------------------------------
# Statistical tests (mirrors Stage 3 of the notebook)
# ---------------------------------------------------------------------------
def run_ttest(df: pd.DataFrame, feature: str = "popularity", alpha: float = 0.05) -> dict:
    success_vals = df.loc[df["success"] == 1, feature]
    failure_vals = df.loc[df["success"] == 0, feature]
    t_stat, p_value = ttest_ind(success_vals, failure_vals, equal_var=False)
    conclusion = (
        "Reject the Null Hypothesis: significant difference found."
        if p_value < alpha
        else "Fail to Reject the Null Hypothesis: no significant difference found."
    )
    return {
        "null_hypothesis": f"No significant difference in average {feature} between successful and unsuccessful movies.",
        "alt_hypothesis": f"There IS a significant difference in average {feature} between the two groups.",
        "p_value": p_value,
        "conclusion": conclusion,
    }


def run_chi_square(df: pd.DataFrame, feature: str = "primary_genre", alpha: float = 0.05) -> dict:
    contingency_table = pd.crosstab(df[feature], df["success"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    conclusion = (
        "Reject the Null Hypothesis: genre is associated with success."
        if p_value < alpha
        else "Fail to Reject the Null Hypothesis: genre and success are independent."
    )
    return {
        "null_hypothesis": f"{feature} and success are independent.",
        "alt_hypothesis": f"{feature} and success are associated.",
        "p_value": p_value,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Cached model / encoder loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load the model, the MultiLabelBinarizer genre encoder, and the saved
    feature-column order. Returns (model, mlb, feature_columns) or
    (None, None, None) if any file is missing.
    """
    model_path = "models/random_forest.pkl"
    mlb_path = "models/mlb_encoder.pkl"
    cols_path = "models/feature_columns.pkl"

    if not (os.path.exists(model_path) and os.path.exists(mlb_path) and os.path.exists(cols_path)):
        return None, None, None

    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    feature_columns = joblib.load(cols_path)
    return model, mlb, feature_columns


df = load_and_prepare()
model, mlb, feature_columns = load_artifacts()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 MovieIQ Filters")

genre_options = ["All"] + sorted(df["primary_genre"].unique().tolist())
selected_genre = st.sidebar.selectbox("Genre Filter", genre_options)

min_vote = st.sidebar.slider(
    "Minimum Vote Average",
    min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()),
    value=float(df["vote_average"].min()),
    step=0.1,
)

filtered_df = df[df["vote_average"] >= min_vote]
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["primary_genre"] == selected_genre]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** movies.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("A movie is labelled **successful** when Revenue > Budget.")

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
st.header("📊 Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Movies (filtered)", f"{len(filtered_df):,}")
col2.metric("Success Rate", f"{filtered_df['success'].mean() * 100:.1f}%")
col3.metric("Avg. Budget", f"${filtered_df['budget'].mean():,.0f}")
col4.metric("Avg. Revenue", f"${filtered_df['revenue'].mean():,.0f}")

with st.expander("Preview filtered data"):
    st.dataframe(
        filtered_df[["title", "primary_genre", "budget", "revenue", "popularity",
                     "runtime", "vote_average", "success"]].reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# Interactive charts
# ---------------------------------------------------------------------------
st.header("📈 Interactive Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Budget vs Revenue")
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = filtered_df["success"].map({1: "#2ecc71", 0: "#e74c3c"})
    ax.scatter(filtered_df["budget"], filtered_df["revenue"], c=colors, alpha=0.5, s=20)
    if len(filtered_df):
        max_val = max(filtered_df["budget"].max(), filtered_df["revenue"].max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    ax.set_xlabel("Budget ($)")
    ax.set_ylabel("Revenue ($)")
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.subheader("Success Rate by Genre")
    fig, ax = plt.subplots(figsize=(6, 5))
    genre_rate = df.groupby("primary_genre")["success"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=genre_rate.values, y=genre_rate.index, ax=ax, hue=genre_rate.index,
                palette="crest", legend=False)
    ax.set_xlabel("Success Rate (%)")
    ax.set_ylabel("Genre")
    st.pyplot(fig)
    plt.close(fig)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Vote Average vs Success")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(x="success", y="vote_average", data=filtered_df, ax=ax,
                hue="success", palette=["#e74c3c", "#2ecc71"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Failure", "Success"])
    st.pyplot(fig)
    plt.close(fig)

with chart_col4:
    st.subheader("Correlation Heatmap")
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Statistical test results
# ---------------------------------------------------------------------------
st.header("🧪 Statistical Test Results")

stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.subheader("T-Test: Popularity vs Success")
    ttest_result = run_ttest(df, "popularity")
    st.write(f"**H0:** {ttest_result['null_hypothesis']}")
    st.write(f"**H1:** {ttest_result['alt_hypothesis']}")
    st.metric("p-value", f"{ttest_result['p_value']:.4g}")
    st.info(ttest_result["conclusion"])

with stat_col2:
    st.subheader("Chi-Square: Genre vs Success")
    chi2_result = run_chi_square(df, "primary_genre")
    st.write(f"**H0:** {chi2_result['null_hypothesis']}")
    st.write(f"**H1:** {chi2_result['alt_hypothesis']}")
    st.metric("p-value", f"{chi2_result['p_value']:.4g}")
    st.info(chi2_result["conclusion"])

# ---------------------------------------------------------------------------
# Prediction section
# ---------------------------------------------------------------------------
st.header("🔮 Predict a Movie's Success")

if model is None:
    st.warning(
        "Model files not found. Make sure `models/random_forest.pkl`, "
        "`models/mlb_encoder.pkl`, and `models/feature_columns.pkl` all exist "
        "(run the Stage 4/5 cells in the notebook first), then relaunch the app."
    )
else:
    with st.form("prediction_form"):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            input_budget = st.number_input("Budget ($)", min_value=0, value=50_000_000, step=1_000_000)
            input_popularity = st.number_input("Popularity", min_value=0.0, value=50.0, step=1.0)
        with pcol2:
            input_runtime = st.number_input("Runtime (minutes)", min_value=0, value=110, step=1)
            input_vote_avg = st.slider("Vote Average", 0.0, 10.0, 6.5, 0.1)
        with pcol3:
            input_genre = st.selectbox("Genre", list(mlb.classes_))

        submitted = st.form_submit_button("Predict Success")

    if submitted:
        # Build the multi-hot genre vector using the SAME MultiLabelBinarizer
        # that was fit during training, so column order matches exactly.
        genre_encoded = mlb.transform([[input_genre]])[0]
        genre_dict = dict(zip(mlb.classes_, genre_encoded))

        input_row = pd.DataFrame([{
            "budget": input_budget,
            "popularity": input_popularity,
            "runtime": input_runtime,
            "vote_average": input_vote_avg,
            **genre_dict,
        }])[feature_columns]  # enforce exact training column order

        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0][1]

        if prediction == 1:
            st.markdown(
                f'<div class="success-banner">✅ Predicted: SUCCESS '
                f'(probability: {probability * 100:.1f}%)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="failure-banner">❌ Predicted: NOT SUCCESSFUL '
                f'(probability of success: {probability * 100:.1f}%)</div>',
                unsafe_allow_html=True,
            )

        confidence = max(probability, 1 - probability)
        st.metric("Prediction Confidence", f"{confidence * 100:.1f}%")
        st.caption(
            "Confidence reflects how strongly the model leans toward its predicted class; "
            "it is not a guarantee of the real-world outcome."
        )

st.markdown("---")
st.caption("MovieIQ — built with Streamlit, scikit-learn, and a Random Forest classifier.")