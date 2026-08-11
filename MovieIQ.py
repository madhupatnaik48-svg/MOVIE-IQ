"""
MovieIQ.py
MovieIQ - Predictive Analytics on Film Success

Streamlit Dashboard

Target Variable:
    Success = 1 if Revenue > Budget
    Success = 0 if Revenue <= Budget
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieIQ - Success Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. PROJECT PATHS
# ============================================================

# Location of this MovieIQ.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATA_PATH = os.path.join(
    BASE_DIR,
    "movies.csv"
)

# Models folder
MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# Model files
RANDOM_FOREST_PATH = os.path.join(
    MODELS_DIR,
    "random_forest.pkl"
)

MLB_PATH = os.path.join(
    MODELS_DIR,
    "mlb_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_DIR,
    "feature_columns.pkl"
)


# ============================================================
# 5. TITLE
# ============================================================

st.markdown(
    "<div class='main-title'>"
    "🎬 MovieIQ - Predictive Analytics on Film Success"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>"
    "Explore movie data, financial success, "
    "statistical analysis, and Random Forest predictions."
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# 6. CHECK REQUIRED FILES
# ============================================================

missing_files = []


# Check movies.csv
if not os.path.exists(DATA_PATH):
    missing_files.append("movies.csv")


# Check Random Forest model
if not os.path.exists(RANDOM_FOREST_PATH):
    missing_files.append(
        "models/random_forest.pkl"
    )


# Check MultiLabelBinarizer
if not os.path.exists(MLB_PATH):
    missing_files.append(
        "models/mlb_encoder.pkl"
    )


# Check feature columns
if not os.path.exists(FEATURE_COLUMNS_PATH):
    missing_files.append(
        "models/feature_columns.pkl"
    )


# Display missing files
if missing_files:

    st.error(
        "❌ Required MovieIQ files are missing."
    )

    st.write(
        "The following files could not be found:"
    )

    for file_name in missing_files:
        st.write(
            f"- `{file_name}`"
        )

    st.info(
        "Make sure your project folder contains "
        "movies.csv and the models folder with all "
        "three model files."
    )

    st.stop()


# ============================================================
# 7. LOAD AND PREPROCESS DATA
# ============================================================

@st.cache_data
def load_data():

    # Load CSV
    df = pd.read_csv(DATA_PATH)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Keep valid financial records
    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ].copy()

    # --------------------------------------------------------
    # Target Variable
    # --------------------------------------------------------
    # Success = 1 if Revenue > Budget
    # Success = 0 if Revenue <= Budget

    df["success"] = (
        df["revenue"] > df["budget"]
    ).astype(int)

    # --------------------------------------------------------
    # Parse Genres
    # --------------------------------------------------------

    def parse_genres(value):

        if pd.isna(value):
            return []

        try:

            parsed = ast.literal_eval(
                str(value)
            )

            if isinstance(parsed, list):

                return [
                    item["name"]
                    for item in parsed
                    if isinstance(item, dict)
                    and "name" in item
                ]

        except (
            ValueError,
            SyntaxError,
            TypeError
        ):
            return []

        return []

    # Create genre list
    df["genre_list"] = df["genres"].apply(
        parse_genres
    )

    # Primary genre
    df["primary_genre"] = df[
        "genre_list"
    ].apply(
        lambda x:
        x[0] if len(x) > 0 else "Unknown"
    )

    return df


# Load dataset
df = load_data()


# ============================================================
# 8. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load Random Forest
    model = joblib.load(
        RANDOM_FOREST_PATH
    )

    # Load MultiLabelBinarizer
    mlb = joblib.load(
        MLB_PATH
    )

    # Load exact feature column order
    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    return (
        model,
        mlb,
        feature_columns
    )


# Load model artifacts
rf_model, mlb, feature_columns = load_model()


# ============================================================
# 9. SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🎬 MovieIQ Filters"
)


# ------------------------------------------------------------
# Genre Filter
# ------------------------------------------------------------

genre_options = sorted(
    df[
        "primary_genre"
    ]
    .dropna()
    .unique()
    .tolist()
)


selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + genre_options
)


# ------------------------------------------------------------
# Vote Average Filter
# ------------------------------------------------------------

min_vote = float(
    df["vote_average"].min()
)

max_vote = float(
    df["vote_average"].max()
)


selected_vote = st.sidebar.slider(
    "Minimum Vote Average",
    min_value=min_vote,
    max_value=max_vote,
    value=min_vote,
    step=0.1
)


# ------------------------------------------------------------
# Apply Filters
# ------------------------------------------------------------

filtered_df = df[
    df["vote_average"] >= selected_vote
].copy()


if selected_genre != "All":

    filtered_df = filtered_df[
        filtered_df["primary_genre"]
        == selected_genre
    ]


# Sidebar summary
st.sidebar.markdown("---")

st.sidebar.write(
    f"Showing **{len(filtered_df):,}** "
    f"of **{len(df):,}** movies."
)


# ============================================================
# 10. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset Summary",
        "📈 Exploratory Analysis",
        "🧪 Statistical Testing",
        "🔮 Movie Prediction"
    ]
)


# ============================================================
# TAB 1 - DATASET SUMMARY
# ============================================================

with tab1:

    st.header(
        "📊 Dataset Summary"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_movies = len(
        filtered_df
    )

    successful_movies = int(
        filtered_df["success"].sum()
    )

    failed_movies = (
        total_movies -
        successful_movies
    )

    if total_movies > 0:

        success_rate = (
            successful_movies /
            total_movies
        ) * 100

    else:

        success_rate = 0.0


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Movies",
        f"{total_movies:,}"
    )


    col2.metric(
        "Successful Movies",
        f"{successful_movies:,}"
    )


    col3.metric(
        "Failed Movies",
        f"{failed_movies:,}"
    )


    col4.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )


    st.markdown("---")


    # --------------------------------------------------------
    # Dataset Sample
    # --------------------------------------------------------

    st.subheader(
        "🎥 Movie Dataset"
    )


    display_columns = [
        "title",
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average",
        "primary_genre",
        "success"
    ]


    available_columns = [
        col
        for col in display_columns
        if col in filtered_df.columns
    ]


    st.dataframe(
        filtered_df[
            available_columns
        ].head(20),
        use_container_width=True
    )


    # --------------------------------------------------------
    # Success Definition
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🎯 Success Definition"
    )

    st.write(
        "A movie is classified as "
        "**Successful (1)** when Revenue > Budget."
    )

    st.write(
        "A movie is classified as "
        "**Failure (0)** when Revenue <= Budget."
    )


# ============================================================
# TAB 2 - EXPLORATORY DATA ANALYSIS
# ============================================================

with tab2:

    st.header(
        "📈 Exploratory Data Analysis"
    )


    if len(filtered_df) == 0:

        st.warning(
            "No movies match the selected filters."
        )

    else:

        # ====================================================
        # CHART 1 - BUDGET VS REVENUE
        # ====================================================

        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:

            st.subheader(
                "💰 Budget vs Revenue"
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            # Success/failure markers
            success_data = filtered_df[
                filtered_df["success"] == 1
            ]

            failure_data = filtered_df[
                filtered_df["success"] == 0
            ]


            ax.scatter(
                failure_data["budget"],
                failure_data["revenue"],
                alpha=0.5,
                s=25,
                label="Failure"
            )


            ax.scatter(
                success_data["budget"],
                success_data["revenue"],
                alpha=0.5,
                s=25,
                label="Success"
            )


            # Break-even line
            max_value = max(
                filtered_df["budget"].max(),
                filtered_df["revenue"].max()
            )


            ax.plot(
                [0, max_value],
                [0, max_value],
                "k--",
                linewidth=1,
                label="Break-even"
            )


            ax.set_xlabel(
                "Budget ($)"
            )

            ax.set_ylabel(
                "Revenue ($)"
            )

            ax.set_title(
                "Budget vs Revenue"
            )

            ax.legend()


            st.pyplot(fig)

            plt.close(fig)


        # ====================================================
        # CHART 2 - SUCCESS RATE BY GENRE
        # ====================================================

        with chart_col2:

            st.subheader(
                "🎭 Success Rate by Genre"
            )


            genre_rate = (
                filtered_df
                .groupby(
                    "primary_genre"
                )["success"]
                .mean()
                .sort_values(
                    ascending=False
                )
                * 100
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.barplot(
                x=genre_rate.values,
                y=genre_rate.index,
                ax=ax
            )


            ax.set_xlabel(
                "Success Rate (%)"
            )

            ax.set_ylabel(
                "Primary Genre"
            )

            ax.set_title(
                "Movie Success Rate by Genre"
            )


            st.pyplot(fig)

            plt.close(fig)


        st.markdown("---")


        # ====================================================
        # CHART 3 - VOTE AVERAGE VS SUCCESS
        # ====================================================

        chart_col3, chart_col4 = st.columns(2)


        with chart_col3:

            st.subheader(
                "⭐ Vote Average vs Success"
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.boxplot(
                data=filtered_df,
                x="success",
                y="vote_average",
                ax=ax
            )


            ax.set_xticks(
                [0, 1]
            )

            ax.set_xticklabels(
                [
                    "Failure",
                    "Success"
                ]
            )


            ax.set_xlabel(
                "Movie Result"
            )

            ax.set_ylabel(
                "Vote Average"
            )


            ax.set_title(
                "Vote Average by Movie Result"
            )


            st.pyplot(fig)

            plt.close(fig)


        # ====================================================
        # CHART 4 - CORRELATION HEATMAP
        # ====================================================

        with chart_col4:

            st.subheader(
                "🔥 Correlation Heatmap"
            )


            numeric_columns = [
                "budget",
                "revenue",
                "popularity",
                "runtime",
                "vote_average",
                "success"
            ]


            available_numeric = [
                col
                for col in numeric_columns
                if col in filtered_df.columns
            ]


            correlation = (
                filtered_df[
                    available_numeric
                ].corr()
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.heatmap(
                correlation,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax
            )


            ax.set_title(
                "Feature Correlation"
            )


            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# TAB 3 - STATISTICAL TESTING
# ============================================================

with tab3:

    st.header(
        "🧪 Statistical Hypothesis Testing"
    )


    # ========================================================
    # T-TEST
    # ========================================================

    st.subheader(
        "1. Independent T-Test"
    )


    st.write(
        "Testing whether popularity differs between "
        "successful and unsuccessful movies."
    )


    successful_popularity = (
        filtered_df[
            filtered_df["success"] == 1
        ]["popularity"]
        .dropna()
    )


    failed_popularity = (
        filtered_df[
            filtered_df["success"] == 0
        ]["popularity"]
        .dropna()
    )


    if (
        len(successful_popularity) > 1
        and
        len(failed_popularity) > 1
    ):

        t_statistic, t_pvalue = (
            stats.ttest_ind(
                successful_popularity,
                failed_popularity,
                equal_var=False
            )
        )


        st.write(
            "**H₀:** Mean popularity is the same "
            "for successful and unsuccessful movies."
        )


        st.write(
            "**H₁:** Mean popularity is different "
            "between successful and unsuccessful movies."
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "T-Statistic",
            f"{t_statistic:.4f}"
        )


        col2.metric(
            "P-Value",
            f"{t_pvalue:.6f}"
        )


        if t_pvalue < 0.05:

            st.success(
                "✅ Reject H₀: There is a statistically "
                "significant difference in popularity."
            )

        else:

            st.info(
                "ℹ️ Fail to Reject H₀: No statistically "
                "significant difference was detected."
            )


    else:

        st.warning(
            "Not enough successful and unsuccessful "
            "movies to perform the T-Test."
        )


    st.markdown("---")


    # ========================================================
    # CHI-SQUARE TEST
    # ========================================================

    st.subheader(
        "2. Chi-Square Test of Independence"
    )


    st.write(
        "Testing whether movie genre and success "
        "are statistically associated."
    )


    contingency_table = pd.crosstab(
        filtered_df["primary_genre"],
        filtered_df["success"]
    )


    if (
        contingency_table.shape[0] > 1
        and
        contingency_table.shape[1] > 1
    ):

        (
            chi2_stat,
            chi2_pvalue,
            degrees_of_freedom,
            expected
        ) = stats.chi2_contingency(
            contingency_table
        )


        st.write(
            "**H₀:** Genre and movie success "
            "are independent."
        )


        st.write(
            "**H₁:** Genre and movie success "
            "are associated."
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Chi-Square",
            f"{chi2_stat:.4f}"
        )


        col2.metric(
            "Degrees of Freedom",
            f"{degrees_of_freedom}"
        )


        col3.metric(
            "P-Value",
            f"{chi2_pvalue:.6f}"
        )


        if chi2_pvalue < 0.05:

            st.success(
                "✅ Reject H₀: Genre and movie success "
                "have a statistically significant association."
            )

        else:

            st.info(
                "ℹ️ Fail to Reject H₀: No statistically "
                "significant association was detected."
            )


        st.subheader(
            "Contingency Table"
        )


        st.dataframe(
            contingency_table,
            use_container_width=True
        )


    else:

        st.warning(
            "Not enough genre/success variation "
            "to perform the Chi-Square test."
        )


# ============================================================
# TAB 4 - MACHINE LEARNING PREDICTION
# ============================================================

with tab4:

    st.header(
        "🔮 Movie Success Prediction"
    )


    st.write(
        "Enter movie characteristics below and "
        "the trained Random Forest model will predict "
        "whether the movie is likely to be financially successful."
    )


    # ========================================================
    # INPUTS
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        input_budget = st.number_input(
            "Budget ($)",
            min_value=0.0,
            value=50_000_000.0,
            step=1_000_000.0
        )


        input_popularity = st.number_input(
            "Popularity",
            min_value=0.0,
            value=50.0,
            step=1.0
        )


        input_runtime = st.number_input(
            "Runtime (minutes)",
            min_value=0.0,
            value=120.0,
            step=5.0
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        input_vote = st.slider(
            "Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=6.5,
            step=0.1
        )


        # Get genres from trained ML encoder
        genre_options_prediction = sorted(
            [
                str(g)
                for g in mlb.classes_
            ]
        )


        # Default genres
        default_genres = []


        if "Action" in genre_options_prediction:
            default_genres.append("Action")


        if "Drama" in genre_options_prediction:
            default_genres.append("Drama")


        if len(default_genres) == 0:

            default_genres = (
                genre_options_prediction[:2]
            )


        input_genres = st.multiselect(
            "Select Genre(s)",
            options=genre_options_prediction,
            default=default_genres
        )


    st.markdown("---")


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Movie Success",
        type="primary"
    ):

        try:

            # ------------------------------------------------
            # Create empty feature dictionary
            # ------------------------------------------------

            input_dict = {
                column: 0
                for column in feature_columns
            }


            # ------------------------------------------------
            # Numerical Features
            # ------------------------------------------------

            if "budget" in input_dict:

                input_dict["budget"] = (
                    input_budget
                )


            if "popularity" in input_dict:

                input_dict["popularity"] = (
                    input_popularity
                )


            if "runtime" in input_dict:

                input_dict["runtime"] = (
                    input_runtime
                )


            if "vote_average" in input_dict:

                input_dict["vote_average"] = (
                    input_vote
                )


            # ------------------------------------------------
            # Genre Features
            #
            # IMPORTANT:
            # The training script saved genre columns
            # using mlb.classes_ directly.
            # Therefore we use the genre name itself.
            # ------------------------------------------------

            for genre in input_genres:

                if genre in input_dict:

                    input_dict[genre] = 1


            # ------------------------------------------------
            # Create DataFrame
            # ------------------------------------------------

            input_df = pd.DataFrame(
                [input_dict]
            )


            # Ensure exact feature order
            input_df = input_df[
                feature_columns
            ]


            # ------------------------------------------------
            # Model Prediction
            # ------------------------------------------------

            prediction = rf_model.predict(
                input_df
            )[0]


            probabilities = (
                rf_model.predict_proba(
                    input_df
                )[0]
            )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            # Find class positions safely
            class_list = list(
                rf_model.classes_
            )


            if 1 in class_list:

                success_index = (
                    class_list.index(1)
                )

                success_probability = (
                    probabilities[
                        success_index
                    ] * 100
                )

            else:

                success_probability = 0.0


            if 0 in class_list:

                failure_index = (
                    class_list.index(0)
                )

                failure_probability = (
                    probabilities[
                        failure_index
                    ] * 100
                )

            else:

                failure_probability = 0.0


            confidence = max(
                success_probability,
                failure_probability
            )


            # ------------------------------------------------
            # Display Prediction
            # ------------------------------------------------

            st.markdown("---")


            if prediction == 1:

                st.success(
                    "🎉 PREDICTION: "
                    "FINANCIAL SUCCESS"
                )


                result_col1, result_col2 = (
                    st.columns(2)
                )


                result_col1.metric(
                    "Success Probability",
                    f"{success_probability:.1f}%"
                )


                result_col2.metric(
                    "Prediction Confidence",
                    f"{confidence:.1f}%"
                )


                st.write(
                    "The model predicts that "
                    "Revenue is likely to exceed Budget."
                )


            else:

                st.error(
                    "⚠️ PREDICTION: "
                    "FINANCIAL FAILURE / HIGH RISK"
                )


                result_col1, result_col2 = (
                    st.columns(2)
                )


                result_col1.metric(
                    "Failure Probability",
                    f"{failure_probability:.1f}%"
                )


                result_col2.metric(
                    "Prediction Confidence",
                    f"{confidence:.1f}%"
                )


                st.write(
                    "The model predicts that "
                    "Revenue is likely to be less than "
                    "or equal to Budget."
                )


            # ------------------------------------------------
            # Probability Chart
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Probability"
            )


            probability_df = pd.DataFrame(
                {
                    "Outcome": [
                        "Failure",
                        "Success"
                    ],
                    "Probability": [
                        failure_probability,
                        success_probability
                    ]
                }
            )


            st.bar_chart(
                probability_df.set_index(
                    "Outcome"
                )
            )


            st.caption(
                "Model confidence indicates how strongly "
                "the Random Forest model favors the predicted "
                "class. It is not a guarantee of actual movie performance."
            )


        except Exception as error:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.exception(error)


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎬 MovieIQ | Predictive Analytics on Film Success | "
    "Python • Pandas • Scikit-learn • Random Forest • Streamlit"
)
