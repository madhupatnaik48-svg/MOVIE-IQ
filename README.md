# 🎬 MovieIQ – Predictive Analytics on Film Success

## 🚀 Live Dashboard

👉 **[Open the MovieIQ Streamlit Dashboard](https://movie-iq-sutynbdybkllnix7wrt6re.streamlit.app/)**

---

## 📌 Project Overview

**MovieIQ** is an interactive machine learning and predictive analytics project designed to analyze movie financial performance and predict whether a movie is likely to be financially successful.

The project defines movie success using the following business rule:

```text
Success = 1 → Revenue > Budget
Success = 0 → Revenue ≤ Budget
```

The project combines:

* Data cleaning
* Exploratory Data Analysis
* Statistical hypothesis testing
* Feature engineering
* Random Forest classification
* Interactive Streamlit visualization
* Movie success probability prediction

---

## 🎯 Business Objective

The objective of MovieIQ is to provide a data-driven analytical tool for understanding movie financial performance and estimating the likelihood of financial success based on movie characteristics.

The dashboard helps users explore:

* Budget and revenue relationships
* Movie success rates
* Genre-level success patterns
* Popularity differences between success groups
* Relationship between genre and success
* Machine learning-based success predictions

---

## 📊 Dashboard Features

### 1. Dataset Summary

The dashboard provides:

* Total Movies
* Successful Movies
* Failed/High-Risk Movies
* Overall Success Rate
* Cleaned Dataset Sample

---

### 2. Interactive EDA

MovieIQ provides interactive visual analysis including:

#### 💰 Budget vs Revenue

Compares movie budget with revenue and displays a break-even reference line.

#### 🎭 Genre Success Rate

Analyzes financial success rates across primary movie genres.

#### 🔥 Correlation Heatmap

Examines relationships between:

* Budget
* Revenue
* Popularity
* Runtime
* Vote Average
* Success

---

### 3. Statistical Testing

MovieIQ includes two statistical tests.

#### Independent T-Test

The test investigates whether movie popularity differs between successful and unsuccessful movies.

**Null Hypothesis (H₀):**

The mean popularity of successful movies is equal to that of unsuccessful movies.

**Alternative Hypothesis (H₁):**

The mean popularity differs between the two groups.

#### Chi-Square Test

The test investigates whether primary movie genre and financial success are associated.

**Null Hypothesis (H₀):**

Primary Genre and Movie Success are independent.

**Alternative Hypothesis (H₁):**

Primary Genre and Movie Success are associated.

---

## 🤖 Machine Learning

MovieIQ uses a:

### Random Forest Classifier

The model uses the following movie attributes:

* Budget
* Popularity
* Runtime
* Vote Average
* Genre features

Movie genres are transformed using:

```text
MultiLabelBinarizer
```

The trained model is stored using:

```text
Joblib
```

---

## 🔮 Movie Success Predictor

The dashboard allows users to enter prospective movie characteristics such as:

* Budget
* Popularity Score
* Runtime
* Vote Average
* Selected Genre(s)

The model then returns:

### Prediction

```text
Class 1 → Financial Success
Class 0 → Financial Failure / High Risk
```

It also provides:

* Model confidence
* Probability of financial success
* Probability of financial failure

---

## 🗂️ Project Structure

```text
MOVIE-IQ/
│
├── MovieIQ.py
├── movies.csv
├── movies_cleaned.csv
├── requirements.txt
├── README.md
│
├── assets/
│   ├── budget_vs_revenue.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── genre_success.png
│   ├── most_common_genres.png
│   ├── popularity_vs_success.png
│   ├── runtime_vs_success.png
│   ├── success_distribution.png
│   ├── train_test_accuracy.png
│   └── vote_average_vs_success.png
│
└── models/
    ├── random_forest.pkl
    ├── mlb_encoder.pkl
    └── feature_columns.pkl
```

---

## 🧹 Data Preparation

The dataset is cleaned by:

1. Removing duplicate records.
2. Removing movies with invalid or zero budget.
3. Removing movies with invalid or zero revenue.
4. Creating the binary `success` target variable.
5. Parsing movie genre information.
6. Transforming genres into machine-learning features.

---

## 🧠 Machine Learning Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Genre Encoding
      ↓
Train/Test Split
      ↓
Random Forest Classifier
      ↓
Model Prediction
      ↓
Success Probability
      ↓
Streamlit Dashboard
```

---

## 🛠️ Technology Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming               |
| Pandas       | Data manipulation         |
| NumPy        | Numerical computing       |
| Matplotlib   | Visualization             |
| Seaborn      | Statistical visualization |
| SciPy        | Statistical testing       |
| Scikit-learn | Machine learning          |
| Joblib       | Model serialization       |
| Streamlit    | Interactive dashboard     |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/madhupatnaik48-svg/MOVIE-IQ.git
```

Move into the project folder:

```bash
cd MOVIE-IQ
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard Locally

Run:

```bash
streamlit run MovieIQ.py
```

The application will open in your browser.

---

## ☁️ Deployment

MovieIQ is deployed using **Streamlit Community Cloud**.

### Live Application

👉 **https://movie-iq-sutynbdybkllnix7wrt6re.streamlit.app/**

---

## 📈 Business Interpretation

MovieIQ is designed as a decision-support and analytical application.

It can help users investigate:

* Financial performance patterns
* Movie genre performance
* Audience popularity patterns
* Relationships between movie characteristics
* Estimated probability of financial success

The machine learning prediction is an analytical estimate and should not be interpreted as a guaranteed financial outcome.

---

## ⚠️ Limitations

The current model is based on the features available in the project dataset.

Movie financial performance can also depend on factors that may not be represented in the current model, including:

* Marketing expenditure
* Distribution strategy
* Competition
* Release timing
* Audience preferences
* Star power
* Economic conditions
* Critical reception

Therefore, the MovieIQ prediction should be used as **analytical decision support**, not as a guarantee of profitability.

---

## 🚀 Future Enhancements

Future versions of MovieIQ may include:

* XGBoost
* Gradient Boosting
* Logistic Regression
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Advanced feature importance
* Revenue prediction
* Profit prediction
* Time-series analysis
* Release-year analysis
* Interactive model performance dashboard
* Additional movie and market features

---

## 👩‍💻 Project

**Project Name:** MovieIQ
**Domain:** Entertainment Analytics / Data Science
**Application:** Predictive Analytics on Film Success
**Model:** Random Forest Classifier
**Dashboard:** Streamlit

---

## ⭐ Live Project

🎬 **Try MovieIQ here:**

👉 **https://movie-iq-sutynbdybkllnix7wrt6re.streamlit.app/**

If you find this project useful, consider giving the GitHub repository a ⭐.
