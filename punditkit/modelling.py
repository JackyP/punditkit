import streamlit as st
import matplotlib.pyplot as plt
import sys

# Data
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.utils import all_estimators

# Diagnostics
from sklearn.inspection import permutation_importance
from pdpbox import pdp
from yellowbrick.regressor import PredictionError
from yellowbrick.classifier.confusion_matrix import ConfusionMatrix

regressors = all_estimators(type_filter="regressor")
classifiers = all_estimators(type_filter="classifier")

# Contents
# 1. Read data
# 2. Select model
# 3. Exploratory data analysis
# 4. Fit model
# 5. Holdout data
# 6. Variable Importance
# 7. Partial Dependency Plot

st.sidebar.title("PunditKit: Visual Interface to Scikit-learn")

# Hide footer
st.markdown(
    """
        <style>
        .main > footer:nth-child(2) {
            visibility: hidden;
        }
        #MainMenuButton {
            visibility: hidden;
        }
        </style>
    """,
    unsafe_allow_html=True,
)
data_location = sys.argv[1]

# 1. Read data
@st.cache
def get_data(filename_or_url):
    df = pd.read_csv(filename_or_url)
    return df


df = get_data(data_location)

df_cols = df.columns.tolist()

response = st.sidebar.selectbox(
    "Choose response", options=df_cols, index=len(df_cols) - 1
)
features = st.sidebar.multiselect("Choose features", df_cols, df_cols[:-1])

if not features:
    st.error("Please select at least one feature.")

if not response:
    st.error("Please select a response.")

if response in features:
    st.warning("Response is in list of features.")


# 2. Select Model
def get_models(df, response):
    if is_numeric_dtype(df[response]):
        return regressors
    else:
        return classifiers


models = get_models(df, response)
model_names = [x[0] for x in models]

selected_model_name = st.sidebar.selectbox("Choose model", options=model_names)

SelectedModel = models[model_names.index(selected_model_name)][1]

st.write(f"# {selected_model_name}")

# 3. Exploratory Data Analysis
# TODO: More detailed exploratory analysis
show_exploratory = st.checkbox("Show exploratory data analysis")

categorical_columns = [x for x in features if not is_numeric_dtype(df[x])]
numerical_columns = [x for x in features if is_numeric_dtype(df[x])]

if show_exploratory:
    st.write("### Exploratory Data Analysis", df.describe(include="all").T)

X = df[features]
y = df[response]

# 4. Fit model
@st.cache
def tts(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)


test_size = st.sidebar.slider("Test Split %", 0.0, 1.0, step=0.01, value=0.30)

X_train, X_test, y_train, y_test = tts(X, y, test_size)


@st.cache
def make_fit_pipe(
    categorical_columns, numerical_columns, X_train, y_train, SelectedModel
):
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_pipe, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ]
    )

    pipe = Pipeline([("preprocess", preprocessing), ("estimator", SelectedModel())])

    pipe.fit(X_train, y_train)
    return pipe


pipe = make_fit_pipe(
    categorical_columns, numerical_columns, X_train, y_train, SelectedModel
)

# 5. Holdout data
if is_numeric_dtype(df[response]):
    st.write("### Prediction Error (Test Set)")
    visualizer = PredictionError(pipe)
else:
    st.write("### Confusion Matrix (Test Set)")
    visualizer = ConfusionMatrix(pipe)


visualizer.score(X_test, y_test)
visualizer.show()
st.pyplot()

# 6. Variable Importance
@st.cache
def importance(pipe, X_train, y_train):
    perm_imp = permutation_importance(
        pipe, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_idx = perm_imp.importances_mean.argsort()
    return perm_imp, sorted_idx


perm_imp, sorted_idx = importance(pipe, X_train, y_train)

fig, ax = plt.subplots()
ax.boxplot(
    perm_imp.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
)
st.write("### Permutation Importances (Train set)")

fig.tight_layout()

st.pyplot()

# 7. Partial Dependence
st.write("### Model Predictions Plots")

pdp_feature_1 = st.selectbox("Plot Feature", options=features, index=0)
# Todo: Target vs Actual, Two-way plots

if is_numeric_dtype(df[pdp_feature_1]):
    pdp_isol = pdp.pdp_isolate(
        model=pipe, dataset=X, model_features=features, feature=pdp_feature_1
    )

    fig, axes = pdp.pdp_plot(
        pdp_isol, pdp_feature_1, plot_lines=True, frac_to_plot=100, ncols=1
    )
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.margins(0.01, 0.01)
    fig.tight_layout()
    st.pyplot()
else:
    # scikit learn pipeline does not produce named columns
    # however pdpbox requires them. So make own pdp plot.
    size = min(1000, len(X.index))
    df_pdp_other = X.sample(n=size).drop(columns=pdp_feature_1).assign(dummy=1)
    df_pdp_feat = X[[pdp_feature_1]].assign(dummy=1)

    df_pdp = (
        df_pdp_other.merge(df_pdp_feat, on="dummy")[features]
        .assign(predict=lambda df: pipe.predict(df))
        .loc[:, [pdp_feature_1, "predict"]]
        .groupby(pdp_feature_1)
        .agg("mean")
        .reset_index()
    )
    df_pdp.plot(
        title=f'PDP for feature "{pdp_feature_1}"',
        x=pdp_feature_1,
        y="predict",
        kind="bar",
    )
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    plt.margins(0.1, 0.1)
    fig.tight_layout()
    st.pyplot()
