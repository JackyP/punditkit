import streamlit as st
import matplotlib.pyplot as plt
import sys

# Data
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.utils.testing import all_estimators

# Diagnostics
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance, plot_partial_dependence

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

data_location = sys.argv[1]

# 1. Read data
@st.cache
def get_data(filename_or_url):
    df = pd.read_csv(filename_or_url)
    return df


df = get_data(data_location)

df_cols = df.columns.tolist()

features = st.sidebar.multiselect("Choose features", df_cols, df_cols[:-1])

response = st.sidebar.selectbox(
    "Choose response", options=df_cols, index=len(df_cols) - 1
)

if not features:
    st.error("Please select at least one feature.")

if not response:
    st.error("Please select a response.")


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

categorical_columns = [x for x in features if not is_numeric_dtype(df[x])]
numerical_columns = [x for x in features if is_numeric_dtype(df[x])]

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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_pipe, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ]
    )

    pipe = Pipeline([("preprocess", preprocessing), ("classifier", SelectedModel())])

    pipe.fit(X_train, y_train)
    return pipe


pipe = make_fit_pipe(
    categorical_columns, numerical_columns, X_train, y_train, SelectedModel
)

# 5. Holdout data
y_test_predicted = pipe.predict(X_test)

if is_numeric_dtype(df[response]):
    st.write("### Prediction Error (Test Set)")
else:
    confused = confusion_matrix(y_test, y_test_predicted)

    st.write("### Confusion Matrix (Test Set)", confused)

# 6. Variable Importance
perm_imp = permutation_importance(
    pipe, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

sorted_idx = perm_imp.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(
    perm_imp.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
)
st.write("### Permutation Importances (Train set)")

fig.tight_layout()

st.pyplot()

# 7. Partial Dependence
st.write("### Model Predictions Plots")

pdp_feature_1 = st.sidebar.selectbox("Plot Feature", options=features, index=0)

if is_numeric_dtype(df[response]) | (df[response].nunique() <= 2):
    plot_partial_dependence(pipe, X_train, [pdp_feature_1])
else:
    for x in pd.unique(df[response]).tolist():
        plot_partial_dependence(pipe, X_train, [pdp_feature_1], target=x)

fig.tight_layout()
st.pyplot()
