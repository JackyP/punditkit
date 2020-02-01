import streamlit as st
import inspect
import sys
import pandas as pd
import numpy as np
import pickle
import base64
from math import log10, floor
from pandas.api.types import is_numeric_dtype
from backend import PunditKitEstimator, all_regressors, all_classifiers

# Contents
# - Read data
# - Select model
# - Fit model
# - Save model
# - Evaluate predictions
# - Column summary
# - Holdout performance
# - Variable importance
# - Partial dependence

st.sidebar.title("PunditKit")
st.sidebar.markdown("##### Simplify. Visualise. Learn.")

# Hide footer
# Hide button
# Make wider because we have many plots
st.markdown(
    """
        <style>
        .main > footer:nth-child(2) {
            visibility: hidden;
        }
        #MainMenuButton {
            visibility: hidden;
        }
        .reportview-container .main .block-container {
            max-width: 1000px;
        }

        </style>
    """,
    unsafe_allow_html=True,
)

# Read data
@st.cache
def get_data(filename_or_url):
    # Set low_memory = False to avoid mixed dtype errors at the cost of memory.
    df = pd.read_csv(filename_or_url, low_memory=False)
    return df


if len(sys.argv) >= 2:
    data_location = sys.argv[1]
else:
    data_location = st.file_uploader("Choose a CSV file", type="csv")

if data_location is not None:
    df = get_data(data_location)
    st.sidebar.markdown(f"Current file: {data_location}")

df_cols = df.columns.tolist()

response = st.sidebar.selectbox(
    "Choose response", options=df_cols, index=len(df_cols) - 1
)
features = st.sidebar.multiselect("Choose features", df_cols, df_cols[:-1])

numerical_columns = [x for x in features if is_numeric_dtype(df[x])]
categorical_columns = [x for x in features if not is_numeric_dtype(df[x])]

# Reorder to be numerical then categorical to preserve column order in
# the transformers in scikit pipeline

features = numerical_columns + categorical_columns

if not features:
    st.sidebar.error("Error: Please select at least one feature.")

if not response:
    st.sidebar.error("Error: Please select a response.")

if response in features:
    st.sidebar.error("Error: Response is in list of features.")

if df[response].isnull().values.any():
    st.sidebar.warning(
        "Warning: Response contains NaN values. These records will be dropped in both model training and test validation metrics."
    )

# Select Model
@st.cache
def get_models(df, is_numeric):
    if is_numeric:
        return (
            [x[0] for x in all_regressors],
            [x[1] for x in all_regressors],
            [x[0] for x in all_regressors].index("DecisionTreeRegressor"),
        )
    else:
        return (
            [x[0] for x in all_classifiers],
            [x[1] for x in all_classifiers],
            [x[0] for x in all_classifiers].index("DecisionTreeClassifier"),
        )


model_names, models, model_default_ind = get_models(df, is_numeric_dtype(df[response]))

# Default to DecisionTree because it estimates fairly quickly compared to other models
selected_model_name = st.sidebar.selectbox(
    "Choose model", options=model_names, index=model_default_ind
)

SelectedModel = models[model_names.index(selected_model_name)]


st.write(f"# {selected_model_name}")

model_url_str = str(SelectedModel).split("'")[1].split(".")

st.sidebar.markdown(
    "[About this model](https://scikit-learn.org/stable/modules/generated/{}.html)".format(
        ".".join([model_url_str[0], model_url_str[1], model_url_str[-1]])
    )
)

hyperparam_select = st.sidebar.checkbox("Set hyperparameters")

hyperparam = {}
if hyperparam_select:
    signature = inspect.signature(SelectedModel.__init__)
    for k, v in signature.parameters.items():
        if k != "self":
            if type(v.default) == str:
                val = st.sidebar.text_input(k, value=v.default)
            elif type(v.default) == int:
                val = st.sidebar.number_input(k, value=v.default)
            elif type(v.default) == float:
                val = st.sidebar.number_input(k, value=v.default)
            else:
                val = st.sidebar.text_input(k, value="None")

            if val == "None":
                hyperparam[k] = None
            elif type(val) == str:
                try:
                    hyperparam[k] = int(val)
                except ValueError:
                    try:
                        hyperparam[k] = float(val)
                    except ValueError:
                        hyperparam[k] = val
            else:
                hyperparam[k] = val


# Fit model - TODO: cache and load class
pdk = PunditKitEstimator(
    Estimator=SelectedModel,
    numeric_features=numerical_columns,
    categorical_features=categorical_columns,
    response=response,
    kwargs=hyperparam,
)

pdk.fit_data(df)


# Save model
# @st.cache
@st.cache
def save_model_base64(pipe):
    s = pickle.dumps(pipe)
    b64 = base64.b64encode(s).decode()
    href = f'<a download="model.pickle" href="data:file/pickle;base64,{b64}">Download Model as Pickle</a>'
    return href


href = save_model_base64(pdk.pipe)

st.markdown(href, unsafe_allow_html=True)

# Evaluate predictions
st.write(
    """## Predictions
Make and explain predictions with the trained model.
"""
)

# - Build form
st.write(
    """### Input:
Select feature values for the model to predict.
"""
)
list_form = []
default_form = []
for col in features:
    if col in numerical_columns:
        mn = df[col].min().item()
        mx = df[col].max().item()
        defv = df[col].mean()

        if mx > mn:
            sp = (mx - mn) / 100
            sp = round(sp, 1 - int(floor(log10(abs(sp)))) - 1)

            if df[col].dtype == np.int64:
                defv = np.ceil(defv).astype(int).item()
                sp = np.ceil(sp).astype(int).item()

            val = st.slider(
                "{} ({})".format(col, df[col].dtype),
                min_value=mn,
                max_value=mx,
                step=sp,
                value=defv,
            )
        else:
            val = mx

    else:
        defv = df[col].mode().item()
        uniq = pd.unique(df[col]).tolist()
        val = st.selectbox(
            "{} ({})".format(col, df[col].dtype), options=uniq, index=uniq.index(defv)
        )
    list_form += [val]
    default_form += [defv]

df_form = pd.DataFrame([list_form], columns=features)
df_default = pd.DataFrame([default_form], columns=features)

if is_numeric_dtype(df[response]):
    st.write("### Predicted Value:", pdk.pipe.predict(df_form).item())
else:
    st.write(
        "### Predicted Probabilities:",
        pd.DataFrame(
            pdk.pipe.predict_proba(df_form), columns=pdk.pipe.classes_.tolist()
        ),
    )

st.write("### Explanation:")
st.markdown(
    "Explanations use [LIME](https://github.com/marcotcr/lime) to derive a linear interpretation of the prediction."
)

if len(features) > 1:
    num_important = st.slider(
        "Number of features to display:",
        min_value=1,
        max_value=len(features),
        value=min(4, len(features)),
    )
else:
    num_important = 1

# - Lime form result
@st.cache
def cached_explanation(df_form, num_important, pipe):
    return pdk.predict_explanation(df_form, num_features=num_important)


cached_explanation(df_form, num_important, pdk.pipe)

st.pyplot()

st.markdown("--------")
# Exploratory Data Analysis - Column Info
st.write("## Data Description")
st.markdown(
    "A summary of the features is included below for a reasonableness review of the input data."
)
st.table((df.describe(include="all").T).loc[:, ["unique", "top", "mean", "min", "max"]])

has_na_features = [f for f in features if df[f].isnull().values.any()]

if len(has_na_features) > 0:
    st.warning(
        "Warning: {} have missing values. Impute with mean for numeric columns and 'missing' for categorical columns.".format(
            str(has_na_features)
        )
    )
st.write("## Data Sample")

# Limit features to fit on the poge and rows for compute time
sample_row_count = st.slider(
    label="Sample size:",
    min_value=1,
    max_value=len(df.index),
    value=min(100, len(df.index)),
)
df_sample = df.sample(n=sample_row_count, random_state=42)
st.dataframe(df_sample)

st.markdown(
    """To quickly generate results, only a sample of the data is displayed for
the scatter and partial dependence plots in the later sections."""
)

st.markdown("--------")


@st.cache
def cached_prediction_error_plot(X_train, X_test, y_test, pipe):
    return pdk.prediction_error_plot()


@st.cache
def cached_confusion_matrix_plot(X_train, X_test, y_test, pipe):
    return pdk.confusion_matrix_plot()


@st.cache
def cached_standard_error(X_train, X_test, y_test, pipe):
    return np.std(pdk.pipe.predict(pdk.X_test) - pdk.y_test, ddof=1)


@st.cache
def cached_coefficient_of_variation(X_train, X_test, y_test, pipe):
    return np.std(pdk.pipe.predict(pdk.X_test) - pdk.y_test, ddof=1) / np.mean(
        pdk.y_test
    )


# Performance on holdout data
if is_numeric_dtype(df[response]):
    st.write("## Prediction Error - (Test Set)")
    st.markdown(
        """Model performance on holdout test data is presented below
with a [Prediction Error Plot](https://www.scikit-yb.org/en/latest/api/regressor/peplot.html)
comparing actual vs predicted values."""
    )
    cached_prediction_error_plot(pdk.X_train, pdk.X_test, pdk.y_test, pdk.pipe)

    st.write(
        "Standard error: ",
        cached_standard_error(pdk.X_train, pdk.X_test, pdk.y_test, pdk.pipe),
    )
    st.pyplot()
    st.write(
        "Coefficient of variation: ",
        cached_coefficient_of_variation(pdk.X_train, pdk.X_test, pdk.y_test, pdk.pipe),
    )
else:
    st.write("## Confusion Matrix (Test Set)")
    st.markdown(
        """Model performance on holdout test data is summarised below with
a [Confusion Matrix](https://www.scikit-yb.org/en/latest/api/classifier/confusion_matrix.html)
comparing actual vs predicted values. Values along the diagonal indicate a correct prediction.
"""
    )
    cached_confusion_matrix_plot(pdk.X_train, pdk.X_test, pdk.y_test, pdk.pipe)

    st.pyplot()

st.markdown("--------")
st.write("## Permutation Feature Importance (Train Set)")

st.markdown(
    """[Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
is a measure of the contribution of a feature to the model predictions."""
)


@st.cache
def cached_feature_importance_plot(X_train, X_test, y_test, pipe):
    return pdk.feature_importance_plot()


cached_feature_importance_plot(pdk.X_train, pdk.X_test, pdk.y_test, pdk.pipe)
st.pyplot()

# Key variables pairplot.
# Seaborn pair grid
st.markdown("--------")
st.markdown("## Pair Grid")
st.markdown(
    """The grid below shows:
 - Upper triangle: Two way scatter plot of values and decision boundaries,
 - Diagonal: Histogram and distribution of values by feature,
 - Lower triangle: Left blank.
"""
)

pdk.classifier_pair_plot(df=df, num_features=num_important)
st.pyplot()

st.markdown("--------")
st.write("## Model Predictions One-Way Plots")


@st.cache
def cached_top_features(X_train, X_test, y_test, pipe):
    return pdk.top_features


pdp_feature_1 = st.selectbox("Plot Feature", options=features, index=0)

pdk.pdp_plot(pdp_feature_1)
st.pyplot()
