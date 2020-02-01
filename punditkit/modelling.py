import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
import sys
import pickle
import base64

# Data
import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from math import log10, floor

# Explanations
import lime
import lime.lime_tabular

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
from mlxtend.plotting import plot_decision_regions

regressors = all_estimators(type_filter="regressor")
classifiers = all_estimators(type_filter="classifier")

# Drop ensemble, dummy estimators
excluded_regressors = [
    "DummyRegressor",
    "ExtraTreeRegressor",
    "MultiOutputRegressor",
    "RegressorChain",
    "StackingRegressor",
    "TransformedTargetRegressor",
    "VotingRegressor",
    "_SigmoidCalibration",
]
excluded_classifiers = [
    "BaggingClassifier",
    "CalibratedClassifierCV",
    "CheckingClassifier",
    "ClassifierChain",
    "DummyClassifier",
    "ExtraTreeClassifier",
    "MultiOutputClassifier",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "StackingClassifier",
    "VotingClassifier",
]

regressors = [x for x in regressors if x[0] not in excluded_regressors]
classifiers = [x for x in classifiers if x[0] not in excluded_classifiers]


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

if not features:
    st.sidebar.error("Error: Please select at least one feature.")

if not response:
    st.sidebar.error("Error: Please select a response.")

if response in features:
    st.sidebar.error("Error: Response is in list of features.")

if df[response].isnull().values.any():
    st.sidebar.warning(
        "Warning: Response contains NaN values. These records will be dropped in model training."
    )

# Select Model
@st.cache
def get_models(df, is_numeric):
    if is_numeric:
        return (
            [x[0] for x in regressors],
            [x[1] for x in regressors],
            [x[0] for x in regressors].index("DecisionTreeRegressor"),
        )
    else:
        return (
            [x[0] for x in classifiers],
            [x[1] for x in classifiers],
            [x[0] for x in classifiers].index("DecisionTreeClassifier"),
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

# Fit model
@st.cache
def tts(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)


test_size = st.sidebar.slider("Test Split %", 0.0, 1.0, step=0.01, value=0.30)

X = df[features]
y = df[response]

X_train, X_test, y_train, y_test = tts(X, y, test_size)

# Drop NaN in response.
X_train = X_train[~pd.isnull(y_train)]
y_train = y_train[~pd.isnull(y_train)]


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
            ("cat", categorical_pipe, [features.index(x) for x in categorical_columns]),
            ("num", numerical_pipe, [features.index(x) for x in numerical_columns]),
        ]
    )

    pipe = Pipeline([("preprocess", preprocessing), ("estimator", SelectedModel())])

    pipe.fit(X_train, y_train)
    return pipe


numerical_columns = [x for x in features if is_numeric_dtype(df[x])]
categorical_columns = [x for x in features if not is_numeric_dtype(df[x])]

pipe = make_fit_pipe(
    categorical_columns, numerical_columns, X_train, y_train, SelectedModel
)

# Save model
@st.cache
def save_model_base64(pipe):
    s = pickle.dumps(pipe)
    b64 = base64.b64encode(s).decode()
    href = f'<a download="model.pickle" href="data:file/pickle;base64,{b64}">Download Model as Pickle</a>'
    return href


href = save_model_base64(pipe)

st.markdown(href, unsafe_allow_html=True)

# Evaluate predictions
st.write("## Predictions")

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
        mn = df[col].min()
        mx = df[col].max()
        defv = df[col].mean()
        if mx > mn:
            sp = (mx - mn) / 100
            sp = round(sp, 1 - int(floor(log10(abs(sp)))) - 1)

            val = st.slider(col, min_value=mn, max_value=mx, step=sp, value=defv)
        else:
            val = mx

    else:
        defv = df[col].mode()
        val = st.selectbox(col, options=pd.unique(df[col]), value=defv)
    list_form += [val]
    default_form += [defv]

df_form = pd.DataFrame([list_form], columns=features)
df_default = pd.DataFrame([default_form], columns=features)
df_default_no_col = pd.DataFrame([default_form])

if is_numeric_dtype(df[response]):
    st.write("### Predicted Value:", pipe.predict(df_form).item())
else:
    st.write(
        "### Predicted Probabilities:",
        pd.DataFrame(pipe.predict_proba(df_form), columns=pipe.classes_.tolist()),
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
if is_numeric_dtype(df[response]):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=features,
        class_names=[response],
        discretize_continuous=True,
        categorical_features=categorical_columns,
        mode="regression",
    )

    exp = explainer.explain_instance(
        df_form.values[0], pipe.predict, num_features=num_important
    )
    fig = exp.as_pyplot_figure()
else:
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=features,
        class_names=pipe.classes_.tolist(),
        discretize_continuous=True,
        categorical_features=categorical_columns,
    )

    exp = explainer.explain_instance(
        df_form.values[0], pipe.predict_proba, num_features=num_important, top_labels=1
    )

    fig = exp.as_pyplot_figure(
        label=pipe.classes_.tolist().index(pipe.predict(df_form)[0])
    )

fig.tight_layout()
st.pyplot()

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

# Performance on holdout data
if is_numeric_dtype(df[response]):
    st.write("## Prediction Error - (Test Set)")
    st.markdown(
        """Model performance on holdout test data is presented below
with a [Prediction Error Plot](https://www.scikit-yb.org/en/latest/api/regressor/peplot.html)
comparing actual vs predicted values."""
    )
    visualizer = PredictionError(pipe)
else:
    st.write("## Confusion Matrix (Test Set)")
    st.markdown(
        """Model performance on holdout test data is summarised below with
a [Confusion Matrix](https://www.scikit-yb.org/en/latest/api/classifier/confusion_matrix.html)
comparing actual vs predicted values. Values along the diagonal indicate a correct prediction.
"""
    )
    visualizer = ConfusionMatrix(pipe)


visualizer.score(X_test, y_test)
visualizer.show()


st.pyplot()


# Feature Importance
@st.cache
def importance(pipe, X_train, y_train):
    perm_imp = permutation_importance(
        pipe, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_idx = perm_imp.importances_mean.argsort()
    return perm_imp, sorted_idx


perm_imp, sorted_idx = importance(pipe, X_train, y_train)

st.write("## Permutation Feature Importance (Train Set)")

fig, ax = plt.subplots()
ax.boxplot(
    perm_imp.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
)

fig.tight_layout()

st.markdown(
    """[Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
is a measure of the contribution of a feature to the model predictions."""
)

st.pyplot()

# Key variables pairplot.
# Seaborn pair grid
st.markdown("## Pair Grid")
st.markdown(
    """The grid below shows:
 - Upper triangle: Two way scatter plot of values and decision boundaries,
 - Diagonal: Histogram and distribution of values by feature,
 - Lower triangle: Left blank.
"""
)


df_sample[selected_model_name] = pipe.predict(df_sample[features])

top_features = X_train.columns[sorted_idx].tolist()
top_features.reverse()
top_features_n = top_features[:num_important]


def classifier_pair_plot(df, pipe, top_features, features, response):
    """ Returns a pair plot of top features for a classifier
    """
    dim_features = len(top_features)
    mpl.rcParams.update(mpl.rcParamsDefault)
    # df_not_na = df[top_features + [response]].dropna()
    is_numeric = is_numeric_dtype(df[response])

    if is_numeric:
        pal0 = sns.color_palette("GnBu_d", 10)
        pal1 = sns.color_palette("GnBu_r", 10)
    else:
        pal0 = sns.color_palette("muted", df[response].nunique())
        pal1 = sns.color_palette("bright", df[response].nunique())
    # g = sns.PairGrid(df_not_na, vars=top_features, hue=response, palette=pal)

    gs = gridspec.GridSpec(dim_features, dim_features)
    fig = plt.figure()

    # Dummy classifier
    class clf:
        model = pipe
        is_num = is_numeric

        def predict(self, X):
            y = self.model.predict(X)
            if self.is_num:
                return pd.qcut(y, 10, labels=False, duplicates="drop")
            else:
                return np.searchsorted(pipe.classes_, y)

    clf0 = clf()

    if is_numeric:
        resp_plt = pd.qcut(df[response], 10, labels=False, duplicates="drop").values
    else:
        resp_plt = df[response].astype("category").cat.codes.values

    # Diag
    lims = []
    for row_i in range(0, dim_features):
        ax = plt.subplot(gs[row_i, row_i])
        f_ind_i = features.index(top_features[row_i])
        # plt.hist(x=df.iloc[:, f_ind_i])
        ax = sns.distplot(df.iloc[:, f_ind_i])
        plt.ylabel("Count")
        plt.xlabel(features[f_ind_i])
        lims += [plt.xlim()]
        # plt.xlim((df.iloc[:, f_ind_i].min(), df.iloc[:, f_ind_i].max()))

    # Upper:
    indices = zip(*np.triu_indices(dim_features, k=1))

    # indices = zip(*np.triu_indices_from(g.axes, 1))
    for row_i, col_j in indices:

        # ax = g.axes[row_i, col_j]
        plt.subplot(gs[row_i, col_j])
        f_ind_i = features.index(top_features[row_i.item()])
        f_ind_j = features.index(top_features[col_j.item()])
        filler_ind = [
            x for x in list(range(0, dim_features)) if x not in [f_ind_i, f_ind_j]
        ]
        filler = df_default_no_col[filler_ind].to_dict("records")[0]
        plot_decision_regions(
            X=df[features].values,
            y=resp_plt,
            feature_index=(f_ind_j, f_ind_i),
            filler_feature_values=filler,
            clf=clf0,
            colors=",".join(pal0.as_hex()),
        )
        if len(filler_ind) > 0:
            plt.scatter(
                x=df.iloc[:, f_ind_j],
                y=df.iloc[:, f_ind_i],
                c=resp_plt,
                cmap=ListedColormap(pal1.as_hex()),
            )
            plt.xlabel(features[f_ind_j])
            plt.ylabel(features[f_ind_i])
            plt.xlim(lims[col_j])
            plt.ylim(lims[row_i])

    # Lower Diag
    indices = zip(*np.tril_indices(dim_features, k=-1))
    for row_i, col_j in indices:
        ax = plt.subplot(gs[row_i, col_j])
        ax.set_facecolor("grey")
        f_ind_i = features.index(top_features[row_i.item()])
        f_ind_j = features.index(top_features[col_j.item()])
        plt.xlim(lims[col_j])
        plt.ylim(lims[row_i])
        plt.xlabel(features[f_ind_j])
        plt.ylabel(features[f_ind_i])

    fig.tight_layout()


classifier_pair_plot(df, pipe, top_features_n, features, response)
st.pyplot()

st.write("## Model Predictions One-Way Plots")

pdp_feature_1 = st.selectbox("Plot Feature", options=top_features, index=0)

if is_numeric_dtype(df[pdp_feature_1]):
    pdp_isol = pdp.pdp_isolate(
        model=pipe,
        dataset=df_sample[features],
        model_features=features,
        feature=pdp_feature_1,
    )

    fig, axes = pdp.pdp_plot(
        pdp_isol,
        pdp_feature_1,
        plot_lines=True,
        frac_to_plot=100,
        ncols=1,
        figsize=[8.0, 6.0],
    )
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.margins(0.01, 0.01)
    fig.tight_layout()
    st.pyplot()
else:
    # scikit learn pipeline does not produce named columns
    # however pdpbox requires them. So make own pdp plot.
    df_pdp_other = df_sample[features].drop(columns=pdp_feature_1).assign(dummy=1)
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
    st.pyplot()
