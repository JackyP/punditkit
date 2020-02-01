import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
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
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline

from sklearn.utils import all_estimators

# Diagnostics
from sklearn.inspection import permutation_importance
from pdpbox import pdp
from yellowbrick.regressor import PredictionError
from yellowbrick.classifier.confusion_matrix import ConfusionMatrix

try:
    from mlxtend_plotting import plot_decision_regions
except ModuleNotFoundError:
    from .mlxtend_plotting import plot_decision_regions

# All estimators
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

all_regressors = [x for x in regressors if x[0] not in excluded_regressors]
all_classifiers = [x for x in classifiers if x[0] not in excluded_classifiers]


class PunditKitEstimator:
    def __init__(
        self,
        Estimator,
        numeric_features,
        categorical_features,
        response,
        kwargs=None,
        sparse=False,
    ):
        """

        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = numeric_features + categorical_features
        self.response = response

        ordinal_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ordinal", OrdinalEncoder()),
            ]
        )

        # Standardize numeric
        prenumeric = ColumnTransformer(
            [
                (
                    "numimp",
                    SimpleImputer(),
                    [self.features.index(x) for x in numeric_features],
                ),
                (
                    "cat",
                    ordinal_pipe,
                    [self.features.index(x) for x in categorical_features],
                ),
            ]
        )

        # Round to int to avoid issues with decision boundary sampling non integers
        # for the range.
        # One Hot for most Sci-kit Learn functions.
        one_hot_pipe = Pipeline(
            [
                ("rint", FunctionTransformer(np.rint)),
                ("onehot", OneHotEncoder(sparse=sparse, handle_unknown="ignore")),
            ]
        )

        preprocessing = ColumnTransformer(
            [
                (
                    "num",
                    RobustScaler(),
                    [self.features.index(x) for x in numeric_features],
                ),
                (
                    "onepipe",
                    one_hot_pipe,
                    [self.features.index(x) for x in categorical_features],
                ),
            ]
        )

        pipe = Pipeline(
            [
                ("prenumeric", prenumeric),
                ("preprocess", preprocessing),
                ("estimator", Estimator(**kwargs)),
            ]
        )
        self.Estimator = Estimator
        self.pipe = pipe
        self.transform_numeric = pipe.named_steps["prenumeric"]
        self.numeric_pipe = Pipeline(
            [
                ("preprocess", pipe.named_steps["preprocess"]),
                ("estimator", pipe.named_steps["estimator"]),
            ]
        )

    def fit_data(self, df, valid_percentage=0.3):
        X = df[self.features]
        y = df[self.response]

        self.response_is_numeric = is_numeric_dtype(df[self.response])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=valid_percentage, random_state=42
        )

        # Drop NaN in response.
        X_train = X_train[~pd.isnull(y_train)]
        y_train = y_train[~pd.isnull(y_train)]

        X_test = X_test[~pd.isnull(y_test)]
        y_test = y_test[~pd.isnull(y_test)]

        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Train
        self.pipe.fit(X_train, y_train)

        # Importance
        self.perm_imp = permutation_importance(
            self.pipe,
            self.X_train,
            self.y_train,
            n_repeats=10,
            random_state=42,
            n_jobs=2,
        )

        self.sorted_idx = self.perm_imp.importances_mean.argsort()
        self.perm_imp_labels = X_train.columns[self.sorted_idx]
        self.top_features = X_train.columns[self.sorted_idx].tolist()
        self.top_features.reverse()

        # Explainer
        self.X_numeric = self.transform_numeric.transform(X_train)

        categorical_feature_indices = [
            self.features.index(x) for x in self.categorical_features
        ]

        if self.response_is_numeric:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_numeric,
                feature_names=self.features,
                class_names=[self.response],
                discretize_continuous=True,
                categorical_features=categorical_feature_indices,
                mode="regression",
            )

        else:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_numeric,
                feature_names=self.features,
                class_names=self.pipe.classes_.tolist(),
                discretize_continuous=True,
                categorical_features=categorical_feature_indices,
            )

        # Typical Feature Values
        typical_feature_values_list = []
        for col in self.features:
            if col in self.numeric_features:
                mn = df[col].min().item()
                mx = df[col].max().item()
                defv = df[col].mean()

                if mx > mn:
                    sp = (mx - mn) / 100
                    sp = round(sp, 1 - int(floor(log10(abs(sp)))) - 1)

                    if df[col].dtype == np.int64:
                        defv = np.ceil(defv).astype(int).item()
                        sp = np.ceil(sp).astype(int).item()

            else:
                defv = df[col].mode().item()
            typical_feature_values_list += [defv]

        self.typical_feature_values = pd.DataFrame(
            [typical_feature_values_list], columns=self.features
        )

    def pipeline_to_base64(self):
        s = pickle.dumps(self.pipe)
        b64 = base64.b64encode(s).decode()
        return b64

    def prediction_error_plot(self):
        visualizer = PredictionError(self.pipe)
        visualizer.score(self.X_test, self.y_test)
        return visualizer.show()

    def confusion_matrix_plot(self):
        visualizer = ConfusionMatrix(self.pipe)
        visualizer.score(self.X_test, self.y_test)
        return visualizer.show()

    def predict_explanation(self, X, num_features):
        """ Predict and explain """
        df_form_lime = self.transform_numeric.transform(X)  # df_form.values[0]

        if self.response_is_numeric:
            exp = self.explainer.explain_instance(
                df_form_lime[0], self.numeric_pipe.predict, num_features=num_features
            )
            fig = exp.as_pyplot_figure()
        else:

            exp = self.explainer.explain_instance(
                df_form_lime[0],
                self.numeric_pipe.predict_proba,
                num_features=num_features,
                top_labels=1,
            )

            fig = exp.as_pyplot_figure(
                label=self.numeric_pipe.classes_.tolist().index(
                    self.numeric_pipe.predict(df_form_lime)[0]
                )
            )
        fig.tight_layout()
        return fig

    def feature_importance_plot(self):
        fig, ax = plt.subplots()
        ax.boxplot(
            self.perm_imp.importances[self.sorted_idx].T,
            vert=False,
            labels=self.perm_imp_labels,
        )

        fig.tight_layout()
        return fig

    def classifier_pair_plot(self, df, num_features=4, regression_quantiles=10):
        """ Returns a pair plot of top features for a classifier
        """

        # TODO: find replace
        df_default = self.typical_feature_values
        pipe = self.numeric_pipe
        transformer = self.transform_numeric
        features = self.features
        top_features = self.top_features[:num_features]
        response = self.response

        dim_features = len(features)

        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams.update({"font.size": 6})  # For readability
        # df_not_na = df[top_features + [response]].dropna()
        vals_df = self.transform_numeric.transform(df[features])
        vals_default = pd.DataFrame(transformer.transform(df_default))

        is_numeric = is_numeric_dtype(df[response])

        if is_numeric:
            pal0 = sns.color_palette("GnBu_d", regression_quantiles)
            pal1 = sns.color_palette("GnBu_r", regression_quantiles)
        else:
            pal0 = sns.color_palette("muted", df[response].nunique())
            pal1 = sns.color_palette("bright", df[response].nunique())
        # g = sns.PairGrid(df_not_na, vars=top_features, hue=response, palette=pal)

        gs = gridspec.GridSpec(num_features, num_features)
        fig = plt.figure()

        # Dummy estimator that predicts categoricals even for regression
        class DummyEstimator:
            model = pipe
            is_num = is_numeric

            def predict(self, X):
                y = self.model.predict(X)
                if self.is_num:
                    return pd.qcut(
                        y, regression_quantiles, labels=False, duplicates="drop"
                    )
                else:
                    return np.searchsorted(pipe.classes_, y)

        estimator = DummyEstimator()

        if is_numeric:
            resp_plt = pd.qcut(
                df[response], regression_quantiles, labels=False, duplicates="drop"
            ).values
        else:
            resp_plt = df[response].astype("category").cat.codes.values

        # Diagonals: Numeric - Dist Plot, Categorical - Bar Plot
        lims = []
        for row_i in range(0, num_features):
            ax = plt.subplot(gs[row_i, row_i])
            feature_name = top_features[row_i]
            f_ind_i = features.index(feature_name)
            # plt.hist(x=df.iloc[:, f_ind_i])
            if is_numeric_dtype(df.loc[:, feature_name]):
                ax = sns.distplot(df.loc[:, feature_name].dropna())
            else:
                # Truncate long category names to avoid cluttering the axis
                kwargs = {feature_name: lambda df: df[feature_name].str.slice(0, 5)}
                df.groupby(feature_name).size().reset_index(name="counts").assign(
                    **kwargs
                ).plot.bar(ax=ax, x=feature_name, y="counts", legend=False)

            plt.ylabel("Count")
            plt.xlabel(feature_name)
            lims += [plt.xlim()]
            # plt.xlim((df.iloc[:, f_ind_i].min(), df.iloc[:, f_ind_i].max()))

        # Upper:
        indices = zip(*np.triu_indices(num_features, k=1))

        # indices = zip(*np.triu_indices_from(g.axes, 1))
        for row_i, col_j in indices:

            # ax = g.axes[row_i, col_j]
            ax = plt.subplot(gs[row_i, col_j])
            feature_name_i = top_features[row_i.item()]
            feature_name_j = top_features[col_j.item()]
            f_ind_i = features.index(feature_name_i)
            f_ind_j = features.index(feature_name_j)
            filler_ind = [
                x for x in list(range(0, dim_features)) if x not in [f_ind_i, f_ind_j]
            ]
            # mlxtend decision regions requries a dummy set of variables if there
            # are more than two features.
            if len(features) > 2:
                filler = vals_default[filler_ind].to_dict("records")[0]
            else:
                filler = None

            mpl.rcParams.update({"contour.negative_linestyle": "dotted"})
            plot_decision_regions(
                X=transformer.transform(df[features].values),
                y=resp_plt,
                feature_index=(f_ind_j, f_ind_i),
                filler_feature_values=filler,
                clf=estimator,
                colors=",".join(pal0.as_hex()),
                hide_spines=False,
            )

            if len(filler_ind) > 0:
                plt.scatter(
                    x=vals_df[:, features.index(feature_name_j)],
                    y=vals_df[:, features.index(feature_name_i)],
                    s=3,
                    c=resp_plt,
                    cmap=ListedColormap(pal1.as_hex()),
                )
            plt.xlabel(features[f_ind_j])
            plt.ylabel(features[f_ind_i])
            plt.xlim(lims[col_j])
            plt.ylim(lims[row_i])

        # Lower Diag
        indices = zip(*np.tril_indices(num_features, k=-1))
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

    def pdp_plot(self, pdp_feature_1):

        pdp_isol = pdp.pdp_isolate(
            model=self.numeric_pipe,
            dataset=pd.DataFrame(self.X_numeric, columns=self.features),
            model_features=self.features,
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

        return fig
