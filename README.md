# PunditKit:
Simplify. Visualise. Learn.

![Screenshot](Screenshot.jpg)

PunditKit is a toolkit for training machine learning models on tabular data
simply and without code. It is intended for students and experienced data
scientists who are looking to quickly obtain some initial predictive insights
from data.

Included Features:
 - Machine learning using [scikit-learn](https://scikit-learn.org)
 - Interpretable explanations of predictions using [lime](https://github.com/marcotcr/lime)
 - Exploratory data summaries and model performance on holdout data for checks
 - Partial dependence plots

PunditKit is under active development. The goal is to develop an opinionated
modelling framework with best practice modelling and visualisation. If you
encounter any problems, please raise an [issue](https://github.com/JackyP/punditkit/issues).

## Installation (via pip)
PunditKit is developed using Python. First download and install a Python 3.x
distribution such as [Anaconda](https://www.anaconda.com/distribution/#download-section)

PunditKit can then be installed using ``pip`` from the command line (if ``pip``
is added to PATH during installation) or using Anaconda prompt.

```
pip install punditkit
```

This adds the ``punditkit`` command.

## Modelling a dataset
Suppose you have a file called ``iris.csv`` that you would like to model.

To use punditkit on the dataset, run:

```
punditkit iris.csv
```

Currently only Comma Separated Values (CSV) datasets are supported. For Excel
spreadsheets, please Save As CSV to analyse using punditkit.
