# PunditKit

Simplify. Visualise. Learn.

![Screenshot](Screenshot.jpg)


PunditKit is a free toolkit for exploratory data analysis and modelling of
tabular data - such as a database or a spreadsheet - with a simple interface,
visual diagnostics, and the choice of a large number of different models.

Included Features:
 - Machine learning models using [scikit-learn](https://scikit-learn.org)
 - Interpretable explanations of predictions using [lime](https://github.com/marcotcr/lime)
 - Exploratory data summaries for checking datasets
 - Model diagnostics for evaluating the effectiveness of different models
 - Feature importance: identify key drivers of the response
 - Partial dependence plots: understand relationships with a particular feature.

PunditKit is under active development. The goal is to develop an opinionated
modelling framework with best practice modelling and visualisation techniques.
If you encounter any problems or have any feature requests, please consider
raising an [issue](https://github.com/JackyP/punditkit/issues).

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


Currently only Comma Separated Values (CSV) datasets are supported. Datasets
within Excel spreadsheets can be analysed using punditkit, use Save As CSV from
within Excel to convert to the right format.
