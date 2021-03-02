# LASTS
LASTS (Local Agnostic Subsequence-based Time Series explainer) is a method that explains the decisions of black-box time series classifiers. 
The explanation consists of factual and counterfactual rules revealing the reasons for the classification through conditions expressed as subsequences that must or must not be contained in the time series. In addition, a set of exemplar and counter-exemplar time series highlight similarities and differences with the time series under analysis. 

## Install
LASTS can be installed by building a conda environment using the `environment.yml` file. Moreover, for visualizing SAX explanations, a specific fork of the library sktime is needed (https://github.com/fspinna/sktime_forked).

## Notebooks
A full working example can be found in the *notebooks* folder  [`Full LASTS Example.ipynb`](https://github.com/fspinna/LASTS_explainer/blob/main/notebooks/Full%20LASTS%20Example.ipynb)
