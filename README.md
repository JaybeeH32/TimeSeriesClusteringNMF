# TimeSeriesClusteringNMF
Project based on papers "Time Series Clustering via NMF in Networks" by Du et al and "Time Series Clustering via Community Detection in Networks" by Ferreira et al. Machine Learning for time series, Laurent Oudre, MVA

# Outline

This repository hosts multiple files and folders.

- the *requirements.txt* file lists the required libraries tp download before running the repository.
- the *libraries* folder hosts the implementation of clustering algorithms, including NMF variants, DeepNMF variants, Node2Vec and LINE implementation
- the *experiments_UCR.ipynb* file is a notebook used to benchmark all methods on the UCR datasets. Running it takes 40 min approximately, but removing methods or datasets to train on should improve speed.
- the *experiments_PTB.ipynb* file is a notebook used to benchmark all methods on the PTB dataset. Running it takes 30 min approximately, but removing methods to test should improve speed.
- the *noise_outliers_PTB.ipynb* file is a notebook used to assess NMF robustness to noise and outliers. Running it takes 40 minutes approximately, but removing sigma and magnitude values to test should make it faster.
- the *ptbdb_abnormal.csv* and *ptbdb_normal.csv* are the data files from which we extract the PTB dataset.
- the *outdated* folder hosts old notebooks.
- the *image_results* folder host various plots and images

Authors: Jean-Baptiste Himbert and Benjamin Lapostolle