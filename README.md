# probabilistic inverse problems and machine learning - example

[![DOI](https://zenodo.org/badge/530996445.svg)](https://zenodo.org/badge/latestdoi/530996445)


This repository includes python scripts that implements the idea proposed in 

Hansen. T.M. and Finlay, C.C., Use of machine learning to estimate properties of the posterior distribution in probabilistic inverse problems - an application to airborne EM data. JGR - Solid Earth,  20/10/2022.

doi:[https://doi.org/10.1029/2022JB024703](https://doi.org/10.1029/2022JB024703)


Training data for priorA can be downloaded from (https://zenodo.org/record/7254008)

Training data for priorB can be downloaded from (https://zenodo.org/record/7254030)

Training data for priorC can be downloaded from (https://zenodo.org/record/7253825)

The latest updated example is always available at https://github.com/cultpenguin/posterior_statistics_using_ml

## 

If you use conda, use for example the following environment

    conda create --name tf python=3.9 numpy scipy jupyterlab matplotlib h5py tensorflow tensorflow-probability scikit-learn
    conda activate tf

If you use pip, install the following packages
    
    pip install --upgrade numpy scipy jupyterlab matplotlib h5py tensorflow tensorflow-probability scikit-learn
    
A python example is available for type of model parameter (m, n1, n2, n3) -->

    python ip_and_ml_regression_m.py
    python ip_and_ml_classification_n1.py
    python ip_and_ml_regression_n2.py
    python ip_and_ml_classification_n3_m.py

To run a short simple setup/training/prediction use (default)

    useSimpleTest=True

To test a laerge number of networks using varying training image size use

    useSimpleTest=False
