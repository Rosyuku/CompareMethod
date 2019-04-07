# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:11:03 2019

@author: Wakasugi Kazuyuki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston, load_diabetes, load_linnerud
from sklearn.model_selection import train_test_split

from CompareMethod import regression

if __name__ == "__main__":

    np.random.seed(0)
    
    datasets = dict()    
    
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.33, random_state=0)
    datasets['boston'] = [X_train, y_train, X_test, y_test, boston.feature_names]

    diabete = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabete.data, diabete.target, test_size=0.33, random_state=0)
    datasets['diabete'] = [X_train, y_train, X_test, y_test, diabete.feature_names]

#    linnerud = load_linnerud()
#    X_train, X_test, y_train, y_test = train_test_split(linnerud.data, linnerud.target, test_size=0.33, random_state=0)
#    datasets['linnerud Weight'] = [X_train, y_train[:, 0], X_test, y_test[:, 0], linnerud.feature_names]
#    datasets['linnerud Waist'] = [X_train, y_train[:, 1], X_test, y_test[:, 1], linnerud.feature_names]
#    datasets['linnerud Pulse'] = [X_train, y_train[:, 2], X_test, y_test[:, 2], linnerud.feature_names]
    
    algorithms = ['nn', 'knn', 'lasso', 'ridge', 'svr', 'gpr', 'rf', 'xgboost', 'lightgbm']
    metrics = ['RMSE', 'MAE', 'RMSPE', 'MAPE', 'R2']
    
    results = regression.get_summary(datasets, algorithms, metrics, normalize=True, plot=True)
    
    metrics = dict()
    for key in results.keys():
        metrics[key] = results[key]["metrics_test"]
        results[key]['feature_importances'].T.astype(float).round(2).to_html(key+".html", justify="justify-all")
        
    pd.concat(metrics).astype(float).round(2).to_html("test.html", justify="justify-all")
