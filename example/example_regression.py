# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:11:03 2019

@author: Wakasugi Kazuyuki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

from CompareMethod import regression

if __name__ == "__main__":

    np.random.seed(0)
    
    datasets = dict()    

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=0)
    datasets['iris'] = [X_train, y_train, X_test, y_test, iris.feature_names]
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
    datasets['iris_normal'] = [X_train, y_train, X_test, y_test, iris.feature_names, scaler_y]
    
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.33, random_state=0)
    datasets['boston'] = [X_train, y_train, X_test, y_test, boston.feature_names]
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
    datasets['boston_normal'] = [X_train, y_train, X_test, y_test, boston.feature_names, scaler_y]
    
    algorithms = ['nn', 'knn', 'lasso', 'ridge', 'svr', 'gpr', 'rf', 'xgboost', 'lightgbm']
    metrics = ['RMSE', 'MAE', 'RMSPE', 'MAPE', 'R2']
    
    results = regression.get_summary(datasets, algorithms, metrics, plot=True)
    
    metrics = dict()
    for key in results.keys():
        metrics[key] = results[key]["metrics_test"]
        
    pd.concat(metrics).to_html("test.html")
