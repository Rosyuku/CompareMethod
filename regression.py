# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:08:36 2019

@author: Wakasugi Kazuyuki
"""

import pandas as pd
import matplotlib.pyplot as plt
import time

import xgboost as xgb
from lightgbm import LGBMModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

def nn(X_train, y_train, X_test, y_test):

    
    reg = KNeighborsRegressor(n_neighbors=1)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def knn(X_train, y_train, X_test, y_test):

    reg = KNeighborsRegressor()
    
    start = time.time()
    reg_cv = GridSearchCV(reg, {'n_neighbors': [2,3,4,5,6,7,8,9,10]}, verbose=1)
    reg_cv.fit(X_train, y_train)
    print(reg_cv.best_params_, reg_cv.best_score_)
    reg = KNeighborsRegressor(**reg_cv.best_params_)    
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def gpr(X_train, y_train, X_test, y_test):

    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    reg = GaussianProcessRegressor(kernel=kernel)
    
    start = time.time()
#    reg_cv = GridSearchCV(reg, {'alpha': [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2], 'kernel':[kernel]}, verbose=1)
#    reg_cv.fit(X_train, y_train)
#    print(reg_cv.best_params_, reg_cv.best_score_)
#    reg = GaussianProcessRegressor(**reg_cv.best_params_)    
    reg.fit(X_train, y_train)
    print("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, reg.kernel_,
                 reg.log_marginal_likelihood(reg.kernel_.theta)))
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test
    
def xgboost(X_train, y_train, X_test, y_test):
    
    reg = xgb.XGBRegressor()
    
    start = time.time()
    reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50]}, verbose=1)
    reg_cv.fit(X_train, y_train)
    print(reg_cv.best_params_, reg_cv.best_score_)
    reg = xgb.XGBRegressor(**reg_cv.best_params_)
    reg.fit(X_train, y_train)
    time_train = time.time() - start
    
    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def lasso(X_train, y_train, X_test, y_test):
    
    reg = LassoCV(cv=5, random_state=0)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def ridge(X_train, y_train, X_test, y_test):
    
    reg = RidgeCV(cv=5)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def svr(X_train, y_train, X_test, y_test):
    
    reg = SVR()
    reg_cv = GridSearchCV(reg, {'C': [0.5,1.0,2.0], 'epsilon': [0.1, 0.5, 1.0]}, verbose=1)
    reg_cv.fit(X_train, y_train)
    print(reg_cv.best_params_, reg_cv.best_score_) 
    reg = SVR(**reg_cv.best_params_)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def rf(X_train, y_train, X_test, y_test):
    
    reg = RandomForestRegressor(n_estimators=50, random_state=0)
#    reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50]}, verbose=1)
#    reg_cv.fit(X_train, y_train)
#    print(reg_cv.best_params_, reg_cv.best_score_) 
#    reg = RandomForestRegressor(**reg_cv.best_params_)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def lightgbm(X_train, y_train, X_test, y_test):
    
    reg = LGBMModel(objective='regression')
#    reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50]}, verbose=1)
#    reg_cv.fit(X_train, y_train)
#    print(reg_cv.best_params_, reg_cv.best_score_) 
#    reg = xgb.LGBMModel(**reg_cv.best_params_)
    
    start = time.time()
    reg.fit(X_train, y_train)
    time_train = time.time() - start

    pred_train = reg.predict(X_train)
    start = time.time()
    pred_test = reg.predict(X_test)
    time_test = time.time() - start
    
    return pred_train, pred_test, time_train, time_test

def get_summary(datasets, algorithms, metrics, plot=False):

    results = dict()
    
    for key in datasets.keys():
        print(key)
        
        if len(datasets[key]) == 4:
            X_train, y_train, X_test, y_test = datasets[key]
            scaler_y = None
        elif len(datasets[key]) == 5:
            X_train, y_train, X_test, y_test, scaler_y = datasets[key]
        
        res_train = pd.DataFrame(data=y_train, columns=['True'])
        res_test = pd.DataFrame(data=y_test, columns=['True'])
        time_train = pd.Series()
        time_pred = pd.Series()
        metrics_train = pd.DataFrame(index=metrics, columns=algorithms)
        metrics_test = pd.DataFrame(index=metrics, columns=algorithms)
        result = dict()
        
        if 'nn' in algorithms:
            alg = 'nn'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = nn(X_train, y_train, X_test, y_test)        
        
        if 'knn' in algorithms:
            alg = 'knn'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = knn(X_train, y_train, X_test, y_test)        
    
        if 'ridge' in algorithms:
            alg = 'ridge'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = ridge(X_train, y_train, X_test, y_test)
           
        if 'lasso' in algorithms:
            alg = 'lasso'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = lasso(X_train, y_train, X_test, y_test)
    
        if 'svr' in algorithms:
            alg = 'svr'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = svr(X_train, y_train, X_test, y_test)
    
        if 'gpr' in algorithms:
            alg = 'gpr'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = gpr(X_train, y_train, X_test, y_test)
            
        if 'rf' in algorithms:
            alg = 'rf'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = rf(X_train, y_train, X_test, y_test)
    
        if 'xgboost' in algorithms:
            alg = 'xgboost'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = xgboost(X_train, y_train, X_test, y_test)
    
        if 'lightgbm' in algorithms:
            alg = 'lightgbm'
            res_train[alg], res_test[alg], time_train[alg], time_pred[alg] = lightgbm(X_train, y_train, X_test, y_test)
    
        if scaler_y is not None:
            res_train = pd.DataFrame(index=res_train.index, columns=res_train.columns, data=scaler_y.inverse_transform(res_train))
            res_test = pd.DataFrame(index=res_test.index, columns=res_test.columns, data=scaler_y.inverse_transform(res_test))
            
        if 'RMSE' in metrics:
            metrics_train.loc['RMSE'] = ((((res_train.T - res_train['True'].values).T)**2).mean()**0.5).iloc[1:].values
            metrics_test.loc['RMSE'] = ((((res_test.T - res_test['True'].values).T)**2).mean()**0.5).iloc[1:].values
    
        if 'MAE' in metrics:
            metrics_train.loc['MAE'] = ((((res_train.T - res_train['True'].values).T).abs()).mean()).iloc[1:].values
            metrics_test.loc['MAE'] = ((((res_test.T - res_test['True'].values).T).abs()).mean()).iloc[1:].values
    
        if 'RMSPE' in metrics:
            metrics_train.loc['RMSPE'] = ((((res_train.T / res_train['True'].values).T - 1)**2).mean()**0.5).iloc[1:].values
            metrics_test.loc['RMSPE'] = ((((res_test.T / res_test['True'].values).T - 1)**2).mean()**0.5).iloc[1:].values
    
        if 'MAPE' in metrics:
            metrics_train.loc['MAPE'] = ((res_train.T / res_train['True'].values).T - 1).abs().mean().iloc[1:].values
            metrics_test.loc['MAPE'] = ((res_test.T / res_test['True'].values).T - 1).abs().mean().iloc[1:].values
    
        if 'R2' in metrics:
            metrics_train.loc['R2'] = (res_train.corr()['True']**2).iloc[1:].values
            metrics_test.loc['R2'] = (res_test.corr()['True']**2).iloc[1:].values
            
        timer = pd.concat([time_train.rename("train"), time_pred.rename("pred")], axis=1).T
    
        if plot == True:
            res_train.plot(x="True", marker=".", lw=0, grid=True, title="{}    Train data".format(key))
            plt.plot([res_train.values.min(), res_train.values.max()], [res_train.values.min(), res_train.values.max()], 'r')
            plt.show()        
            res_test.plot(x="True", marker=".", lw=0, grid=True, title="{}    Test data".format(key))
            plt.plot([res_test.values.min(), res_test.values.max()], [res_test.values.min(), res_test.values.max()], 'r')
            plt.show()
            metrics_train.T.plot.bar(subplots=True, layout=(1, 5), grid=True, figsize=(15, 4), title="{}    Train data".format(key))
            plt.show()
            metrics_test.T.plot.bar(subplots=True, layout=(1, 5), grid=True, figsize=(15, 4), title="{}    Test data".format(key))
            plt.show()
            timer.plot.bar(logy=True, grid=True, title="{}    Time".format(key))
            plt.show()
        
        result["res_train"] = res_train
        result["res_test"] = res_test
        result["metrics_train"] = metrics_train
        result["metrics_test"] = metrics_test
        result["time"] = timer
        
        results[key] = result
        
    return results