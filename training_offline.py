# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:26:40 2024

@author: 78689
"""


#import streamlit as st 
import numpy as np
from matplotlib import pyplot as plt
#from pyDOE import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
#import pandas as pd


def kriging_model(X,y):
    # X  (n x d)
    # y  (n,)   
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2)) 
    #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    #gp.fit(X, y)
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)

    return gp

# def problem(x):
    
#     y = (np.square(x[:,0:1])+4)-np.sin(2.5*x[:,0:1])-2
#     y = y.reshape((-1,1))
#     return -y

Xdata = np.loadtxt("D:\Python\Web_cable\X.txt",dtype=np.float32,delimiter='	')
Ydata = np.loadtxt("D:\Python\Web_cable\Y.txt",dtype=np.float32,delimiter='	')
Ydata[:,0:10] = Ydata[:,0:10]/1e6
Ydata[:,10] = Ydata[:,10]/1e2

rng = np.random.RandomState(1)
gp = kriging_model(Xdata, Ydata)

r = np.array([[0.2,12,13,0.72,0.29,0.7]])
mean_prediction, std_prediction = gp.predict(r, return_std=True)
output1 = mean_prediction


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(np.linspace(0, 10, num = 10), mean_prediction[0,0:10], label="索力预测值", linestyle="dotted")
#plt.plot()
ax.scatter(np.linspace(0, 10, num = 10), mean_prediction[0,0:10], label="Observations")


# save 
joblib.dump(gp, 'D:/Python/Web_cable/gp.pkl')