# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:04:11 2024

@author: 78689
"""

import streamlit as st 
import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd

# 设置全局属性
st.set_page_config(
    page_title='斜拉桥成桥状态轻量级优化器',
    page_icon=' ',
    layout='wide'
)

# 正文
#st.title('hello world')
#st.markdown('> Streamlit 支持通过 st.markdown 直接渲染 markdown')

with st.sidebar:
    #st.title('欢迎使用并反馈宝贵意见')
    st.image('D:/Python/Web_cable/tongji.png', caption=' ')
    
    st.markdown('---')
    st.markdown('目录：\n- 使用界面\n- 帮助文档\n- 开发团队')
    


# 设置标题
st.title('关于输入功能的提示')
st.image('D:/Python/Web_cable/tongji.png', caption=' ')


## 默认渲染到主界面
st.title('请输入有关设计变量')


# 读取输入并输出
input1 = st.number_input('请输入拉索间距', 3)

# calculator
output1 = input1+1


def kriging_model(X,y):
    # X  (n x d)
    # y  (n,)   
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2)) 
    #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    #gp.fit(X, y)
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X_train, y_train)

    return gp

# def problem(x):
    
#     y = (np.square(x[:,0:1])+4)-np.sin(2.5*x[:,0:1])-2
#     y = y.reshape((-1,1))
#     return -y

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]
gp = kriging_model(X_train, y_train)

r = np.array(input1).reshape(1, -1) 
mean_prediction, std_prediction = gp.predict(r, return_std=True)
output1 = mean_prediction


# 输出
with st.container(): #height=1000
    st.write("**  \n  \n  \n  \n  以下为预测索力分布")
   
    #st.write('The current movie title is', output1)
    ## 画图
    #fig, ax = plt.subplots(2,1, figsize=(10, 10))
    #fig, ax = plt.subplots(1,2,figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    #plt.plot()
    ax.scatter(X_train, y_train, label="Observations")
    ax.scatter(r, mean_prediction.reshape(1, -1), label="Mean prediction")
    ax.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
        )
    # ax[0].legend()
    # ax[0].xlabel("$x$")
    # ax[0].ylabel("$f(x)$")
    # _ = ax.title("Gaussian process regression on noise-free dataset")
    fig.set_size_inches(4, 3) 

    st.pyplot(fig, use_container_width=False)
    
    
    ## 列表
    df = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))
    st.table(df)
    