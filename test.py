# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:04:11 2024

@author: 78689
"""

import streamlit as st 
import numpy as np
from matplotlib import pyplot as plt
#from pyDOE import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import joblib
import os
os.chdir(os.path.dirname(__file__))

# 设置全局属性
st.set_page_config(
    page_title='斜拉桥成桥状态轻量级优化器',
    page_icon=' ',
    layout='wide'
)


def GetUX (SideSpanCableDistanse,MidSpanCableDistanse):

    CableNum = np.size(SideSpanCableDistanse)
    UXside = np.zeros((CableNum-1,1))
    UXmid = np.zeros((CableNum-1,1))
    for i in range(0,CableNum-1):
        UXside[i,0] = SideSpanCableDistanse[i+1,0]-SideSpanCableDistanse[0,0]
        UXmid[i,0] = MidSpanCableDistanse[i+1,0]-MidSpanCableDistanse[0,0]
        
    UXside = UXside/(SideSpanCableDistanse[CableNum-1,0]-SideSpanCableDistanse[0,0])
    UXmid = UXmid/(MidSpanCableDistanse[CableNum-1,0]-MidSpanCableDistanse[0,0])
    
    return UXside,UXmid


def CalCableDistanse(MidSpanLength,SideMidSpanRatio,StanCableDistanse):
    
    SideSpanSymRatio = 0.65        # %边跨标准索距区占边跨比例
    MidSpanNonCableDis = StanCableDistanse       #  %跨中无索区长度
    SideSpanLength = MidSpanLength*SideMidSpanRatio

    CableNum = np.int16(np.round(MidSpanLength/2/StanCableDistanse)-1)  #  %中跨拉索个数
    SideCableNum1 = np.int16(np.round(SideSpanSymRatio*SideSpanLength/StanCableDistanse))    # %边跨标准索距区拉索个数
    SideSymmLength = SideCableNum1*StanCableDistanse         #  %边跨标准索距区长度
    SideCableNum2 = CableNum-SideCableNum1   #  %边跨非标准索距区拉索个数
    MidSpanCableDistanse = np.zeros((CableNum,1))
    SideSpanCableDistanse = np.zeros((CableNum,1))

    ZeroCableDis = np.round(((MidSpanLength-MidSpanNonCableDis)/2/StanCableDistanse-np.floor((MidSpanLength-MidSpanNonCableDis)/2/StanCableDistanse)+1)*StanCableDistanse)
    MidSpanCableDistanse[0] = ZeroCableDis
    SideSpanCableDistanse[0] = ZeroCableDis
    for i in range(1,CableNum):
        
        MidSpanCableDistanse[i] = MidSpanCableDistanse[i-1]+StanCableDistanse
        
        if i<=SideCableNum1:
            SideSpanCableDistanse[i] = MidSpanCableDistanse[i]
        else:
            SideSpanCableDistanse[i] = SideSpanCableDistanse[SideCableNum1]+(SideSpanLength-SideSymmLength)/(SideCableNum2+1)*(i-SideCableNum1)

        
    CableNum = np.int16(np.size(MidSpanCableDistanse))
    
    return MidSpanCableDistanse,SideSpanCableDistanse,CableNum


def BaseFunction(i, k , u, NodeVector):
    
    if k == 0:       # % 0次B样条
        if (u > NodeVector[i]) and (u <= NodeVector[i+1]):
            Nik_u = 1.0
        else:     
            Nik_u = 0.0

    else:
        
        Length1 = NodeVector[i+k] - NodeVector[i]
        Length2 = NodeVector[i+k+1] - NodeVector[i+1]     #% 支撑区间的长度
        if Length1 == 0.0:      # % 规定0/0 = 0
            Length1 = 1.0

        if Length2 == 0.0:
            Length2 = 1.0

        Nik_u = (u - NodeVector[i]) / Length1 * BaseFunction(i, k-1, u, NodeVector)+ (NodeVector[i+k+1] - u) / Length2 * BaseFunction(i+1, k-1, u, NodeVector)

    return Nik_u

def B_spline(U,P) :
    n = 3      #n+1个控制点
    k = 2     #k次、k+1阶
    NodeVector = np.array([0,0,0,0.5,1,1,1])     #%n+k+1+1个结点矢量，
    dist=0.01
    uu=np.arange(0,1+dist,dist)      #%参数坐标
    puy = np.zeros((np.size(uu),2))      #%笛卡尔坐标
    for i in range(0,np.size(uu)):
        for j in range(0,n+1):
            puy[i,0] = puy[i,0] + BaseFunction(j, k , uu[i], NodeVector)*P[j,0]        #%x轴坐标
            puy[i,1] = puy[i,1] + BaseFunction(j, k , uu[i], NodeVector)*P[j,1]       # %y轴坐标

    res = np.zeros((np.size(U),1))
    flag_j = 2
    for i in range(0,np.size(U)):
        for j in range(flag_j,np.size(uu)):
            if(U[i]<=puy[j,0]):    #寻找距离最近的插值点
               if( (U[i]-puy[j-1,0])<(puy[j,0]-U[i]) ):
                    res[i]=puy[j-1,1]   
               else:
                    res[i]=puy[j,1] 
               flag_j = j
               break
           
    return res


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


# 正文
#st.title('hello world')
#st.markdown('> Streamlit 支持通过 st.markdown 直接渲染 markdown')

with st.sidebar:
    #st.title('欢迎使用并反馈宝贵意见')
    #st.image('tongji.png', caption=' ')
    
    st.markdown('---')
    st.markdown('目录：\n- 使用界面\n- 帮助文档\n- 开发团队')
    


# 设置标题
#st.title('关于输入功能的提示')
#st.image('tongji.png', caption=' ')


## 默认渲染到主界面
st.title('请输入有关设计变量')


# 读取输入并输出
input1 = st.number_input('请桥塔高跨比(目前0.2-0.26)', 0.2)

# 读取输入并输出
input2 = st.number_input('请输入梁上索间距(目前12-18)', 12)

# 读取输入并输出
input3 = st.number_input('桥塔顺桥尺寸(13-14)', 13)

# 读取输入并输出
input4 = st.number_input('桥塔横桥尺寸(0.65-0.79)', 0.72)

# 读取输入并输出
input5 = st.number_input('壁厚(0.15-0.29)', 0.29)

# 读取输入并输出
input6 = st.number_input('截面缩小系数(0.5-0.9)', 0.7)

# calculator
output1 = input1+1


# def kriging_model(X,y):
#     # X  (n x d)
#     # y  (n,)   
#     #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2)) 
#     #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
#     #gp.fit(X, y)
#     kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
#     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#     gp.fit(X_train, y_train)

#     return gp

# def problem(x):
    
#     y = (np.square(x[:,0:1])+4)-np.sin(2.5*x[:,0:1])-2
#     y = y.reshape((-1,1))
#     return -y

# X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
# y = np.squeeze(X * np.sin(X))

# rng = np.random.RandomState(1)
# training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
# X_train, y_train = X[training_indices], y[training_indices]
# gp = kriging_model(X_train, y_train)

# r = np.array(input1).reshape(1, -1) 
# mean_prediction, std_prediction = gp.predict(r, return_std=True)
# output1 = mean_prediction

gp = joblib.load('gp.pkl')
r = np.array([[0.2,12,13,0.72,0.29,0.7]])
r = np.array([[input1,input2,input3,input4,input5,input6]])

# 预测B-spline 插值点数据、和跨中索和压重
Ini1 = r
InterPY = gp.predict(Ini1, return_std=False)

#InterPY = np.array([InterPY(1:9,1)/1e7;InterPY(10,1)/(GirderSectionHeight*GirderDeckWidth*50000*0.5);InterPY(11,1)/(SideMidSpanRatio*MidSpanLength*0.5)])

(MidSpanCableDistanse,SideSpanCableDistanse,CableNum) = CalCableDistanse(818,0.4377,r[0,1])

[UXside,UXmid] = GetUX(SideSpanCableDistanse,MidSpanCableDistanse);     #索距归一化

# 插值点到控制点
InterPX = np.array([[0,0.3,0.7,1.0]])         # %插值点的X坐标

InterP_S = np.append(InterPX,InterPY[:,0:4],axis=0)     #     %边跨插值点坐标
InterP_M = np.append(InterPX,InterPY[:,4:8],axis=0)     #     %中跨插值点坐标

InterP_S = np.mat(np.transpose(InterP_S))
InterP_M = np.mat(np.transpose(InterP_M))
    
A = np.mat([[1,    0,     0 ,    0], [0.16,  0.66,  0.18,  0], [0,  0.18,  0.66,  0.16],  [0 ,   0,     0  ,   1]])

ConP_side = np.dot(np.linalg.inv(A),InterP_S)
ConP_mid = np.dot(np.linalg.inv(A),InterP_M)


SideCableForce = B_spline(UXside,ConP_side)           #%拉索索力插值，0~1
MidCableForce = B_spline(UXmid,ConP_mid)

SideCableForce = np.append(InterPY[:,8].reshape(1,1),SideCableForce)
MidCableForce = np.append(InterPY[0,8].reshape(1,1),MidCableForce)


# 输出
with st.container(): #height=1000
    st.write("**  \n  \n  \n  \n  以下为预测索力分布")
   
    #st.write('The current movie title is', output1)
    ## 画图
    #fig, ax = plt.subplots(2,1, figsize=(10, 10))
    #fig, ax = plt.subplots(1,2,figsize=(10, 5))
    fig, ax = plt.subplots(2,1,dpi=300)
    ax[0].bar (-1*np.flip(SideSpanCableDistanse.reshape(-1)), np.flip(SideCableForce.reshape(-1)),width=2, align='center')
    ax[1].bar ((MidSpanCableDistanse.reshape(-1)), (MidCableForce.reshape(-1)),width=2, align='center')


    st.pyplot(fig, use_container_width=False)
    
    
    # ## 列表
    # df = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))
    # st.table(df)
    