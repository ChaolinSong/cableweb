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

# def problem(x):
    
#     y = (np.square(x[:,0:1])+4)-np.sin(2.5*x[:,0:1])-2
#     y = y.reshape((-1,1))
#     return -y

Xdata = np.loadtxt("D:\Python\Web_cable\X.txt",dtype=np.float32,delimiter='	')
Ydata = np.loadtxt("D:\Python\Web_cable\Y.txt",dtype=np.float32,delimiter='	')
Ydata[:,0:10] = Ydata[:,0:10]/1e7
Ydata[:,10] = Ydata[:,10]/1e2

rng = np.random.RandomState(1)
gp = kriging_model(Xdata, Ydata)

r = np.array([[0.2,12,13,0.72,0.29,0.7]])
mean_prediction, std_prediction = gp.predict(r, return_std=True)
output1 = mean_prediction


#fig, ax = plt.subplots(figsize=(10, 10))
#ax.plot(np.linspace(0, 10, num = 10), mean_prediction[0,0:10], label="索力预测值", linestyle="dotted")
#plt.plot()
#ax.scatter(np.linspace(0, 10, num = 10), mean_prediction[0,0:10], label="Observations")


# save 
joblib.dump(gp, 'D:/Python/Web_cable/gp.pkl')

##%  calculator 
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

fig, ax = plt.subplots(2,1,dpi=300)
ax[0].bar (-1*np.flip(SideSpanCableDistanse.reshape(-1)), np.flip(SideCableForce.reshape(-1)),width=2, align='center')
ax[1].bar ((MidSpanCableDistanse.reshape(-1)), (MidCableForce.reshape(-1)),width=2, align='center')
# fig, ax = plt.subplots(2,1)
# ax[0].plot(np.flip(SideSpanCableDistanse.reshape(1,-1)), np.flip(SideCableForce), label="索力预测值", linestyle="dotted")
# ax[1].bar(np.flip(MidSpanCableDistanse.reshape(1,-1)), np.flip(MidCableForce), label="索力预测值", linestyle="dotted")

