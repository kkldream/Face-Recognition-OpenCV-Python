# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:02:39 2021

@author: Canvas
@function: Kalman Filter Demo
"""

import numpy as np
import matplotlib.pyplot as plt

"""
X(k) = AX(k-1) + BU(k) + w(k-1)
Z(k) = HX(k) + e(k)
p(w) = N(0, Q)
p(e) = N(0, R)
"""

def kf_predict(X0, P0, A, Q, B, U1):
    X10 = np.dot(A,X0) + np.dot(B,U1)
    P10 = np.dot(np.dot(A,P0),A.T)+ Q
    return (X10, P10)

def kf_update(X10, P10, Z, H, R):
    V = Z - np.dot(H,X10)
    K = np.dot(np.dot(P10,H.T),np.linalg.pinv(np.dot(np.dot(H,P10),H.T) + R))
    X1 = X10 + np.dot(K,V)
    P1 = np.dot(1 - np.dot(K,H),P10)
    return (X1, P1, K)

"""
加速度白噪声建模
状态方程：
x' = v'
v' = a'
a' = 0 
离散化得到；
x(k) = x(k-1)+t*v(k)+0.5*t^2*a(k)
v(k) = v(k-1)+t*a(k)
a(k) = a(k-1)

观测方程：
z(k) = x(k) + e

"""

n = 101 #数据量
nx = 3 #变量数量
t = np.linspace(0,5,n) #时间序列
dt = t[1] - t[0]

#真实函数关系
a_true = np.ones(n)*9.8
v_true = a_true*t
x_true = 0.5*a_true*(t**2)
X_true = np.concatenate([x_true, v_true, a_true]).reshape([nx,-1])
# 观测噪声协方差！！！！！！！！！！！！！！！！！！！！（可调整）
R = np.diag([5**2,0,0])

#仿真观测值
e = np.random.normal(0,np.sqrt(R[0][0]),n)
x_obs = X_true[0,:]
x_obs += e
Z = np.zeros([nx,n])
Z[0,:] = x_obs

# 计算系数
A = np.array([1,dt,0.5*dt**2, 0,1,dt, 0,0,1]).reshape([nx,nx])
B = 0
U1 = 0

#状态假设（观测）初始值
x0 = -1.0
v0 = 1.0
a0 = 9.0
X0 = np.array([x0,v0,a0]).reshape(-1,1)

#初始状态不确定度！！！！！！！！！！！！！！！！（可调整）
P0 = np.diag([5**2,2**2,1**2])

#状态递推噪声协方差！！！！！！！！！！！！！！！！！！（可调整）
Q = np.diag([0,0,1.0**2])

###开始处理
X1_np = np.copy(X0)
P1_list = [P0]
X10_np = np.copy(X0)
P10_list = [P0]

for i in range(n):
    Zi = np.array(Z[:,i]).reshape([-1,1])
    Hi = np.array([1,0,0, 0,0,0, 0,0,0]).reshape([nx,nx])
    if (i == 0):
        continue
    else:
        Xi = X10_np[:,i-1].reshape([-1,1])
        Pi = P10_list[i-1]
        X10, P10 = kf_predict(Xi, Pi, A, Q, B, U1)
        X10_np = np.concatenate([X10_np, X10], axis=1)
        P10_list.append(P10)
        X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
        X1_np = np.concatenate([X1_np, X1], axis=1)
        P1_list.append(P1)

#结束，绘图
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x_true, 'k-', label="Truth")
ax1.plot(X1_np[0,:], 'go--', label="Kalman Filter")
ax1.plot(X10_np[0,:], 'ro--', label="Prediction")
ax1.scatter(np.arange(n), Z[0,:], label="Observation", marker='*')

plt.legend()
plt.show()