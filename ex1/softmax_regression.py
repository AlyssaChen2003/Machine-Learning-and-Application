# coding=utf-8
import numpy as np
from numba import jit

@jit(nopython=True)
def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    m=x.shape[0]
    
    #Y[m,k]OneHot编码
    for i in range(iters):
        theta_xi=np.dot(x,theta.T)
        fenzi=np.exp(theta_xi) #分子
        fenmu=np.sum(np.exp(theta_xi),axis=1,keepdims=True)#分母
        h_theta_xi=fenzi/fenmu
        J_theta=-np.sum(np.multiply(np.log(h_theta_xi),y.T).reshape(y.T.size,-1))/m #reshape控制为Y.size=m行，列数系统自己定
        f=J_theta
        print("i=",i,"loss=",f)
        g=np.dot((h_theta_xi-y.T).T,x)/m #下降的梯度
        theta=theta-alpha*g
    return theta
