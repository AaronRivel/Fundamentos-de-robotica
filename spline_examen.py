#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def inicio(x,y):
    def poliSpline(x,y):
        n=len(x)-1
        A=np.zeros([4*n,4*n])
        b=np.zeros(4*n)
        
        for i in range (n):
            A[ i , i*4 : 4*(i+1) ]=np.array([x[ i ]**3,x[ i ]**2,x[ i ],1])
            b[ i ]=y[ i ]
            A[ i+n , 4*i : (i+1)*4 ]=np.array([x[ i+1 ]**3,x[ i+1 ]**2,x[ i+1 ], 1 ])
            b[ i+n ]=y[ i+1 ]
            
        for i in range (n-1):
            A[ i+2*n , i*4 : 4*(i+2) ]=np.array([3*x[ i+1 ]**2,2*x[ i+1 ],1,0,-3*x[ i+1 ]**2,-2*x[ i+1 ],-1,0])
            A[ 3*n-1+i , i*4 : 4*(i+2) ]=np.array([6*x[ i+1 ],2,0,0,-6*x[ i+1 ],-2,0,0])
        A[4*n-2,0:2]=np.array([6*x[0],2])
        A[4*n-1,4*n-4:4*n-2]=np.array([6*x[n],2])
        q = np.dot(np.linalg.inv(A),b)
        C=[]#coeficientes 
        for k in range(len(x)-1):
            C.append(q[4*k:4*k+4])
        return C   
    def evalSpline(xx,C,x):
        yy=np.zeros(len(xx))
        for k in range(len(C)):
            yy +=(C[k][0]*np.power(xx,3)+C[k][1]*np.power(xx,2)+C[k][2]*xx+C[k][3])*(x[k]<=xx) * (xx < x[k+1])
        yy +=(C[len(C)-1][0]*np.power(xx,3)+C[len(C)-1][1]*np.power(xx,2)+C[len(C)-1][2]*xx+C[len(C)-1][3])*(x[len(C)]==xx)
        return yy

    t=range(len(x))
    tt=np.linspace(t[0],t[len(t)-1],500)
    X_Coeficientes=poliSpline(t,x)
    Y_Coeficientes=poliSpline(t,y)
    xx=evalSpline(tt,X_Coeficientes,t)
    yy=evalSpline(tt,Y_Coeficientes,t)
    plt.axis('equal')
    plt.plot(x,y,'*',xx,yy,'-')
    plt.show()
figura = plt.figure()
plt.title('Examen')


x=[0.203, 0.109, 0.111, 0.201, 0.211, 0.326, 0.256, 0.328, 0.357, 0.361, 0.436, 0.404, 0.486, 0.488, 0.551, 0.607, 0.656, 0.598, 0.66, 0.719, 0.713, 0.779, 0.789, 0.852]
y=[0.404, 0.529, 0.234, 0.378, 0.255, 0.38, 0.315, 0.24, 0.365, 0.232, 0.37, 0.352, 0.346, 0.234, 0.362, 0.229, 0.378, 0.409, 0.359, 0.391, 0.224, 0.406, 0.245, 0.255]

inicio(x,y)