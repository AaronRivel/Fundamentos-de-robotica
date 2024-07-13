#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def inicio(x,y):
    def poliSpline(x,y):
        n=len(x)-1
        A=np.zeros([4*n,4*n])
        b=np.zeros(4*n)
        #Primera condición
        A[ 0 , 0 : 4 ]=np.array([x[ 0 ]**3,x[ 0 ]**2,x[ 0 ],1])
        A[ 1 , 4 : 8 ]=np.array([x[ 1 ]**3,x[ 1 ]**2,x[ 1 ],1])
        A[ 2 , 8 : 12 ]=np.array([x[ 2 ]**3,x[ 2 ]**2,x[ 2 ],1])
        A[ 3 , 12 : 16 ]=np.array([x[ 3 ]**3,x[ 3 ]**2,x[ 3 ],1])
        b[ 0 ]=y[ 0 ]
        b[ 1 ]=y[ 1 ]
        b[ 2 ]=y[ 2 ]
        b[ 3 ]=y[ 3 ]
        #condición 2
        A[ 4 , 0 : 4 ]=np.array([x[ 1 ]**3,x[ 1 ]**2,x[ 1 ], 1 ])
        A[ 5 , 4 : 8 ]=np.array([x[ 2 ]**3,x[ 2 ]**2,x[ 2 ], 1 ])
        A[ 6 , 8 : 12 ]=np.array([x[ 3 ]**3,x[ 3 ]**2,x[ 3 ], 1 ])
        A[ 7 , 12 : 16 ]=np.array([x[ 4 ]**3,x[ 4 ]**2,x[ 4 ], 1 ])
        b[ 4 ]=y[ 1 ]
        b[ 5 ]=y[ 2 ]
        b[ 6 ]=y[ 3 ]
        b[ 7 ]=y[ 4 ]
        #Tercera condición
        A[ 8 , 0 : 8 ]=np.array([3*x[ 1 ]**2,2*x[ 1 ],1,0,-3*x[ 1 ]**2,-2*x[ 1 ],-1,0])
        A[ 9 , 4 : 12 ]=np.array([3*x[ 2 ]**2,2*x[ 2 ],1,0,-3*x[ 2 ]**2,-2*x[ 2 ],-1,0])
        A[ 10 , 8 : 16 ]=np.array([3*x[ 3 ]**2,2*x[ 3 ],1,0,-3*x[ 3 ]**2,-2*x[ 3 ],-1,0])
        #cuarta condición
        A[ 11 , 0 : 8 ]=np.array([6*x[ 1 ],2,0,0,-6*x[ 1 ],-2,0,0])
        A[ 12 , 4 : 12 ]=np.array([6*x[ 2 ],2,0,0,-6*x[ 2 ],-2,0,0])
        A[ 13 , 8 : 16 ]=np.array([6*x[ 3 ],2,0,0,-6*x[ 3 ],-2,0,0])

        A[14,0:2]=np.array([6*x[0],2])
        A[15,4*n-4:4*n-2]=np.array([6*x[n],2])

        q = np.dot(np.linalg.inv(A),b)
        C=[]
        for k in range(len(x)-1):
            C.append(q[4*k:4*k+4])
        return C
    def evalSpline(xx,C):
        yy=np.zeros(len(xx))
        for k in range(len(C)):
            yy +=(C[k][0]*np.power(xx,3)+C[k][1]*np.power(xx,2)+C[k][2]*xx+C[k][3])*(x[k]<=xx) * (xx < x[k+1])
        yy +=(C[len(C)-1][0]*np.power(xx,3)+C[len(C)-1][1]*np.power(xx,2)+C[len(C)-1][2]*xx+C[len(C)-1][3])*(x[len(C)]==xx)
        return yy

    xx=np.linspace(x[0],x[len(x)-1],100)
    C=poliSpline(x,y)

    yy=evalSpline(xx,C)

    plt.plot(x,y,'o',xx,yy)
    plt.show()
figura = plt.figure()
plt.title('Spline cúbico usando 5 puntos (hacer caso general)')
t=np.array([0,1,2,3,4])
x=np.array([0,4,5,4,0])
inicio(t,x)