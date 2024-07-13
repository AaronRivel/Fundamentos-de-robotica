#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def inicio(x,y):
    def poliSpline(x,y):
        k=len(x)
        n=len(x)-1
        A=np.zeros([4*n,4*n])
        b=np.zeros(4*n)


        for i in range (n):
            A[ i , i*4 : 4*(i+1) ] = np.array([x[ i ]**3,x[ i ]**2,x[ i ],1])
            b[ i ]=  y[ i ]
            A[ i+n , 4*i : (i+1)*4 ] = np.array([x[ i+1 ]**3,x[ i+1 ]**2,x[ i+1 ], 1 ])
            b[ i+n ] = y[ i+1 ]

        for i in range (n-1):
            A[ i+2*n , i*4 : 4*(i+2) ] = np.array([3*x[ i+1 ]**2,2*x[ i+1 ],1,0,-3*x[ i+1 ]**2,-2*x[ i+1 ],-1,0])
            A[ 3*n-1+i , i*4 : 4*(i+2) ] = np.array([6*x[ i+1 ],2,0,0,-6*x[ i+1 ],-2,0,0])
            
        
        A[4*n-2,0:2]=np.array([6*x[0],2])
        A[4*n-1,4*n-4:4*n-2]=np.array([6*x[n],2])
  
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

    xx=np.linspace(x[0],x[len(x)-1],1000)
    C=poliSpline(x,y)
    yy=evalSpline(xx,C)

    plt.plot(x,y,'o',xx,yy)
    plt.show()
figura = plt.figure()
plt.title('Spline general (cualquier cantidad de puntos)')
t=np.array([0,1,2,3,4,5,6,8,9,10,11,12,13])
x=np.array([0,2,2,4,9,2,3,2,6,4,2,5,9])
inicio(t,x)