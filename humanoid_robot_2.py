#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def DrawT(T):
    esc=.4
    ax.plot3D(  T[0,3]+[0,esc*T[0,0]],
                T[1,3]+[0,esc*T[1,0]],
                T[2,3]+[0,esc*T[2,0]],'r')
    ax.plot3D(  T[0,3]+[0,esc*T[0,1]],
                T[1,3]+[0,esc*T[1,1]],
                T[2,3]+[0,esc*T[2,1]],'g')
    ax.plot3D(  T[0,3]+[0,esc*T[0,2]],
                T[1,3]+[0,esc*T[1,2]],
                T[2,3]+[0,esc*T[2,2]],'b')

def Rotx(alpha):
    c,s=np.cos(alpha),np.sin(alpha)
    return np.array([[1,0, 0,0],
                        [0,c,-s,0],
                        [0,s, c,0],
                        [0,0, 0,1]])
def Transx(a):
    return np.array([[1,0,0,a],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
def Transz(d):
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,d],
                        [0,0,0,1]])
def Rotz(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0,0],
                        [s, c,0,0],
                        [0, 0,1,0],
                        [0, 0,0,1]])

fig = plt.figure()

ax= plt.axes(projection='3d')
beta=np.pi/7

T=np.identity(4)
DrawT(T)
theta,d,a,alpha=[0,2,0,-np.pi/2]
Rz=Rotz(theta)
Tz=Transz(d)
Tx=Transx(a)
Rx=Rotx(alpha)
A=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
T=np.dot(T,A)
DrawT(T)
#Para el derecho
theta,d,a,alpha=[0,0,0,np.pi/2-beta]
Rz=Rotz(theta)
Tz=Transz(d)
Tx=Transx(a)
Rx=Rotx(alpha)
A=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
TD=np.dot(T,A)
DrawT(TD)
theta,d,a,alpha=[np.pi/2,1,0,np.pi/2]
Rz=Rotz(theta)
Tz=Transz(d)
Tx=Transx(a)
Rx=Rotx(alpha)
A=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
TD=np.dot(TD,A)
DrawT(TD)
#Para el izquierdo
theta,d,a,alpha=[0,0,0,np.pi/2+beta]
Rz=Rotz(theta)
Tz=Transz(d)
Tx=Transx(a)
Rx=Rotx(alpha)
A=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
TI=np.dot(T,A)
DrawT(TI)
theta,d,a,alpha=[-np.pi/2,1,0,-np.pi/2]
Rz=Rotz(theta)
Tz=Transz(d)
Tx=Transx(a)
Rx=Rotx(alpha)
A=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
TI=np.dot(TI,A)
DrawT(TI)


ax.set_aspect('equal')
plt.show()