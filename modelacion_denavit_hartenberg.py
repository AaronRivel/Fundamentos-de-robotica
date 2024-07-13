#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d



def drawT(T):#Esta función dibuja los marcos dada la transformación
    ax.plot3D(T[0,3]+esc*np.array([0,T[0,0]]),T[1,3]+esc*np.array([0,T[1,0]]),T[2,3]+esc*np.array([0,T[2,0]]),'r')
    ax.plot3D(T[0,3]+esc*np.array([0,T[0,1]]),T[1,3]+esc*np.array([0,T[1,1]]),T[2,3]+esc*np.array([0,T[2,1]]),'g')
    ax.plot3D(T[0,3]+esc*np.array([0,T[0,2]]),T[1,3]+esc*np.array([0,T[1,2]]),T[2,3]+esc*np.array([0,T[2,2]]),'b')

esc=0.5
fig = plt.figure
#ax = plt.axes(projection='3d')
ax = plt.axes(projection = "3d")
#Definción del marco del mundo
T=np.identity(4)
drawT(T)


#1. Parámetros DH
theta, d, a, alpha = [np.pi/2, 1, 0, np.pi/2]

Rz=np.array([[np.cos(theta), -np.sin(theta),0,0],
             [np.sin(theta),  np.cos(theta),0,0],
             [0,0,1,0],
             [0,0,0,1]])

Tz=np.array([[1,0,0,0],
             [0,1,0,0],
             [0,0,1,d],
             [0,0,0,1]])
Tx=np.array([[1,0,0,a],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]])
Rx=np.array([[1,0,0,0],
             [0,np.cos(alpha),-np.sin(alpha),0],
             [0,np.sin(alpha), np.cos(alpha),0],
             [0,0,0,1]])

T=np.dot(T,np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx)))
drawT(T)
#2. Parámetros DH
theta, d, a, alpha = [0, 0, 1, 0]

Rz=np.array([[np.cos(theta), -np.sin(theta),0,0],
             [np.sin(theta),  np.cos(theta),0,0],
             [0,0,1,0],
             [0,0,0,1]])

Tz=np.array([[1,0,0,0],
             [0,1,0,0],
             [0,0,1,d],
             [0,0,0,1]])
Tx=np.array([[1,0,0,a],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]])
Rx=np.array([[1,0,0,0],
             [0,np.cos(alpha),-np.sin(alpha),0],
             [0,np.sin(alpha), np.cos(alpha),0],
             [0,0,0,1]])

T=np.dot(T,np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx)))
print(T)
drawT(T)
#3. Parámetros DH
theta, d, a, alpha = [0, 0, 1, 0]

Rz=np.array([[np.cos(theta), -np.sin(theta),0,0],
             [np.sin(theta),  np.cos(theta),0,0],
             [0,0,1,0],
             [0,0,0,1]])

Tz=np.array([[1,0,0,0],
             [0,1,0,0],
             [0,0,1,d],
             [0,0,0,1]])
Tx=np.array([[1,0,0,a],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]])
Rx=np.array([[1,0,0,0],
             [0,np.cos(alpha),-np.sin(alpha),0],
             [0,np.sin(alpha), np.cos(alpha),0],
             [0,0,0,1]])

T=np.dot(T,np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx)))
print(T)
drawT(T)

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
plt.show()