#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

def update(val):
    q=[]
    for i in range(3):
        q.append(Slide[i].val)
    
    theta = np.array([90,q[0],q[1],q[2]])*np.pi/180
    d = [2,0,0,0]
    
    for i in range (4):
        Rz=np.array([[np.cos(theta[i]), -np.sin(theta[i]),0,0],
                [np.sin(theta[i]),  np.cos(theta[i]),0,0],
                [0,0,1,0],
                [0,0,0,1]])

        Tz=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,d[i]],
                    [0,0,0,1]])
        Tx=np.array([[1,0,0,a[i]],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Rx=np.array([[1,0,0,0],
                    [0,np.cos(alpha[i]),-np.sin(alpha[i]),0],
                    [0,np.sin(alpha[i]), np.cos(alpha[i]),0],
                    [0,0,0,1]])
        T[i]= np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
    
    T0_1 = T[0]
    T0_2 = np.dot(T0_1,T[1])
    T0_3 = np.dot(T0_2,T[2])
    T0_E = np.dot(T0_3,T[3])

    T_array = np.array([T0_1,T0_2,T0_3,T0_E])   
        
    for i in range (4):
        T_aux = T_array[i]
        lines_x[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,0]]))
        lines_x[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,0]]))
        lines_x[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,0]]))
        lines_y[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,1]]))
        lines_y[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,1]]))
        lines_y[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,1]]))
        lines_z[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,2]]))
        lines_z[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,2]]))
        lines_z[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,2]]))
        


        
fig = plt.figure()
fig.subplots_adjust(right=0.6)
ax =[fig.add_axes([0.05,0.15,.80,0.9],projection = '3d'),
     fig.add_axes([0.03,0.1,.17,0.05]),
     fig.add_axes([0.28,0.1,.17,0.05]),
     fig.add_axes([0.54,0.1,.17,0.05])]

ax[0].plot3D([0,1],[0,0],[0,0],'r')
ax[0].plot3D([0,0],[0,1],[0,0],'g')
ax[0].plot3D([0,0],[0,0],[0,1],'b')
T = np.array([np.identity(4),np.identity(4),np.identity(4),np.identity(4)])

theta = np.array([90,0,0,0])*np.pi/180
d = [2,0,0,0]
a = [0,2,2,2]
alpha = np.array([90,0,0,0])*np.pi/180
for i in range (4):
    Rz=np.array([[np.cos(theta[i]), -np.sin(theta[i]),0,0],
             [np.sin(theta[i]),  np.cos(theta[i]),0,0],
             [0,0,1,0],
             [0,0,0,1]])

    Tz=np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,d[i]],
                [0,0,0,1]])
    Tx=np.array([[1,0,0,a[i]],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
    Rx=np.array([[1,0,0,0],
                [0,np.cos(alpha[i]),-np.sin(alpha[i]),0],
                [0,np.sin(alpha[i]), np.cos(alpha[i]),0],
                [0,0,0,1]])
    T[i]= np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
    
    
T0_1 = T[0]
T0_2 = np.dot(T0_1,T[1])
T0_3 = np.dot(T0_2,T[2])
T0_E = np.dot(T0_3,T[3])

T_array = np.array([T0_1,T0_2,T0_3,T0_E])
lines_x=[]
lines_y = []
lines_z = []
colors = ['r','g','b']
for j in range (3):
    for i in range (4):
        T_aux = T_array[i]
        line, = ax[0].plot3D(T_aux[0,3]+np.array([0,T_aux[0,j]]),
                            T_aux[1,3]+np.array([0,T_aux[1,j]]),
                            T_aux[2,3]+np.array([0,T_aux[2,j]]),colors[j])
        if j == 0:
            lines_x.append(line)
        if j == 1:
            lines_y.append(line)
        if j ==2:
            lines_z.append(line)

Slide=[]
label=['O1','O2','O3']
for i in range(3):
    Slide.append( Slider(
        ax=ax[i+1],
        label=label[i],
        valmin=0,
        valmax=360,
        valinit=0,
    ))
    Slide[i].on_changed(update)
ax[0].set_xlim([-8,8])
ax[0].set_ylim([-8,8])
ax[0].set_zlim([0,10])
plt.show()