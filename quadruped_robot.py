#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

numero_marcos = 7 #sin contar el marco 0
joints = 2 #variables
delta = 0
def update(val):
    q=[]
    for i in range(joints):
        q.append(Slide[i].val)
    
    fi = (360-2*(q[0]-q[1]))/2

    theta = np.array([q[0],0,fi,q[1],0,-fi,-delta])*np.pi/180
    d = [0,0,0,0,0,0,0]
    for i in range (numero_marcos):
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

    T_array = []
    T_array.append(T[0])
    for i in range (numero_marcos-1):
        if(i == 2):
            T_array.append(T[3])
        else: 
            T_array.append(np.dot(T_array[i],T[i+1]))
    
        
    for i in range (numero_marcos):
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
        
        if(i < numero_marcos-1 and (i != 2 and i+1 != 3)):
            T_aux = T_array[i]
            T_aux2 = T_array[i+1]
            if(i >= 3):
                vectors[i-1].set_xdata([T_aux[0][3],T_aux2[0][3]])
                vectors[i-1].set_ydata([T_aux[1][3],T_aux2[1][3]])
                vectors[i-1].set_3d_properties([T_aux[2][3],T_aux2[2][3]])
            else:
                vectors[i].set_xdata([T_aux[0][3],T_aux2[0][3]])
                vectors[i].set_ydata([T_aux[1][3],T_aux2[1][3]])
                vectors[i].set_3d_properties([T_aux[2][3],T_aux2[2][3]])

            if (i == 5):
                T_aux = T_array[4]
                T_aux2 = T_array[6]
                vectors[i].set_xdata([T_aux[0][3],T_aux2[0][3]])
                vectors[i].set_ydata([T_aux[1][3],T_aux2[1][3]])
                vectors[i].set_3d_properties([T_aux[2][3],T_aux2[2][3]])
            


        
fig = plt.figure()
fig.subplots_adjust(right=0.6)
ax =[fig.add_axes([0.05,0.15,.80,0.9],projection = '3d'),
     fig.add_axes([0.13,0.1,.30,0.05]),
     fig.add_axes([0.58,0.1,.30,0.05])]

ax[0].plot3D([0,1],[0,0],[0,0],'r')
ax[0].plot3D([0,0],[0,1],[0,0],'g')
ax[0].plot3D([0,0],[0,0],[0,1],'b')
T = []
for i in range (numero_marcos):
    T.append(np.identity(4))

theta = np.array([135,0,90,45,0,-90,0])*np.pi/180
d = [0,0,0,0,0,0,0]
a = [0,-7,7,0,-7,7,7]
alpha = np.array([0,0,0,0,0,0,0])*np.pi/180

for i in range (numero_marcos):
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
    
T_array = []
T_array.append(T[0])
for i in range (numero_marcos-1):
    if(i == 2):
        T_array.append(T[3])
    else: 
        T_array.append(np.dot(T_array[i],T[i+1]))
lines_x=[]
lines_y = []
lines_z = []

vectors = []
for j in range (3):
    for i in range (numero_marcos):
        colors = ['r','g','b']
        if( i == 2):
            colors = ['k','k','k']
        if (i == 5):
            colors = ['m','m','m']
        if(i == 6):
            colors = ['y','y','y']
        
        
        T_aux = T_array[i]
        line, = ax[0].plot3D(T_aux[0,3]+np.array([0,T_aux[0,j]]),
                            T_aux[1,3]+np.array([0,T_aux[1,j]]),
                            T_aux[2,3]+np.array([0,T_aux[2,j]]),colors[j])
        vector_color = []
        if (i <= 1):
            vector_color = 'k'
        if (i >= 2):
            vector_color = 'm'
        
        if(i < numero_marcos-1 and (i != 2 and i+1 != 3)and j == 0):
            T_aux = T_array[i]
            T_aux2 = T_array[i+1]
            vector, = ax[0].plot3D([T_aux[0][3],T_aux2[0][3]],[T_aux[1][3],T_aux2[1][3]],[T_aux[2][3],T_aux2[2][3]],vector_color)
            vectors.append(vector)
            if (i == 5):
                T_aux = T_array[4]
                T_aux2 = T_array[6]
                vector, = ax[0].plot3D([T_aux[0][3],T_aux2[0][3]],[T_aux[1][3],T_aux2[1][3]],[T_aux[2][3],T_aux2[2][3]],vector_color)
                vectors.append(vector)
                
        if j == 0:
            lines_x.append(line)
        if j == 1:
            lines_y.append(line)
        if j ==2:
            lines_z.append(line)

Slide=[]
label=['theta P','theta F']
for i in range(joints):
    if(i == 0):
        valmin_ = 0
        valmax_ = 720
        valinit_ = 360+135
    if (i == 1):
        valmin_ = 0
        valmax_ = 720
        valinit_ = 360+45
    Slide.append( Slider(
        ax=ax[i+1],
        label=label[i],
        valmin=valmin_,
        valmax=valmax_,
        valinit=valinit_,
    ))
    Slide[i].on_changed(update)
ax[0].set_xlim([-15,15])
ax[0].set_ylim([-20,2.5])
ax[0].set_zlim([0,10])
plt.show()