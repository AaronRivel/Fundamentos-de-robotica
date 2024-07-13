#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

def update(val):
    q=[]
    for k in range(3):
        q.append(Slide[k].val*np.pi/180)

    RY=np.array([[np.cos(q[0]), 0, np.sin(q[0])],
		 [0,1,0],
		 [-np.sin(q[0]),0,  np.cos(q[0])]])##Hacer las otras rotaciones
    
    RZ = np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
		 [np.sin(q[2]),np.cos(q[2]),0],
		 [0,0,  1]])
    
    RX = np.array([[1, 0, 0],
		 [0,np.cos(q[1]),-np.sin(q[1])],
		 [0,np.sin(q[1]),  np.cos(q[1])]])
    RZY=np.dot(RZ,RY)
    RZYZ = np.dot(RZY,RX)
    Q = np.dot(RZYZ,P)
    line.set_xdata(Q[0])
    line.set_ydata(Q[1])
    line.set_3d_properties(Q[2])


fig = plt.figure()
fig.subplots_adjust(right=0.6)
ax = plt.axes(projection='3d')


P=np.array([[1,0,-1, 0,0, 1, 0,-1,0,0],
	    [0,1, 0,-1,0, 0,-1, 0,0,1],
	    [0,0, 0, 0,1, 0, 0, 0,1,0]])
theta=0
R=np.array([[np.cos(theta), -np.sin(theta), 0],
	    [np.sin(theta),  np.cos(theta), 0],
	    [0, 0, 1]])
Q=np.dot(R,P)


ax.plot3D([0,1],[0,0],[0,0],'r')
ax.plot3D([0,0],[0,1],[0,0],'g')
ax.plot3D([0,0],[0,0],[0,1],'b')
ax.plot3D(P[0],P[1],P[2])
line, =ax.plot3D(Q[0],Q[1],Q[2])
Slide=[]
label=['Y','X','Z']
for k in range(3):
    axrot = fig.add_axes([0.7, 0.8-.1*k, 0.2, 0.03])
    Slide.append( Slider(
        ax=axrot,
        label=label[k],
        valmin=-180,
        valmax=180,
        valinit=0,
    ))
    Slide[k].on_changed(update)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
#ax.set_aspect('equal', 'box')
plt.show()