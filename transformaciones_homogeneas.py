#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:00:31 2024

@author: aaron
"""

import matplotlib.pyplot as plt
import numpy as np
 
from matplotlib.widgets import Slider
 
def update(val):
    q=[]
    for k in range(6):
        q.append(Slide[k].val)
    print(q)
    A1=np.array([[np.cos(q[0]), -np.sin(q[0]), 0],
                 [np.sin(q[0]), np.cos(q[0]), 1],
                 [0           ,0            , 1]])
    
    A2=np.array([[np.cos(q[1]), -np.sin(q[1]), 1],
                 [np.sin(q[1]), np.cos(q[1]), 0],
                 [0           , 0           , 1]])
    
    A3=np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                 [np.sin(q[2]), np.cos(q[2]), -1],
                 [0           , 0           , 1]])
    T=np.dot(A1,A2)
    T2 = np.dot(T,A3)
    print("T=",T)
    print("T2 = ", T2)
    M1[0].set_xdata(A1[0,2]+esc*np.array([0,A1[0,0]]))
    M1[0].set_ydata(A1[1,2]+esc*np.array([0,A1[1,0]]))
 
    M1[1].set_xdata(A1[0,2]+esc*np.array([0,A1[0,1]]))
    M1[1].set_ydata(A1[1,2]+esc*np.array([0,A1[1,1]]))
 
    M1[2].set_xdata(T[0,2]+esc*np.array([0,T[0,0]]))
    M1[2].set_ydata(T[1,2]+esc*np.array([0,T[1,0]]))
 
    M1[3].set_xdata(T[0,2]+esc*np.array([0,T[0,1]]))
    M1[3].set_ydata(T[1,2]+esc*np.array([0,T[1,1]]))
    
    M1[4].set_xdata(T2[0,2]+esc*np.array([0,T2[0,0]]))
    M1[4].set_ydata(T2[1,2]+esc*np.array([0,T2[1,0]]))
 
    M1[5].set_xdata(T2[0,2]+esc*np.array([0,T2[0,1]]))
    M1[5].set_ydata(T2[1,2]+esc*np.array([0,T2[1,1]]))
    
    
 
    
def plotM(a,A1,A2,A3):
    a.plot(esc*np.array([0,1]),
           esc*np.array([0,0]),'r')
    a.plot(esc*np.array([0,0]),
           esc*np.array([0,1]),'g')
    lines=[]
    T=np.dot(A1,A2)
    print("T=",T)

    line, =a.plot(A1[0,2]+esc*np.array([0,A1[0,0]]),
                  A1[1,2]+esc*np.array([0,A1[1,0]]),'r')
    lines.append(line)
    line, =a.plot(A1[0,2]+esc*np.array([0,A1[0,1]]),
                  A1[1,2]+esc*np.array([0,A1[1,1]]),'g')
    lines.append(line)
    line, =a.plot(T[0,2]+esc*np.array([0,T[0,0]]),
                  T[1,2]+esc*np.array([0,T[1,0]]),'r')
    lines.append(line)
    line, =a.plot(T[0,2]+esc*np.array([0,T[0,1]]),
                  T[1,2]+esc*np.array([0,T[1,1]]),'g')
    lines.append(line)
    line, =a.plot(A3[0,2]+esc*np.array([0,A3[0,0]]),
                  A3[1,2]+esc*np.array([0,A3[1,0]]),'r')
    lines.append(line)
    line, =a.plot(A3[0,2]+esc*np.array([0,A3[0,1]]),
                  A3[1,2]+esc*np.array([0,A3[1,1]]),'g')
    lines.append(line)

    
    return lines
    
 
esc=0.5
S0=np.identity(3)
theta=0
A1=np.array([[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta),  1],
             [0,             0,              1]])
    
A2=np.array([[np.cos(theta), -np.sin(theta), 1],
             [np.sin(theta), np.cos(theta),  0],
             [0,             0,              1]])
    
A3=np.array([[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta),  -1],
             [0            ,0               ,1]])
 
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)
M1=plotM(ax,A1,A2,A3)
Slide=[]
for k in range(6):
    axrot = fig.add_axes([0.7, 0.8-.1*k, 0.2, 0.03])
    Slide.append( Slider(
        ax=axrot,
        label='L%d'%k,
        valmin=-np.pi,
        valmax=np.pi,
        valinit = 0
    ))
    Slide[k].on_changed(update)
 
update(theta)
ax.grid()
ax.axis('equal')
ax.set_xlim([-5, 7])
ax.set_ylim([-3, 3])
plt.show()
 