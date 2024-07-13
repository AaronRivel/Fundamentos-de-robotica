#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

import matplotlib.animation as animation2

flag = False

def plot_matrix(f,T,handel_figure,s):
    lines_x=[]
    lines_y = []
    lines_z = []

    vectors = []
    for j in range (3):
        for i in range (f):
            colors = ['r','g','b']

            T_aux = T[i]
            line, = handel_figure.plot3D(T_aux[0,3]+s*np.array([0,T_aux[0,j]]),
                                T_aux[1,3]+s*np.array([0,T_aux[1,j]]),
                                T_aux[2,3]+s*np.array([0,T_aux[2,j]]),colors[j])
            
            if(i < f-1 and j == 0):
                T_aux = T[i]
                T_aux2 = T[i+1]
                vector, = handel_figure.plot3D([T_aux[0][3],T_aux2[0][3]],[T_aux[1][3],T_aux2[1][3]],[T_aux[2][3],T_aux2[2][3]],'k')
                vectors.append(vector)
                    
            if j == 0:
                lines_x.append(line)
            if j == 1:
                lines_y.append(line)
            if j ==2:
                lines_z.append(line)
    return lines_x,lines_y,lines_z,vectors       
def frame_transformation(t,dz,da,a,f,T_b):
    T = []
    T.append(T_b)
    for i in range (f-1):
        T.append(np.identity(4))

    for i in range (f):
        Rz=np.array([[np.cos(t[i]), -np.sin(t[i]),0,0],
                [np.sin(t[i]),  np.cos(t[i]),0,0],
                [0,0,1,0],
                [0,0,0,1]])

        Tz=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,dz[i]],
                    [0,0,0,1]])
        Tx=np.array([[1,0,0,da[i]],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Rx=np.array([[1,0,0,0],
                    [0,np.cos(a[i]),-np.sin(a[i]),0],
                    [0,np.sin(a[i]), np.cos(a[i]),0],
                    [0,0,0,1]])
        if (i == 0):
            T[i]= np.dot(T_b,np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx)))
        else:
           T[i]=np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
    return T
def update_frames(T,hx,hy,hz,hv,f,s):
    for i in range (f):
        T_aux = T[i]
        hx[i].set_xdata(T_aux[0,3]+s*np.array([0,T_aux[0,0]]))
        hx[i].set_ydata(T_aux[1,3]+s*np.array([0,T_aux[1,0]]))
        hx[i].set_3d_properties(T_aux[2,3]+s*np.array([0,T_aux[2,0]]))
        hy[i].set_xdata(T_aux[0,3]+s*np.array([0,T_aux[0,1]]))
        hy[i].set_ydata(T_aux[1,3]+s*np.array([0,T_aux[1,1]]))
        hy[i].set_3d_properties(T_aux[2,3]+s*np.array([0,T_aux[2,1]]))
        hz[i].set_xdata(T_aux[0,3]+s*np.array([0,T_aux[0,2]]))
        hz[i].set_ydata(T_aux[1,3]+s*np.array([0,T_aux[1,2]]))
        hz[i].set_3d_properties(T_aux[2,3]+s*np.array([0,T_aux[2,2]]))

        if(i < f-1):
                T_aux = T[i]
                T_aux2 = T[i+1]
                hv[i].set_xdata([T_aux[0][3],T_aux2[0][3]])
                hv[i].set_ydata([T_aux[1][3],T_aux2[1][3]])
                hv[i].set_3d_properties([T_aux[2][3],T_aux2[2][3]])
def gate_patern(Q,x,a,b):
    x0 = Q[0]
    y0 = Q[1]
    y =(b*np.sqrt(a**2-x**2+2*x*x0-x0**2))/(a)+y0
    return y
def main(plot,matrix,frame_update,gate):
    def animation(frame):
        x_value = x2[round(frame)]-20
        y_value = walk[round(frame)]-20
        c = np.sqrt(x_value**2+y_value**2)
        B = np.arctan2(x_value,y_value)*180/np.pi
        phi=  B-90
        alpha2 = np.arccos((lt**2-lf**2-c**2)/(-2*lf*c))*180/np.pi
        theta_f = 180-phi-alpha2
        theta_d_2  = np.arccos((c**2-lf**2-lt**2)/(-2*lf*lt))*180/np.pi
        beta = 90 - theta_d_2/2
        theta_p = theta_f +2*beta

        phi3 = (360-2*(theta_p-theta_f))/2

        theta = np.array([theta_p,0,-phi3,theta_f,0,-phi3,-delta])*np.pi/180

        Transformation = matrix(theta,d,a,alpha,frames,base)

        T_array = []
        T_array.append(Transformation[0])
        for i in range (frames-1):
            if(i == 2):
                T_array.append(np.dot(base,Transformation[3]))
            else: 
                T_array.append(np.dot(T_array[i],Transformation[i+1]))


        update_frames(T_array,handel_xaxis,handel_yaxis,handel_zaxis,handel_vectors,frames,scale)

        theta_p_array.append(theta_p)
        theta_f_array.append(theta_f)
        global flag
        if(round(frame) == len(walk)-1 and flag == False):
            flag = True
            print('______________________________________________________________________________________')
            print('Theta P = ',theta_p_array)
            print('----------------------')
            print('Theta F = ',theta_f_array)
            print('______________________________________________________________________________________')
        
    #---------------------------------------------------------------------------------------------------------------
    frames = 7 
    delta = 0
    scale = 2
    ld1,ld2,lf,lp,lt = [7,7,7,7,14]

    fig = plt.figure()
    fig.subplots_adjust(right=0.6)
    ax =fig.add_axes([0.05,0.15,.80,0.9],projection = '3d')

    ax.plot3D([0,1],[0,0],[0,0],'r')
    ax.plot3D([0,0],[0,1],[0,0],'g')
    ax.plot3D([0,0],[0,0],[0,1],'b')

    path_size = 20

    Q0 = [20,5]
    ar = 10
    br = 5
    x = np.linspace(Q0[0]+ar,Q0[0]-ar,path_size)
    elipse = gate(Q0,x,ar,br)
    line = np.linspace(Q0[0]-ar,Q0[0]+ar,path_size)
    
    walk = []
    x2 = []
    for i in range (len(x)*2):
        if (i < path_size):
            x2.append(x[i])
            walk.append(elipse[i])
        else:
            x2.append(line[i-path_size])
            walk.append(5)
    ax.plot(x,elipse,'-b',line,np.zeros(len(line))+5,'-b')
    ax.plot(x2,walk,'--r')

    theta = np.array([45+90,0,-90,-45+90,0,-90,0])*np.pi/180
    d = [0,0,0,0,0,0,0]
    a = [0,-lp,ld2,0,-lf,ld1,lt/2]
    alpha = np.array([180,0,0,0,0,0,0])*np.pi/180
    base = np.identity(4)

    base[0][3] = 20
    base [1][3] = 20

    Transformation = matrix(theta,d,a,alpha,frames,base)

    T_array = []
    T_array.append(Transformation[0])

    for i in range (frames-1):
        if(i == 2):
            T_array.append(np.dot(base,Transformation[3]))
        else: 
            T_array.append(np.dot(T_array[i],Transformation[i+1]))

    handel_xaxis,handel_yaxis,handel_zaxis,handel_vectors = plot(frames,T_array,ax,scale)

    ax.set_xlim([0,40])
    ax.set_ylim([0,25])
    ax.set_zlim([0,10])
    ax.set_aspect('equal', 'box')

    h = 0.1
    ani = animation2.FuncAnimation(fig=fig, func=animation, frames=len(x2), interval=1000*h)
    plt.show()

theta_p_array = []
theta_f_array = []
main(plot_matrix,frame_transformation,update_frames,gate_patern)
