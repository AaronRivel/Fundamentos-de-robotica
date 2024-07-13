#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
from cmath import sqrt as sq
from cmath import atan as arctan

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
            
def main(plot,matrix,frame_update):
    def update(val):
        q=[]
        for i in range(joints):
            q.append(Slide[i].val)

        S1 = (q[0]**2 + q[1]**2 + lf**2 - lt**2)/(2*lf)
        theta_f = 2*np.arctan((q[1]+np.sqrt(q[1]**2 + q[0]**2-S1**2))/(q[0]+S1))
        theta_f = theta_f*180.0/np.pi

        theta_d1 = np.arctan2(q[1]-lf*np.sin(theta_f),q[0]-lf*np.cos(theta_f))-delta

        rd = np.array([lf*np.cos(theta_f)+ld1*np.cos(theta_d1),lf*np.sin(theta_f)+ld1*np.sin(theta_d1)])

        S2 = (rd[0]**2 + rd[1]**2 + lt**2 - lf**2)/(2*lt)
        theta_d2 = 2*np.arctan((rd[1]+np.sqrt(rd[1]**2 + rd[0]**2-S2**2))/(rd[0]+S2))
        theta_d2 = theta_d1*180.0/np.pi

        S3 = (rd[0]**2 + lp**2 +rd[1]**2 -ld1**2)/(2*lp)
        theta_p = 2*np.arctan((rd[1]+np.sqrt(rd[1]**2 + rd[0]**2-S3**2))/(rd[0]+S3))
        theta_p = theta_p*180.0/np.pi

        

        phi = (360-2*(theta_p-theta_f))/2

        theta = np.array([theta_p+90,0,phi,theta_f+90,0,-phi,-delta])*np.pi/180

        Transformation = matrix(theta,d,a,alpha,frames,base)

        T_array = []
        T_array.append(Transformation[0])
        for i in range (frames-1):
            if(i == 2):
                T_array.append(np.dot(base,Transformation[3]))
            else: 
                T_array.append(np.dot(T_array[i],Transformation[i+1]))

        rd = T_array[2]

        update_frames(T_array,handel_xaxis,handel_yaxis,handel_zaxis,handel_vectors,frames,scale)
        
    #---------------------------------------------------------------------------------------------------------------
    frames = 7 
    joints = 2
    delta = 0
    scale = 2
    ld1,ld2,lf,lp,lt = [7,7,7,7,14]

    fig = plt.figure()
    fig.subplots_adjust(right=0.6)
    ax =[fig.add_axes([0.05,0.15,.80,0.9],projection = '3d'),
        fig.add_axes([0.05,0.1,.30,0.05]),
        fig.add_axes([0.58,0.1,.30,0.05])]

    ax[0].plot3D([0,1],[0,0],[0,0],'r')
    ax[0].plot3D([0,0],[0,1],[0,0],'g')
    ax[0].plot3D([0,0],[0,0],[0,1],'b')

    theta = np.array([135,0,90,45,0,-90,0])*np.pi/180
    d = [0,0,0,0,0,0,0]
    a = [0,-lp,ld2,0,-lf,ld1,lt/2]
    alpha = np.array([0,0,0,0,0,0,0])*np.pi/180
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

    handel_xaxis,handel_yaxis,handel_zaxis,handel_vectors = plot(frames,T_array,ax[0],scale)

    Slide=[]
    label=['Xrp','Yrp']
    for i in range(joints):
        if(i == 0 ):
            valmin_ = -20
            valmax_ = 20
            valinit_ = 10#25
        if(i == 1 ):
            valmin_ = -20
            valmax_ = 20
            valinit_ = 10#5
        Slide.append( Slider(
            ax=ax[i+1],
            label=label[i],
            valmin=valmin_,
            valmax=valmax_,
            valinit=valinit_,
        ))
        Slide[i].on_changed(update)

    ax[0].set_xlim([0,40])
    ax[0].set_ylim([0,40])
    ax[0].set_zlim([0,10])
    ax[0].set_aspect('equal', 'box')
    plt.show()
    
main(plot_matrix,frame_transformation,update_frames)