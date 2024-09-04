#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time as tm

init_time = tm.time()

window_closed = False

def on_close(event):
    global window_closed
    window_closed = True

def Ref(t):
    T = 5
    a = 5
    b = 2
    
    ref = np.array([a*np.sin(np.pi/T*t)  , b*np.cos(np.pi/T*t) - 15 , 0])
    vel_ref = np.array([np.pi*a*np.cos(np.pi/T*t)*(1/T) ,  -np.pi*b*np.sin(np.pi/T*t)*(1/T) , 0])

    return ref, vel_ref

def DHParameters(q):
    phi =  np.pi - abs(q[3]) + abs(q[1])
    return [[ phi,  0, -7,  0],
            [q[1],  0,  0,  0],
            [   0,  0, -7,  0],
            [q[3],  0,  7,  0],
            [   0,  0,  7,  0]]

def MatrixT(q):
    DH =  DHParameters(q)
    T = []
    for i in range (len(q)):
        T.append(np.identity(4))
        
    for i in range (len(q)):
        
        Rz=np.array([[np.cos(DH[i][0]), -np.sin(DH[i][0]),0,0],
                    [np.sin(DH[i][0]),  np.cos(DH[i][0]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Tz=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,DH[i][1]],
                    [0,0,0,1]])
        Tx=np.array([[1,0,0,DH[i][2]],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Rx=np.array([[1,0,0,0],
                    [0,np.cos(DH[i][3]),-np.sin(DH[i][3]),0],
                    [0,np.sin(DH[i][3]), np.cos(DH[i][3]),0],
                    [0,0,0,1]])
        
        T[i]= np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))

    T_array = []
    T_array.append(T[0])
    T_array.append(T[1])
    for i in range (1,4):
        T_array.append(np.dot(T_array[i],T[i+1]))

    return T_array, T_array


def jacobian(q):
    T, _= MatrixT(q)
    Pe = T[len(q)-1][:3,3]

    J = np.zeros((6,len(q)))
    for k in range(len(q)):
        zk = T[k][:3,2]
        Pk = T[k][:3,3]
        
        J[:,k] = np.concatenate((np.cross(zk , (Pe-Pk)), zk)) 
    return J , Pe

def f(t,q):
    J , Pe = jacobian(q)
    R,Rv=Ref(t)
    Kp=1
    e=R-Pe
    u=np.dot(np.linalg.pinv( J[:3,:]),Kp*e + Rv)
    #u[4] = u[1] - u[0]
    u = u if np.linalg.norm(u)<20 else 20*u/np.linalg.norm(u)
    return u

def rk4(y0, t, h):

    k1 = f(t, y0)
    k2 = f(t + 0.5*h, y0 + 0.5*k1*h)
    k3 = f(t + 0.5*h, y0 + 0.5*k2*h)
    k4 = f(t + h, y0 + k3*h)
    y1 = y0 + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return y1, (k1 + 2*k2 + 2*k3 + k4)/6

def animation(t,q):
    R,_ = Ref(t)

    T , c_c= MatrixT(q)
    
    ax.clear()

    ax.plot3D(R[0],R[1],R[2],'xm')


    ax.plot3D([c_c[1][0,3],c_c[0][0,3]],[c_c[1][0,3],c_c[0][1,3]],[c_c[1][0,3],c_c[0][2,3]],'--k')
    ax.plot3D([c_c[0][0,3],c_c[3][0,3]],[c_c[0][1,3],c_c[3][1,3]],[c_c[0][2,3],c_c[3][2,3]],'--k')
    ax.plot3D([c_c[1][0,3],c_c[2][0,3]],[c_c[1][1,3],c_c[2][1,3]],[c_c[1][2,3],c_c[2][2,3]],'--k')
    ax.plot3D([c_c[2][0,3],c_c[3][0,3]],[c_c[2][1,3],c_c[3][1,3]],[c_c[2][2,3],c_c[3][2,3]],'--k')
    ax.plot3D([c_c[3][0,3],c_c[4][0,3]],[c_c[3][1,3],c_c[4][1,3]],[c_c[3][2,3],c_c[4][2,3]],'--k')


    for k in range(len(q0)):
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,0]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,0]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,0]],'-r')
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,1]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,1]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,1]],'-g')
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,2]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,2]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,2]],'-b')


    ax.set_xlim([-20,20])
    ax.set_ylim([-20,10])
    ax.set_zlim([-2,2])
    plt.title('t = {:.2f} s'.format(t))
    ax.set_aspect('equal', 'box')

esc= 2
fig = plt.figure()
fig.canvas.mpl_connect('close_event', on_close)

ax , marcos = plt.axes(projection='3d'),[]

q0 = np.array([0,
                np.pi/4    ,
               0          ,
               -np.pi/2          ,  
               0            ])

h = 1

while( not window_closed):
    actual_time = tm.time()  - init_time 

    q , q_dot = rk4(q0 , actual_time , h )
    print('theta f: ', q[1]*57.3,'vel f: ', q_dot[1]*57.3,'theta p: ', (np.pi - abs(q[3]) + abs(q[1]))*57.3, 'vel p: ', (q_dot[3] + q_dot[1])*57.3 )
    RPM = q_dot*60/2*np.pi
    dxl_vel = RPM/0.229

    q0 = q

    animation(actual_time,q)

    plt.draw()
    plt.pause(0.01)
    