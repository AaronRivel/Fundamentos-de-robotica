#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as fani

def Ref(t):
    T = 5
    a = 5
    b = 2
    
    ref = np.array([a*np.sin(np.pi/T*t)  , b*np.cos(np.pi/T*t) - 15 , 0])

    vel_ref = np.array([np.pi*a*np.cos(np.pi/T*t)*(1/T) ,  -np.pi*b*np.sin(np.pi/T*t)*(1/T) , 0])

    return ref, vel_ref

def DHParameters(q):
    phi = (2*np.pi - 2*(q[1] - q[0]))/2
    
    return [[q[0],  0,  0,  0],
            [0,     0, -7,  0],
            [-phi,  0,  7,  0],
            [0,     0,  7,  0],
            [q[1],  0,  0,  0],
            [0,     0, -7,  0]]

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
    for i in range (3):
        T_array.append(np.dot(T_array[i],T[i+1]))

    T_array.append(T[4])
    T_array.append(np.dot(T_array[4],T[5]))

    
    return T_array, T_array


def jacobian(q):
    T, _= MatrixT(q)
    Pe = T[3][:3,3]
    global J
    for k in range(len(q)):
        zk = T[k][:3,2]
        Pk = T[k][:3,3] 
        J[:,k] = np.concatenate((np.cross(zk , (Pe-Pk)), zk))
    return J , Pe

def f(t,q):
    J , Pe = jacobian(q)
    R,Rv=Ref(t)
    Kp=2
    e=R-Pe
    u=np.dot(np.linalg.pinv( J[:3,:]),Kp*e+Rv)
    u = u if np.linalg.norm(u)<20 else 20*u/np.linalg.norm(u)
    return u

def rk4(y0, tf, h):
    n_steps=round(tf/h)
    t = np.linspace(0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    for i in range(1, n_steps):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*k1*h)
        k3 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*k2*h)
        k4 = f(t[i-1] + h, y[i-1] + k3*h)
        print(k4)
        y[i] = y[i-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6
        print((k1 + 2*k2 + 2*k3 + k4)/6)
    return t, y

def animation(frame):
    tk = t[frame]
    qk = q[frame]
    R,_ = Ref(tk)

    T , c_c= MatrixT(qk)
    
    ax.clear()

    ax.plot3D(R[0],R[1],R[2],'xm')


    ax.plot3D([c_c[0][0,3],c_c[1][0,3]],[c_c[0][0,3],c_c[1][1,3]],[c_c[0][0,3],c_c[1][2,3]],'--k')
    ax.plot3D([c_c[1][0,3],c_c[2][0,3]],[c_c[1][1,3],c_c[2][1,3]],[c_c[1][2,3],c_c[2][2,3]],'--k')
    ax.plot3D([c_c[2][0,3],c_c[3][0,3]],[c_c[2][1,3],c_c[3][1,3]],[c_c[2][2,3],c_c[3][2,3]],'--k')
    ax.plot3D([c_c[4][0,3],c_c[5][0,3]],[c_c[4][0,3],c_c[5][1,3]],[c_c[4][0,3],c_c[5][2,3]],'--k')
    ax.plot3D([c_c[5][0,3],c_c[2][0,3]],[c_c[5][1,3],c_c[2][1,3]],[c_c[5][2,3],c_c[2][2,3]],'--k')


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
    plt.title('t = {:.2f} s'.format(tk))
    ax.set_aspect('equal', 'box')

esc=3
fig = plt.figure()
ax , marcos = plt.axes(projection='3d'),[]

q0 = np.array([-np.pi/4     +np.pi/2,
               0            +np.pi/2,
               -np.pi/2     +np.pi/2,
               0            +np.pi/2,
               np.pi/4      +np.pi/2,
               0            +np.pi/2])
tf=100
h=1
J = np.zeros((6,len(q0)))

t , q = rk4(q0 , tf , h )

ani = fani(fig=fig, func=animation, frames=len(t), interval=100)
plt.show()
