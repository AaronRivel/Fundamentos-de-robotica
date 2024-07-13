#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as fani

def Ref(t):
    d=1
    Ref = np.array([(d*np.sqrt(2)*np.cos(t))/(np.sin(t)**2+1)+0.5,
                    (d*np.sqrt(2)*np.cos(t)*np.sin(t))/(np.sin(t)**2+1)+0.5,
                    1])

    Ref_derivada=np.array([(d*np.sqrt(2)*(-np.sin(t)**3-np.sin(t)-np.cos(t)*np.sin(2*t)))/(np.sin(t)**2+1)**2,
                        (d*np.sqrt(2)*(-np.sin(t)**2+np.cos(2*t)))/(np.sin(t)**2+1)**2,
                        0])
    return Ref,Ref_derivada
def animation(frame):
    def MatrixT(q):
        DH = DHParameters()
        T = []
        for i in range (len(q)+1):
                T.append(np.identity(4))
        for i in range (1,len(q)+1):
            Rz=np.array([[np.cos(DH[i][0]+q[i-1]), -np.sin(DH[i][0]+q[i-1]),0,0],
                    [np.sin(DH[i][0]+q[i-1]),  np.cos(DH[i][0]+q[i-1]),0,0],
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
        for i in range (len(q)):
            T_array.append(np.dot(T_array[i],T[i+1]))
        T = T_array
          
        return T,T
    ##Reconstruir las transformaciones
    tk = t[frame]
    qk = q[frame]
    n = len(qk)
    R,_ = Ref(tk)

    T , c_c= MatrixT(qk)
    
    ax.clear()

    ax.plot3D(R[0],R[1],R[2],'xm')


    ax.plot3D([c_c[0][0,3],c_c[1][0,3]],[c_c[0][0,3],c_c[1][1,3]],[c_c[0][0,3],c_c[1][2,3]],'--k')
    ax.plot3D([c_c[1][0,3],c_c[2][0,3]],[c_c[1][1,3],c_c[2][1,3]],[c_c[1][2,3],c_c[2][2,3]],'--k')
    ax.plot3D([c_c[2][0,3],c_c[3][0,3]],[c_c[2][1,3],c_c[3][1,3]],[c_c[2][2,3],c_c[3][2,3]],'--k')


    for k in range(n+1):
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,0]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,0]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,0]],'-r')
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,1]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,1]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,1]],'-g')
        ax.plot3D([T[k][0,3], T[k][0,3]+esc*T[k][0,2]],
                  [T[k][1,3], T[k][1,3]+esc*T[k][1,2]],
                  [T[k][2,3], T[k][2,3]+esc*T[k][2,2]],'-b')


    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    plt.title('t = {:.2f} s'.format(tk))
    ax.set_aspect('equal', 'box')


def DHParameters():
    return [[0,0,0,0,'r'],
            [0,1,0,np.pi/2,'r'],
            [np.pi/2,0,1,0,'r'],
            [0,0,1,0,'r']]

def simulacion(q0,tf,h):
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
            y[i] = y[i-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6
        return t, y

    def f(t,q):
        def MatrixT(q):
            DH = DHParameters()
            T = []
            for i in range (len(q)+1):
                T.append(np.identity(4))
            for i in range (1,len(q)+1):
                
                Rz=np.array([[np.cos(DH[i][0]+q[i-1]), -np.sin(DH[i][0]+q[i-1]),0,0],
                        [np.sin(DH[i][0]+q[i-1]),  np.cos(DH[i][0]+q[i-1]),0,0],
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
            for i in range (len(q)):
                T_array.append(np.dot(T_array[i],T[i+1]))
            T = T_array
            return T
        def jacobiano(q):
            DH = DHParameters()
            T = MatrixT(q)
            Pe = T[len(q)][:3,3]
            global J
            for k in range(len(q)):
                zk = T[k][:3,2]
                Pk = T[k][:3,3]
                dummy=np.concatenate((np.cross(zk , (Pe-Pk)), zk)) if DH[k][4] == 'r' else np.concatenate((zk , [0,0,0 ]))
                J[:,k] = dummy
            return J , Pe

        J , Pe = jacobiano(q)
        R,Rv=Ref(t)
        Kp=1
        e=R-Pe
        u=np.dot(np.linalg.pinv( J[:3,:]),Kp*e+Rv)
        u = u if np.linalg.norm(u)<20 else 20*u/np.linalg.norm(u)
        return u

    t , q = rk4(q0,tf,h)
    return t , q




esc=0.1
fig = plt.figure()
ax , marcos = plt.axes(projection='3d'),[]

q0 = np.array([0,0,0])
tf=10
h=0.05
J = np.zeros((6,len(q0)))

t , q = simulacion(q0 , tf , h )

R,Rv = Ref(0)

ref_vector = []

ani = fani(fig=fig, func=animation, frames=len(t), interval=100)
plt.show()