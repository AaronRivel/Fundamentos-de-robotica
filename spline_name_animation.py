#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation2
def inicio(x,y):
    def poliSpline(x,y):
        n=len(x)-1
        A=np.zeros([4*n,4*n])
        b=np.zeros(4*n)
        
        for i in range (n):
            A[ i , i*4 : 4*(i+1) ]=np.array([x[ i ]**3,x[ i ]**2,x[ i ],1])
            b[ i ]=y[ i ]
            A[ i+n , 4*i : (i+1)*4 ]=np.array([x[ i+1 ]**3,x[ i+1 ]**2,x[ i+1 ], 1 ])
            b[ i+n ]=y[ i+1 ]
            
        for i in range (n-1):
            A[ i+2*n , i*4 : 4*(i+2) ]=np.array([3*x[ i+1 ]**2,2*x[ i+1 ],1,0,-3*x[ i+1 ]**2,-2*x[ i+1 ],-1,0])
            A[ 3*n-1+i , i*4 : 4*(i+2) ]=np.array([6*x[ i+1 ],2,0,0,-6*x[ i+1 ],-2,0,0])
        A[4*n-2,0:2]=np.array([6*x[0],2])
        A[4*n-1,4*n-4:4*n-2]=np.array([6*x[n],2])
        q = np.dot(np.linalg.inv(A),b)
        C=[]#coeficientes 
        for k in range(len(x)-1):
            C.append(q[4*k:4*k+4])
        return C   
    def evalSpline(xx,C,x):
        yy=np.zeros(len(xx))
        vy=np.zeros(len(xx))
        for k in range(len(C)):
            yy +=(C[k][0]*np.power(xx,3)+C[k][1]*np.power(xx,2)+C[k][2]*xx+C[k][3])*(x[k]<=xx) * (xx < x[k+1])
            vy += (C[k][0]*3*np.power(xx,2)+C[k][1]*2*np.power(xx,1)+C[k][2])*(x[k]<=xx) * (xx < x[k+1])
        yy +=(C[len(C)-1][0]*np.power(xx,3)+C[len(C)-1][1]*np.power(xx,2)+C[len(C)-1][2]*xx+C[len(C)-1][3])*(x[len(C)]==xx)
        vy += (C[len(C)-1][0]*3*np.power(xx,2)+C[len(C)-1][1]*2*np.power(xx,1)+C[len(C)-1][2])*(x[len(C)]==xx)
        return yy , vy
    def rk4(f,y0,tf,h,k,com):
        n = round(tf/h)
        t = np.linspace(0,tf,n)
        y = np.zeros((n,len(y0)))
        y[0]=y0
        for i in range(n-1):
            
            K1 = f(t[i],y[i],k,com)*h
            K2 = f(t[i]+h*0.5, y[i] + K1*0.5,k,com)*h
            K3 = f(t[i]+h*0.5, y[i] + K2*0.5,k,com)*h
            K4 = f(t[i]+h, y[i] + K3,k,com)*h

            y[i+1] = y[i] + (K1+2*K2+2*K3+K4)/6
        return t,y
    def spline_control(C,xx):
            k = round(xx)
            if (k < len(C)):
                yy =(C[k][0]*np.power(xx,3)+C[k][1]*np.power(xx,2)+C[k][2]*xx+C[k][3])
                vy = (C[k][0]*3*np.power(xx,2)+C[k][1]*2*np.power(xx,1)+C[k][2]) 
            else:
                yy =(C[len(C)-1][0]*np.power(xx,3)+C[len(C)-1][1]*np.power(xx,2)+C[len(C)-1][2]*xx+C[len(C)-1][3])
                vy = (C[len(C)-1][0]*3*np.power(xx,2)+C[len(C)-1][1]*2*np.power(xx,1)+C[len(C)-1][2]) 
            return yy , vy
    def f(t,Q,k,com):
        rx,vx=spline_control(ax_c,t)
        ry,vy=spline_control(ay_c,t)
        e = np.array([rx-Q[0], ry-Q[1]])
        v = np.array([vx,vy])
        u = k*e+v*com
        return u
    def animation(frame):
        ta=t1[frame]
        plot_ani[0].set_data([rx_reference[frame]],[ry_reference[frame]])
        plot_ani[1].set_data([rx_reference[:frame]],[ry_reference[:frame]])
        plot_ani[2].set_data([Q1[frame,0]],[Q1[frame,1]])
        plot_ani[3].set_data([Q1[:frame,0]],[Q1[:frame,1]])
        plot_ani[4].set_data([Q2[frame,0]],[Q2[frame,1]])
        plot_ani[5].set_data([Q2[:frame,0]],[Q2[:frame,1]])
        plot_ani[6].set_data([Q3[frame,0]],[Q3[frame,1]])
        plot_ani[7].set_data([Q3[:frame,0]],[Q3[:frame,1]])
        plt.title('t = {0:.2f} s'.format(ta))
        return (plot_ani)
    
    t=range(len(x))
    tt = np.linspace(0,len(t)-1,230)

    ax_c=poliSpline(t,x)
    ay_c=poliSpline(t,y)
    
    rx_reference,_ = evalSpline(tt,ax_c,t)
    ry_reference,_ = evalSpline(tt,ay_c,t)

    Q0 = np.array([0.1,0.55])
    h = 0.1

    compensation_velocity = False
    kp = 0.1
    t1,Q1=rk4(f,Q0,len(x)-1,h,kp,compensation_velocity)

    compensation_velocity = True
    kp = 0.1
    t2,Q2=rk4(f,Q0,len(x)-1,h,kp,compensation_velocity)

    compensation_velocity = False
    kp = 10
    t3,Q3=rk4(f,Q0,len(x)-1,h,kp,compensation_velocity)

    fig=plt.figure()
    ax= fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot()

    plot_ani = [ax.plot(rx_reference[0],ry_reference[1],'ok')[0],
                ax.plot(rx_reference[0],ry_reference[1],'-k')[0],
                ax.plot(Q0[0],Q0[1],'or')[0],
                ax.plot(Q1[0,0],Q1[0,1],'--r')[0],
                ax.plot(Q0[0],Q0[1],'ob')[0],
                ax.plot(Q2[0,0],Q2[0,1],'--b')[0],
                ax.plot(Q0[0],Q0[1],'om')[0],
                ax.plot(Q3[0,0],Q3[0,1],'--m')[0]]
    
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ani = animation2.FuncAnimation(fig=fig, func=animation, frames=len(tt), interval=1000*h)
    plt.show()

x=[0.203, 0.109, 0.111, 0.201, 0.211, 0.326, 0.256, 0.328, 0.357, 0.361, 0.436, 0.404, 0.486, 0.488, 0.551, 0.607, 0.656, 0.598, 0.66, 0.719, 0.713, 0.779, 0.789, 0.852]
y=[0.404, 0.529, 0.234, 0.378, 0.255, 0.38, 0.315, 0.24, 0.365, 0.232, 0.37, 0.352, 0.346, 0.234, 0.362, 0.229, 0.378, 0.409, 0.359, 0.391, 0.224, 0.406, 0.245, 0.255]

inicio(x,y)