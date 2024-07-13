#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation2
def Referencia(t):
     r,w,cx,cy =1,2,2.5,2.5
     Ref = np.array([r*np.cos(w*t)+cx,r*np.sin(w*t)+cy])
     Ref_derivada=np.array([-r*w*np.sin(w*t),r*w*np.cos(w*t)])
     '''
     #Lemniscata
     d=1
     Ref = np.array([(d*np.sqrt(2)*np.cos(t))/(np.sin(t)**2+1)+2.5,
                     (d*np.sqrt(2)*np.cos(t)*np.sin(t))/(np.sin(t)**2+1)+2.5])
     Ref_derivada=np.array([?,?])
     '''
     return Ref,Ref_derivada
def rk4(f, y0, tf, h):
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
def f(t, Q):
  R,Rp=Referencia(t)
  Kp=2
  activarCompensacion=False #True

  e=R-Q
  u=Kp*e+activarCompensacion*Rp
  return u

Q0 = np.array([2.5, 2.5])
tf = 10
h = 0.01

t, Q = rk4(f, Q0, tf, h)
fig=plt.figure()
ax= fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot()
plot_ani = [ax.plot(2.5,2.5,'ob')[0],
            ax.plot(Q[0,0],Q[0,1],'-b')[0],
            ax.plot(Q[0,0],Q[0,1],'or')[0],
            ax.plot(Q[0,0],Q[0,1],'--r')[0]]
ax.axis('equal')
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)



def animation(frame):
    ta=t[frame]
    R=Referencia(ta)[0]
    path,_=Referencia(t[:frame])
    plot_ani[0].set_data([R[0]],[R[1]])
    plot_ani[1].set_data([path[0]],[path[1]])
    plot_ani[2].set_data([Q[frame,0]],[Q[frame,1]])
    plot_ani[3].set_data([Q[:frame,0]],[Q[:frame,1]])
    plt.title('t = {0:.2f} s'.format(ta))
    return (plot_ani)


ani = animation2.FuncAnimation(fig=fig, func=animation, frames=len(t), interval=1000*h)
plt.show()