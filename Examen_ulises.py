#Ulises Sifuentes Herrera
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import sqrt
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
fig = plt.figure()
def update(val):
    def kcross(k):
       
        return np.array([[0, -k[2], k[1]],
                        [k[2], 0, -k[0]],
                        [-k[1], k[0], 0 ]])
    q=[]
    for K in range(3):
        q.append(Slide[K].val*np.pi/180)
        
    RX = np.array([[1, 0, 0],
		 [0,np.cos(q[0]),-np.sin(q[0])],
		 [0,np.sin(q[0]),  np.cos(q[0])]])
    RX2 = np.array([[1, 0, 0],
		 [0,np.cos(q[1]),-np.sin(q[1])],
		 [0,np.sin(q[1]),  np.cos(q[1])]]) 
    RZ= np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                  [np.sin(q[2]),  np.cos(q[2]), 0], 
                  [0, 0, 1]])

    R = np.dot(np.dot(RX, RZ),RX2)
    Q = np.dot(R,P)
    line.set_xdata(Q[0])
    line.set_ydata(Q[1])
    line.set_3d_properties(Q[2])
    
    theta_rodrigues = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1)/2)
    k = [(R[2][1] - R[1][2])/(2*np.sin(theta_rodrigues)), (R[0][2] - R[2][0])/(2*np.sin(theta_rodrigues)), (R[1][0] - R[0][1])/(2*np.sin(theta_rodrigues))]   
    kx = kcross(k)
    I = np.identity(3)
    R_Rodrigues = I + np.sin(theta_rodrigues)*kx + (1-np.cos(theta_rodrigues))*np.dot(kx,kx)
    R2 = np.dot(R_Rodrigues, P)
    
    line2.set_xdata(R2[0])
    line2.set_ydata(R2[1])
    line2.set_3d_properties(R2[2])
    line3.set_xdata([0,k[0]])
    line3.set_ydata([0,k[1]])
    line3.set_3d_properties([0,k[2]])
    
    
fig.subplots_adjust(right=0.6)
ax =[fig.add_axes([0.05,0.30,.40,0.5],projection = '3d'),
     fig.add_axes([0.55,0.30,.40,0.5],projection = '3d'),
     fig.add_axes([0.37,0.17,.25,0.05]),
     fig.add_axes([0.37,0.01,.25,0.05]),
     fig.add_axes([0.37,0.09,.25,0.05])]
ax[0].plot3D([0,1],[0,0],[0,0],'r')
ax[0].plot3D([0,0],[0,1],[0,0],'g')
ax[0].plot3D([0,0],[0,0],[0,1],'b')
ax[1].plot3D([0,1],[0,0],[0,0],'r')
ax[1].plot3D([0,0],[0,1],[0,0],'g')
ax[1].plot3D([0,0],[0,0],[0,1],'b')


theta = 0
P=np.array([[1,0,-1, 0,0, 1, 0,-1,0,0],
	    [0,1, 0,-1,0, 0,-1, 0,0,1],
	    [0,0, 0, 0,1, 0, 0, 0,1,0]])
RZ = np.array([[np.cos(theta), -np.sin(theta), 0],
		 [np.sin(theta),np.cos(theta),0],
		 [0,0,  1]])
Q = np.dot(RZ,P)
k = [0,0,1]
line, =ax[0].plot3D(Q[0],Q[1],Q[2])
line2, = ax[1].plot3D(Q[0],Q[1],Q[2])
line3,=ax[1].plot3D([0,0],[0,0],[0,1],'k')

Slide=[]
label=['X','X2','Z']
for K in range(3):
    Slide.append( Slider(
        ax=ax[2+K],
        label=label[K],
        valmin=-180,
        valmax=180,
        valinit=0,
    ))
    Slide[K].on_changed(update)
ax[0].set_aspect('equal','box')
ax[1].set_aspect('equal','box')
plt.show()