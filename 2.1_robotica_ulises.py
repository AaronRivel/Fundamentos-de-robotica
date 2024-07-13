import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

numero_marcos = 5 
def update(val):
    q=[]
    for i in range(5):
        q.append(Slide[i].val)    
    theta = np.array([q[0],q[1],q[2],q[3],q[4]])*np.pi/180
    d = [2,2,0,2,0]
    a = [0,2,0,2,2]
    alpha = np.array([-90,0,90,90,0])*np.pi/180
    for i in range (numero_marcos):
        Rz=np.array([[np.cos(theta[i]), -np.sin(theta[i]),0,0],
                [np.sin(theta[i]),  np.cos(theta[i]),0,0],
                [0,0,1,0],
                [0,0,0,1]])

        Tz=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,d[i]],
                    [0,0,0,1]])
        Tx=np.array([[1,0,0,a[i]],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Rx=np.array([[1,0,0,0],
                    [0,np.cos(alpha[i]),-np.sin(alpha[i]),0],
                    [0,np.sin(alpha[i]), np.cos(alpha[i]),0],
                    [0,0,0,1]])
        T[i]= np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))

        T_array[0] = T[0]
    for i in range (numero_marcos-1):
        T_array[i+1] = np.dot(T_array[i],T[i+1])  
        
    for i in range (numero_marcos):
        T_aux = T_array[i]
        lines_x[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,0]]))
        lines_x[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,0]]))
        lines_x[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,0]]))
        lines_y[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,1]]))
        lines_y[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,1]]))
        lines_y[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,1]]))
        lines_z[i].set_xdata(T_aux[0,3]+np.array([0,T_aux[0,2]]))
        lines_z[i].set_ydata(T_aux[1,3]+np.array([0,T_aux[1,2]]))
        lines_z[i].set_3d_properties(T_aux[2,3]+np.array([0,T_aux[2,2]]))
                
fig = plt.figure()
fig.subplots_adjust(right=0.6)
ax =[fig.add_axes([0.08,0.06,.80,0.9],projection = '3d'),
     fig.add_axes([0.08,0.7,.17,0.05]),
     fig.add_axes([0.08,0.6,.17,0.05]),
     fig.add_axes([0.08,0.5,.17,0.05]),
     fig.add_axes([0.08,0.4,.17,0.05]),
     fig.add_axes([0.08,0.3,.17,0.05])]
ax[0].plot3D([0,1],[0,0],[0,0],'r')
ax[0].plot3D([0,0],[0,1],[0,0],'g')
ax[0].plot3D([0,0],[0,0],[0,1],'b')
T = []
for i in range (numero_marcos):
    T.append(np.identity(4))

theta = np.array([0,-90,90,90,0])*np.pi/180
d = [2,2,0,2,0]
a = [0,2,0,2,2]
alpha = np.array([-90,0,90,90,0])*np.pi/180

for i in range (numero_marcos):
    Rz=np.array([[np.cos(theta[i]), -np.sin(theta[i]),0,0],
             [np.sin(theta[i]),  np.cos(theta[i]),0,0],
             [0,0,1,0],
             [0,0,0,1]])

    Tz=np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,d[i]],
                [0,0,0,1]])
    Tx=np.array([[1,0,0,a[i]],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
    Rx=np.array([[1,0,0,0],
                [0,np.cos(alpha[i]),-np.sin(alpha[i]),0],
                [0,np.sin(alpha[i]), np.cos(alpha[i]),0],
                [0,0,0,1]])
    T[i]= np.dot(np.dot(Rz,Tz),np.dot(Tx,Rx))
    
T_array = []
T_array.append(T[0])
for i in range (numero_marcos-1):
    T_array.append(np.dot(T_array[i],T[i+1]))

lines_x=[]
lines_y = []
lines_z = []
colors = ['r','g','b']
for j in range (3):
    for i in range (numero_marcos):
        T_aux = T_array[i]
        line, = ax[0].plot3D(T_aux[0,3]+np.array([0,T_aux[0,j]]),
                            T_aux[1,3]+np.array([0,T_aux[1,j]]),
                            T_aux[2,3]+np.array([0,T_aux[2,j]]),colors[j])
        if j == 0:
            lines_x.append(line)
        if j == 1:
            lines_y.append(line)
        if j ==2:
            lines_z.append(line)

Slide=[]
label=['O1','O2','O3','O4','O5']
for i in range(5):
    Slide.append( Slider(
        ax=ax[i+1],
        label=label[i],
        valmin=0,
        valmax=360,
        valinit=0,
    ))
    Slide[i].on_changed(update)
ax[0].set_xlim([-8,8])
ax[0].set_ylim([-8,8])
ax[0].set_zlim([0,10])
plt.show()