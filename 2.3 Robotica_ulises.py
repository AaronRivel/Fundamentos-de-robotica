import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
def update(val):
    q=[]
    for i in range(2):
        q.append(Slide[i].val*5/360)
    q.append(Slide[2].val*np.pi/180)
        
    T0_1 = np.array([[np.cos(q[2]), -np.sin(q[2]), 0, 0],
                     [np.sin(q[2]), np.cos(q[2]), 0, 0],
                     [0,            0,            1, L1],
                     [0,            0,            0, 1]])
    T1_2 = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, q[0]],
                     [0, 0, 0, 1]])
    T2_E = np.array([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, q[1]],
                     [0, 0, 0, 1]])
    T0_2 = np.dot(T0_1,T1_2)
    T0_E = np.dot(T0_2,T2_E)
    T = np.array([T0_1,T0_2,T0_E])
    for i in range (3):
        T_aux = T[i]
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
ax =[fig.add_axes([0.05,0.15,.80,0.9],projection = '3d'),
     fig.add_axes([.35,0.01,.25,0.05]),
     fig.add_axes([0.35,0.06,.25,0.05]),
     fig.add_axes([0.35,0.11,.25,0.05])]
ax[0].plot3D([0,1],[0,0],[0,0],'r')
ax[0].plot3D([0,0],[0,1],[0,0],'g')
ax[0].plot3D([0,0],[0,0],[0,1],'b')
theta = 0
L1 = 2
d2 = 2
d3 = 2
T0_1 = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                 [0,            0,            1, L1],
                 [0,            0,            0, 1]])
T1_2 = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, d2],
                 [0, 0, 0, 1]])
T2_E = np.array([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, d3],
                 [0, 0, 0, 1]])
T0_2 = np.dot(T0_1,T1_2)
T0_E = np.dot(T0_2,T2_E)

T = [T0_1,T0_2,T0_E]
lines_x=[]
lines_y = []
lines_z = []
colors = ['r','g','b']
for j in range (3):
    for i in range (3):
        T_aux = T[i]
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
label=['d2','d3','theta']
for i in range(3):
    Slide.append( Slider(
        ax=ax[i+1],
        label=label[i],
        valmin=0,
        valmax=360,
        valinit=0,
    ))
    Slide[i].on_changed(update)
ax[0].set_xlim([-5,5])
ax[0].set_ylim([-5,5])
ax[0].set_zlim([0,10])
plt.show()