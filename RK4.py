#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

x0,y0=[0,0]
n = 20
h = 0.5


def function(x,y):
    return -x**2

xi = np.zeros(n, dtype=float)
yi = np.zeros(n, dtype=float)
xi[0] = x0
yi[0] = y0

for i in range(n-1):

    K1 = h * function(xi[i],yi[i])
    K2 = h * function(xi[i]+h/2, yi[i] + K1/2)
    K3 = h * function(xi[i]+h/2, yi[i] + K2/2)
    K4 = h * function(xi[i]+h, yi[i] + K3)

    yi[i+1] = yi[i] + 1/6*(K1+2*K2+2*K3+K4)
    xi[i+1] = xi[i] +h

plt.plot(xi,yi)
plt.show()

