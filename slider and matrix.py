#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:03 2024

@author: aaron
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

P= np.array ([[-4, 2, 2, 4, 2, 2, -4, -4], [1, 1, 3, 0, -3, -1, -1, 1]])

# The parametrized function to be plotted
def f(radian):
    theta = 2*np.pi*radian
    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    Q = np.dot(R, P)
    return Q
# Define initial parameters
init_value = 0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line1, = ax.plot(P[0, :],P[1, :])
line2, = ax.plot(f(init_value)[0,:],f(init_value)[1,:] , lw=2)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axrad = fig.add_axes([0.25, 0.1, 0.65, 0.03])
rad_slider = Slider(
    ax=axrad,
    label='radians',
    valmin=0.1,
    valmax=6.30,
    valinit=init_value,
)




# The function to be called anytime a slider's value changes
def update(val):
    line2.set_xdata(f(rad_slider.val)[0,:])
    line2.set_ydata(f(rad_slider.val)[1,:])
    
    fig.canvas.draw_idle()


# register the update function with each slider
rad_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    rad_slider.reset()
button.on_clicked(reset)
plt.ion()
plt.show()