#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:06:21 2023

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 14.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}

def pretty_plot(ax, xlims, ylims, zlims, dx, dy, dz):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
    ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
    ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    ax.zaxis.set_major_locator(MultipleLocator(dz["major"]))
    ax.zaxis.set_minor_locator(MultipleLocator(dz["minor"]))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.zaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax


plt.close('all')
omega = 5.
N = 11
s_range = np.linspace(-1., 1., N)
t_range = np.arange(0., 20., 0.1)
MC_states = np.load('./MC_states_w_' + str(int(omega)) + '.npy')
rmse = np.zeros((N, N, N, 8))
for i in range(N):
    for j in range(N):
        for k in range(N):
            for z in range(8):
                # if np.max(np.abs(MC_states[i, j, k, z, :])) < 1e-3:
                #     rmse[i, j, k, z] = np.mean(MC_states[i, j, k, z, t_range > 15.])
                # else:
                rmse[i, j, k, z] = np.mean(MC_states[i, j, k, z, t_range > 15.])/np.max(np.abs(MC_states[..., z, :]))

# z_ticks = [0, 2, 5, 7, 9]

# X, Y = np.meshgrid(s_range, s_range)
# # for z in range(N):
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# # cmap = colors.ListedColormap('black')
# markers = ['o', '<', '>', '^', 'v', 'd', '1', '2']
# dummy_lines = []
# for i in range(8):
#     mask = abs(rmse[:, :, :, i]) > 0.05
#     options = {"zorder": [i], 'marker': markers[i], 's': 40.0*mask}
#     # options_clabel = {"inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, "zorder": 2}
#     idx = np.arange(int(np.prod(rmse[:, :, :, i].shape)))
#     x, y, z = np.unravel_index(idx, rmse[:, :, :, i].shape)
#     CS = ax.scatter(x/19.*2. - 1, y/19.*2. - 1, z/19.*2. - 1, c=str(i*0.08), **options)
#     dummy_lines += [ax.scatter([], [], [], c=str(i*0.1), marker=markers[i])]
# ax.set_xlabel(r'\boldmath$s_x$', fontsize=16)
# ax.set_ylabel(r'\boldmath$s_y$', fontsize=16)
# ax.set_zlabel(r'\boldmath$s_z$', fontsize=16)
# ax.legend(dummy_lines, [r'$u$', r'$v$', r'$w$', r'$p$', r'$q$', r'$r$', r'$\phi$', r'$\theta$'], loc='lower right', fontsize=16, bbox_to_anchor=(0.0, 0.05))
# ax.view_init(15, -50)
# xlims = (-1.25, 1.25)
# dx = {'major': 0.5, 'minor': 0.5/4}
# ax = pretty_plot(ax, xlims, xlims, xlims, dx, dx, dx)
# plt.tight_layout()
# plt.savefig('./Monte_Carlo/visualization.pdf', dpi=1000)

# fig, ax = plt.subplots()
# dummy_lines = []
# mask = np.where(abs(rmse)>0.05)
# y_loc = mask[1]
# z_loc = mask[2]
# ax.scatter(s_range[y_loc], s_range[z_loc], marker='o')

# fig, ax = plt.subplots()
# mask = damping_rate < 0
# negative_damping_rates = damping_rate[mask]

# # Plot the negative damping rates
# plt.scatter(negative_damping_rates[:, 0], negative_damping_rates[:, 1],
#             c=negative_damping_rates[:, -1], cmap='Reds')
# plt.colorbar()
# plt.show()

