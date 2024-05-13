#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:15:40 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import json
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator)

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 14.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}

def pretty_plot(ax, xlims, ylims, dx, dy):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
    ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
    ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    # ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax

def model(coeff, dB, sin=True, freq=2., square=False):
    if sin:
        phi = 0.
    else:
        phi = np.pi/2.
    if not square:
        m = lambda x : x[0]*np.sin(freq*dB + phi) + x[1]
        e = lambda x : m(x) - coeff
        res = optimize.leastsq(e, [300, np.average(coeff)])
        A = res[0][0]
        z = res[0][1]
        return A, freq, phi, z
    else:
        m = lambda x : x[0]*np.abs(np.sin(freq*dB))
        e = lambda x : m(x) - coeff
        A = optimize.leastsq(e, [0])[0][0]
        return A, freq, phi, -A

plt.close('all')
Ixx = np.array([9280.]*13)
Iyy = np.array([58449., 58427., 58368., 58288., 58207., 58149., 58127., 58149., 58207., 58288., 58368., 58427., 58449.])
Izz = np.array([65445., 65466., 65525., 65606., 65686., 65745., 65766., 65745., 65686., 65606., 65525., 65466., 65445.])
Ixy = np.zeros(13)
Ixz = np.array([-5.]*13)
Iyz = np.array([0., -80., -139., -161., -139., -80., 0., -80., -139., -161., -139., -80., 0.])

dB = np.arange(-90., 95., 15.)
dB_rad = np.deg2rad(dB)


fig, ax = plt.subplots(5, 1, sharex=True)
ixx = ax[2].scatter(dB, Ixx, ec='k', fc='none', marker='o')
iyy = ax[1].scatter(dB, Iyy, ec='k', fc='none', marker='^')
izz = ax[0].scatter(dB, Izz, ec='k', fc='none', marker='v')
ixy = ax[3].scatter(dB, Ixy, ec='k', fc='none', marker='<')
ixz = ax[3].scatter(dB, Ixz, ec='k', fc='none', marker='>')
iyz = ax[4].scatter(dB, Iyz, ec='k', fc='none', marker='d')
ax[3].scatter(dB, Iyz, ec='k', fc='none', marker='d')

# A, freq, phi, z = model(Iyy, dB_rad, sin=False, freq=2.)
# plt.plot(dB, A*abs(np.sin(freq*dB_rad + phi) + z))
# plt.plot(dB, A*np.sin(freq*dB_rad + phi) + z)

model_coeff_keys = ["A", "w", "phi", "z"]
model_coeff_dict = {key: 0. for key in model_coeff_keys}

models_dict = {"Ixx": model_coeff_dict,
               "Iyy": model_coeff_dict,
               "Izz": model_coeff_dict,
               "Ixy": model_coeff_dict,
               "Ixz": model_coeff_dict,
               "Iyz": model_coeff_dict}

dB = np.linspace(-90, 90, 1000)
models_dict["Ixx"] = {key: x for key, x in zip(model_coeff_keys, [0., 0., 0., np.average(Ixx)])}
ax[2].axhline(np.average(Ixx), -180, 180, color='k')
A, freq, phi, z = model(Iyy, dB_rad, sin=False, freq=2.)
ax[1].plot(dB, A*np.sin(freq*dB*np.pi/180 + phi) + z, color='k')
models_dict["Iyy"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
A, freq, phi, z = model(Izz, dB_rad, sin=False, freq=2.)
ax[0].plot(dB, A*np.sin(freq*dB*np.pi/180 + phi) + z, color='k')
models_dict["Izz"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
A, freq, phi, z = model(Iyz, dB_rad, freq=2.,square=True)
ax[4].plot(dB, A*np.abs(np.sin(freq*dB*np.pi/180)), color='k')
ax[3].plot(dB, A*np.abs(np.sin(freq*dB*np.pi/180)), color='k')
models_dict["Iyz"] = {key: x for key, x in zip(model_coeff_keys, [A, freq, phi, z])}
models_dict["Ixz"] = {key: x for key, x in zip(model_coeff_keys, [0., 0., 0., np.average(Ixz)])}
ax[3].axhline(np.average(Ixz), -180, 180, color='k')
ax[3].axhline(0., -180, 180, color='k')

xlims = (-95., 95.)
ylims = (64800., 66100.)
dx = {"major": 30, "minor": 30/4}
dy = {"major": 500, "minor": 500/4}
ax[0] = pretty_plot(ax[0], xlims, ylims, dx, dy)
ylims = (57900., 58600.)
dy = {"major": 250, "minor": 250/4}
ax[1] = pretty_plot(ax[1], xlims, ylims, dx, dy)
ylims = (9250., 9310.)
dy = {"major": 20, "minor": 20/4}
ax[2] = pretty_plot(ax[2], xlims, ylims, dx, dy)
ylims = (-8., 2.)
dy = {"major": 3, "minor": 3/4}
ax[3] = pretty_plot(ax[3], xlims, ylims, dx, dy)
ylims = (-232., -8)
dy = {"major": 60, "minor": 60/4}
ax[4] = pretty_plot(ax[4], xlims, ylims, dx, dy)

ax[0].xaxis.tick_top()
ax[0].spines['bottom'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].tick_params(axis='x', which='both', bottom=False)
ax[2].spines['top'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)
ax[2].tick_params(axis='x', which='both', bottom=False)
ax[3].spines['top'].set_visible(False)
ax[3].spines['bottom'].set_visible(False)
ax[3].tick_params(axis='x', which='both', bottom=False)
ax[4].spines['top'].set_visible(False)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
ax[0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

kwargs = dict(transform=ax[1].transAxes, color='k', clip_on=False)
ax[1].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[2].transAxes)  # switch to the bottom axes
ax[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

kwargs = dict(transform=ax[2].transAxes, color='k', clip_on=False)
ax[2].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax[2].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[3].transAxes)  # switch to the bottom axes
ax[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[3].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

kwargs = dict(transform=ax[3].transAxes, color='k', clip_on=False)
ax[3].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax[3].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[4].transAxes)  # switch to the bottom axes
ax[4].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[4].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.supxlabel(r'\textbf{BIRE Rotation Angle, }\boldmath$\delta_B$\textbf{ [deg]}', fontsize=14)
fig.supylabel(r'\textbf{Inertia, }\boldmath$I$\textbf{ [slug-ft\textsuperscript{2}]}', fontsize=14)
fig.legend([ixx, iyy, izz, ixy, ixz, iyz], [r'$I_{xx}$', r'$I_{yy}$', r'$I_{zz}$', r'$I_{xy}$', r'$I_{xz}$', r'$I_{yz}$'], loc=(0.8, 0.38), fontsize=14)
for a in ax:
    a.grid()
plt.tight_layout()
plt.savefig('./Inertia Figure/inertia_fig.pdf', dpi=1000)


with open("bire_inertia_model.json", "w") as outfile:
    json.dump(models_dict, outfile, indent=4)
