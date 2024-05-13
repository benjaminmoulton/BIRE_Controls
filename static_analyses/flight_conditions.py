#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:10:59 2022

@author: christian
"""
import matplotlib.pyplot as plt
import numpy as np
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

def pretty_plot(ax, xlims, ylims, dx, dy):
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
    ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
    ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax


plt.close('all')
fe_data_l = np.genfromtxt("./Flight Conditions/FlightEnvelope_left.csv", delimiter=',')
fe_data_r = np.genfromtxt("./Flight Conditions/FlightEnvelope_right.csv", delimiter=',')
fig, ax = plt.subplots()
ax.plot(fe_data_l[:, 0], fe_data_l[:, 1], color='k')
ax.plot(fe_data_r[:, 0], fe_data_r[:, 1], color='k')
ax.hlines(np.zeros(len(fe_data_l[0, :])), fe_data_l[0, 0], fe_data_r[0, 0], color='k')
ax.hlines([50000., 50000.], fe_data_l[-1, 0], 2.0, color='k')
ax.vlines([2.0, 2.0], fe_data_r[-1, 1], 50000., color='k')
ax.axvline(0.8, color='0.5', linestyle='--')
ax.axvline(1.2, color='0.5', linestyle='--')
ax.annotate(r'\textbf{Subsonic}', [0.42, 35000], fontsize=14)
ax.annotate(r'\textbf{Transonic}', [0.8, 40000], fontsize=14)
ax.annotate(r'\textbf{Supersonic}', [1.22, 35000], fontsize=14)
case_altitudes = [1000., 15000., 15000., 1000., 30000.]
case_Machs = [0.2, 0.2, 0.6, 0.8, 0.8]
ax.annotate(r"\textbf{T1}", [0.23, 2000.], fontsize=14)
ax.annotate(r"\textbf{T2}", [0.23, 16000.], fontsize=14)
ax.annotate(r"\textbf{C1}", [0.65, 2000.], fontsize=14)
ax.annotate(r"\textbf{C2}", [0.63, 16000.], fontsize=14)
ax.annotate(r"\textbf{C3}", [0.83, 31000.], fontsize=14)
ax.set_ylabel(r"\textbf{Altitude, }\boldmath$H$\textbf{ [ft]}", fontsize=14)
ax.set_xlabel(r"\textbf{Mach Number, }\boldmath$M$", fontsize=14)
xlim = (-0.05, 2.05)
ylim = (-1250, 52500.)
dx = {'major': 0.2, 'minor': 0.05}
dy = {'major': 5000., 'minor': 5000/4}
ax = pretty_plot(ax, xlim, ylim, dx, dy)
ax.grid()
ax.scatter(case_Machs[1:], case_altitudes[1:], fc='k', ec='k', zorder=2)
ax.scatter(case_Machs[0], case_altitudes[0], fc='k', ec='k', zorder=2)
plt.tight_layout()
plt.savefig('./Flight Conditions/Flight_Envelope.pdf', dpi=1000)