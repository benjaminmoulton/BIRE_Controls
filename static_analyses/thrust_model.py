#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:12:20 2022

@author: christian
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from hunsaker_atm import stdatm_english
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 16.,
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


def find_coeffs(C, T_data, rho, V):
    [a, T0, T1, T2] = C
    T = (rho/rho_0)**a*(T0 + T1*V + T2*np.square(V))
    return np.linalg.norm(T - T_data)

def a_coeff(a, T_data, rho):
    T = (rho/rho_0)**a*(T_data)
    print(a)
    print(np.linalg.norm(T - T_data))
    return np.linalg.norm(T - T_data)

T_idle = np.array([[635, 425, 690, 1010, 1330, 1700],
                   [60, 25, 345, 755, 1130, 1525],
                   [-1020, -710, -300, 350, 910, 1360],
                   [-2700, -1900, -1300, -247, 600, 1100],
                   [-3600, -1400, -595, -342, -200, 700]])
T_mil = np.array([[12680, 9150, 6313, 4040, 2470, 1400],
                  [12610, 9312, 6610, 4290, 2600, 1560],
                  [12640, 9839, 7090, 4660, 2840, 1660],
                  [12390, 10176, 7750, 5320, 3250, 1930],
                  [11680, 9848, 8050, 6100, 3800, 2310]])
T_max = np.array([[21420, 15700, 11225, 7323, 4435, 2600],
                  [22700, 16860, 12250, 8154, 5000, 2835],
                  [24240, 18910, 13760, 9285, 5700, 3215],
                  [26070, 21075, 15975, 11115, 6860, 3950],
                  [28886, 23319, 18300, 13484, 8642, 5057]])
M = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
H = np.arange(0., 60000., 10000.)
rho_0 = stdatm_english(0.)[-2]
rho = np.zeros(len(H))
a = np.zeros(len(H))
V = np.zeros_like(T_idle)
for i in range(len(H)):
    rho[i], a[i] = stdatm_english(H[i])[-2:]
    for j in range(len(M)):
        V[j, i] = M[j]*a[i]
T0_i = np.zeros_like(H)
T1_i = np.zeros_like(H)
T2_i = np.zeros_like(H)
a_i = np.zeros_like(H)
for i in range(len(H)):
    [a_i[i], T0_i[i], T1_i[i], T2_i[i]] = optimize.minimize(find_coeffs, [1.]*4, args=(T_idle[:, i], rho[i], V[:, i])).x
T_idle_fit = np.zeros_like(T_idle)
for i in range(len(H)):
    for j in range(len(M)):
        T_idle_fit[j, i] = (rho[i]/rho_0)**a_i[i]*(T0_i[i] + T1_i[i]*V[j, i] + T2_i[i]*V[j, i]**2)

T0_mil = np.zeros_like(H)
T1_mil = np.zeros_like(H)
T2_mil = np.zeros_like(H)
a_mil = np.zeros_like(H)
for i in range(len(H)):
    [a_mil[i], T0_mil[i], T1_mil[i], T2_mil[i]] = optimize.minimize(find_coeffs, [1.]*4, args=(T_mil[:, i], rho[i], V[:, i])).x
T_mil_fit = np.zeros_like(T_mil)
for i in range(len(H)):
    for j in range(len(M)):
        T_mil_fit[j, i] = (rho[i]/rho_0)**a_mil[i]*(T0_mil[i] + T1_mil[i]*V[j, i] + T2_mil[i]*V[j, i]**2)

T0_max = np.zeros_like(H)
T1_max = np.zeros_like(H)
T2_max = np.zeros_like(H)
a_max = np.zeros_like(H)
for i in range(len(H)):
    [a_max[i], T0_max[i], T1_max[i], T2_max[i]] = optimize.minimize(find_coeffs, [1.]*4, args=(T_max[:, i], rho[i], V[:, i])).x
T_max_fit = np.zeros_like(T_max)
for i in range(len(H)):
    for j in range(len(M)):
        T_max_fit[j, i] = (rho[i]/rho_0)**a_max[i]*(T0_max[i] + T1_max[i]*V[j, i] + T2_max[i]*V[j, i]**2)

T0_i_fit = np.polyfit(H, T0_i, 2)
T1_i_fit = np.polyfit(H, T1_i, 2)
T2_i_fit = np.polyfit(H, T2_i, 2)
a_i_fit = np.polyfit(H, a_i, 2)
T0_mil_fit = np.polyfit(H, T0_mil, 2)
T1_mil_fit = np.polyfit(H, T1_mil, 2)
T2_mil_fit = np.polyfit(H, T2_mil, 2)
a_mil_fit = np.polyfit(H, a_mil, 2)
T0_max_fit = np.polyfit(H, T0_max, 2)
T1_max_fit = np.polyfit(H, T1_max, 2)
T2_max_fit = np.polyfit(H, T2_max, 2)
a_max_fit = np.polyfit(H, a_max, 2)

"""
Figures
"""
plt.close('all')
linestyles = ['-', '--', '-.', ':', (0, (1, 10)), (5, (10, 3))]
markers = ['o', '^', '<', '>', 'v', 'd']
fig_idle, ax_idle = plt.subplots()
for i in range(len(T_idle[0, :])):
    ax_idle.scatter(M, T_idle[:, i], ec='k', fc='None', marker=markers[i], label=r'$H = $' + str(int(H[i])) + ' ft')
    ax_idle.plot(M, T_idle_fit[:, i], color='k', linestyle=linestyles[i])
xlims = (0.1, 1.1)
ylims = (-4500, 2500)
dx = {'major': 0.2, 'minor': 0.2/4}
dy = {'major': 1000, 'minor': 1000/4}
ax_idle = pretty_plot(ax_idle, xlims, ylims, dx, dy)
ax_idle.set_xlabel(r'\textbf{Mach Number, }\boldmath$M$', fontsize=16)
ax_idle.set_ylabel(r'\textbf{Idle Thrust, }\boldmath$T_\mathrm{idle}$\textbf{ [lbf]}', fontsize=16)
ax_idle.grid()
ax_idle.legend(fontsize=16)
fig_idle.tight_layout()
plt.savefig('./Thrust Model/T_idle.pdf', dpi=1000)

fig_mil, ax_mil = plt.subplots()
for i in range(len(T_mil[0, :])):
    ax_mil.scatter(M, T_mil[:, i], ec='k', fc='None', marker=markers[i], label=r'$H = $' + str(int(H[i])) + ' ft')
    ax_mil.plot(M, T_mil_fit[:, i], color='k', linestyle=linestyles[i])
xlims = (0.1, 1.1)
ylims = (1000, 15000)
dx = {'major': 0.2, 'minor': 0.2/4}
dy = {'major': 2000, 'minor': 2000/4}
ax_mil = pretty_plot(ax_mil, xlims, ylims, dx, dy)
ax_mil.set_xlabel(r'\textbf{Mach Number, }\boldmath$M$', fontsize=16)
ax_mil.set_ylabel(r'\textbf{Military Thrust, }\boldmath$T_\mathrm{mil}$\textbf{ [lbf]}', fontsize=16)
ax_mil.legend(fontsize=16)
ax_mil.grid()
fig_mil.tight_layout()
plt.savefig('./Thrust Model/T_mil.pdf', dpi=1000)

fig_max, ax_max = plt.subplots()
for i in range(len(T_max[0, :])):
    ax_max.scatter(M, T_max[:, i], ec='k', fc='None', marker=markers[i], label=r'$H = $' + str(int(H[i])) + ' ft')
    ax_max.plot(M, T_max_fit[:, i], color='k', linestyle=linestyles[i])
xlims = (0.1, 1.1)
ylims = (-2500, 32500)
dx = {'major': 0.2, 'minor': 0.2/4}
dy = {'major': 5000, 'minor': 5000/4}
ax_max = pretty_plot(ax_max, xlims, ylims, dx, dy)
ax_max.set_xlabel(r'\textbf{Mach Number, }\boldmath$M$', fontsize=16)
ax_max.set_ylabel(r'\textbf{Max Thrust, }\boldmath$T_\mathrm{max}$\textbf{ [lbf]}', fontsize=16)
ax_max.legend(fontsize=16)
ax_max.grid()
fig_max.tight_layout()
plt.savefig('./Thrust Model/T_max.pdf', dpi=1000)

fig_params, ax_params = plt.subplots(2, 2, sharex=True)
ax_params[0, 0].scatter(H, T0_i, ec='k', fc='None', marker=markers[0])
ax_params[0, 0].plot(H, T0_i_fit[2] + T0_i_fit[1]*H + T0_i_fit[0]*np.square(H), color='k', linestyle=linestyles[0])
ax_params[0, 1].scatter(H, T1_i, ec='k', fc='None', marker=markers[0])
ax_params[0, 1].plot(H, T1_i_fit[2] + T1_i_fit[1]*H + T1_i_fit[0]*np.square(H), color='k', linestyle=linestyles[0])
ax_params[1, 0].scatter(H, T2_i, ec='k', fc='None', marker=markers[0])
ax_params[1, 0].plot(H, T2_i_fit[2] + T2_i_fit[1]*H + T2_i_fit[0]*np.square(H), color='k', linestyle=linestyles[0])
ax_params[1, 1].scatter(H, a_i, ec='k', fc='None', marker=markers[0], label=r'$T_\mathrm{idle}$')
ax_params[1, 1].plot(H, a_i_fit[2] + a_i_fit[1]*H + a_i_fit[0]*np.square(H), color='k', linestyle=linestyles[0])
ax_params[0, 0].scatter(H, T0_mil, ec='k', fc='None', marker=markers[1])
ax_params[0, 0].plot(H, T0_mil_fit[2] + T0_mil_fit[1]*H + T0_mil_fit[0]*np.square(H), color='k', linestyle=linestyles[1])
ax_params[0, 1].scatter(H, T1_mil, ec='k', fc='None', marker=markers[1])
ax_params[0, 1].plot(H, T1_mil_fit[2] + T1_mil_fit[1]*H + T1_mil_fit[0]*np.square(H), color='k', linestyle=linestyles[1])
ax_params[1, 0].scatter(H, T2_mil, ec='k', fc='None', marker=markers[1])
ax_params[1, 0].plot(H, T2_mil_fit[2] + T2_mil_fit[1]*H + T2_mil_fit[0]*np.square(H), color='k', linestyle=linestyles[1])
ax_params[1, 1].scatter(H, a_mil, ec='k', fc='None', marker=markers[1], label=r'$T_\mathrm{mil}$')
ax_params[1, 1].plot(H, a_mil_fit[2] + a_mil_fit[1]*H + a_mil_fit[0]*np.square(H), color='k', linestyle=linestyles[1])
ax_params[0, 0].scatter(H, T0_max, ec='k', fc='None', marker=markers[2])
ax_params[0, 0].plot(H, T0_max_fit[2] + T0_max_fit[1]*H + T0_max_fit[0]*np.square(H), color='k', linestyle=linestyles[2])
ax_params[0, 1].scatter(H, T1_max, ec='k', fc='None', marker=markers[2])
ax_params[0, 1].plot(H, T1_max_fit[2] + T1_max_fit[1]*H + T1_max_fit[0]*np.square(H), color='k', linestyle=linestyles[2])
ax_params[1, 0].scatter(H, T2_max, ec='k', fc='None', marker=markers[2])
ax_params[1, 0].plot(H, T2_max_fit[2] + T2_max_fit[1]*H + T2_max_fit[0]*np.square(H), color='k', linestyle=linestyles[2])
ax_params[1, 1].scatter(H, a_max, ec='k', fc='None', marker=markers[2], label=r'$T_\mathrm{max}$')
ax_params[1, 1].plot(H, a_max_fit[2] + a_max_fit[1]*H + a_max_fit[0]*np.square(H), color='k', linestyle=linestyles[2])
xlims = (-5000, 55000)
dx = {'major': 15000, 'minor': 15000/4}
dy = {'major': 15000, 'minor': 15000/4}
ylims = (0 - dy['minor']*2, 60000 + dy['minor']*2)
ax_params[0, 0] = pretty_plot(ax_params[0, 0], xlims, ylims, dx, dy)
ax_params[0, 0].grid()
ax_params[0, 0].annotate(r'\boldmath$T_0$', (30000, 45000), fontsize=16)
dy = {'major': 10, 'minor': 10/4}
ylims = (-30 - dy['minor']*2, 10 + dy['minor']*2)
ax_params[0, 1] = pretty_plot(ax_params[0, 1], xlims, ylims, dx, dy)
ax_params[0, 1].grid()
ax_params[0, 1].annotate(r'\boldmath$T_1$', (30000, -20), fontsize=16)
dy = {'major': 0.03, 'minor': 0.03/4}
ylims = (-0.03 - dy['minor']*2, 0.09 + dy['minor']*2)
ax_params[1, 0] = pretty_plot(ax_params[1, 0], xlims, ylims, dx, dy)
ax_params[1, 0].grid()
ax_params[1, 0].annotate(r'\boldmath$T_2$', (30000, 0.06), fontsize=16)
dy = {'major': 0.2, 'minor': 0.2/4}
ylims = (1.0 - dy['minor']*2, 1.8 + dy['minor']*2)
ax_params[1, 1] = pretty_plot(ax_params[1, 1], xlims, ylims, dx, dy)
ax_params[1, 1].grid()
ax_params[1, 1].annotate(r'\boldmath$a$', (30000, 1.65), fontsize=16)
ax_params[1, 1].legend(fontsize=16)
fig_params.supylabel(r'\textbf{Thrust Model Coefficients}', fontsize=16)
fig_params.supxlabel(r'\textbf{Altitude, }\boldmath$H$\textbf{ [ft]}', fontsize=16)
plt.tight_layout()
plt.savefig('./Thrust Model/T_fits.pdf', dpi=1000)
