#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:20:56 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator)
import airfoil_db as afdb

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

def biconvex(p, c=1.):
    x = np.linspace(0., c, 100)
    p = p/100.
    y_u = 2.*p*x*(1. - x)
    y_l = -y_u
    return x, y_u, y_l

plt.close('all')
af_input = {'geometry': {'outline_points': './64A204.txt'}}
A204 = afdb.Airfoil('64A204', af_input)
N0004 = afdb.Airfoil('0004', {'geometry': {'NACA': '0004'}})
N0005 = afdb.Airfoil('0004', {'geometry': {'NACA': '0005'}})
A204_xy = A204.get_outline_points(N=25)
N0004_xy = N0004.get_outline_points(N=25)
N0005_xy = N0005.get_outline_points(N=25)
np.savetxt('./Airfoil Data/NACA64A204.csv', A204_xy, delimiter=',')
np.savetxt('./Airfoil Data/NACA0004.csv', N0004_xy, delimiter=',')
np.savetxt('./Airfoil Data/NACA0005.csv', N0005_xy, delimiter=',')
A204_xy = A204.get_outline_points()
x_c = np.linspace(0., 1., 50)
y_c = A204.get_camber(x_c)
x_c.shape = (50, 1)
y_c.shape = (50, 1)
np.savetxt('./Airfoil Data/NACA64A204_camber.csv', np.hstack((x_c, y_c)), delimiter=',')
x_c = np.linspace(0., 1., 100)
y_c = A204.get_camber(x_c)
fig, ax = plt.subplots(3, 1, sharex=True)
top_mask = len(A204_xy)//2
ax[0].plot(A204_xy[:top_mask, 0]*100, A204_xy[:top_mask, 1]*100, color='k')
ax[0].plot(A204_xy[top_mask:, 0]*100, A204_xy[top_mask:, 1]*100, color='k')
ax[0].plot(x_c*100, y_c*100, color='k', linestyle='--')
N0004_xy = N0004.get_outline_points()
y_c = N0004.get_camber(x_c)
x, y_u, y_l = biconvex(5.3)
ax[2].plot(N0004_xy[:top_mask, 0]*100, N0004_xy[:top_mask, 1]*100, color='k')
ax[2].plot(N0004_xy[top_mask:, 0]*100, N0004_xy[top_mask:, 1]*100, color='k')
ax[2].plot(x_c*100, y_c*100, color='k', linestyle='--')
# ax[2].plot(x, y_u, color='0.5')
# ax[2].plot(x, y_l, color='0.7')
# x, y_u, y_l = biconvex(3)
# ax[2].plot(x, y_u, color='0.5')
# ax[2].plot(x, y_l, color='0.7')
N0005_xy = N0005.get_outline_points()
y_c = N0005.get_camber(x_c)
ax[1].plot(N0005_xy[:top_mask, 0]*100, N0005_xy[:top_mask, 1]*100, color='k')
ax[1].plot(N0005_xy[top_mask:, 0]*100, N0005_xy[top_mask:, 1]*100, color='k')
ax[1].plot(x_c*100, y_c*100, color='k', linestyle='--')
# x, y_u, y_l = biconvex(6)
# ax[1].plot(x, y_u, color='0.5')
# ax[1].plot(x, y_l, color='0.7')
# x, y_u, y_l = biconvex(3.5)
# ax[1].plot(x, y_u, color='0.5')
# ax[1].plot(x, y_l, color='0.7')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
xlims = (-1, 101)
ylims = (-7.5, 7.5)
dx = {"major": 10, "minor": 10/4}
dy = {"major": 5, "minor": 5/4}
ax[0] = pretty_plot(ax[0], xlims, ylims, dx, dy)
ax[0].grid()
ax[1] = pretty_plot(ax[1], xlims, ylims, dx, dy)
ax[1].grid()
ax[2] = pretty_plot(ax[2], xlims, ylims, dx, dy)
ax[2].grid()
fig.supylabel(r'\textbf{Ordinate, }\boldmath$y/c$\textbf{ \%}', fontsize=16)
ax[2].set_xlabel(r'\textbf{Station, }\boldmath$x/c$\textbf{ \%}', fontsize=16)
ax[0].annotate(r'\textbf{NACA 64A204}', (70, -5), fontsize=16)
ax[1].annotate(r'\textbf{NACA 0005}', (70, -5), fontsize=16)
ax[2].annotate(r'\textbf{NACA 0004}', (70, -5), fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/airfoils.pdf', dpi=1000)
x_c = np.linspace(0., 1., 100)
y_c = A204.get_camber(x_c)

dycdx = np.diff(y_c)/(x_c[1] - x_c[0])
theta = np.linspace(0., np.pi, 99)
dtheta = theta[1] - theta[0]
CLa_tat = 2.*np.pi
Cmc4_tat = 0.5*np.trapz(dycdx*(np.cos(2.*theta) - np.cos(theta)), theta, dtheta)
aL0_tat = (1./np.pi)*np.trapz(dycdx*(1. - np.cos(theta)), theta, dtheta)


alpha_0 = np.loadtxt('./Airfoil Data/aL0_v_Thickness_64A.csv', delimiter=',')
CLa = np.loadtxt('./Airfoil Data/CLa_v_Thickness_64A.csv', delimiter=',')
Cmc4 = np.loadtxt('./Airfoil Data/Cmc4_a0_v_Thickness_64A.csv', delimiter=',')
CD0 = np.loadtxt('./Airfoil Data/MinDrag_v_Thickness_64A.csv', delimiter=',')
Cmc4_64A204 = np.loadtxt('./Airfoil Data/64A210_Cm_c4_data.csv', delimiter=',')
CD_64A210 = np.loadtxt('./Airfoil Data/64A210_Drag_data.csv', delimiter=',')
CD_64A212 = np.loadtxt('./Airfoil Data/64A212_Drag_data.csv', delimiter=',')
CD_64_206 = np.loadtxt('./Airfoil Data/64206_Drag_data.csv', delimiter=',')
CD_64_212 = np.loadtxt('./Airfoil Data/64212_Drag_data.csv', delimiter=',')
CD_0006 = np.loadtxt('./Airfoil Data/64206_Drag_data.csv', delimiter=',')
CD_0009 = np.loadtxt('./Airfoil Data/64212_Drag_data.csv', delimiter=',')

thickness = np.linspace(2., 24., 100)

fig, ax = plt.subplots()
ax.scatter(alpha_0[:, 0], alpha_0[:, 1], ec='k', fc='none', label='Loftin [97]')
print(aL0_tat)

ax.axhline(aL0_tat*180./np.pi, 0., 24., color='0.5', linestyle='--')
ax.annotate(r'\textbf{THIN AIRFOIL THEORY}', (11., -0.022*180./np.pi), fontsize=16)
xlims = (2., 22.)
ylims = (-2.75, 0.75)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.5, "minor": 0.5/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
ax.set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax.set_ylabel(r'\textbf{Zero-Lift Angle of Attack, }\boldmath$\alpha_{L0}$\textbf{ [deg]}', fontsize=16)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/64A204_a0.pdf', dpi=1000)

fig, ax = plt.subplots()
ax.scatter(CLa[:, 0], CLa[:, 1], ec='k', fc='none', label='Loftin [97]')
print(CLa_tat)

ax.axhline(CLa_tat*np.pi/180., 0., 24., color='0.5', linestyle='--')
ax.annotate(r'\textbf{THIN AIRFOIL THEORY}', (11., 0.11), fontsize=16)
xlims = (2., 22.)
ylims = (0.0925, 0.1275)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.005, "minor": 0.005/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
ax.set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax.set_ylabel(r'\textbf{Lift Slope, }\boldmath$C_{L,\alpha}$\textbf{ [1/deg]}', fontsize=16)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/64A204_CLa.pdf', dpi=1000)

fig, ax = plt.subplots()
ax.scatter(Cmc4[:, 0], Cmc4[:, 1], ec='k', fc='none', label='Loftin [97]')
print(Cmc4_tat)

ax.axhline(Cmc4_tat, 0., 24., color='0.5', linestyle='--')
ax.annotate(r'\textbf{THIN AIRFOIL THEORY}', (11., -0.0335), fontsize=16)
xlims = (2., 22.)
ylims = (-0.1725, 0.1725)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.05, "minor": 0.05/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
ax.set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax.set_ylabel(r'\textbf{Quarter-Chord Pitching Moment}' '\n' r'\textbf{ Coefficient, }' r'\boldmath$C_{m_{c/4}}(\alpha=0)$', fontsize=16)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/64A204_Cmc4.pdf', dpi=1000)

fig, ax = plt.subplots()
ax.scatter(CD0[:, 0], CD0[:, 1], ec='k', fc='none', label='Loftin [97]')

CD0_t, CD0_0 = np.polyfit(CD0[:, 0], CD0[:, 1], 1)
ax.plot(thickness, CD0_0 + CD0_t*thickness, color='0.5')
CD0_64A204 = CD0_0 + CD0_t*4
print(CD0_64A204)
ax.scatter(6., 0.0038404, ec='k', fc='none', marker='s', label='Abbott et al. [98]')
ax.scatter(4., CD0_64A204, ec='k', fc='none', marker='d', label='64A204 Estimate')

xlims = (2., 22.)
ylims = (-0.001, 0.013)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.002, "minor": 0.002/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
ax.set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax.set_ylabel(r'\textbf{Minimum Section Drag}' '\n' r'\textbf{ Coefficient, }' r'\boldmath$C_{D_0}$', fontsize=16)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/64A204_CD0.pdf', dpi=1000)


fig, ax = plt.subplots(2, 1, sharex=True)
CL = np.linspace(-1.5, 1.5)
CD2_A210, CD1_A210, CD0_A210 = np.polyfit(CD_64A210[:, 0], CD_64A210[:, 1], 2)
CD2_206, CD1_206, CD0_206 = np.polyfit(CD_64_206[:, 0], CD_64_206[:, 1], 2)
CD2_A212, CD1_A212, CD0_A212 = np.polyfit(CD_64A212[:, 0], CD_64A212[:, 1], 2)
CD2_212, CD1_212, CD0_212 = np.polyfit(CD_64_212[:, 0], CD_64_212[:, 1], 2)
ax[0].scatter([10, 12], [CD1_A210, CD1_A212], ec='k', fc='none', label='Loftin [97]')
# ax[0].scatter([6, 12], [CD1_206, CD1_212], ec='k', fc='none', label='Abbott et al. [98]', marker='s')
ax[1].scatter([10, 12], [CD2_A210, CD2_A212], ec='k', fc='none', label='Loftin [97]')
# ax[1].scatter([6, 12], [CD2_206, CD2_212], ec='k', fc='none', label='Abbott et al. [98]', marker='s')

CD1_t, CD1_0 = np.polyfit([10, 12], [CD1_A210, CD1_A212], 1)
CD2_t, CD2_0 = np.polyfit([10, 12], [CD2_A210, CD2_A212], 1)
CD1_64A204 = CD1_0 + CD1_t*4
print(CD1_64A204)
CD2_64A204 = CD2_0 + CD2_t*4
print(CD2_64A204)
ax[0].scatter(4., CD1_64A204, ec='k', fc='none', marker='d', label='64A204 Estimate')
ax[1].scatter(4., CD2_64A204, ec='k', fc='none', marker='d', label='64A204 Estimate')
ax[0].plot(thickness, CD1_0 + CD1_t*thickness, color='0.5')
ax[1].plot(thickness, CD2_0 + CD2_t*thickness, color='0.5')

xlims = (2., 22.)
ylims = (-0.0015,-0.0005)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.0002, "minor": 0.0002/4}
ax[0] = pretty_plot(ax[0], xlims, ylims, dx, dy)
ax[0].grid()
ylims = (0.0025,0.0075)
dy = {"major": 0.001, "minor": 0.001/4}
ax[1] = pretty_plot(ax[1], xlims, ylims, dx, dy)
ax[1].grid()
fig.supylabel(r'\textbf{Drag Derivatives}', fontsize=16)
ax[1].set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax[0].set_ylabel(r'\boldmath$C_{D,L}$', fontsize=16)
ax[1].set_ylabel(r'\boldmath$C_{D,L^2}$', fontsize=16)
ax[0].legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/64A204_CD1_2.pdf', dpi=1000)

fig, ax = plt.subplots(3, 1, sharex=True)
CL = np.linspace(-1.5, 1.5)
CD2_0006, CD1_0006, CD0_0006 = np.polyfit(CD_0006[:, 0], CD_0006[:, 1], 2)
CD2_0009, CD1_0009, CD0_0009 = np.polyfit(CD_0009[:, 0], CD_0009[:, 1], 2)
ax[0].scatter([6, 9], [CD0_0006, CD0_0009], ec='k', fc='none', label='Abbott et al. [98]')
ax[1].scatter([6, 9], [CD1_0006, CD1_0009], ec='k', fc='none')
ax[2].scatter([6, 9], [CD2_0006, CD2_0009], ec='k', fc='none')

CD0_t, CD0_0 = np.polyfit([6, 9], [CD0_0006, CD0_0009], 1)
CD1_t, CD1_0 = np.polyfit([6, 9], [CD1_0006, CD1_0009], 1)
CD2_t, CD2_0 = np.polyfit([6, 9], [CD2_0006, CD2_0009], 1)
CD0_0004 = CD0_0 + CD0_t*4
print(CD0_0004)
CD1_0004 = CD1_0 + CD1_t*4
print(CD1_0004)
CD2_0004 = CD2_0 + CD2_t*4
print(CD2_0004)
CD0_0005 = CD0_0 + CD0_t*5
print(CD0_0005)
CD1_0005 = CD1_0 + CD1_t*5
print(CD1_0005)
CD2_0005 = CD2_0 + CD2_t*5
print(CD2_0005)
ax[0].scatter(4., CD0_0004, ec='k', fc='none', marker='^', label='0004 Estimate')
ax[1].scatter(4., CD1_0004, ec='k', fc='none', marker='^', label='0004 Estimate')
ax[2].scatter(4., CD2_0004, ec='k', fc='none', marker='^', label='0004 Estimate')
ax[0].scatter(5., CD0_0005, ec='k', fc='none', marker='d', label='0005 Estimate')
ax[1].scatter(5., CD1_0005, ec='k', fc='none', marker='d', label='0005 Estimate')
ax[2].scatter(5., CD2_0005, ec='k', fc='none', marker='d', label='0005 Estimate')
ax[0].plot(thickness, CD0_0 + CD0_t*thickness, color='0.5')
ax[1].plot(thickness, CD1_0 + CD1_t*thickness, color='0.5')
ax[2].plot(thickness, CD2_0 + CD2_t*thickness, color='0.5')

xlims = (2., 22.)
ylims = (0.00435,0.00525)
dx = {"major": 4, "minor": 1}
dy = {"major": 0.0003, "minor": 0.0003/4}
ax[0] = pretty_plot(ax[0], xlims, ylims, dx, dy)
ax[0].grid()
ylims = (-0.005, 0.001)
dy = {"major": 0.002, "minor": 0.002/4}
ax[1] = pretty_plot(ax[1], xlims, ylims, dx, dy)
ax[1].grid()
ylims = (0.0015,0.0105)
dy = {"major": 0.003, "minor": 0.003/4}
ax[2] = pretty_plot(ax[2], xlims, ylims, dx, dy)
ax[2].grid()
fig.supylabel(r'\textbf{Drag Derivatives}', fontsize=16)
ax[2].set_xlabel(r'\textbf{Maximum Airfoil Thickness, }\boldmath$\frac{t_\mathrm{max}}{c}$\textbf{ \%}', fontsize=16)
ax[0].set_ylabel(r'\boldmath$C_{D_0}$', fontsize=16)
ax[1].set_ylabel(r'\boldmath$C_{D,L}$', fontsize=16)
ax[2].set_ylabel(r'\boldmath$C_{D,L^2}$', fontsize=16)
ax[0].legend(fontsize=16)
plt.tight_layout()
plt.savefig('./Airfoil Data/0004_5_CD0_1_2.pdf', dpi=1000)

