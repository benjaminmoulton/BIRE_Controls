#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:28:49 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import aero_trim
from matplotlib import colors
from hunsaker_atm import gravity_english, stdatm_english
import scipy.optimize as optimize
from bire_aero import BIREAero
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

plt.close('all')
H = 15000.
CLmax = 1.9
rho = stdatm_english(H)[3]
W = 20500.
S_w = 300.
V_stall = np.sqrt(2.*W/S_w/CLmax/rho)

aft_cg_limit = 11.32*(0.35 - 0.4)
x_shifts = np.linspace(1.5, aft_cg_limit)
# x_shifts = np.arange(1.5, -0.6, -0.5)
deB = np.zeros(len(x_shifts))
de = np.zeros(len(x_shifts))
dB = np.zeros(len(x_shifts))
dr = np.zeros(len(x_shifts))
phi = np.zeros(len(x_shifts))
FM = np.zeros((len(x_shifts), 6))

gamma = 0.
Gamma = 0.1
Gamma_B = 0.1
generate_data = False
case = BIREAero()
n_target = 1.

def find_target_g(phi, x_shift):
    solution = aero_trim.trim(V_stall, H, gamma, phi[0], Gamma, shss=False, cg_shift=[x_shift, 0., 0.], verbose=False, fixed_point=True)
    n_a = solution.load
    return abs(n_a - n_target)**2

if generate_data:
    asymptote = False
    trim_0 = np.zeros(6)
    phi_0 = 0.
    for i in range(len(x_shifts)):
        res = optimize.minimize(find_target_g, 0., args=(x_shifts[i]), method='Nelder-Mead', options={'gtol': 1e-6, 'return_all': True})
        phi[i] = res.x[0]
        # phi[i] = 0.
        try:
            solution_base = aero_trim.trim(V_stall, H, gamma, phi[i], Gamma, shss=False, cg_shift=[x_shifts[i], 0., 0.], verbose=False)
            state_na = solution_base.x
        except TypeError:
            state_na = np.array([np.nan]*6)
        de[i] = state_na[4]*180./np.pi
        dr[i] = state_na[5]*180./np.pi

        solution_bire = aero_trim.trim(V_stall, H, gamma, phi[i], Gamma, shss=False, cg_shift=[x_shifts[i], 0., 0.], verbose=False, bire=True, fixed_point=False, trim_0=trim_0)
        state_na_bire = solution_bire.x
        deB[i] = state_na_bire[4]*180./np.pi
        dB[i] = state_na_bire[5]*180./np.pi
        trim_0 = state_na_bire
        print(dB[i])
        phi_0 = phi[i]
    np.save(f"./SCT Data/SCT_CG_elevator_H{int(H):2d} {int(n_target):1d}g.npy", de)
    np.save(f"./SCT Data/SCT_CG_rudder_H{int(H):2d} {int(n_target):1d}g.npy", dr)
    np.save(f"./SCT Data/SCT_CG_BIRE_rotation_H{int(H):2d} {int(n_target):1d}g.npy", dB)
    np.save(f"./SCT Data/SCT_CG_BIRE_elevator_H{int(H):2d} {int(n_target):1d}g.npy", deB)
    np.save(f"./SCT Data/SCT_CG_phi_H{int(H):2d} {int(n_target):1d}g.npy", phi)
else:
    de = np.load(f"./SCT Data/SCT_CG_elevator_H{int(H):2d} {int(n_target):1d}g.npy")
    dr = np.load(f"./SCT Data/SCT_CG_rudder_H{int(H):2d} {int(n_target):1d}g.npy")
    dB = np.load(f"./SCT Data/SCT_CG_BIRE_rotation_H{int(H):2d} {int(n_target):1d}g.npy")
    deB = np.load(f"./SCT Data/SCT_CG_BIRE_elevator_H{int(H):2d} {int(n_target):1d}g.npy")
    phi = np.load(f"./SCT Data/SCT_CG_phi_H{int(H):2d} {int(n_target):1d}g.npy")

fig, ax = plt.subplots()
ax2 = ax.twinx()
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
colors = ['0.0', '0.4', '0.8']
i = 0
for n in [1., 5., 9.]:
    de = np.load(f"./SCT Data/SCT_CG_elevator_H{int(H):2d} {int(n):1d}g.npy")
    dr = np.load(f"./SCT Data/SCT_CG_rudder_H{int(H):2d} {int(n):1d}g.npy")
    dB = np.load(f"./SCT Data/SCT_CG_BIRE_rotation_H{int(H):2d} {int(n):1d}g.npy")
    deB = np.load(f"./SCT Data/SCT_CG_BIRE_elevator_H{int(H):2d} {int(n):1d}g.npy")
    phi = np.load(f"./SCT Data/SCT_CG_phi_H{int(H):2d} {int(n):1d}g.npy")
    dr_line = ax.plot(x_shifts, dr, color=colors[i], label="$\delta_r$", linestyle='-')
    de_line = ax.plot(x_shifts, de, color=colors[i], label="$\delta_e$", linestyle='--')
    phi_line = ax.plot(x_shifts, phi*180/np.pi, color=colors[i], label='$\phi$', linestyle=(5, (10, 3)))

    for j in range(len(dB)):
        while abs(dB[j]) > 180.:
            if dB[j] > 180.:
                dB[j] -= 360.
            else:
                dB[j] += 360.
    dB_line = ax2.plot(x_shifts, dB, color=colors[i], label="$\delta_B$", linestyle=':')
    # dB2_line = ax2.plot(x_shifts, dB + 180., color='b', linestyle='--')
    # dB2_line = ax2.plot(x_shifts, dB - 180., color='b', linestyle='--')
    deB_line = ax.plot(x_shifts, deB, color=colors[i], label="$\delta_e^B$", linestyle='-.')
    # deB2_line = ax.plot(x_shifts, deB*-1, color='r', linestyle='--')
    if i == 0:
        lns = dr_line + de_line + phi_line + dB_line + deB_line
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=16, loc='upper right')
    i += 1
dummy_lines = []
i = 0
for n in [1., 5., 9.]:
    dummy_lines.append(ax.plot([], [], c=colors[i], ls='-')[0])
    i += 1
legend2 = plt.legend([dummy_lines[i] for i in [0,1, 2]], ["n = 1", "n = 5", "n = 9"], loc='upper left', fontsize=16)
ax.set_xlabel(r'\textbf{CG Shift, }\boldmath$\Delta x_\mathrm{cg}$\textbf{ [ft]}', fontsize=16)
ax.set_ylabel(r'\textbf{Angle [deg]}', fontsize=16)
ax2.set_ylabel(r'\textbf{BIRE Rotation Angle, }\boldmath$\delta_B$\textbf{ [deg]}', fontsize=16)

# ax.axhline(0., color='0.5')

ylims = (-140., 140.)
dy = {'major': 40, 'minor': 40/4}
ylims2 = (-210., 210.)
dy2 = {'major': 60, 'minor': 60/4}
xlims = (-0.5, 1.5)
dx = {'major': 0.5, 'minor': 0.5/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
ax.grid()
ax2.grid()
plt.tight_layout()
plt.show()
plt.savefig(f"./CG Figures/SCT_Altitude{int(H):2d}.pdf")
