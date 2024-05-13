#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:39:07 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import aero_trim
from hunsaker_atm import stdatm_english
import scipy.optimize as optimize
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
H = 30000.
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

def find_Xwind(phi, x_shift):
    solution = aero_trim.trim(V_stall, H, gamma, phi[0], Gamma, shss=True, cg_shift=[x_shift + 1., 0., 0.], verbose=False)
    trim_state = solution.x
    dr = trim_state[5]*180./np.pi
    return (dr - 30.)**2

if generate_data:
    trim_0 = np.zeros(6)
    phi_0 = 0.
    for i in range(len(x_shifts)):
        res = optimize.minimize(find_Xwind, 0., args=(x_shifts[i]), options={'gtol': 1e-6, 'return_all': True})
        print(res.message)
        phi[i] = res.x[0]
        # phi[i] = 0.
        try:
            solution_base = aero_trim.trim(V_stall, H, gamma, phi[i], Gamma, shss=True, cg_shift=[x_shifts[i] + 1., 0., 0.], verbose=False)
            xwind_state = solution_base.x
        except TypeError:
            xwind_state = np.array([np.nan]*6)
        de[i] = xwind_state[4]*180./np.pi
        dr[i] = xwind_state[5]*180./np.pi

        solution_bire = aero_trim.trim(V_stall, H, gamma, phi[i], Gamma_B, shss=True, cg_shift=[x_shifts[i] + 1., 0., 0.], verbose=False, bire=True, fixed_point=False, trim_0=trim_0)
        trim_state = solution_bire.x
        deB[i] = trim_state[4]*180./np.pi
        dB[i] = trim_state[5]*180./np.pi
        trim_0 = trim_state
        print(dB[i])
        phi_0 = phi[i]
    np.save(f"./Crosswind Data/XWind_CG_elevator_H{int(H):2d}.npy", de)
    np.save(f"./Crosswind Data/XWind_CG_rudder_H{int(H):2d}.npy", dr)
    np.save(f"./Crosswind Data/XWind_CG_BIRE_rotation_H{int(H):2d}.npy", dB)
    np.save(f"./Crosswind Data/XWind_CG_BIRE_elevator_H{int(H):2d}.npy", deB)
    np.save(f"./Crosswind Data/XWind_CG_phi_H{int(H):2d}.npy", phi)
else:
    de = np.load(f"./Crosswind Data/XWind_CG_elevator_H{int(H):2d}.npy")
    dr = np.load(f"./Crosswind Data/XWind_CG_rudder_H{int(H):2d}.npy")
    dB = np.load(f"./Crosswind Data/XWind_CG_BIRE_rotation_H{int(H):2d}.npy")
    deB = np.load(f"./Crosswind Data/XWind_CG_BIRE_elevator_H{int(H):2d}.npy")
    phi = np.load(f"./Crosswind Data/XWind_CG_phi_H{int(H):2d}.npy")

fig, ax = plt.subplots()
ax2 = ax.twinx()
dr_line = ax.plot(x_shifts, dr, color='k', linestyle='-')
de_line = ax.plot(x_shifts, de, color='k', linestyle='--', label=r"$\delta_e$")
phi_line = ax.plot(x_shifts, phi*180./np.pi, color='k', linestyle=(5, (10, 3)), label=r"$\phi$")

dB_line = ax2.plot(x_shifts, dB, color='k', linestyle=':', label="$\delta_B$")
deB_line = ax.plot(x_shifts, deB, color='k', linestyle='-.', label="$\delta_e^B$")

ax.set_xlabel(r'\textbf{CG Shift, }\boldmath$\Delta x_\mathrm{cg}$\textbf{ [ft]}', fontsize=16)
ax.set_ylabel(r'\textbf{Angle [deg]}', fontsize=16)
ax2.set_ylabel(r'\textbf{BIRE Rotation Angle, }\boldmath$\delta_B$\textbf{ [deg]}', fontsize=16)
lns = dr_line + de_line + phi_line + dB_line + deB_line
labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, fontsize=16)

ylims = (-35., 35.)
dy = {'major': 10, 'minor': 10/4}
ylims2 = (-105., 105.)
dy2 = {'major': 30, 'minor': 30/4}
xlims = (-0.5, 1.5)
dx = {'major': 0.5, 'minor': 0.5/4}

ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax2 = pretty_plot(ax2, xlims, ylims2, dx, dy2)
ax.annotate(r'\boldmath$\delta_r$', (0.5, 26), fontsize=16)
ax.annotate(r'\boldmath$\phi$', (0.5, 8), fontsize=16)
ax2.annotate(r'\boldmath$\delta_B$', (0.6, -20), fontsize=16)
ax.annotate(r'\boldmath$\delta_e$', (1, -20), fontsize=16)
ax.annotate(r'\boldmath$\delta_e^B$', (0.3, -27), fontsize=16)
ax.grid()
plt.tight_layout()
plt.savefig(f"./CG Figures/Altitude{int(H):2d}.pdf")
