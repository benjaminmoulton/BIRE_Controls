#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:46:01 2022

@author: christian
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import aero_trim
from matplotlib import colors
from hunsaker_atm import gravity_english, stdatm_english
import scipy.optimize as optimize
import itertools
from multiprocessing import Pool
import ZachsModules as zm
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import time

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
Gamma = 0.8
Gamma_B = 0.5
H = 30000.
N = 50
phi = np.linspace(0., 45., N)
CLmax = 1.9
rho, a = stdatm_english(H)[3:]
W = 20500.
S_w = 300.
V_stall = np.sqrt(2.*W/S_w/CLmax/rho)
M = np.linspace(0.2, 0.8, N)
V = M*a
gamma = np.deg2rad(0.)
cg_shift = [1., 0., 0.]
generate_data = False
parallel = False
np.save('./Crosswind Data/SHSS_Mach.npy', M)
np.save('./Crosswind Data/SHSS_Bank_Angle.npy', phi)
print("SHSS", H, " CG", cg_shift[0])

def shss_trim(params):
    V = params[0]
    phi = params[1]
    try:
        solution_base = aero_trim.trim(V, H, gamma, np.deg2rad(phi), Gamma, shss=True, cg_shift=cg_shift, verbose=False, compressible=True)
        trim_state = solution_base.x
        CL_base = solution_base.FM[2]
        phi_ij = solution_base.orient[0]
        theta_ij = solution_base.orient[1]
        rudder_deg = trim_state[5]*180./np.pi
        elevator_deg = trim_state[4]*180./np.pi
        phi_deg = phi_ij*180./np.pi
        theta_deg = theta_ij*180./np.pi
    except TypeError:
        trim_state = np.array([np.nan]*6)
        CL_base = np.nan
        phi_deg = np.nan
        theta_deg = np.nan
        rudder_deg = np.nan
        elevator_deg = np.nan
    try:
        solution_bire = aero_trim.trim(V, H, gamma, np.deg2rad(phi), Gamma_B, shss=True, cg_shift=cg_shift, verbose=False, bire=True, fixed_point=False, compressible=True)
        trim_state_bire = solution_bire.x
        CL_BIRE = solution_bire.FM[2]
        phi_ij = solution_bire.orient[0]
        theta_ij = solution_bire.orient[1]
        BIRE_rotation_deg = trim_state_bire[5]*180./np.pi
        BIRE_elevator_deg = trim_state_bire[4]*180./np.pi
        BIRE_phi_deg = phi_ij*180./np.pi
        BIRE_theta_deg = theta_ij*180./np.pi
    except TypeError:
        trim_state_bire = np.array([np.nan]*6)
        CL_BIRE = np.nan
        BIRE_phi_deg = np.nan
        BIRE_theta_deg = np.nan
        BIRE_elevator_deg = np.nan
        BIRE_rotation_deg = np.nan
    return V, phi, rudder_deg, elevator_deg, phi_deg, theta_deg, CL_base, BIRE_rotation_deg, BIRE_elevator_deg, BIRE_phi_deg, BIRE_theta_deg, CL_BIRE



if generate_data:
    if parallel:
        cases = list(itertools.product(V, phi))
        fn = './Crosswind Data/SHSS_analysis.csv'
        f = open(fn, 'w')
        f.write(zm.io.csvLineWrite('V',
                                   'phi',
                                   'base_dr',
                                   'base_de',
                                   'base_phi',
                                   'base_theta',
                                   'base_CL',
                                   'BIRE_dB',
                                   'BIRE_de',
                                   'BIRE_phi',
                                   'BIRE_theta',
                                   'BIRE_CL'))
        f.close()
        bat = 100
        chu = 10
        zm.nm.runCases(shss_trim, cases, fn, nBatch=bat, chunkSize=chu,
                       progKW={'title': 'Running Cases: {}/batch, {}/chunck'.format(bat, chu)})
        data = np.genfromtxt(fn, delimiter=',', skip_header=1)
        df = pd.DataFrame(data, columns = ['V', 'phi', 'base_dr',
                                   'base_de',
                                   'base_phi',
                                   'base_theta',
                                   'base_CL',
                                   'BIRE_dB',
                                   'BIRE_de',
                                   'BIRE_phi',
                                   'BIRE_theta',
                                   'BIRE_CL'])
        df = df.sort_values(by = ['V', 'phi'], ascending = [True, True], na_position = 'first')
        data = df.to_numpy()
        rudder_deg = np.reshape(data[:, 2], (N, N))
        elevator_deg = np.reshape(data[:, 3], (N, N))
        phi_deg = np.reshape(data[:, 4], (N, N))
        theta_deg = np.reshape(data[:, 5], (N, N))
        CL_base = np.reshape(data[:, 6], (N, N))
        BIRE_rotation_deg = np.reshape(data[:, 7], (N, N))
        BIRE_elevator_deg = np.reshape(data[:, 8], (N, N))
        BIRE_phi_deg = np.reshape(data[:, 9], (N, N))
        BIRE_theta_deg = np.reshape(data[:, 10], (N, N))
        CL_BIRE = np.reshape(data[:, 11], (N, N))
    else:
        t0 = time.time()
        rudder_deg = np.zeros((len(V), len(phi)))
        V_cross = np.zeros((len(V), len(phi)))
        elevator_deg = np.zeros((len(V), len(phi)))
        trim_state = np.zeros(6)
        CL_base = np.zeros((len(V), len(phi)))
        CD_base = np.zeros_like(CL_base)
        Cn_base = np.zeros_like(CL_base)
        phi_deg = np.zeros((len(V), len(phi)))
        theta_deg = np.zeros((len(V), len(phi)))
        BIRE_rotation_deg = np.zeros((len(V), len(phi)))
        BIRE_V_cross = np.zeros((len(V), len(phi)))
        BIRE_elevator_deg = np.zeros((len(V), len(phi)))
        BIRE_phi_deg = np.zeros((len(V), len(phi)))
        BIRE_theta_deg = np.zeros((len(V), len(phi)))
        CL_BIRE = np.zeros((len(V), len(phi)))
        CD_BIRE = np.zeros((len(V), len(phi)))
        Cn_BIRE = np.zeros_like(CL_BIRE)
        trim_state_bire = np.zeros(6)
        for i in range(len(V)):
            print(V[i])
            trim_0 = np.zeros(6)
            trim_state_bire = np.zeros(6)
            trim_state = np.zeros(6)
            for j in range(len(phi)):
                if trim_state[5]*180./np.pi > 35.:
                    rudder_deg[i, j] = np.nan
                    elevator_deg[i, j] = np.nan
                    phi_deg[i, j] = np.nan
                    theta_deg[i, j] = np.nan
                    V_cross[i, j] = np.nan
                    BIRE_rotation_deg[i, j] = np.nan
                    BIRE_elevator_deg[i, j] = np.nan
                    BIRE_phi_deg[i, j] = np.nan
                    BIRE_theta_deg[i, j] = np.nan
                    BIRE_V_cross[i, j] = np.nan
                else:
                    try:
                        solution_base = aero_trim.trim(V[i], H, gamma, np.deg2rad(phi[j]), Gamma, shss=True, cg_shift=cg_shift, verbose=True)
                        trim_state = solution_base.x
                        CL_base[i, j] = solution_base.FM[2]
                        CD_base[i, j] = solution_base.FM[0]
                        Cn_base[i, j] = solution_base.FM[5]
                        phi_ij = solution_base.orient[0]
                        theta_ij = solution_base.orient[1]
                        [u, v, w] = solution_base.velocity
                        rudder_deg[i, j] = trim_state[5]*180./np.pi
                        elevator_deg[i, j] = trim_state[4]*180./np.pi
                        phi_deg[i, j] = phi_ij*180./np.pi
                        theta_deg[i, j] = theta_ij*180./np.pi
                        c_a = np.cos(trim_state[1])
                        s_a = np.sin(trim_state[1])
                        c_b = np.cos(trim_state[2])
                        s_b = np.sin(trim_state[2])
                        V_cross[i, j] = -c_a*s_b*u + c_b*v - s_a*s_b*w
                    except TypeError:
                        trim_state = np.array([np.nan]*6)
                        CL_base[i, j] = np.nan
                        CD_base[i, j] = np.nan
                        Cn_base[i, j] = np.nan
                        phi_deg[i, j] = np.nan
                        theta_deg[i, j] = np.nan
                        V_cross[i, j] = np.nan
                    try:
                        solution_bire = aero_trim.trim(V[i], H, gamma, np.deg2rad(phi[j]), Gamma_B, shss=True, cg_shift=cg_shift, verbose=True, bire=True, fixed_point=False, trim_0=trim_0)
                        trim_state_bire = solution_bire.x
                        CL_BIRE[i, j] = solution_bire.FM[2]
                        CD_BIRE[i, j] = solution_bire.FM[0]
                        Cn_BIRE[i, j] = solution_bire.FM[5]
                        phi_ij = solution_bire.orient[0]
                        theta_ij = solution_bire.orient[1]
                        [u, v, w] = solution_bire.velocity
                        BIRE_rotation_deg[i, j] = trim_state_bire[5]*180./np.pi
                        BIRE_elevator_deg[i, j] = trim_state_bire[4]*180./np.pi
                        BIRE_phi_deg[i, j] = phi_ij*180./np.pi
                        BIRE_theta_deg[i, j] = theta_ij*180./np.pi
                        c_a = np.cos(trim_state_bire[1])
                        s_a = np.sin(trim_state_bire[1])
                        c_b = np.cos(trim_state_bire[2])
                        s_b = np.sin(trim_state_bire[2])
                        BIRE_V_cross[i, j] = -c_a*s_b*u + c_b*v - s_a*s_b*w
                        trim_0 = trim_state_bire
                    except TypeError:
                        trim_state_bire = np.array([np.nan]*6)
                        CL_BIRE[i, j] = np.nan
                        phi_deg[i, j] = np.nan
                        theta_deg[i, j] = np.nan
                        BIRE_V_cross[i, j] = np.nan
                print(BIRE_rotation_deg[i, j])
    np.save(f"./Crosswind Data/SHSS_base_rudder{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", rudder_deg)
    np.save(f"./Crosswind Data/SHSS_base_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", elevator_deg)
    np.save(f"./Crosswind Data/SHSS_base_phi{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", phi_deg)
    np.save(f"./Crosswind Data/SHSS_base_theta{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", theta_deg)
    np.save(f"./Crosswind Data/SHSS_base_Vcross{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", V_cross)
    np.save(f"./Crosswind Data/SHSS_base_CL{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", CL_base)
    np.save(f"./Crosswind Data/SHSS_base_CD{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", CD_base)
    np.save(f"./Crosswind Data/SHSS_base_Cn{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", Cn_base)
    np.save(f"./Crosswind Data/SHSS_BIRE_rotation{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", BIRE_rotation_deg)
    np.save(f"./Crosswind Data/SHSS_BIRE_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", BIRE_elevator_deg)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_phi{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", BIRE_phi_deg)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_theta{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", BIRE_theta_deg)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_Vcross{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", BIRE_V_cross)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_CL{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", CL_BIRE)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_CD{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", CD_BIRE)
    np.save(f"./Crosswind Data/Tail_Strike_BIRE_Cn{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy", Cn_BIRE)
    t1 = time.time()
    print(t1 - t0)

if generate_data == False:
    rudder_deg = np.load(f"./Crosswind Data/SHSS_base_rudder{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
    elevator_deg = np.load(f"./Crosswind Data/SHSS_base_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
    CL_base = np.load(f"./Crosswind Data/SHSS_base_CL{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
    BIRE_rotation_deg = np.load(f"./Crosswind Data/SHSS_BIRE_rotation{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
    BIRE_elevator_deg = np.load(f"./Crosswind Data/SHSS_BIRE_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")


# for i in range(len(V)):
#     for j in range(len(V)):
#         if BIRE_rotation_deg[i, j] >= 360.:
#             while BIRE_rotation_deg[i, j] >= 180.:
#                 BIRE_rotation_deg[i, j] -= 360.
#         elif BIRE_rotation_deg[i, j] > 90.:
#             while BIRE_rotation_deg[i, j] > 90.:
#                 BIRE_rotation_deg[i, j] -= 180.
#                 BIRE_elevator_deg[i, j] *= -1.
#         if BIRE_rotation_deg[i, j] <= -360.:
#             while BIRE_rotation_deg[i, j] <= -180.:
#                 BIRE_rotation_deg[i, j] += 360.
#         elif BIRE_rotation_deg[i, j] < -90.:
#             while BIRE_rotation_deg[i, j] < -90.:
#                 BIRE_rotation_deg[i, j] += 180.
#                 BIRE_elevator_deg[i, j] *= -1.
# try:
#     V_stall = V[CL_base[:, 0] >= CLmax][-1]
#     Vstall_indx = CL_base[:, 0] >= CLmax
#     phistall_indx = rudder_deg[Vstall_indx, :] <= 43.
#     phi_stall = phi[phistall_indx[-1, :]]
#     V_stall = np.zeros_like(phi_stall)
#     V_stall[:] = M[Vstall_indx][-1]
#     stalled = True
# except IndexError:
#     stalled = False


X, Y = np.meshgrid(phi, M)
fig, ax = plt.subplots()
cmap = colors.ListedColormap('black')
lvls = np.arange(5., 35., 5.)
options_base = {"cmap": cmap, "levels": lvls, "linestyles": '-', "zorder": 2}
options_clabel = {"levels": lvls, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, "zorder": 2}
CS = ax.contour(X, Y, rudder_deg, **options_base)
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        manual_locations = [(8, 0.66), (8, 0.48), (11.8, 0.47), (19, 0.51), (26.5, 0.53), (28, 0.5)] # 1000
    elif H == 15000.:
        manual_locations = [(5, 0.7), (9, 0.66), (7.75, 0.49), (14.5, 0.60), (22, 0.64), (28, 0.65)] # 15000
    else:
        manual_locations = [(2.9, 0.76), (6, 0.73), (7.5, 0.66), (9.6, 0.66), (13.5, 0.71), (17, 0.73)] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        manual_locations = [(5., 0.6), (12, 0.57), (20, 0.6), (27.5, 0.66), (24, 0.54), (28, 0.55)] # 1000
    elif H == 15000.:
        manual_locations = [(5, 0.7), (9, 0.66), (15, 0.68), (17.6, 0.68), (23, 0.7), (28, 0.65)] # 15000
    else:
        manual_locations = [(3, 0.68), (6, 0.75), (7, 0.7), (10, 0.7), (12, 0.73), (14, 0.68)] # 30000
plt.clabel(CS, **options_clabel, manual=manual_locations)

mask = (rudder_deg > 30.)
X[mask] = np.nan
Y[mask] = np.nan
elevator_deg[mask] = np.nan
BIRE_rotation_deg[mask] = np.nan
BIRE_elevator_deg[mask] = np.nan
mask = (phi > 45.)
X[mask] = np.nan
Y[mask] = np.nan
elevator_deg[mask] = np.nan
BIRE_rotation_deg[mask] = np.nan
BIRE_elevator_deg[mask] = np.nan

BIRE_elevator_deg[BIRE_rotation_deg > 90.] *= -1.
BIRE_rotation_deg[BIRE_rotation_deg > 90.] -= 180.
BIRE_elevator_deg[BIRE_rotation_deg < -90.] *= -1.
BIRE_rotation_deg[BIRE_rotation_deg < -90.] += 180.

cmap2 = colors.ListedColormap('black')
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        lvls2 = [-0.5, 0., 1., 2., 3.] # 1000
    elif H == 15000.:
        lvls2 = [-0.5, 0., 2., 6.] # 15000
    else:
        lvls2 = [0., 1., 3., 5., 7.] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        lvls2 = [-4., -3., -2., -1.5] # 1000
    elif H == 15000.:
        lvls2 = [-6., -3., -2.] # 15000
    else:
        lvls2 = [-8., -5., -3.] # 30000
options_base2 = {"cmap": cmap2, "levels": lvls2, "linestyles": '--', "zorder": 2}
options_clabel2 = {"levels": lvls2, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, "zorder": 2}
CS2 = ax.contour(X, Y, elevator_deg, **options_base2)
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        manual_locations = [(15, 0.6), (4.4, 0.23), (12.5, 0.44), (6.3, 0.26), (8.6, 0.32)] # 1000
    elif H == 15000.:
        manual_locations = [(15, 0.77), (5.5, 0.58), (6.5, 0.34), (2.5, 0.22)] # 15000
    else:
        manual_locations = [(1.5, 0.29), (1.5, 0.33), (4.3, 0.41), (8.6, 0.57), (18.4, 0.77)] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        manual_locations = [(2.6, 0.67), (4.8, 0.24), (2.3, 0.32), (8., 0.44)] # 1000
    elif H == 15000.:
        manual_locations = [(7.75, 0.57), (5.3, 0.4), (3, 0.24)] # 15000
    else:
        manual_locations = [(3.5, 0.27), (4.2, 0.37), (5.3, 0.54)] # 30000
plt.clabel(CS2, **options_clabel2, manual=manual_locations)

cmap_B = colors.ListedColormap('black')
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        lvls_B = [-5., 2., 5., 10.] # 1000
    elif H == 15000.:
        lvls_B = [-5., -2., 2., 5., 10., 20.] # 15000
    else:
        lvls_B = [-2., 5., 10., 20.] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        lvls_B = [-4, -2., -1., -0.5] # 1000
    elif H == 15000.:
        lvls_B = [ -3., -2., -1., -0.5] # 15000
    else:
        lvls_B = [-6., -3., -1.] # 30000
options_BIRE = {"cmap": cmap_B, "levels": lvls_B, "linestyles": ':', "zorder": 2}
options_clabel_B = {"levels": lvls_B, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, "zorder": 2}
CS_BIRE = ax.contour(X, Y, BIRE_rotation_deg, **options_BIRE)
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        manual_locations = [(3.4, 0.53), (8.3, 0.54), (18.7, 0.55), (2.6, 0.3)] # 1000
    elif H == 15000.:
        manual_locations = [(2.7, 0.67), (8.3, 0.71), (4.8, 0.55), (11.1, 0.54), (2.6, 0.39), (2.1, 0.33)] # 15000
    else:
        manual_locations = [(1.75, 0.73), (5.5, 0.76), (11, 0.75), (3.8, 0.37)] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        manual_locations = [(12.4, 0.47), (11.9, 0.36), (6.57, 0.28), (7.58, 0.53)] # 1000
    elif H == 15000.:
        manual_locations = [(5.3, 0.61), (12.7, 0.62), (9.5, 0.47), (3.2, 0.34)] # 15000
    else:
        manual_locations = [(2.9, 0.59), (9.3, 0.57), (2.3, 0.35)] # 30000
plt.clabel(CS_BIRE, **options_clabel_B, manual=manual_locations)

cmap_B2 = colors.ListedColormap('black')
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        lvls_B2 = [-3., 0., 1., 3., 5.] # 1000
    elif H == 15000.:
        lvls_B2 = [-4., -1., 0., 3., 5.] # 15000
    else:
        lvls_B2 = [-5., -3., 1., 5., 10.] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        lvls_B2 = [-6., -4., -2., -1.] # 1000
    elif H == 15000.:
        lvls_B2 = [-10., -5., -3., -1.5] # 15000
    else:
        lvls_B2 = [-15., -10., -5., -3.] # 30000
options_BIRE2 = {"cmap": cmap_B2, "levels": lvls_B2, "linestyles": '-.', "zorder": 2}
options_clabel_B2 = {"levels": lvls_B2, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, "zorder": 2}
CS_BIRE2 = ax.contour(X, Y, BIRE_elevator_deg, **options_BIRE2)
if cg_shift[0] == 0.:
    # Nominal CG
    if H == 1000.:
        manual_locations = [(10.2, 0.69), (4.5, 0.41), (2.8, 0.25), (27.7, 0.64), (14, 0.55)] # 1000
    elif H == 15000.:
        manual_locations = [(7.2, 0.75), (13.8, 0.68), (16.2, 0.66), (3.9, 0.38)] # 15000
    else:
        manual_locations = [(5, 0.47), (2.5, 0.55), (9.4, 0.69), (8.5, 0.75), (4.3, 0.74)] # 30000
else:
    # CG +1 ft
    if H == 1000.:
        manual_locations = [(3, 0.71), (9.9, 0.5), (8, 0.37), (7.9, 0.32)] # 1000
    elif H == 15000.:
        manual_locations = [(2.7, 0.75), (14.9, 0.59), (6.3, 0.42), (5.1, 0.32)] # 15000
    else:
        manual_locations = [(3.9, 0.71), (4.2, 0.56), (6.6, 0.46), (2.8, 0.32)] # 30000
plt.clabel(CS_BIRE2, **options_clabel_B2, manual=manual_locations)

# if stalled:
#     ax.plot(phi_stall, V_stall, color='k', linestyle='--', linewidth=2)
#     ax.annotate("Stall Speed", (phi[phi > phi_stall[-1]][0] + 0.5, V[Vstall_indx][-2]), (phi[phi > phi_stall[-1]][0] + 0.5, V[Vstall_indx][-2]), fontsize=16)

ax.set_ylabel(r'\textbf{Mach Number, }\boldmath$M$', fontsize=16)
ax.set_xlabel(r'\textbf{Bank Angle, }\boldmath$\phi$\textbf{ [deg]}', fontsize=16)

h1,_ = CS.legend_elements()
h2,_ = CS2.legend_elements()
h3,_ = CS_BIRE.legend_elements()
h4,_ = CS_BIRE2.legend_elements()
ax.legend([h1[0], h2[0], h3[0], h4[0]], [r'$\delta_r$', r'$\delta_e$', r'$\delta_B$', r'$\delta_e^B$'], loc='lower right', fontsize=16)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax.set_xlim(0., 45)
ax.set_ylim(0.2, 0.8)
ax.set_yticks(np.arange(0.2, 0.9, 0.1))
ax.axvline(0., color='k', linewidth=2)
xlim = (0., 45.)
ylim = (0.2, 0.8)
dx = {'major': 5., 'minor': 5./4.}
dy = {'major': 0.1, 'minor': 0.1/4.}
ax = pretty_plot(ax, xlim, ylim, dx, dy)
ax.grid()

plt.tight_layout()
plt.savefig(f"./SHSS Figures/Altitude{int(H):2d}CG{int(cg_shift[0]):2d}.pdf")

