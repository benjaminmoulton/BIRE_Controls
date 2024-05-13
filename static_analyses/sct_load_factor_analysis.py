#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:21:49 2022

@author: christian
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import aero_trim as trim
from hunsaker_atm import gravity_english, stdatm_english
import scipy.optimize as optimize
from matplotlib import colors
import alphashape
import itertools
from multiprocessing import Pool
import ZachsModules as zm
import pandas as pd
import time
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

H = 1000.
gamma = 0.
Gamma = 0.8
Gamma_B = 0.5
generate_data = False
parallel = False
N = 50
CLmax = 1.9
rho, a = stdatm_english(H)[3:]
W = 20500.
S_w = 300.
V_stall = np.sqrt(2.*W/S_w/CLmax/rho)
M = np.load('./Crosswind Data/SHSS_Mach.npy')
V = M*a
n = np.linspace(1., 9., N)
cg_shift = [0., 0., 0.]
np.save(f'./SCT Data/Loads Data/Load_Factor{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy', n)
print("SCT", H, " CG", cg_shift[0])

def find_loadfactor(phi, n, V, bire):
    if bire:
        try:
            solution_bire = trim.trim(V, H, gamma, phi[0], Gamma_B, shss=False, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
            n_a = solution_bire.load
        except np.linalg.LinAlgError:
            solution_bire = trim.trim(V, H, gamma, phi[0], 0.1, shss=False, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
            n_a = solution_bire.load
        return (n - n_a)**2
    else:
        try:
            solution_base = trim.trim(V, H, gamma, phi[0], Gamma, shss=False, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
            n_a = solution_base.load
        except np.linalg.LinAlgError:
            solution_base = trim.trim(V, H, gamma, phi[0], 0.1, shss=False, bire=bire, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
            n_a = solution_base.load
        return (n - n_a)**2

def sct_trim(params):
    V = params[0]
    n = params[1]
    phi_0 = np.arccos(1./n)
    phi_base = optimize.minimize(find_loadfactor, phi_0, args=(n, V, False), method='Nelder-Mead').x[0]
    CW = W/(0.5*rho*V**2*S_w)
    n_stall = CLmax/CW
    if n > n_stall:
        dr = np.nan
        de = np.nan
        deB = np.nan
        dB = np.nan
        CL_base = np.nan
        CL_BIRE = np.nan
        CD_base = np.nan
        Cn_base = np.nan
        CD_BIRE = np.nan
        Cn_BIRE = np.nan
    else:
        solution_base = trim.trim(V, H, gamma, phi_base, Gamma, shss=False, bire=False, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
        trim_base = solution_base.x
        CL_base = solution_base.FM[2]
        CD_base = solution_base.FM[0]
        Cn_base = solution_base.FM[5]
        na_base = solution_base.load
        solution_BIRE = trim.trim(V, H, gamma, phi_base, Gamma, shss=False, bire=True, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
        trim_BIRE = solution_BIRE.x
        CL_BIRE = solution_BIRE.FM[2]
        CD_BIRE = solution_BIRE.FM[0]
        Cn_BIRE = solution_BIRE.FM[5]
        na_BIRE = solution_BIRE.load
        dr = np.rad2deg(trim_base[5])
        de = np.rad2deg(trim_base[4])
        deB = np.rad2deg(trim_BIRE[4])
        dB = np.rad2deg(trim_BIRE[5])
        print(n - na_base, n - na_BIRE)
    return V, n, phi_base, dr, de, dB, deB, CL_base, CL_BIRE, CD_base, CD_BIRE, Cn_base, Cn_BIRE

if generate_data:
    if parallel:
        cases = list(itertools.product(V, n))
        fn = './SCT Data/V-n Data/load_factor_analysis.csv'
        f = open(fn, 'w')
        f.write(zm.io.csvLineWrite('V',
                                   'n',
                                   'phi_base',
                                   'phi_BIRE',
                                   'dr',
                                   'de',
                                   'dB',
                                   'deB',
                                   'CL_base',
                                   'CL_BIRE'))
        f.close()
        bat = 50
        chu = 3
        zm.nm.runCases(sct_trim, cases, fn, nBatch=bat, chunkSize=chu,
                       progKW={'title': 'Running Cases: {}/batch, {}/chunck'.format(bat, chu)})
        data = np.genfromtxt(fn, delimiter=',', skip_header=1)
        df = pd.DataFrame(data, columns = ['V','n','phi', 'dr', 'de', 'dB', 'deB', 'CL_base', 'CL_BIRE'])
        df = df.sort_values(by = ['V', 'n'], ascending = [True, True], na_position = 'first')
        data = df.to_numpy()
        phi_base = np.reshape(data[:, 2], (N, N))
        dr = np.reshape(data[:, 3], (N, N))
        de = np.reshape(data[:, 4], (N, N))
        dB = np.reshape(data[:, 5], (N, N))
        deB = np.reshape(data[:, 6], (N, N))
        CL_base = np.reshape(data[:, 7], (N, N))
        CL_BIRE = np.reshape(data[:, 8], (N, N))
        np.save(f"./SCT Data/V-n Data/base_phi{int(H):2d}CG{int(cg_shift[0]):2d}.npy", phi_base)
    else:
        t0 = time.time()
        dr = np.zeros((N, N))
        de = np.zeros((N, N))
        deB = np.zeros((N, N))
        dB = np.zeros((N, N))
        CL_base = np.zeros((N, N))
        CL_BIRE = np.zeros((N, N))
        CD_base = np.zeros((N, N))
        CD_BIRE = np.zeros((N, N))
        Cn_base = np.zeros((N, N))
        Cn_BIRE = np.zeros((N, N))
        phi_base = np.zeros((N, N))
        phi_0 = 0.1
        for i in range(len(V)):
            print(V[i])
            CW = W/(0.5*rho*V[i]**2*S_w)
            n_stall = CLmax/CW
            for j in range(len(n)):
                print(n[j])
                if n[j] > n_stall:
                    dr[i, j] = np.nan
                    de[i, j] = np.nan
                    deB[i, j] = np.nan
                    dB[i, j] = np.nan
                    CL_base[i, j] = np.nan
                    CL_BIRE[i, j] = np.nan
                    CD_base[i, j] = np.nan
                    CD_BIRE[i, j] = np.nan
                    Cn_base[i, j] = np.nan
                    Cn_BIRE[i, j] = np.nan
                    phi_base[i, j] = np.nan
                    print('stalled')
                else:
                    if j == 0:
                        phi_0 = np.arccos(1./n[j])
                    else:
                        phi_0 = phi_base[i, j-1]
                    phi_base[i, j] = optimize.minimize(find_loadfactor, phi_0, args=(n[j], V[i], False), method='Nelder-Mead', options={'fatol': 1e-12}).x[0]
                    solution_base = trim.trim(V[i], H, gamma, phi_base[i, j], Gamma, shss=False, bire=False, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
                    trim_base = solution_base.x
                    CL_base[i, j] = solution_base.FM[2]
                    CD_base[i, j] = solution_base.FM[0]
                    Cn_base[i, j] = solution_base.FM[5]
                    na_base = solution_base.load
                    try:
                        solution_BIRE = trim.trim(V[i], H, gamma, phi_base[i, j], Gamma, shss=False, bire=True, cg_shift=cg_shift, verbose=False, fixed_point=False, compressible=True)
                        trim_BIRE = solution_BIRE.x
                        CL_BIRE[i, j] = solution_BIRE.FM[2]
                        CD_BIRE[i, j] = solution_BIRE.FM[0]
                        Cn_BIRE[i, j] = solution_BIRE.FM[5]
                        na_BIRE = solution_BIRE.load
                    except np.linalg.LinAlgError:
                        trim_BIRE = [np.nan]*6
                        CL_BIRE[i, j] = np.nan
                        CD_BIRE[i, j] = np.nan
                        Cn_BIRE[i, j] = np.nan
                        na_BIRE = np.nan
                    dr[i, j] = np.rad2deg(trim_base[5])
                    de[i, j] = np.rad2deg(trim_base[4])
                    deB[i, j] = np.rad2deg(trim_BIRE[4])
                    dB[i, j] = np.rad2deg(trim_BIRE[5])
                    print(na_base - n[j], na_BIRE - n[j])
                phi_0 = phi_base[i, j]
        np.save(f"./SCT Data/V-n Data/base_phi{int(H):2d}CG{int(cg_shift[0]):2d}.npy", phi_base)
    np.save(f"./SCT Data/V-n Data/base_dr{int(H):2d}CG{int(cg_shift[0]):2d}.npy", dr)
    np.save(f"./SCT Data/V-n Data/base_de{int(H):2d}CG{int(cg_shift[0]):2d}.npy", de)
    np.save(f"./SCT Data/V-n Data/base_CL{int(H):2d}CG{int(cg_shift[0]):2d}.npy", CL_base)
    np.save(f"./SCT Data/V-n Data/base_CD{int(H):2d}CG{int(cg_shift[0]):2d}.npy", CD_base)
    np.save(f"./SCT Data/V-n Data/base_Cn{int(H):2d}CG{int(cg_shift[0]):2d}.npy", Cn_base)
    np.save(f"./SCT Data/V-n Data/BIRE_deB{int(H):2d}CG{int(cg_shift[0]):2d}.npy", deB)
    np.save(f"./SCT Data/V-n Data/BIRE_dB{int(H):2d}CG{int(cg_shift[0]):2d}.npy", dB)
    np.save(f"./SCT Data/V-n Data/BIRE_CL{int(H):2d}CG{int(cg_shift[0]):2d}.npy", CL_BIRE)
    np.save(f"./SCT Data/V-n Data/BIRE_CD{int(H):2d}CG{int(cg_shift[0]):2d}.npy", CD_BIRE)
    np.save(f"./SCT Data/V-n Data/BIRE_Cn{int(H):2d}CG{int(cg_shift[0]):2d}.npy", Cn_BIRE)
    t1 = time.time()
    print(t1 - t0)
else:
    dr = np.load(f"./SCT Data/V-n Data/base_dr{int(H):2d}CG{int(cg_shift[0]):2d}.npy")
    de = np.load(f"./SCT Data/V-n Data/base_de{int(H):2d}CG{int(cg_shift[0]):2d}.npy")
    CL_base = np.load(f"./SCT Data/V-n Data/base_CL{int(H):2d}CG{int(cg_shift[0]):2d}.npy")
    deB = np.load(f"./SCT Data/V-n Data/BIRE_deB{int(H):2d}CG{int(cg_shift[0]):2d}.npy")
    dB = np.load(f"./SCT Data/V-n Data/BIRE_dB{int(H):2d}CG{int(cg_shift[0]):2d}.npy")
    CL_BIRE = np.load(f"./SCT Data/V-n Data/BIRE_CL{int(H):2d}CG{int(cg_shift[0]):2d}.npy")

X, Y = np.meshgrid(n, M)

mask_base = CL_base > CLmax
mask_BIRE = CL_BIRE > CLmax
dr[mask_base] = np.nan
de[mask_base] = np.nan
deB[mask_BIRE] = np.nan
dB[mask_BIRE] = np.nan
deB[dB > 90.] *= -1.
dB[dB > 90.] -= 180.
dB[dB < -90.] += 180.
dB[abs(dB) > 5.] = np.nan
deB[abs(deB) > 50.] = np.nan

stall_boundary = np.zeros(N)
for i in range(N):
    try:
        stall_boundary[i] = M[np.isnan(CL_base[:, i])][-1]
    except IndexError:
        stall_boundary[i] = M[0]
a = np.polyfit(n, stall_boundary, 3)
model = lambda x : a[0]*x**3 + a[1]*x**2 + a[2]*x + a[3]

fig, ax = plt.subplots()
cmap_dr = colors.ListedColormap('black')
if cg_shift[0] == 0.:
    if H == 1000.:
        lvls_dr = [-0.2, -0.1, 0., 0.1]
        manual_locations_dr = [(1.8, 0.332), (2.7, 0.52), (4.5, 0.44), (6.7, 0.45)]
        lvls_de = [-0.5, 0., 0.5, 1., 2.]
        manual_locations_de = [(2.2, 0.71), (6.3, 0.74), (3.6, 0.54), (4.85, 0.53), (3.82, 0.40)]
        lvls_dB = np.array([-1., -0.5, 0., 0.5, 1.])
        manual_locations_dB = [(3.5, 0.64), (5.1, 0.71), (4, 0.6), (5.83, 0.49), (3.8, 0.36)]
        lvls_deB = [-4, -3, -2]
        manual_locations_deB= [(2.7, 0.66), (2.7, 0.45), (4.55, 0.46)]
    elif H == 15000.:
        lvls_dr = [-0.05, 0., 0.05]
        manual_locations_dr = [(1.6, 0.74), (6.6, 0.63), (6.8, 0.58)]
        lvls_de = [0, 1, 2, 3]
        manual_locations_de = [(2.95, 0.75), (1.9, 0.49), (5.3, 0.62), (8, 0.65)]
        lvls_dB = [-0.5, 0.5]
        manual_locations_dB = [(2.55, 0.69), (3.64, 0.68), (1.25, 0.39)]
        lvls_deB = [-3, -2, -1, 0]
        manual_locations_deB= [(2.5, 0.74), (2, 0.46)]
    else:
        lvls_dr = [-0.05, 0., 0.05]
        manual_locations_dr = [(2.3, 0.56), (4.3, 0.69), (4.68, 0.65)]
        lvls_de = [0, 1, 2, 3]
        manual_locations_de = [(1.2, 0.76), (2.3, 0.72), (3.77, 0.73), (3, 0.6)]
        lvls_dB = [-0.5, 0.5]
        manual_locations_dB = [(2.5, 0.75), (3.6, 0.57)]
        lvls_deB = [-4, -3, -2]
        manual_locations_deB= [(1.54, 0.74), (3.3, 0.73), (5.6, 0.76)]
else:
    if H == 1000.:
        lvls_dr = [-0.2, -0.1, 0., 0.1]
        manual_locations_dr = [(1.8, 0.332), (2.7, 0.52), (4.5, 0.44), (7.1, 0.47)]
        lvls_de = [-8, -6, -4, -2]
        manual_locations_de = [(2.28, 0.69), (4.8, 0.63), (6.9, 0.58), (6.5, 0.48)]
        lvls_dB = [-2., -0.5, -0.1]
        manual_locations_dB = [(4.5, 0.7), (5, 0.58), (7.3, 0.58)]
        lvls_deB = [-8, -6, -4, -2]
        manual_locations_deB= [(5, 0.53), (6.1, 0.67), (3.9, 0.7), (1.3, 0.75)]
    elif H == 15000.:
        lvls_dr = [-0.1, 0., 0.1]
        manual_locations_dr = [(1.5, 0.51), (4, 0.51), (6.3, 0.55)]
        lvls_de = [-8, -6, -4, -2]
        manual_locations_de = [(2.28, 0.69), (5.2, 0.52), (3.2, 0.5), (3.5, 0.66)]
        lvls_dB = [-4, -2., -0.5, -0.1]
        manual_locations_dB = [(3.5, 0.76), (4.6, 0.68), (4.28, 0.58), (5.1, 0.59)]
        lvls_deB = [-8, -6, -4]
        manual_locations_deB= [(2.7, 0.73), (2.9, 0.61), (3.9, 0.59)]
    else:
        lvls_dr = [-0.05, 0., 0.05]
        manual_locations_dr = [(2.6, 0.67), (2.5, 0.54), (4.7, 0.66)]
        lvls_de = [-8, -6, -4, -2]
        manual_locations_de = [(1.16, 0.77), (1.95, 0.64), (2.69, 0.58), (3.31, 0.54)]
        lvls_dB = [-2., -0.5, -0.1]
        manual_locations_dB = [(1.6, 0.72), (3.2, 0.74), (3.65, 0.7)]
        lvls_deB = [-10, -7, -5]
        manual_locations_deB= [(1.8, 0.7), (1.75, 0.58), (1.3, 0.43)]
options_dr = {"cmap": cmap_dr, "levels": lvls_dr, 'linestyles': '-', 'zorder': 2}
options_clabel_dr = {"levels": lvls_dr, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, 'zorder': 2}
CS_dr = ax.contour(X, Y, dr, **options_dr)
CS_dr.collections[0].set_label('$\delta_r$')
plt.clabel(CS_dr, **options_clabel_dr, manual=manual_locations_dr)

cmap_de = colors.ListedColormap('black')
options_de = {"cmap": cmap_de, "levels": lvls_de, 'linestyles': '--', 'zorder': 2}
options_clabel_de = {"levels": lvls_de, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, 'zorder': 2}
CS_de = ax.contour(X, Y, de, **options_de)
CS_de.collections[0].set_label('$\delta_e$')
plt.clabel(CS_de, **options_clabel_de, manual=manual_locations_de)

cmap_dB = colors.ListedColormap('black')
options_dB = {"cmap": cmap_dB, "levels": lvls_dB, 'linestyles': ':', 'zorder': 2}
options_clabel_dB = {"levels": lvls_dB, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, 'zorder': 2}
CS_dB = ax.contour(X, Y, dB, **options_dB)
CS_dB.collections[0].set_label('$\delta_B$')
plt.clabel(CS_dB, **options_clabel_dB, manual=manual_locations_dB)

cmap_deB = colors.ListedColormap('black')
options_deB = {"cmap": cmap_deB, "levels": lvls_deB, 'linestyles': '-.', 'zorder': 2}
options_clabel_deB = {"levels": lvls_deB, "inline_spacing": 3, "fmt": "%4.3g", "fontsize": 16, 'zorder': 2}
CS_deB = ax.contour(X, Y, deB, **options_deB)
CS_deB.collections[0].set_label('$\delta_e^B$')
plt.clabel(CS_deB, **options_clabel_deB, manual=manual_locations_deB)

ax.plot(np.linspace(0, 9, 100), model(np.linspace(0, 9, 100)), color='0.5', linewidth=4)
ax.fill_between(np.linspace(0, 9, 100), np.full(100, M[0]), model(np.linspace(0, 9, 100)), facecolor='1.', hatch='xx', alpha=1.0, ec='0.5')
ax.set_ylabel(r'\textbf{Mach Number, }\boldmath$M$', fontsize=16)
ax.set_xlabel(r'\textbf{Load Factor, }\boldmath$n$', fontsize=16)
ax.annotate("Stall Region", (5., (0.2 + model(5.))/2.), weight='bold', size=20).set_bbox(dict(facecolor='1.', alpha=1.0, edgecolor='0.5'))
h1,_ = CS_dr.legend_elements()
h2,_ = CS_de.legend_elements()
h3,_ = CS_dB.legend_elements()
h4,_ = CS_deB.legend_elements()
ax.legend([h1[0], h2[0], h3[0], h4[0]], [r'$\delta_r$', r'$\delta_e$', r'$\delta_B$', r'$\delta_e^B$'], loc='lower right', fontsize=16)
xlims = (1., 9.)
ylims = (0.2, 0.8)
dy = {'major': 0.1, 'minor': 0.1/4}
dx = {'major': 1, 'minor': 1/4}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.grid()
plt.tight_layout()
plt.savefig(f"./SCT Figures/Altitude{int(H):2d}CG{int(cg_shift[0]):2d}.pdf")

