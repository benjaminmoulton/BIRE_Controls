#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:26:18 2022

@author: christian
"""

import f16_linearization as f16
from control import lqr, ctrb
import aero_trim as trim
import numpy as np
from hunsaker_atm import stdatm_english
from state_control_simulator import simulate
from control_plots import simulation_results
import matplotlib.pyplot as plt
from frequency_analysis import mimo_io_transfer
import matplotlib as mpl
from matplotlib.ticker import (LogLocator, MultipleLocator)

mpl.rcParams['axes.linewidth'] = 1.75 #set the value globally
mpl.rcParams["font.family"] = "serif"
plt.rc('font', weight='bold')

major_dict = {"width" : 1.25, "size" : 7., "labelsize" : 16.,
             "direction" : 'in', "which" : 'major'}
minor_dict = {"width" : 1.25, "size" : 4.,
             "direction" : 'in', "which" : 'minor'}


def pretty_plot(ax, xlims, ylims, dx, dy, **kwargs):
    log = kwargs.get('log', False)
    set_ticks = kwargs.get('set_ticks', True)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.yaxis.set_major_locator(MultipleLocator(dy["major"]))
    ax.yaxis.set_minor_locator(MultipleLocator(dy["minor"]))
    if log:
        ax.xaxis.set_major_locator(LogLocator())
        ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(0.1, 1., 0.1)))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(dx["major"]))
        ax.xaxis.set_minor_locator(MultipleLocator(dx["minor"]))
    if set_ticks:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    ax.tick_params(**major_dict)
    ax.tick_params(**minor_dict)
    return ax

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

plt.close('all')
H = 15000.
a = stdatm_english(H)[-1]
M = 0.6
V = M*a
gamma = np.deg2rad(0.)
phi = np.deg2rad(0.)
Gamma = 0.5
cg_shift = [0., 0., 0.]
aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
trim_solution = trim.trim(V, H, gamma, phi, Gamma,
                          shss=False, bire=False,
                          cg_shift=cg_shift,
                          fixed_point=False,
                          compressible=False,
                          aero_dir=aero_dir)
x_hat = trim_solution.states
alpha_hat = trim_solution.x[1]
beta_hat = trim_solution.x[2]
u_hat = trim_solution.inputs[1:]
tau = u_hat[0]
FM_hat = trim_solution.FM
props = trim.AircraftProperties(V, H, Gamma, aero_dir)
linearization = f16.LinearizationBaseline(props, aero_dir)
linearization.set_linearization_point(x_hat, u_hat, alpha_hat, beta_hat, FM_hat, cg_shift, tau)
A = linearization.create_A_matrix()
B = linearization.create_B_matrix()
C = linearization.create_C_matrix()
Q = np.zeros_like(A)
R = np.eye(len(B[0]))*0.1
# Q[0, 0] = 1.
# Q[1, 1] = 1.
# Q[2, 2] = 0.1
Q[3, 3] = 10.
Q[4, 4] = 10.
Q[5, 5] = 10.
Q[7, 7] = 20.
R[0, 0] = 2.
R[1, 1] = 1.
R[2, 2] = 2.
np.set_printoptions(formatter={'float_kind': "{:.4f}".format, 'complexfloat': "{:.4f}".format})
print(bmatrix(Q))
print(bmatrix(R))
K, S, eigs_controlled = lqr(A, B, Q, R)
eigs_uncontrolled = np.linalg.eigvals(A)
lin_results = f16.Lin_Results(len(A), len(B[0]))
lin_results.A = A
lin_results.B = B
lin_results.C = C
lin_results.eigs = eigs_controlled
lin_results.K = K
t_range = np.arange(0., 20., 0.1)
model_gust = {"type": "gust", "params": {"A": 80.,
                                         "gamma": 1.,
                                         "w": 5.,
                                         "s_x": 1.,
                                         "s_y": 1.,
                                         "s_z": 0.5,
                                         "t_0": 1.}}
simulate(trim_solution, t_range, lin_results, props, cg_shift, True, model=model_gust)
simulate(trim_solution, t_range, lin_results, props, cg_shift, False, model=model_gust)

figs, axs = simulation_results(False, cg_shift,  dt=0.1)
[fig_V, fig_Vz, fig_R, fig_Rz, fig_O, fig_Oz, fig_X, fig_u, fig_udot, fig_w] = figs
[ax_V1, ax_V2, ax_uz, ax_vz, ax_wz, ax_R1, ax_R2, ax_R3, ax_pz, ax_qz, ax_rz, ax_phi, ax_theta, ax_psi, ax_phiz, ax_thetaz, ax_X, ax_Y, ax_Z, ax_da, ax_de, ax_dr, ax_dadot, ax_dedot, ax_drdot, ax_w] = axs

xlims = (-1, 21)
dx = {'major': 2., 'minor': 2/4.}

ylims = (535, 725)
dy = {'major': 50, 'minor': 50/4}
ax_V1 = pretty_plot(ax_V1, xlims, ylims, dx, dy, set_ticks=False)
ax_V1.spines['bottom'].set_visible(False)
ax_V2.spines['top'].set_visible(False)
ax_V1.xaxis.tick_top()
ax_V1.yaxis.set_ticks_position('both')
ax_V1.tick_params(labeltop=False)  # don't put tick labels at the top
ax_V2.xaxis.tick_bottom()
ax_V2.yaxis.set_ticks_position('both')
ax_V1.grid()
ylims = (-125, 125)
dy = {'major': 50, 'minor': 50/4}
ax_V2 = pretty_plot(ax_V2, xlims, ylims, dx, dy, set_ticks=False)
ax_V2.grid()
fig_V.supylabel(r'\textbf{Velocity Components [ft/s]}', fontsize=16)
ax_V2.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_V1.annotate(r'\boldmath$u$', (14, 640), fontsize=16)
ax_V2.annotate(r'\boldmath$v$', (14, 5), fontsize=16)
ax_V2.annotate(r'\boldmath$w$', (14, 40), fontsize=16)
dummy_lines = [ax_V2.plot([], [], c='k', linestyle='-')[0],
               ax_V2.plot([], [], c='0.5', linestyle='-')[0]]
ax_V2.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='lower right', fontsize=16)
fig_V.tight_layout()
fig_V.savefig('./Control Development/base_uvw.pdf', dpi=1000)

ylims = (-25., 25.)
dy = {'major': 10., 'minor': 10./4.}
ax_R1 = pretty_plot(ax_R1, xlims, ylims, dx, dy)
ax_R1.annotate(r'\boldmath$p$', (6, 5), fontsize=16)
fig_R.supylabel(r'\textbf{Rotational Velocity Components [deg/s]}', fontsize=16)
ax_R3.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_R1.grid()
ax_R2 = pretty_plot(ax_R2, xlims, ylims, dx, dy)
ax_R2.annotate(r'\boldmath$q$', (16, 5), fontsize=16)
ax_R2.grid()
ax_R3 = pretty_plot(ax_R3, xlims, ylims, dx, dy)
ax_R3.annotate(r'\boldmath$r$', (8, 5), fontsize=16)
ax_R3.grid()
dummy_lines = [ax_R1.plot([], [], c='k', linestyle='-')[0],
               ax_R1.plot([], [], c='0.5', linestyle='-')[0]]
ax_R1.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_R.tight_layout()
fig_R.savefig('./Control Development/base_pqr.pdf', dpi=1000)

ylims = (-7500., 17500.)
dy = {'major': 5000., 'minor': 5000./4.}
ax_X = pretty_plot(ax_X, xlims, ylims, dx, dy)
ax_X.annotate(r'\boldmath$x$', (8, 8000), fontsize=16)
fig_X.supylabel(r'\textbf{Aircraft Position [ft]}', fontsize=16)
ax_Z.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_X.grid()
ylims = (-150, 150)
dy = {'major': 100., 'minor': 100./4.}
ax_Y = pretty_plot(ax_Y, xlims, ylims, dx, dy)
ax_Y.annotate(r'\boldmath$y$', (14, 0.), fontsize=16)
ax_Y.grid()
ylims = (-15150, -14850)
dy = {'major': 100., 'minor': 100/4.}
ax_Z = pretty_plot(ax_Z, xlims, ylims, dx, dy)
ax_Z.annotate(r'\boldmath$z$', (14, -15050), fontsize=16)
ax_Z.grid()
dummy_lines = [ax_X.plot([], [], c='k', linestyle='-')[0],
               ax_X.plot([], [], c='0.5', linestyle='-')[0]]
ax_X.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_X.tight_layout()
fig_X.savefig('./Control Development/base_xyz.pdf', dpi=1000)

ylims = (-12.5, 12.5)
dy = {'major': 5., 'minor': 5./4.}
ax_phi = pretty_plot(ax_phi, xlims, ylims, dx, dy)
ax_phi.annotate(r'\boldmath$\phi$', (8, 2), fontsize=16)
fig_O.supylabel(r'\textbf{Aircraft Orientation [deg]}', fontsize=16)
ax_psi.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_phi.grid()
ax_theta = pretty_plot(ax_theta, xlims, ylims, dx, dy)
ax_theta.annotate(r'\boldmath$\theta$', (8, 5), fontsize=16)
ax_theta.grid()
ax_psi = pretty_plot(ax_psi, xlims, ylims, dx, dy)
ax_psi.annotate(r'\boldmath$\psi$', (10, 2), fontsize=16)
ax_psi.grid()
dummy_lines = [ax_phi.plot([], [], c='k', linestyle='-')[0],
               ax_phi.plot([], [], c='0.5', linestyle='-')[0]]
ax_phi.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_O.tight_layout()
fig_O.savefig('./Control Development/base_orientation.pdf', dpi=1000)

ylims = (-75, 75)
dy = {'major': 50., 'minor': 50./4.}
ax_uz = pretty_plot(ax_uz, xlims, ylims, dx, dy)
ax_uz.annotate(r'\boldmath$\Delta u$', (8, 10), fontsize=16)
fig_Vz.supylabel(r'\textbf{Velocity Deviation [ft]}', fontsize=16)
ax_wz.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_uz.grid()
ylims = (-75, 75)
dy = {'major': 50, 'minor': 50/4.}
ax_vz = pretty_plot(ax_vz, xlims, ylims, dx, dy)
ax_vz.annotate(r'\boldmath$\Delta v$', (14, 10), fontsize=16)
ax_vz.grid()
ylims = (-30, 30)
dy = {'major': 20, 'minor': 20/4.}
ax_wz = pretty_plot(ax_wz, xlims, ylims, dx, dy)
ax_wz.annotate(r'\boldmath$\Delta w$', (14, 5), fontsize=16)
ax_wz.grid()
dummy_lines = [ax_uz.plot([], [], c='k', linestyle='-')[0],
               ax_uz.plot([], [], c='0.5', linestyle='-')[0]]
ax_uz.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_Vz.tight_layout()
fig_Vz.savefig('./Control Development/base_d_uvw.pdf', dpi=1000)

ylims = (-37.5, 37.5)
dy = {'major': 15., 'minor': 15./4.}
ax_pz = pretty_plot(ax_pz, xlims, ylims, dx, dy)
ax_pz.annotate(r'\boldmath$\Delta p$', (6, 7.5), fontsize=16)
fig_Rz.supylabel(r'\textbf{Rotational Velocity Deviation [deg/s]}', fontsize=16)
ax_rz.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_pz.grid()
ax_qz = pretty_plot(ax_qz, xlims, ylims, dx, dy)
ax_qz.annotate(r'\boldmath$\Delta q$', (7, 7.5), fontsize=16)
ax_qz.grid()
ax_rz = pretty_plot(ax_rz, xlims, ylims, dx, dy)
ax_rz.annotate(r'\boldmath$\Delta r$', (8, 7.5), fontsize=16)
ax_rz.grid()
dummy_lines = [ax_pz.plot([], [], c='k', linestyle='-')[0],
               ax_pz.plot([], [], c='0.5', linestyle='-')[0]]
ax_pz.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_Rz.tight_layout()
fig_Rz.savefig('./Control Development/base_d_pqr.pdf', dpi=1000)

ylims = (-12.5, 12.5)
dy = {'major': 5., 'minor': 5./4.}
ax_phiz = pretty_plot(ax_phiz, xlims, ylims, dx, dy)
ax_phiz.annotate(r'\boldmath$\Delta \phi$', (8, 2.5), fontsize=16)
fig_Oz.supylabel(r'\textbf{Aircraft Orientation Deviation [deg]}', fontsize=16)
ax_rz.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_phiz.grid()
ax_thetaz = pretty_plot(ax_thetaz, xlims, ylims, dx, dy)
ax_thetaz.annotate(r'\boldmath$\Delta \theta$', (12, 2.5), fontsize=16)
ax_thetaz.grid()
dummy_lines = [ax_phiz.plot([], [], c='k', linestyle='-')[0],
               ax_phiz.plot([], [], c='0.5', linestyle='-')[0]]
ax_phiz.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_Oz.tight_layout()
fig_Oz.savefig('./Control Development/base_d_orientation.pdf', dpi=1000)

xlims2 = (-1, 21)
dx2 = {'major': 5., 'minor': 5/4.}
ylims = (-5, 5)
dy = {'major': 2, 'minor': 2/4.}
ax_da = pretty_plot(ax_da, xlims2, ylims, dx2, dy)
fig_u.supylabel(r'\textbf{Control Settings}\textbf{ [deg]}', fontsize=16)
ax_de.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_da.annotate(r'\boldmath$\delta_a$', (7, 1), fontsize=16)
ax_dr.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_da.grid()
ylims = (-2.5, 2.5)
dy = {'major': 1, 'minor': 1/4.}
ax_de = pretty_plot(ax_de, xlims2, ylims, dx2, dy)
ax_de.annotate(r'\boldmath$\delta_e$', (7, 0.5), fontsize=16)
ax_de.grid()
ylims = (-12.5, 12.5)
dy = {'major': 5, 'minor': 5/4.}
ax_dr = pretty_plot(ax_dr, xlims2, ylims, dx2, dy)
ax_dr.annotate(r'\boldmath$\delta_r$', (7, 3), fontsize=16)
ax_dr.grid()
dummy_lines = [ax_da.plot([], [], c='k', linestyle='-')[0],
               ax_da.plot([], [], c='0.5', linestyle='-')[0]]
ax_da.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_u.tight_layout()
fig_u.savefig('./Control Development/base_control.pdf', dpi=1000)

xlims2 = (-1, 21)
dx2 = {'major': 5., 'minor': 5/4.}
fig_udot.supylabel(r'\textbf{Control Rates [deg]}', fontsize=16)
ax_drdot.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ylims = (-2.5, 2.5)
dy = {'major': 1, 'minor': 1/4.}
ax_dadot = pretty_plot(ax_dadot, xlims2, ylims, dx2, dy)
ax_dadot.annotate(r'\boldmath$\dot{\delta_a}$', (7, 0.5), fontsize=16)
ax_dadot.grid()
ax_dedot = pretty_plot(ax_dedot, xlims2, ylims, dx2, dy)
ax_dedot.annotate(r'\boldmath$\dot{\delta_e}$', (7, 0.5), fontsize=16)
ax_dedot.grid()
ylims = (-7.5, 7.5)
dy = {'major': 3, 'minor': 3/4.}
ax_drdot = pretty_plot(ax_drdot, xlims2, ylims, dx2, dy)
ax_drdot.annotate(r'\boldmath$\dot{\delta_r}$', (7, 1.5), fontsize=16)
ax_drdot.grid()
dummy_lines = [ax_dadot.plot([], [], c='k', linestyle='-')[0],
                ax_dadot.plot([], [], c='0.5', linestyle='-')[0]]
ax_dadot.legend(dummy_lines, [r'Controlled', r'Uncontrolled'], loc='upper right', fontsize=16)
fig_udot.tight_layout()
fig_udot.savefig('./Control Development/base_control_rates.pdf', dpi=1000)

ylims = (-70, 70)
dy = {'major': 20., 'minor': 20/4}
ax_w = pretty_plot(ax_w, xlims, ylims, dx, dy)
ax_w.set_ylabel(r'\textbf{Gust Velocity Components, }\boldmath$V_g$\textbf{ [ft/s]}', fontsize=16)
ax_w.set_xlabel(r'\textbf{Time, }\boldmath$t$\textbf{ [s]}', fontsize=16)
ax_w.legend(fontsize=16)
ax_w.grid()
fig_w.tight_layout()
fig_w.savefig('./Control Development/base_wind.pdf', dpi=1000)

# s = np.logspace(-3, 3, 100)*2.*np.pi
# s, G, svd_G, L, S, T, svd_S = mimo_io_transfer(lin_results, s, closed_loop=True)
# fig, ax = plt.subplots()
# svd_S_max = [np.max(idx) for idx in zip(*svd_S)]
# svd_S_min = [np.min(idx) for idx in zip(*svd_S)]
# ax.semilogx(s, 20.*np.log10(np.absolute(svd_S_max)), color='k', linestyle='-', label='Max')
# ax.semilogx(s, 20.*np.log10(np.absolute(svd_S_min)), color='k', linestyle='--', label='Min')
# # ax.axhline(-10., color='0.5', linestyle=':')
# # ax.axhline(10., color='0.5', linestyle=':')
# # max_mask = 20.*np.log10(np.absolute(svd_S_max)) < -10.
# # min_mask = 20.*np.log10(np.absolute(svd_S_min)) < 10.
# # ax.axvline(s[max_mask][0], 0, (90 - 10.)/180, color='0.5', linestyle=':')
# # ax.axvline(s[min_mask][0], 0, (90 + 10.)/180., color='0.5', linestyle=':')
# xlims = (s[0], s[-1])
# dx = {}
# ylims = (-105, 105)
# dy = {'major': 30, 'minor': 30/4.}
# ax = pretty_plot(ax, xlims, ylims, dx, dy, log=True)
# ax.grid()
# ax.legend(fontsize=16)
# ax.set_xlabel(r'\textbf{Frequency, }\boldmath$\omega$\textbf{ [rad/s]}', fontsize=16)
# ax.set_ylabel(r'\textbf{SVD Magnitude, }\boldmath$\vert \sigma (S) \vert$\textbf{ [dB]}', fontsize=16)
# fig.tight_layout()
# fig.savefig('./Control Development/base_svd.pdf', dpi=1000)

# s = np.logspace(-1, 3, 100)*2.*np.pi
# s, G, svd_G, L, S, T, svd_S = mimo_io_transfer(lin_results, s, closed_loop=True)
# fig, ax = plt.subplots()
# svd_S_max = [np.max(idx) for idx in zip(*svd_G)]
# svd_S_min = [np.min(idx) for idx in zip(*svd_G)]
# ax.semilogx(s, 20.*np.log10(np.absolute(svd_S_max)), color='k', linestyle='-')
# ax.semilogx(s, 20.*np.log10(np.absolute(svd_S_min)), color='k', linestyle='--')
