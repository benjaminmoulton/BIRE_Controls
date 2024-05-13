#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:46:01 2022

@author: christian
"""
import numpy as np
import matplotlib.pyplot as plt
import aero_trim
from matplotlib import colors
from hunsaker_atm import gravity_english, stdatm_english
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
H = 1000.
M = np.load('./Crosswind Data/SHSS_Mach.npy')
a = stdatm_english(H)[-1]
V = M*a
phi = np.load('./Crosswind Data/SHSS_Bank_Angle.npy')
CLmax = 1.9
gamma = 0.
cg_shift = [1., 0., 0.]
BIRE_rotation_deg = np.load(f"./Crosswind Data/SHSS_BIRE_rotation{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
BIRE_elevator_deg = np.load(f"./Crosswind Data/SHSS_BIRE_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
phi_rad = np.load(f"./Crosswind Data/Tail_Strike_BIRE_phi{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")*np.pi/180.
theta = np.load(f"./Crosswind Data/Tail_Strike_BIRE_theta{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")*np.pi/180.
base_elevator_deg = np.load(f"./Crosswind Data/SHSS_base_elevator{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
base_phi_rad = np.load(f"./Crosswind Data/SHSS_base_phi{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")*np.pi/180.
base_theta = np.load(f"./Crosswind Data/SHSS_base_theta{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")*np.pi/180.
rudder_deg = np.load(f"./Crosswind Data/SHSS_base_rudder{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
CL_base = np.load(f"./Crosswind Data/SHSS_base_CL{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
V_cross = np.load(f"./Crosswind Data/SHSS_base_Vcross{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
BIRE_V_cross = np.load(f"./Crosswind Data/Tail_Strike_BIRE_Vcross{int(H):2d}CG{int(cg_shift[0] - 1):2d}.npy")
z_LG_E_base = np.zeros_like(rudder_deg)
z_LG_TEL_base = np.zeros_like(rudder_deg)
z_LG_TER_base = np.zeros_like(rudder_deg)
z_LG_E_bire = np.zeros_like(rudder_deg)
z_LG_TEL_bire = np.zeros_like(rudder_deg)
z_LG_TER_bire = np.zeros_like(rudder_deg)

h_intake = 2.906  # From Nguyen Drawing Scaled from Centerline
h_landing = h_intake*2.  # From centerline to ground is ~ two intakes
b_h = 9.2
c_rh = 7.9833 # stab root chord
s_fh = 3.4  # semispan of fuselage portion of h-stab
l_h = 13.13  # from CG to QC of h-stab
G_h = -10.*np.pi/180.  # Anhedral of baseline tail
z_LG = 5.812
y_LG = 1.557
x_LG = -0.3063
x_E = -18.71
y_E = 0.
z_E = 1.679
x_TE = -l_h
y_TE = b_h  # based on BIRE
z_TE = 0.  # based on BIRE
x_SP = -l_h - 0.21*c_rh
y_SP = s_fh
z_SP = 0.
p_LG = np.array([-x_LG, y_LG, -z_LG])  # LG to CG
p_E = np.array([x_E, y_E, z_E])  # CG to engine
# Base
base_P_L = np.array([x_SP, -y_SP, z_SP])  # CG to left stab pivot
base_P_R = np.array([x_SP, y_SP, z_SP])  # CG to right stab pivot
base_TE_L = np.array([-0.54*c_rh, -(y_TE - y_SP)*np.cos(G_h), z_TE - (b_h - s_fh)*np.sin(G_h)])  # left stab pivot to left TE
base_TE_R = np.array([-0.54*c_rh, (y_TE - y_SP)*np.cos(G_h), z_TE - (b_h - s_fh)*np.sin(G_h)])  # right stab pivot to right TE
# BIRE
bire_EMP = np.array([x_SP, 0., z_SP]) # CG to center of empennage rotation
bire_E_P_R = np.array([0., y_SP, 0.])  # center of empennage to right stab pivot
bire_E_P_L = np.array([0., -y_SP, 0.])  # center of empennage to left stab pivot
bire_P_TE_R = np.array([-0.54*c_rh, (y_TE - y_SP), z_TE])  # right pivot to tip corner
bire_P_TE_L = np.array([-0.54*c_rh, -(y_TE - y_SP), z_TE])  # left pivot to tip corner

for i in range(len(V)):
    for j in range(len(phi)):
        dB = BIRE_rotation_deg[i, j]*np.pi/180.
        de_BIRE = BIRE_elevator_deg[i, j]*np.pi/180.
        de_base = base_elevator_deg[i, j]*np.pi/180.
        R_theta = np.array([[np.cos(theta[i, j]), 0., np.sin(theta[i, j])],
                            [0., 1., 0.],
                            [-np.sin(theta[i, j]), 0., np.cos(theta[i, j])]])
        R_phi = np.array([[1., 0., 0.],
                          [0., np.cos(phi_rad[i, j]), -np.sin(phi_rad[i, j])],
                          [0., np.sin(phi_rad[i, j]), np.cos(phi_rad[i, j])]])
        R_dB = np.array([[1., 0., 0.],
                          [0., np.cos(dB), -np.sin(dB)],
                          [0., np.sin(dB), np.cos(dB)]])
        R_de = np.array([[np.cos(de_BIRE), 0., np.sin(de_BIRE)],
                          [0., 1., 0.],
                          [-np.sin(de_BIRE), 0., np.cos(de_BIRE)]])
        P_LG_E = np.matmul(R_theta, np.matmul(R_phi, p_LG + p_E))
        P_LG_TEL = np.matmul(R_theta, np.matmul(R_phi, p_LG + bire_EMP + np.matmul(R_dB, bire_E_P_L + np.matmul(R_de, bire_P_TE_L))))
        P_LG_TER = np.matmul(R_theta, np.matmul(R_phi, p_LG + bire_EMP + np.matmul(R_dB, bire_E_P_R + np.matmul(R_de, bire_P_TE_R))))
        z_LG_E_bire[i, j] = P_LG_E[2]
        z_LG_TEL_bire[i, j] = P_LG_TEL[2]
        z_LG_TER_bire[i, j] = P_LG_TER[2]


        R_phi = np.array([[1., 0., 0.],
                          [0., np.cos(base_phi_rad[i, j]), -np.sin(base_phi_rad[i, j])],
                          [0., np.sin(base_phi_rad[i, j]), np.cos(base_phi_rad[i, j])]])
        R_theta = np.array([[np.cos(base_theta[i, j]), 0., np.sin(base_theta[i, j])],
                            [0., 1., 0.],
                            [-np.sin(base_theta[i, j]), 0., np.cos(base_theta[i, j])]])
        R_de = np.array([[np.cos(de_base), 0., np.sin(de_base)],
                         [0., 1., 0.],
                         [-np.sin(de_base), 0., np.cos(de_base)]])
        P_LG_E = np.matmul(R_theta, np.matmul(R_phi, p_LG + p_E))
        P_LG_TEL = np.matmul(R_theta, np.matmul(R_phi, p_LG + base_P_L + np.matmul(R_de, base_TE_L)))
        P_LG_TER = np.matmul(R_theta, np.matmul(R_phi, p_LG + base_P_R + np.matmul(R_de, base_TE_R)))
        z_LG_E_base[i, j] = P_LG_E[2]
        z_LG_TEL_base[i, j] = P_LG_TEL[2]
        z_LG_TER_base[i, j] = P_LG_TER[2]

mask = base_phi_rad[0, :]*180/np.pi < 10.

fig, ax = plt.subplots()
ax.plot(base_phi_rad[0, mask]*180/np.pi, -z_LG_E_base[0, mask], color='k', linestyle='-')
ax.plot(base_phi_rad[0, mask]*180/np.pi, -z_LG_TEL_base[0, mask], color='k', linestyle='--')
ax.plot(base_phi_rad[0, mask]*180/np.pi, -z_LG_TER_base[0, mask], color='k', linestyle=':')
ax.plot(phi_rad[0, mask]*180/np.pi, -z_LG_E_bire[0, mask], color='0.5', linestyle='-')
ax.plot(phi_rad[0, mask]*180/np.pi, -z_LG_TEL_bire[0, mask], color='0.5', linestyle='--')
ax.plot(phi_rad[0, mask]*180/np.pi, -z_LG_TER_bire[0, mask], color='0.5', linestyle=':')
ax.axhline(0., color='k', linewidth='5.')
ax.annotate(r'\textbf{Ground}', (8, 0.2), fontsize=16)
dummy_lines = [ax.plot([], [], c='k')[0], ax.plot([], [], c='0.5')[0]]
legend1 = plt.legend(dummy_lines, ['Baseline', 'BIRE'], loc='upper right',
                     fontsize=16)
dummy_lines = [ax.plot([], [], c='k', linestyle='-')[0],
               ax.plot([], [], c='k', linestyle='--')[0],
               ax.plot([], [], c='k', linestyle=':')[0]]
legend2 = plt.legend(dummy_lines, [r'$z_\mathrm{E}$',
                                   r'$z_\mathrm{TE,L}$',
                                   r'$z_\mathrm{TE,R}$'],
                     loc='upper left', fontsize=16)
ax.add_artist(legend1)
ax.add_artist(legend2)

xlims = (0., 10.)
dx = {'major': 2., 'minor': 2./4.}
if cg_shift[0] == 0.:
    if H == 1000.:
        ylims = (-4.5, 4.5)
        dy = {'major': 1., 'minor': 1./4.}
    else:
        ylims = (-7, 5)
        dy = {'major': 2., 'minor': 2./4.}
else:
    if H == 1000.:
        ylims = (-5., 7.)
        dy = {'major': 2., 'minor': 2./4.}
    else:
        ylims = (-7, 7)
        dy = {'major': 2., 'minor': 2./4.}
ax = pretty_plot(ax, xlims, ylims, dx, dy)
ax.set_ylabel(r'\textbf{Distance to Ground, }\boldmath$z_\mathrm{g}$\textbf{ [ft]}', fontsize=16)
ax.set_xlabel(r'\textbf{Bank Angle, }\boldmath$\phi$\textbf{ [deg]}', fontsize=16)
ax.grid()

plt.savefig(f"./Tail Strike Figures/Altitude{int(H):2d}CG{int(cg_shift[0]):2d}.pdf")

