#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 15:54:09 2021

@author: christian
"""
import sys
aero_directory = '../aerodynamics_model/'
loads_directory = '../trim/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, loads_directory)

import numpy as np
from f16_aero import F16Aero
from bire_aero import BIREAero
from hunsaker_atm import gravity_english, stdatm_english
import json
import matplotlib.pyplot as plt
import scipy.optimize as optimize


class AircraftProperties:
    def __init__(self, V, H, Gamma, path='./', bire=False, **kwargs):
        if bire:
            fn = kwargs.get('filename', 'BIRE_props_for_loads.json')
            prop_dict = json.load(open(path + fn))
        else:
            fn = kwargs.get('filename', 'F16_props_for_loads.json')
            prop_dict = json.load(open(path + fn))
        self.S_w = prop_dict["geometry"]["S_w"]
        self.b_w = prop_dict["geometry"]["b_w"]
        self.c_w = prop_dict["geometry"]["c_w"]
        self.l_h = prop_dict["geometry"]["l_h"]
        self.RA_w = prop_dict["geometry"]["RA_w"]
        self.Lam_w = prop_dict["geometry"]["Lam_w"]
        self.RA_v = prop_dict["geometry"]["RA_v"]
        self.Lam_v = prop_dict["geometry"]["Lam_v"]
        self.RA_h = prop_dict["geometry"]["RA_h"]
        self.Lam_h = prop_dict["geometry"]["Lam_h"]
        self.W = prop_dict["inertia"]["W"]
        self.hz = prop_dict["inertia"]["h_z"]
        self.hy = prop_dict["inertia"]["h_y"]
        self.hx = prop_dict["inertia"]["h_x"]
        if bire:
            I_model = json.load(open(path + 'bire_inertia_model.json'))
            Ixx = I_model["Ixx"]
            Iyy = I_model["Iyy"]
            Izz = I_model["Izz"]
            Ixz = I_model["Ixz"]
            Ixy = I_model["Ixy"]
            Iyz = I_model["Iyz"]
            self.I_xx = lambda dB : Ixx["A"]*np.sin(Ixx["w"]*dB + Ixx["phi"]) + Ixx["z"]
            self.I_yy = lambda dB : Iyy["A"]*np.sin(Iyy["w"]*dB + Iyy["phi"]) + Iyy["z"]
            self.I_zz = lambda dB : Izz["A"]*np.sin(Izz["w"]*dB + Izz["phi"]) + Izz["z"]
            self.I_yz = lambda dB : Iyz["A"]*np.sin(Iyz["w"]*dB + Iyz["phi"]) + Iyz["z"]
            self.I_xy = lambda dB : Ixy["A"]*np.sin(Ixy["w"]*dB + Ixy["phi"]) + Ixy["z"]
            self.I_xz = lambda dB : Ixz["A"]*np.sin(Ixz["w"]*dB + Ixz["phi"]) + Ixz["z"]
            self.dI_xx = lambda dB : np.array([0., 0., 0., Ixx["A"]*Ixx["w"]*np.cos(Ixx["w"]*dB + Ixx["phi"])])
            self.dI_yy = lambda dB : np.array([0., 0., 0., Iyy["A"]*Iyy["w"]*np.cos(Iyy["w"]*dB + Iyy["phi"])])
            self.dI_zz = lambda dB : np.array([0., 0., 0., Izz["A"]*Izz["w"]*np.cos(Izz["w"]*dB + Izz["phi"])])
            self.dI_yz = lambda dB : np.array([0., 0., 0., Iyz["A"]*Iyz["w"]*np.cos(Iyz["w"]*dB + Iyz["phi"])])
            self.dI_xy = lambda dB : np.array([0., 0., 0., Ixy["A"]*Ixy["w"]*np.cos(Ixy["w"]*dB + Ixy["phi"])])
            self.dI_xz = lambda dB : np.array([0., 0., 0., Ixz["A"]*Ixz["w"]*np.cos(Ixz["w"]*dB + Ixz["phi"])])
        else:
            self.Ixx = prop_dict["inertia"]["I_xx"]
            self.Ixy = prop_dict["inertia"]["I_xy"]
            self.Iyx = self.Ixy
            self.Ixz = prop_dict["inertia"]["I_xz"]
            self.Izx = self.Ixz
            self.Iyy = prop_dict["inertia"]["I_yy"]
            self.Iyz = prop_dict["inertia"]["I_yz"]
            self.Izy = self.Iyz
            self.Izz = prop_dict["inertia"]["I_zz"]
        self.g = gravity_english(H)
        dummyz, dummyT, dummyp, self.rho, self.a = stdatm_english(H)
        dummyz, dummyT, dummyp, self.rho_0, self.a_0 = stdatm_english(H)
        self.nondim_const = 0.5*self.rho*V*V*self.S_w
        self.V = V
        self.H = H
        self.Gamma = Gamma
        self.M = self.V/self.a

    def calc_BIRE_inertia(self, dB):
        self.Ixx = self.I_xx(dB)
        self.Ixy = self.I_xy(dB)
        self.Ixz = self.I_xz(dB)
        self.Iyy = self.I_yy(dB)
        self.Iyz = self.I_yz(dB)
        self.Izz = self.I_zz(dB)

class TrimSolution:
    def __init__(self):
        self.FM = np.zeros(6)
        self.rates = np.zeros(3)
        self.velocity = np.zeros(3)
        self.load = 0.
        self.load_s = 0.
        self.x = np.zeros(6)
        self.orient = np.zeros(3)
        self.num_iters = 0.
        self.vehicle = "Baseline"

def climb_2_elev(u, v, w, phi, gamma, V):
    V = np.sqrt(u**2 + v**2 + w**2)
    n_1 = u*V*np.sin(gamma)
    n_2 = (v*np.sin(phi) + w*np.cos(phi))
    n_3 = np.sqrt(u*u + n_2**2 - V**2*np.sin(gamma)**2)
    d = u**2 + n_2**2
    th_plus = np.arcsin((n_1 + n_2*n_3)/d)
    th_minus = np.arcsin((n_1 - n_2*n_3)/d)
    check_plus = (V*np.sin(gamma) - u*np.sin(th_plus) - n_2*np.cos(th_plus) < 1e-8)
    check_minus = (V*np.sin(gamma) - u*np.sin(th_minus) - n_2*np.cos(th_minus) < 1e-8)
    if check_plus:
        return th_plus
    elif check_minus:
        return th_minus

def v_comp(alpha, beta, V):
    u = V*np.cos(alpha)*np.cos(beta)
    v = V*np.sin(beta)
    w = V*np.sin(alpha)*np.cos(beta)
    return u, v, w

def load_2_bank(n_a, Fx, W, p, q, u, v, alpha, theta, props):
    num = n_a - Fx*np.sin(alpha)/W - (q*u - p*v)*np.cos(alpha)/props.g
    denom = np.cos(theta)*np.cos(alpha)
    phi = np.arccos(num/denom)
    return phi

def load_factor(theta, phi, alpha, p, q, r, u, v, w, props):
    C1 = np.cos(theta)*np.cos(phi) + (q*u - p*v)/props.g
    C2 = np.sin(theta) - (r*v - q*w)/props.g
    n_a = C1*np.cos(alpha) + C2*np.sin(alpha)
    return n_a

def rotation_rates(phi, theta, u, w, props):
    C_num = props.g*np.sin(phi)*np.cos(theta)
    C_denom = u*np.cos(theta)*np.cos(phi) + w*np.sin(theta)
    C = C_num/C_denom
    p = -C*np.sin(theta)
    q = C*np.sin(phi)*np.cos(theta)
    r = C*np.cos(phi)*np.cos(theta)
    return p, q, r

def thrust(V, props):
    T0 = 29550.
    T1 = 0.
    T2 = 0.
    a = 0.84
    C1 = (props.rho/props.rho_0)**a
    C2 = T0 + T1*V + T2*V**2
    T = C1*C2
    return T

def _tau_p1(tau, Fx, theta, q, r, v, w, T, props):
    num = Fx - props.W*np.sin(theta) + (r*v - q*w)*props.W/props.g
    denom = T
    tau_p1 = tau - props.Gamma*num/denom
    return tau_p1, num

def _beta_p1(beta, Fy, phi, theta, p, r, u, w, CSb, props):
    num = Fy + props.W*np.sin(phi)*np.cos(theta) + (p*w - r*u)*props.W/props.g
    denom = props.nondim_const*CSb*np.cos(beta)
    beta_p1 = beta - props.Gamma*num/denom
    return beta_p1, num

def _alpha_p1(alpha, Fz, phi, theta, p, q, u, v, CLa, props):
    num = Fz + props.W*np.cos(phi)*np.cos(theta) + (q*u - p*v)*props.W/props.g
    denom = props.nondim_const*CLa*np.cos(alpha)
    alpha_p1 = alpha + props.Gamma*num/denom
    return alpha_p1, num

def _da_p1(da, Mx, p, q, r, Clda, props):
    num_1 = Mx - props.hz*q + props.hy*r + (props.Iyy - props.Izz)*q*r
    num_2 = props.Iyz*(q**2 - r**2) + props.Ixz*p*q - props.Ixy*p*r
    num = num_1 + num_2
    denom = props.nondim_const*props.b_w*Clda
    da_p1 = da - props.Gamma*num/denom
    return da_p1, num

def _de_p1(de, My, p, q, r, Cmde, props):
    num_1 = My + props.hz*p - props.hx*r + (props.Izz - props.Ixx)*p*r
    num_2 = props.Ixz*(r**2 - p**2) + props.Ixy*q*r - props.Iyz*p*q
    num = num_1 + num_2
    denom = props.nondim_const*props.c_w*Cmde
    de_p1 = de - props.Gamma*num/denom
    return de_p1, num

def _dr_p1(dr, Mz, p, q, r, Cndr, props):
    num_1 = Mz - props.hy*p + props.hx*q + (props.Ixx - props.Iyy)*p*q
    num_2 = props.Ixy*(p**2 - q**2) + props.Iyz*p*r - props.Ixz*q*r
    num = num_1 + num_2
    denom = props.nondim_const*props.b_w*Cndr
    dr_p1 = dr - props.Gamma*num/denom
    return dr_p1, num

def _dB_p1(dB, Mz, p, q, r, CndB, props):
    num_1 = Mz - props.hy*p + props.hx*q + (props.Ixx - props.Iyy)*p*q
    num_2 = props.Ixy*(p**2 - q**2) + props.Iyz*p*r - props.Ixz*q*r
    num = num_1 + num_2
    denom = props.nondim_const*props.b_w*CndB
    dB_p1 = dB - props.Gamma*num/denom
    return dB_p1, num

def _f1(Fx, theta, phi, pqr, uvw, props):
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fx - props.W*np.sin(theta) + (r*v - q*w)*props.W/props.g

def _f2(Fy, theta, phi, pqr, uvw, props):
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fy + props.W*np.sin(phi)*np.cos(theta) + (p*w - r*u)*props.W/props.g

def _f3(Fz, theta, phi, pqr, uvw, props):
    [u, v, w] = uvw
    [p, q, r] = pqr
    return Fz + props.W*np.cos(phi)*np.cos(theta) + (q*u - p*v)*props.W/props.g

def _f4(Mx, theta, phi, pqr, uvw, props):
    [p, q, r] = pqr
    C1 = Mx - props.hz*q + props.hy*r + (props.Iyy - props.Izz)*q*r
    C2 = props.Iyz*(q**2 - r**2) + props.Ixz*p*q - props.Ixy*p*r
    return C1 + C2

def _f5(My, theta, phi, pqr, uvw, props):
    [p, q, r] = pqr
    C1 = My + props.hz*p - props.hx*r + (props.Izz - props.Ixx)*p*r
    C2 = props.Ixz*(r**2 - p**2) + props.Ixy*q*r - props.Iyz*p*q
    return C1 + C2

def _f6(Mz, theta, phi, pqr, uvw, props):
    [p, q, r] = pqr
    C1 = Mz - props.hy*p + props.hx*q + (props.Ixx - props.Iyy)*p*q
    C2 = props.Ixy*(p**2 - q**2) + props.Iyz*p*r - props.Ixz*q*r
    return C1 + C2

def _recalc_forces(state, phi, gamma, coeffs, props, shss):
    V = props.V
    [tau, alpha, beta, da, de, dr] = state
    u, v, w = v_comp(alpha, beta, V)
    theta = climb_2_elev(u, v, w, phi, gamma, V)
    if not shss:
        p, q, r = rotation_rates(phi, theta, u, w, props)
        pbar = p*props.b_w/(2.*V)
        qbar = q*props.c_w/(2.*V)
        rbar = r*props.b_w/(2.*V)
    else:
        p, q, r = [0., 0., 0.]
        pbar, qbar, rbar = [0., 0., 0.]
    FM = coeffs.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dr)
    [CL, CS, CD, Cl, Cm, Cn] = FM
    CX = -(CD*np.cos(alpha)*np.cos(beta) + CS*np.cos(alpha)*np.sin(beta) - CL*np.sin(alpha))
    CY = CS*np.cos(beta) - CD*np.sin(beta)
    CZ = -(CD*np.sin(alpha)*np.cos(beta) + CS*np.sin(alpha)*np.sin(beta) + CL*np.cos(alpha))
    Fx = CX*props.nondim_const + tau*thrust(V, props)
    Fy = CY*props.nondim_const
    Fz = CZ*props.nondim_const
    Mx = Cl*props.nondim_const*props.b_w - Fz*props.y_shift + Fy*props.z_shift
    My = Cm*props.nondim_const*props.c_w - Fx*props.z_shift + Fz*props.x_shift
    Mz = Cn*props.nondim_const*props.b_w - Fy*props.x_shift + Fx*props.y_shift
    FM = [Fx, Fy, Fz, Mx, My, Mz]
    return FM, [u, v, w], [p, q, r], theta


def fpi(tau, alpha, beta, rot_rates, de, da, dr, vel_comp,
        phi, theta, coeffs, FM, props, bire, dm_E=0., dn_E=0.):
    V = props.V
    [Fx, Fy, Fz, Mx, My, Mz] = FM
    [p, q, r] = rot_rates
    [u, v, w] = vel_comp
    T = thrust(V, props)
    if bire:
        dB = dr
        CLa = coeffs._CL_alpha(0.)
        CLde = coeffs._CL_de(0.)
        CSb = coeffs._CS_beta(0.)
        Clda = coeffs._Cl_da(0.)
        Cmde = coeffs._Cm_de(0.)
        pbar = p*props.b_w/(2.*V)
        qbar = q*props.c_w/(2.*V)
        rbar = r*props.b_w/(2.*V)
        CndB = coeffs.Cn_dB(alpha, beta, pbar, qbar, rbar, da, de, 0.)
    else:
        CLa = coeffs.CLa
        CSb = coeffs.CSb
        Clda = coeffs.Clda
        Cmde = coeffs.Cmde
        Cndr = coeffs.Cndr
    tau_p1, num_tau = _tau_p1(tau, Fx, theta, q, r, v, w, T, props)
    beta_p1, num_beta = _beta_p1(beta, Fy, phi, theta, p, r, u, w, CSb, props)
    alpha_p1, num_alpha = _alpha_p1(alpha, Fz, phi, theta, p, q, u, v, CLa, props)
    da_p1, num_da = _da_p1(da, Mx, p, q, r, Clda, props)
    if bire:
        de_p1, num_de = _de_p1(de, My, p, q, r, Cmde, props)
        dB_p1, num_dB = _dB_p1(dB, Mz, p, q, r, CndB, props)
        error = np.array([num_tau, num_beta, num_alpha, num_da, num_de, num_dB])
        return np.array([tau_p1, alpha_p1, beta_p1, da_p1, de_p1, dB_p1]), error
    else:
        de_p1, num_de = _de_p1(de, My, p, q, r, Cmde, props)
        dr_p1, num_dr = _dr_p1(dr, Mz, p, q, r, Cndr, props)
        error = np.array([num_tau, num_beta, num_alpha, num_da, num_de, num_dr])
        return np.array([tau_p1, alpha_p1, beta_p1, da_p1, de_p1, dr_p1]), error

def jacobian(trim_state, phi, theta, gamma, coeffs, props, shss, delta=0.001):
    [tau, alpha, beta, de, da, dr] = trim_state
    J = np.zeros((6, 6))
    f = [_f1, _f2, _f3, _f4, _f5, _f6]
    for i in range(6):
        delta_state = np.zeros(6)
        delta_state[i] = delta
        for j in range(6):
            FM_p, vcomp_p, rotrates_p, theta_p = _recalc_forces([t + d for t,d in zip(trim_state, delta_state)], phi, gamma, coeffs, props, shss)
            FM_m, vcomp_m, rotrates_m, theta_m = _recalc_forces([t - d for t,d in zip(trim_state, delta_state)], phi, gamma, coeffs, props, shss)
            f_p = f[j](FM_p[j], theta_p, phi, rotrates_p, vcomp_p, props)
            f_m = f[j](FM_m[j], theta_m, phi, rotrates_m, vcomp_m, props)
            J[j, i] = (f_p - f_m)/(2.*delta)
    return J


def trim_optimize(trim_state, V, H, gamma, phi, cg_shift, shss=True):
    props = AircraftProperties(V, H, 0.)
    x_shift, y_shift, z_shift = cg_shift
    props.x_shift = x_shift
    props.y_shift = y_shift
    props.z_shift = z_shift
    [tau, alpha, beta, da, de, dB] = trim_state
    coeffs = BIREAero()
    p, q, r = [0., 0., 0.]
    pbar, qbar, rbar = p, q, r
    u, v, w = v_comp(alpha, beta, V)
    theta = climb_2_elev(u, v, w, phi, gamma, V)
    if not shss:
        p, q, r = rotation_rates(phi, theta, u, w, props)
        pbar = p*props.b_w/(2.*V)
        qbar = q*props.c_w/(2.*V)
        rbar = r*props.b_w/(2.*V)
    FM = coeffs.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dB)
    [CL, CS, CD, Cl, Cm, Cn] = FM
    CX = -(CD*np.cos(alpha)*np.cos(beta) + CS*np.cos(alpha)*np.sin(beta) - CL*np.sin(alpha))
    CY = CS*np.cos(beta) - CD*np.sin(beta)
    CZ = -(CD*np.sin(alpha)*np.cos(beta) + CS*np.sin(alpha)*np.sin(beta) + CL*np.cos(alpha))
    Fx = CX*props.nondim_const + tau*thrust(V, props)
    Fy = CY*props.nondim_const
    Fz = CZ*props.nondim_const
    Mx = Cl*props.nondim_const*props.b_w - Fz*y_shift + Fy*z_shift
    My = Cm*props.nondim_const*props.c_w - Fx*z_shift + Fz*x_shift
    Mz = Cn*props.nondim_const*props.b_w - Fy*x_shift + Fx*y_shift
    FM = [Fx, Fy, Fz, Mx, My, Mz]
    X_zero = Fx + props.W*np.sin(theta) - props.W/props.g*(r*v - q*w)
    Y_zero = Fy + props.W*np.sin(phi)*np.cos(theta) - props.W/props.g*(p*w - r*u)
    Z_zero = Fz + props.W*np.cos(phi)*np.cos(theta) - props.W/props.g*(q*u - p*v)
    return np.array([X_zero, Y_zero, Z_zero, Mx, My, Mz])

def compressible_correction(a0, Lambda, AR, M):
    num = a0*np.cos(Lambda)
    denom_1 = np.sqrt(1. - M**2*np.cos(Lambda)**2 +
                      (num/(np.pi*AR))**2)
    denom_2 = num/(np.pi*AR)
    denom = denom_1 + denom_2
    return num/denom

def trim(V, H, gamma, phi, Gamma, trim_0=np.zeros(6),
         shss=False, bire=False, cg_shift=[0., 0., 0.], verbose=True,
         fixed_point=True, aero_dir='./', **kwargs):
    props_fn = kwargs.get('props_filename', False)
    if not props_fn:
        props = AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire)
    else:
        props = AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire, filename=props_fn)
    trim_state = trim_0
    x_shift, y_shift, z_shift = cg_shift
    props.x_shift = x_shift
    props.y_shift = y_shift
    props.z_shift = z_shift
    comp_correction = kwargs.get("compressible", False)
    if bire:
        [tau, alpha, beta, da, de, dB] = trim_state
        coeffs = kwargs.get("coeffs", BIREAero(aero_dir))
    else:
        [tau, alpha, beta, da, de, dr] = trim_state
        coeffs = kwargs.get("coeffs", F16Aero(aero_dir))
    p, q, r = [0., 0., 0.]
    pbar, qbar, rbar = p, q, r
    error = 100.
    number_of_iterations = 0
    while (error > 1e-9)*(number_of_iterations <= 1000):
        number_of_iterations += 1
        u, v, w = v_comp(alpha, beta, V)
        theta = climb_2_elev(u, v, w, phi, gamma, V)
        if not shss:
            p, q, r = rotation_rates(phi, theta, u, w, props)
            pbar = p*props.b_w/(2.*V)
            qbar = q*props.c_w/(2.*V)
            rbar = r*props.b_w/(2.*V)
        if bire:
            FM = coeffs.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dB)
            props.calc_BIRE_inertia(dB)
        else:
            FM = coeffs.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dr)
        [CL, CS, CD, Cl, Cm, Cn] = FM
        if comp_correction:
            if props.M < 1.:
                if bire:
                    CL = compressible_correction(CL, props.Lam_w, props.RA_w, props.M)
                    CS = compressible_correction(CS, props.Lam_h, props.RA_h, props.M)
                    Cl = compressible_correction(Cl, props.Lam_w, props.RA_w, props.M)
                    Cm = compressible_correction(Cm, props.Lam_w, props.RA_w, props.M)
                    Cn = compressible_correction(Cn, props.Lam_h, props.RA_h, props.M)
                else:
                    CL = compressible_correction(CL, props.Lam_w, props.RA_w, props.M)
                    CS = compressible_correction(CS, props.Lam_v, props.RA_v, props.M)
                    Cl = compressible_correction(Cl, props.Lam_v, props.RA_v, props.M)
                    Cm = compressible_correction(Cm, props.Lam_w, props.RA_w, props.M)
                    Cn = compressible_correction(Cn, props.Lam_v, props.RA_v, props.M)
            else:
                CL = CL/np.sqrt(props.M**2 - 1.)
                CS = CS/np.sqrt(props.M**2 - 1.)
                Cl = Cl/np.sqrt(props.M**2 - 1.)
                Cm = Cm/np.sqrt(props.M**2 - 1.)
                Cn = Cn/np.sqrt(props.M**2 - 1.)
        CX = -(CD*np.cos(alpha)*np.cos(beta) + CS*np.cos(alpha)*np.sin(beta) - CL*np.sin(alpha))
        CY = CS*np.cos(beta) - CD*np.sin(beta)
        CZ = -(CD*np.sin(alpha)*np.cos(beta) + CS*np.sin(alpha)*np.sin(beta) + CL*np.cos(alpha))
        Fx = CX*props.nondim_const + trim_state[0]*thrust(V, props)
        Fy = CY*props.nondim_const
        Fz = CZ*props.nondim_const
        Mx = Cl*props.nondim_const*props.b_w - Fz*y_shift + Fy*z_shift
        My = Cm*props.nondim_const*props.c_w - Fx*z_shift + Fz*x_shift
        Mz = Cn*props.nondim_const*props.b_w - Fy*x_shift + Fx*y_shift
        FM = [Fx, Fy, Fz, Mx, My, Mz]
        if fixed_point:
            if bire:
                trimstate_p1, nums = fpi(tau, alpha, beta, [p, q, r], de, da, dB, [u, v, w],
                                          phi, theta, coeffs, FM, props, bire)
            else:
                trimstate_p1, nums = fpi(tau, alpha, beta, [p, q, r], de, da, dr, [u, v, w],
                                          phi, theta, coeffs, FM, props, bire)
        else:
            f = [_f1, _f2, _f3, _f4, _f5, _f6]
            nums = np.array([f(FM[idx], theta, phi, [p, q, r], [u, v, w], props) for idx, f in enumerate(f)])
            try:
                J = jacobian(trim_state, phi, theta, gamma, coeffs, props, shss)
                D_G = np.linalg.solve(-J, nums)
            except np.linalg.LinAlgError:
                J = jacobian(trim_state, phi, theta, gamma, coeffs, props, shss, delta=0.1)
                D_G = np.linalg.solve(-J, nums)
            trimstate_p1 = trim_state + Gamma*D_G
        error = np.max(np.abs(nums))
        trim_state = trimstate_p1
        if bire:
            [tau, alpha, beta, da, de, dB] = trim_state
        else:
            [tau, alpha, beta, da, de, dr] = trim_state
    T = trim_state[0]*thrust(V, props)
    n_a = ((np.cos(theta)*np.cos(phi) + (q*u - p*v)/props.g)*np.cos(alpha) +
           (np.sin(theta) - (r*v - q*w)/props.g)*np.sin(alpha))
    n_sa = CL/(props.W/(0.5*props.rho*V**2*props.S_w))
    if bire:
        while abs(dB) > np.pi:
            if dB >= 2.*np.pi:
                while dB >= np.pi:
                    dB -= 2.*np.pi
            if dB > np.pi:
                while dB > np.pi:
                    dB -= np.pi
                    de *= -1.
            if dB <= -2.*np.pi:
                while dB <= -np.pi:
                    dB += 2.*np.pi
            if dB < -np.pi:
                while dB < -np.pi:
                    dB += np.pi
                    de *= -1.
    if verbose:
        print("------ Trim Solution ------")
        print(f"Elevation Angle (deg.) : {theta*180./np.pi:1.12g}")
        print(f"Bank Angle (deg.) : {phi*180./np.pi:1.12g}")
        print(f"Alpha (deg.) : {alpha*180./np.pi:1.12g}")
        print(f"Beta (deg.) : {beta*180./np.pi:1.12g}")
        print(f"p (deg./s) : {p*180./np.pi:1.12g}")
        print(f"q (deg./s) : {q*180./np.pi:1.12g}")
        print(f"r (deg./s) : {r*180./np.pi:1.12g}")
        print(f"Aileron (deg.) : {da*180./np.pi:1.12g}")
        print(f"Elevator (deg.) : {de*180./np.pi:1.12g}")
        if bire:
            print(f"BIRE Rotation (deg.) : {dB*180./np.pi:1.12g}")
        else:
            print(f"Rudder (deg.) : {dr*180./np.pi:1.12g}")
        print(f"Throttle : {tau:1.12g}")
        print(f"Thrust (lbf.) : {T:1.12f}")
        print(f"Load Factor : {n_a:1.12f}")
        print(f"Stability Axis Load Factor : {n_sa:1.12f}")
        print(f"Number of Iterations : {number_of_iterations:d}")
    solution = TrimSolution()
    solution.FM = np.array([CD, CS, CL, Cl, Cm, Cn])
    solution.FM_dim = np.array([Fx, Fy, Fz, Mx, My, Mz])
    solution.load = n_a
    solution.load_s = n_sa
    # replace the following
    solution.x = trim_state
    # if not bire:
    #     solution.x = np.concatenate((trim_state, [tau, da, de, dr]))
    # else:
    #     solution.x = np.concatenate((trim_state, [tau, da, de, dB]))
    solution.num_iters = number_of_iterations
    solution.orient = np.array([phi, theta, 0.])
    solution.velocity = np.array([u, v, w])
    solution.rates = np.array([p, q, r])
    solution.states = np.array([u, v, w, p, q, r, phi, theta])
    solution.aero = coeffs
    if bire:
        solution.vehicle = "BIRE"
        solution.inputs = np.array([tau, da, de, dB])
    else:
        solution.vehicle = "Baseline"
        solution.inputs = np.array([tau, da, de, dr])
    if verbose:
        print(solution.x)
    return solution

if __name__ == "__main__":
    V = 222.5211
    # print("V = ", V)
    gamma = np.deg2rad(0.)
    phi = np.deg2rad(5.)
    H = 1000.
    Gamma = 0.1
    cg_shift = [1., 0., 0.]
    solution = trim(V, H, gamma, phi, Gamma, shss=True, bire=True, cg_shift=cg_shift, fixed_point=False, compressible=True)
