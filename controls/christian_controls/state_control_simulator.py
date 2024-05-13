#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:49:57 2022

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from f16_aero import F16Aero
from bire_aero import BIREAero
import aero_trim as trim
from hunsaker_atm import stdatm_english
import f16_linearization as control_design
import bire_linearization as control_design_bire
import control as ct
from os.path import exists
import pickle

def gust_model(t, **kwargs):
    A = kwargs.get("A", 80.)
    g_t = kwargs.get("gamma", 1.)
    w = kwargs.get("w", 2.)
    s_x = kwargs.get("s_x", 1.)
    s_y = kwargs.get("s_y", 1.)
    s_z = kwargs.get("s_z", 0.5)
    t_0 = kwargs.get("t_0", 1.)
    if t >= t_0:
        V_wx = s_x*A*np.exp(-g_t*(t - t_0))*np.sin(w*(t - t_0))
        V_wy = s_y*A*np.exp(-g_t*(t - t_0))*np.sin(w*(t - t_0))
        V_wz = s_z*A*np.exp(-g_t*(t - t_0))*np.sin(w*(t - t_0))
        Vd_wx = s_x*A*np.exp(-g_t*(t - t_0))*(w*np.cos(w*(t - t_0)) - g_t*(np.sin(w*(t - t_0))))
        Vd_wy = s_y*A*np.exp(-g_t*(t - t_0))*(w*np.cos(w*(t - t_0)) - g_t*(np.sin(w*(t - t_0))))
        Vd_wz = s_z*A*np.exp(-g_t*(t - t_0))*(w*np.cos(w*(t - t_0)) - g_t*(np.sin(w*(t - t_0))))
    else:
        V_wx = 0.
        V_wy = 0.
        V_wz = 0.
        Vd_wx = 0.
        Vd_wy = 0.
        Vd_wz = 0.
    return V_wx, V_wy, V_wz, Vd_wx, Vd_wy, Vd_wz

def eqs_of_motion(t, x, xhat, uhat, props, cg_shift, linearization, bire, control_sys, dist_model, tau):
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    K = linearization.K
    if dist_model["type"] == "gust":
        V_wx, V_wy, V_wz, Vd_wx, Vd_wy, Vd_wz = gust_model(t, **dist_model["params"])
    else:
        V_wx = 0.
        V_wy = 0.
        V_wz = 0.
        Vd_wx = 0.
        Vd_wy = 0.
        Vd_wz = 0.
        dist = dist_model["params"]["dist"]
        mag = dist_model["params"]["mag"]
        t_0 = dist_model["params"]["t_0"]
        if (t >= t_0):
            x += [d*m for d,m in zip(dist, mag)]
    if bire:
        u, v, w, p, q, r, x_f, y_f, z_f, phi, theta, psi = x
    else:
        u, v, w, p, q, r, x_f, y_f, z_f, phi, theta, psi = x
    H = -z_f
    if H < 0.:
        H = 0.
        z_f = 0.
    x_shift, y_shift, z_shift = cg_shift
    rho = stdatm_english(H)[3]
    V = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan(w/u)
    beta = np.arcsin(v/V)
    c_a = np.cos(alpha)
    s_a = np.sin(alpha)
    c_b = np.cos(beta)
    s_b = np.sin(beta)
    s_p = np.sin(phi)
    c_p = np.cos(phi)
    s_t = np.sin(theta)
    c_t = np.cos(theta)
    s_ps = np.sin(psi)
    c_ps = np.cos(psi)
    z = np.array([u - xhat[0], v - xhat[1], w - xhat[2],
                  p - xhat[3], q - xhat[4], r - xhat[5],
                  phi - xhat[6], theta - xhat[7]])
    g = props.g
    W = props.W
    b_w = props.b_w
    c_w = props.c_w
    S_w = props.S_w
    dim_coeff = 0.5*rho*V*V*S_w
    pbar = p*b_w/(2.*V)
    qbar = q*c_w/(2.*V)
    rbar = r*b_w/(2.*V)
    if bire:
        aero = BIREAero(aero_dir)
        if control_sys:
            [da, de, dB] = uhat - np.matmul(K, z)
        else:
            [da, de, dB] = uhat
        params = [alpha, beta, pbar, qbar, rbar, da, de, dB]
        I_inv = control_design_bire.LinearizationBIRE(props)._I_inv(dB)
        Ixx = props.I_xx(dB)
        Iyy = props.I_yy(dB)
        Izz = props.I_zz(dB)
        Ixy = props.I_xy(dB)
        Ixz = props.I_xz(dB)
        Iyz = props.I_yz(dB)
    else:
        if control_sys:
            [da, de, dr] = uhat - np.matmul(K, z)
        else:
            [da, de, dr] = uhat
        aero = F16Aero(aero_dir)
        params = [alpha, beta, pbar, qbar, rbar, da, de, dr]
        I_inv = control_design.LinearizationBaseline(props)._I_inv()
        Ixx = props.Ixx
        Iyy = props.Iyy
        Izz = props.Izz
        Ixy = props.Ixy
        Ixz = props.Ixz
        Iyz = props.Iyz
    [CL, CS, CD, Cl, Cm, Cn] = aero.aero_results(*params)
    Fx = -(CD*c_a*c_b + CS*c_a*s_b - CL*s_a)*dim_coeff + trim.thrust(tau, V, props)
    Fy = (CS*c_b - CD*s_b)*dim_coeff
    Fz = -(CD*s_a*c_b + CS*s_a*s_b + CL*c_a)*dim_coeff
    Mx = Cl*dim_coeff*b_w - Fz*y_shift + Fy*z_shift
    My = Cm*dim_coeff*c_w - Fx*z_shift + Fz*x_shift
    Mz = Cn*dim_coeff*b_w - Fy*x_shift + Fx*y_shift
    f1 = g/W*Fx - g*s_t + r*v - q*w - Vd_wx
    f2 = g/W*Fy + g*s_p*c_t + p*w - r*u - Vd_wy
    f3 = g/W*Fz + g*c_p*c_t + q*u - p*v - Vd_wz
    M_vec = np.zeros(3)
    M_vec[0] = Mx + (Iyy - Izz)*q*r + Iyz*(q**2 - r**2) + Ixz*p*q - Ixy*p*r
    M_vec[1] = My + (Izz - Ixx)*p*r + Ixz*(r**2 - p**2) + Ixy*q*r - Iyz*p*q
    M_vec[2] = Mz + (Ixx - Iyy)*p*q + Ixy*(p**2 - q**2) + Iyz*p*r - Ixz*q*r
    f4, f5, f6 = np.matmul(I_inv, M_vec)
    f7 = c_t*c_ps*u + (s_p*s_t*c_ps - c_p*s_ps)*v + (c_p*s_t*c_ps + s_p*s_ps)*w + V_wx
    f8 = c_t*s_ps*u + (s_p*s_t*s_ps + c_p*c_ps)*v + (c_p*s_t*s_ps - s_p*c_ps)*w + V_wy
    f9 = -s_t*u + s_p*c_t*v + c_p*c_t*w + V_wz
    f10 = p + (s_p*s_t/c_t)*q + (c_p*s_t/c_t)*r
    f11 = c_p*q - s_p*r
    f12 = s_p/c_t*q + c_p/c_t*r
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

def control(t, x, xhat, uhat, K, bire, control_sys):
    z = np.zeros((len(t), 8))
    u = np.zeros((len(t), 3))
    for i in range(len(t)):
        z[i, 0:6] = x[i, 0:6] - xhat[0:6]
        z[i, 6] = x[i, 9] - xhat[6]
        z[i, 7] = x[i, 10] - xhat[7]
        if control_sys:
            if not bire:
                u[i, :] = uhat - np.matmul(K, z[i, :])
            else:
                u[i, :] = uhat - np.matmul(K, z[i, :])

        else:
            u[i, :] = uhat
    return u

def wind(t, model):
    Vx = np.zeros_like(t)
    Vy = np.zeros_like(t)
    Vz = np.zeros_like(t)
    for i in range(len(t)):
        Vx[i], Vy[i], Vz[i] = gust_model(t[i], **model['params'])[:3]
    return Vx, Vy, Vz



def design_controller(H, M, gamma, phi, Gamma, cg_shift, bire, save_trim_lin=True):
    a = stdatm_english(H)[-1]
    V = M*a
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    trim_solution = trim.trim(V, H, gamma, phi, Gamma,
                              shss=True, bire=bire,
                              cg_shift=cg_shift,
                              fixed_point=False,
                              compressible=False,
                              aero_dir=aero_dir)
    Q = np.eye(8)
    Q[4, 4] = 25.
    Q[5, 5] = 25.
    Q[6, 6] = 10.
    Q[7, 7] = 10.
    R = np.eye(4)
    R[0, 0] = 20.
    R[3, 3] = 20.
    if not bire:
        lin_control = control_design.create_feedback_control(trim_solution, V, H, Gamma, cg_shift,  lqr_flag=True, Q=Q, R=R)
    else:
        lin_control = control_design_bire.create_feedback_control(trim_solution, V, H, Gamma, cg_shift, Q=Q, R=R)
    if save_trim_lin:
        save_dir = './Simulation Data/'
        if bire:
            save_dir += 'BIRE/'
        else:
            save_dir += 'Baseline/'
    with open(save_dir + 'Linearization/CG_' + str(cg_shift[0] - 1.) + '.lin', 'wb') as lin_file:
        pickle.dump(lin_control, lin_file)
    with open(save_dir + 'Solution/CG_' + str(cg_shift[0] - 1.) + '.trim', 'wb') as trim_file:
        pickle.dump(trim_solution, trim_file)
    return trim_solution, lin_control


def simulate(trim_solution, t_range, linearization, props, cg_shift, control_sys, **kwargs):
    dist_model = kwargs.get("model", {"type": "step", "params": {"dist": [1.]*8, "mag": 0.}})
    if linearization.aircraft == "BIRE":
        bire = True
    else:
        bire = False
    K = linearization.K
    H = props.H
    x_hat = trim_solution.states
    u_hat = trim_solution.inputs[1:]
    tau = trim_solution.inputs[0]
    a = stdatm_english(H)[-1]
    M = props.V/a
    """Simulation"""
    len_t = len(t_range)
    t0 = t_range[0]
    t1 = t_range[-1]
    dt = t_range[1] - t_range[0]
    r = ode(eqs_of_motion).set_integrator('dopri5')
    x = np.zeros((len_t, 12))
    x[0, :6] = x_hat[:6]
    x[0, 6:9] = [0., 0., -H]
    x[0, 9:] = trim_solution.orient
    if bire:
        r.set_initial_value(x[0, :], t0).set_f_params(x_hat, u_hat, props, cg_shift, linearization, True, control_sys, dist_model, tau)
    else:
        r.set_initial_value(x[0, :], t0).set_f_params(x_hat, u_hat, props, cg_shift, linearization, False, control_sys, dist_model, tau)
    i = 1
    z = np.zeros((len_t, 8))
    while r.successful() and t_range[i] < t_range[-1]:
        x[i, :] = r.integrate(r.t+dt)
        z[i, 0:6] = x[i, 0:6] - x_hat[0:6]
        z[i, 6] = x[i, 9] - x_hat[6]
        z[i, 7] = x[i, 10] - x_hat[7]
        i += 1
    x[i, :] = r.integrate(r.t+dt)
    z[i, 0:6] = x[i, 0:6] - x_hat[0:6]
    z[i, 6] = x[i, 9] - x_hat[6]
    z[i, 7] = x[i, 10] - x_hat[7]
    u = control(t_range, x, x_hat, u_hat, K, bire, control_sys)
    Vx, Vy, Vz = wind(t_range, dist_model)
    if bire:
        save_dir = './Simulation Data/BIRE/'
    else:
        save_dir = './Simulation Data/Baseline/'
    if control_sys:
        save_dir += 'Controlled/'
    else:
        save_dir += 'Uncontrolled/'
    np.save(save_dir + 'wind_CG_' + str(cg_shift[0]) + '.npy', [Vx, Vy, Vz])
    np.save(save_dir + 'time_range_CG_' + str(cg_shift[0]) + '.npy', t_range)
    np.save(save_dir + 'states_CG_' + str(cg_shift[0]) + '.npy', x)
    np.save(save_dir + 'inputs_CG_' + str(cg_shift[0]) + '.npy', u)
    np.save(save_dir + 'trim_states_CG_' + str(cg_shift[0]) + '.npy', x_hat)
    np.save(save_dir + 'trim_inputs_CG_' + str(cg_shift[0]) + '.npy', u_hat)
    np.save(save_dir + 'shifted_states_CG_' + str(cg_shift[0]) + '.npy', z)

if __name__ == "__main__":
    H = 15000.
    M = 0.6
    a = stdatm_english(H)[-1]
    V = M*a
    Gamma = 0.1
    gamma = np.deg2rad(0.)
    phi = np.deg2rad(0.)
    t0 = 0.
    t1 = 20.
    dt = 0.1
    t_range = np.arange(t0, t1 + dt, dt)
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    lqr = True
    control_sys = False
    comp = True
    model_gust = {"type": "gust", "params": {"A": 80.,
                                             "gamma": 1.,
                                             "w": 5.,
                                             "s_x": 1.,
                                             "s_y": 1.,
                                             "s_z": 0.5,
                                             "t_0": 1.}}
    for shift in np.arange(0., 1.1, 0.25):
        cg_shift = [shift, 0., 0.]
        print(cg_shift)
        bire = False
        base_props = trim.AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire)
        if exists('./Simulation Data/Baseline/Linearization/CG_' + str(cg_shift[0] - 1.) + '.lin')*exists('./Simulation Data/Baseline/Solution/CG_' + str(cg_shift[0] - 1.) + '.trim'):
            with open('./Simulation Data/Baseline/Linearization/CG_' + str(cg_shift[0] - 1.) + '.lin', 'rb') as lin_file:
                base_lin = pickle.load(lin_file)
            with open('./Simulation Data/Baseline/Solution/CG_' + str(cg_shift[0] - 1.) + '.trim', 'rb') as trim_file:
                base_solution = pickle.load(trim_file)
        else:
            base_solution, base_lin = design_controller(H, M, gamma, phi, Gamma, cg_shift, bire)
        simulate(base_solution, t_range, base_lin, base_props, cg_shift, control_sys, model=model_gust)
        bire = True
        bire_props = trim.AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire)
        if exists('./Simulation Data/BIRE/Linearization/CG_' + str(cg_shift[0] - 1.) + '.lin')*exists('./Simulation Data/BIRE/Solution/CG_' + str(cg_shift[0] - 1.) + '.trim'):
            with open('./Simulation Data/BIRE/Linearization/CG_' + str(cg_shift[0] - 1.) + '.lin', 'rb') as lin_file:
                bire_lin = pickle.load(lin_file)
            with open('./Simulation Data/BIRE/Solution/CG_' + str(cg_shift[0] - 1.) + '.trim', 'rb') as trim_file:
                bire_solution = pickle.load(trim_file)
        else:
            bire_solution, bire_lin = design_controller(H, M, gamma, phi, Gamma, cg_shift, bire)
        simulate(bire_solution, t_range, bire_lin, bire_props, cg_shift, control_sys, model=model_gust)
