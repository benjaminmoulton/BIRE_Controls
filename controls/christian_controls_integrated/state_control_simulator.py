#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:49:57 2022

@author: christian
"""

import sys
aero_directory = '../aerodynamics_model/'
mass_directory = '../mass_properties/'
trim_directory = '../trim/'
ctrl_directory = "../controls/"

sys.path.insert(1, aero_directory)
sys.path.insert(1, mass_directory)
sys.path.insert(1, trim_directory)
sys.path.insert(1, ctrl_directory)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from f16_aero import F16Aero
from bire_aero import BIREAero
import aero_trim as trim
from hunsaker_atm import stdatm_english, gravity_english
import f16_linearization as control_design
import bire_linearization as control_design_bire
import control as ct
from os.path import exists
import pickle
import json
from controller_simulation import Aircraft as ben_sim

def gust_model(t, **kwargs):
    A = kwargs.get("A", 80.)
    g_t = kwargs.get("gamma", 1.)
    w = kwargs.get("w", 5.)
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

def eqs_of_motion(t, x, xhat, uhat, props, cg_shift, linearization, bire, control_sys, dist_model, tau, compressible, stall):
    aero_dir = aero_directory # '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
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
    rho,a = stdatm_english(H)[3:5]
    V = np.sqrt((u+V_wx)**2 + (v+V_wy)**2 + (w+V_wz)**2); M = V / a
    alpha = np.arctan((w+V_wz)/(u+V_wx))
    beta = np.arcsin((v+V_wy)/V)
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
    # g = gravity_english(H)
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
            [da, de, dB] = uhat - np.matmul(K, z)[1:4] # this is a fudgy addition, but ok
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
        aero = F16Aero(aero_dir)
        if control_sys:
            [da, de, dr] = uhat - np.matmul(K, z)[1:4] # this is a fudgy addition, but ok
        else:
            [da, de, dr] = uhat
        params = [alpha, beta, pbar, qbar, rbar, da, de, dr]
        I_inv = control_design.LinearizationBaseline(props)._I_inv()
        Ixx = props.Ixx
        Iyy = props.Iyy
        Izz = props.Izz
        Ixy = props.Ixy
        Ixz = props.Ixz
        Iyz = props.Iyz
    ######
    hx = props.hx
    hy = props.hy
    hz = props.hz
    ######
    [CL, CS, CD, Cl, Cm, Cn] = aero.aero_results(*params,compressible=compressible,M=M,enforce_stall=stall)
    Fx = -(CD*c_a*c_b + CS*c_a*s_b - CL*s_a)*dim_coeff + aero.get_thrust(tau,H,V)
    Fy = (CS*c_b - CD*s_b)*dim_coeff
    Fz = -(CD*s_a*c_b + CS*s_a*s_b + CL*c_a)*dim_coeff
    Mx = Cl*dim_coeff*b_w - Fz*y_shift + Fy*z_shift
    My = Cm*dim_coeff*c_w - Fx*z_shift + Fz*x_shift
    Mz = Cn*dim_coeff*b_w - Fy*x_shift + Fx*y_shift
    f1 = g/W*Fx - g*s_t + r*v - q*w # - Vd_wx
    f2 = g/W*Fy + g*s_p*c_t + p*w - r*u # - Vd_wy
    f3 = g/W*Fz + g*c_p*c_t + q*u - p*v # - Vd_wz
    M_vec = np.zeros(3)
    M_vec[0] = Mx - hz*q + hy*r + (Iyy - Izz)*q*r + Iyz*(q**2 - r**2) + Ixz*p*q - Ixy*p*r
    M_vec[1] = My + hz*p - hx*r + (Izz - Ixx)*p*r + Ixz*(r**2 - p**2) + Ixy*q*r - Iyz*p*q
    M_vec[2] = Mz - hy*p + hx*q + (Ixx - Iyy)*p*q + Ixy*(p**2 - q**2) + Iyz*p*r - Ixz*q*r
    f4, f5, f6 = np.matmul(I_inv, M_vec)
    f7 = c_t*c_ps*(u+V_wx) + (s_p*s_t*c_ps - c_p*s_ps)*(v+V_wy) + (c_p*s_t*c_ps + s_p*s_ps)*(w+V_wz)
    f8 = c_t*s_ps*(u+V_wx) + (s_p*s_t*s_ps + c_p*c_ps)*(v+V_wy) + (c_p*s_t*s_ps - s_p*c_ps)*(w+V_wz)
    f9 = -s_t*(u+V_wx) + s_p*c_t*(v+V_wy) + c_p*c_t*(w+V_wz)
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
                u[i, :] = uhat - np.matmul(K, z[i, :])[1:4] # this is fudgy, but ok
            else:
                u[i, :] = uhat - np.matmul(K, z[i, :])[1:4] # this is fudgy, but ok

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


class empty():
    pass

def design_controller(H, M, gamma, phi, Gamma, cg_shift, bire, save_trim_lin=True,
    compressible=True, stall = True):
    a = stdatm_english(H)[-1]
    V = M*a
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    trim_solution = trim.trim(V, H, gamma, phi, Gamma,
                              shss=True, bire=bire,
                              cg_shift=cg_shift,
                              fixed_point=False,
                              compressible=compressible,
                              stall=stall,
                              threshold=1e-9)#compressible)#False)#,
                            #   aero_dir=aero_dir)

    # trim_solution = empty()
    # trim_solution.FM = np.array([
    #     0.022687626760060312,
    #     0.0,
    #     0.2259085707209069,
    #     0.0,
    #     -6.416357517419512e-16,
    #     0.0
    # ])
    # trim_solution.FM_dim = np.array([
    #     945.0863567209703,
    #     0.0,
    #     -20478.203333747337,
    #     0.0,
    #     -6.560665474375797e-10,
    #     0.0
    # ])
    # trim_solution.load = 1.0
    # trim_solution.load_s = 0.9953865293409758
    # trim_solution.x = np.array([
    #     0.10241742585810601,
    #     0.04611811971867537,
    #     0.0,
    #     0.0,
    #     -0.002650199552657236,
    #     0.0
    # ])
    # trim_solution.num_iters = 290
    # trim_solution.orient = np.array([
    #     0.0,
    #     0.04611811971867508,
    #     0.0
    # ])
    # trim_solution.velocity = np.array([
    #     633.7387741170639,
    #     0.0,
    #     29.247578968811375
    # ])
    # trim_solution.rates = np.array([0.0,0.0,0.0])
    # trim_solution.states = np.array([
    #     633.7387741170639,
    #     0.0,
    #     29.247578968811375,
    #     0.0,
    #     0.0,
    #     0.0,
    #     0.0,
    #     0.04611811971867508
    # ])
    # trim_solution.vehicle = "Baseline"
    # trim_solution.inputs = np.array([
    #     0.10241742585810601,
    #     0.0,
    #     -0.002650199552657236,
    #     0.0
    # ])



    Q = np.eye(8)
    if bire:
        Q = Q * 0.0
        Q[3,3] = 20.0
        Q[4,4] = Q[5,5] = Q[6,6] = Q[7,7] = 10.0
        # # christians old values for both
        # Q[4, 4] = 25.
        # Q[5, 5] = 25.
        # Q[6, 6] = 10.
        # Q[7, 7] = 10.
    else:
        Q = Q * 0.0
        Q[3,3] = Q[4,4] = Q[5,5] = 10.0
        Q[7,7] = 20.0
        # # christians old values for both
        # Q[4, 4] = 25.
        # Q[5, 5] = 25.
        # Q[6, 6] = 10.
        # Q[7, 7] = 10.
    R = np.eye(4)
    if bire:
        R = R * 11111111.11111111
        R[1,1] = 2.0
        R[2,2] = R[3,3] = 0.1
        # # christians old values for both
        # R[0, 0] = 20.
        # R[3, 3] = 20.
    else:
        R = R * 11111111.11111111
        R[2,2] = 1.0
        R[1,1] = R[3,3] = 2.0
        # # christians old values for both
        # R[0, 0] = 20.
        # R[3, 3] = 20.
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
    with open(save_dir + 'Linearization/CG_' + str(cg_shift[0]) + '.lin', 'wb') as lin_file:
        pickle.dump(lin_control, lin_file)
    with open(save_dir + 'Solution/CG_' + str(cg_shift[0]) + '.trim', 'wb') as trim_file:
        pickle.dump(trim_solution, trim_file)
    # rep2D(lin_control.A,"A")
    # rep2D(lin_control.B[:,1:],"B")
    return trim_solution, lin_control


def _rk4(t0,x0,dt,inputs):

    # calculate k values
    ht = 0.5 * dt
    k1 = np.array( eqs_of_motion(t0       ,x0,*inputs)           )
    k2 = np.array(       eqs_of_motion(t0+ht,x0 + ht*k1,*inputs) )
    k3 = np.array(       eqs_of_motion(t0+ht,x0 + ht*k2,*inputs) )

    # calculate derivatives
    ks = (k1 + 2.*(k2 + k3) + np.array(eqs_of_motion(t0+dt,x0 + dt*k3,*inputs))) / 6.

    # update x1
    x1 = (np.array(x0) + dt*ks).tolist()

    return x1


def simulate(trim_solution, t_range, linearization, props, cg_shift, control_sys, **kwargs):
    dist_model = kwargs.get("model", {"type": "step", "params": {"dist": [1.]*8, "mag": 0.}})
    if linearization.aircraft == "BIRE":
        bire = True
    else:
        bire = False
    compressible = kwargs.get("compressible",False)
    stall = kwargs.get("stall",False)
    K = linearization.K
    # rep2D(K[1:,:],"K")
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
    r = ode(eqs_of_motion).set_integrator('dopri5',atol=1e-10,rtol=1e-10)
    x = np.zeros((len_t, 12))
    x[0, :6] = x_hat[:6]
    x[0, 6:9] = [0., 0., -H]
    x[0, 9:] = trim_solution.orient

    # add shift
    velshift = 10.
    shift = np.deg2rad(5.)
    delta_x0 = np.array([
        0.0, velshift, velshift,
        shift, shift, shift,
        0.0, 0.0, 0.0,
        shift, shift, 0.0
    ])
    delta_x0 = np.array([
        0.0, velshift, velshift,
        shift, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]) # anything more than this causes the BIRE aircraft controller to go crazy
    x[0] = x[0] + delta_x0


    # frint("t =", 0.0)
    # rep2D(x[0,:,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
    # print()
    use_rk4 = False


    if bire:
        r.set_initial_value(x[0, :], t0).set_f_params(x_hat, u_hat, props, cg_shift, linearization, True, control_sys, dist_model, tau, compressible, stall)
    else:
        r.set_initial_value(x[0, :], t0).set_f_params(x_hat, u_hat, props, cg_shift, linearization, False, control_sys, dist_model, tau, compressible, stall)
    i = 1
    if bire:
        inputs = [x_hat, u_hat, props, cg_shift, linearization, True, control_sys, dist_model, tau, compressible, stall]
    else:
        inputs = [x_hat, u_hat, props, cg_shift, linearization, False, control_sys, dist_model, tau, compressible, stall]
    z = np.zeros((len_t, 8))
    while r.successful() and t_range[i] < t_range[-1]:
        if use_rk4:
            x[i, :] = _rk4(t_range[i],x[i-1,:],dt,inputs) # 
        else:
            x[i, :] = r.integrate(r.t+dt) # 
        z[i, 0:6] = x[i, 0:6] - x_hat[0:6]
        z[i, 6] = x[i, 9] - x_hat[6]
        z[i, 7] = x[i, 10] - x_hat[7]
        # frint("t =", t_range[i])
        # rep2D(x[i,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
        # print()

        # if i >= 78000:#72: #94: # 11
        #     quit()

        i += 1
    if use_rk4:
        x[i, :] = _rk4(t_range[i],x[i-1,:],dt,inputs) # 
    else:
        x[i, :] = r.integrate(r.t+dt) # 
    z[i, 0:6] = x[i, 0:6] - x_hat[0:6]
    z[i, 6] = x[i, 9] - x_hat[6]
    z[i, 7] = x[i, 10] - x_hat[7]
    # print(t_range[i])
    # rep2D(x[i,np.newaxis].reshape((3,4)),"x",predecimals=7,decimals=16)
    # print()
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


    print("running ben sim...")
    base_file = "base_sim_input.json"
    bire_file = "bire_sim_input.json"
    if bire:
        file_name = ctrl_directory + bire_file
    else:
        file_name = ctrl_directory + base_file
    # import json file from file path
    json_string = open(file_name).read()
    # save to vals dictionary
    input_dict = json.loads(json_string)
    # get linear nonlinear parameter and initialize aircraft
    input_dict["simulation"] = input_dict.get("simulation",{})
    input_dict["simulation"]["nonlinear_dynamics"] = True
    input_dict["simulation"]["time_step[sec]"] = t_range[1] - t_range[0]
    input_dict["simulation"]["total_time[sec]"] = t_range[-1]
    if use_rk4:
        method = "rk4"
    else:
        method = "ode"
    input_dict["simulation"]["integrator"] = method
    input_dict["simulation"]["nonlinear_dynamics"] = True
    input_dict["simulation"]["use_quaternions"] = False
    input_dict["simulation"]["limit_input"] = True
    input_dict["simulation"]["limit_input_rates"] = True
    input_dict["simulation"]["include_compressibility"] = True
    input_dict["simulation"]["use_Anderson_corrections"] = True
    input_dict["simulation"]["include_stall"] = True
    input_dict["simulation"]["simulate_uncontrolled"] = False
    input_dict["simulation"]["use_fitted_thrust_model"] = True
    input_dict["simulation"]["BIRE"] = bire
    base = ben_sim(input_dict)
    mrrr = [6,7,8,11] # None # 
    mrrc = [3] # None # 
    mrrc_opp = [0,1,2]
    # # force same trim condition
    # x_trim = base.x_trim*0.
    # x_trim[:6] = trim_solution.states[:6]*1.
    # x_trim[9:11] = trim_solution.states[6:8]*1.
    # x_trim_deg = x_trim*1.
    # indices = [3,4,5,9,10,11]
    # x_trim_deg[indices] = np.rad2deg(x_trim[indices])
    # Brng = [1,2,3,0]
    # u_trim = trim_solution.inputs[Brng]*1.
    # base.x_trim = x_trim*1.
    # base.x0 = x_trim*1.
    # base.x_trim_euler_deg = x_trim_deg*1.
    # base.u_trim = u_trim
    base.run_simulation(mrrr=mrrr,save_matrices=False,delta_x0=delta_x0)#,mrrc=mrrc)
    # rep2D(base.Lin_Model.A_min,"A")
    # rep2D(base.Lin_Model.B_min[:,mrrc_opp],"B")
    # rep2D(base.Lin_Model.K[mrrc_opp,:],"K")

    # print(base.xarr.shape,"==",x.T.shape)
    # print(base.uarr.shape,"==",u.T.shape)

    # print("differences")
    # rep2D(base.Lin_Model.A_min - linearization.A,"A")
    # rep2D(base.Lin_Model.B_min - linearization.B[:,Brng],"B")
    # rep2D(base.Lin_Model.K - linearization.K[Brng,:],"K")

    x_ind = [3,4,5,9,10,11]
    x_deg = x.T*1.
    x_deg[x_ind,:] = np.rad2deg(x_deg[x_ind,:])
    u_ind = [0,1,2]
    u_deg = u.T*1.
    u_deg[u_ind,:] = np.rad2deg(u_deg[u_ind,:])

    base.zarr  = x_deg *1.
    u_w_thr = base.uarr*1.
    u_w_thr[0:3,:] = u_deg*1.
    base.varr  = u_w_thr *1.
    base.aeroz = base.aerox*0.
    Vzarr = (base.zarr[0]**2. + base.zarr[1]**2. + base.zarr[2]**2.)**0.5
    azarr = np.rad2deg(np.arctan2(base.zarr[2],base.zarr[0]))
    bzarr = np.rad2deg(np.arcsin(base.zarr[1]/Vzarr)) # experimental beta
    base.aeroz = np.array([Vzarr,azarr,bzarr])

    plot_dict = {
        "show" : False,
        "plot_full" : False,
        "plot_delta" : True,
        "zoom_deltas" : True,
        "zoom_fraction" : 0.05,
        "transparent" : False,
        "format" : "png"
    }
    plot_dict["first_set_label"] = "b"
    plot_dict["second_set_label"] = "c"
    plot_dict["plot_second_set"] = True
    plot_dict["plotting_directory"] = ctrl_directory

    # plot using plot_results
    base.plot_results(**plot_dict)
    plot_dict["plot_full"] = True
    plot_dict["plot_delta"] = False
    plot_dict["zoom_deltas"] = False
    plot_dict["plot_second_set"] = False
    base.xarr = base.xarr - base.zarr
    base.uarr = base.uarr - base.varr
    base.aerox = base.aerox - base.aeroz
    base.plot_results(**plot_dict)



def rep2D(array, name = "ans", predecimals = 5, decimals = 4):

    printname = "{} = ".format(name)
    lenname = len(printname)
    for i in range(array.shape[0]):
        if i == 0:
            print(printname,end="")
        else:
            print(" "*lenname,end="")
        
        for j in range(array.shape[1]):
            print("{:> {}.{}e}".format(array[i,j],decimals+predecimals+4,decimals),end="")
            if j != array.shape[1]-1:
                print(",",end="")
        print()
    print()

def frint(name,value,predecimals=4,decimals=16):
    print(name + " {:> {}.{}e}".format(value,decimals+predecimals+4,decimals))



if __name__ == "__main__":
    H = 15000.
    M = 0.6
    a = stdatm_english(H)[-1]
    V = M*a
    # print(V)
    Gamma = 0.1
    gamma = np.deg2rad(0.)
    phi = np.deg2rad(0.)
    t0 = 0.
    t1 = 10.
    dt = 0.01
    t_range = np.arange(t0, t1 + dt, dt)
    aero_dir = aero_directory#'/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    lqr = True
    control_sys = True
    comp = True
    stall = True
    model_gust = {"type": "gust", "params": {"A": 80.,
                                             "gamma": 1.,
                                             "w": 5.,
                                             "s_x": 1.,
                                             "s_y": 1.,
                                             "s_z": 0.5,
                                             "t_0": 10000.}}
    for shift in [0.0]:#np.arange(0., 1.1, 0.25):
        cg_shift = [shift, 0., 0.]
        print(cg_shift)
        # print("BASE")
        # bire = False
        # base_props = trim.AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire)
        # if False*exists('./Simulation Data/Baseline/Linearization/CG_' + str(cg_shift[0]) + '.lin')*exists('./Simulation Data/Baseline/Solution/CG_' + str(cg_shift[0]) + '.trim'):
        #     with open('./Simulation Data/Baseline/Linearization/CG_' + str(cg_shift[0]) + '.lin', 'rb') as lin_file:
        #         base_lin = pickle.load(lin_file)
        #     with open('./Simulation Data/Baseline/Solution/CG_' + str(cg_shift[0]) + '.trim', 'rb') as trim_file:
        #         base_solution = pickle.load(trim_file)
        # else:
        #     base_solution, base_lin = design_controller(H, M, gamma, phi, Gamma, cg_shift, bire,compressible=comp, stall=stall)
        # simulate(base_solution, t_range, base_lin, base_props, cg_shift, control_sys, model=model_gust,compressible=comp, stall=stall)
        

        print("BIRE")
        bire = True
        bire_props = trim.AircraftProperties(V, H, Gamma, path=aero_dir, bire=bire)
        if False*exists('./Simulation Data/BIRE/Linearization/CG_' + str(cg_shift[0]) + '.lin')*exists('./Simulation Data/BIRE/Solution/CG_' + str(cg_shift[0]) + '.trim'):
            with open('./Simulation Data/BIRE/Linearization/CG_' + str(cg_shift[0]) + '.lin', 'rb') as lin_file:
                bire_lin = pickle.load(lin_file)
            with open('./Simulation Data/BIRE/Solution/CG_' + str(cg_shift[0]) + '.trim', 'rb') as trim_file:
                bire_solution = pickle.load(trim_file)
        else:
            bire_solution, bire_lin = design_controller(H, M, gamma, phi, Gamma, cg_shift, bire,compressible=comp, stall=stall)
        simulate(bire_solution, t_range, bire_lin, bire_props, cg_shift, control_sys, model=model_gust,compressible=comp, stall=stall)
        print()