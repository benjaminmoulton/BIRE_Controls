from simulator_physics import simulator, rewrite_input_file
from fit_damped_sinusoid import *
import json
import matplotlib
import matplotlib.pyplot as plt
import sys

dynam_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/dynamic_stability/'
sys.path.insert(1, dynam_directory)

import numpy as np
from dynamics_analysis import dynamicAnalysis
from fit_damped_sinusoid import damped_sinusoid, estimate_amp, normalize_sim_data, get_peak_align_index




BIRE = False

V = 634 #ft/s
gamma = 0.0 #deg
phi = 0.0 #deg
H = 15000. #ft
cg_shift = [1.0, 0.0, 0.0] #ft

SHSS = False
COMP = False
STALL = False
NUM_DERIVS = False
SIMPLE_THRUST = True

if SHSS == False:
    trim_type = 'sct'
else:
    trim_type = 'shss'

ti_SP = 1.1
ti_LP = 1.1
ti_DR = 2.0
t_total = 400

de = 0.0
dr = 0.0
dq = 1.0


'''FLIGHT SIMULATION DATA'''

rewrite_input_file(t_total = t_total, phi = phi, V = V, H = H, trim_type = trim_type,
                   de = de, dr = dr, dq = dq, BIRE = BIRE, cg_shift = cg_shift, STALL = STALL,
                   COMP = COMP, SIMPLE_THRUST = SIMPLE_THRUST, filename='simulator_input.json')


simulator_class = simulator(init_filename = 'simulator_input.json')
simulator_class.run_sim()

dt = simulator_class.dt
time = simulator_class.time_plot
airspeed = simulator_class.airspeed_plot
alt = simulator_class.z_plot

u_velo = simulator_class.u_plot
v_velo = simulator_class.v_plot
w_velo = simulator_class.w_plot

p = simulator_class.p_plot
q = simulator_class.q_plot
r = simulator_class.r_plot

x_pos = simulator_class.x_plot
theta = simulator_class.theta_plot


'''DYNAMIC MODE DATA'''

directory = './Studies/'

case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eigenvector_comparision_test.txt',
                        BIRE=BIRE, shss=SHSS, compressible=COMP,
                        stall=STALL, cg_shift=cg_shift, simple_thrust= SIMPLE_THRUST)
    
case.update_aircraft_properties(V, H, dB = 0.0)
case.solve_equilibrium_state(V, H, np.deg2rad(gamma), np.deg2rad(phi), cg_shift)    
case.solve_derivatives(num_derivs=NUM_DERIVS)
case.solve_dynamics_system()

eig_val_real = case.eigreal
eig_val_imag = case.eigimag
omega_d = case.omegad
period = case.period
sigma = case.sigma

eig_vecs = case.eigvecs
amps = case.amps
phases = case.phase

LP_index = 5
SP_index = 7
DR_index = 10

LP_amps = amps[:,LP_index]
LP_phases = phases[:,LP_index]

SP_amps = amps[:,SP_index]
SP_phases = phases[:,SP_index]

DR_amps = amps[:,DR_index]
DR_phases = phases[:,DR_index]


sp_Li, sp_Ui = int(ti_SP//dt), int(10.0//dt) # start at 1 sec, end at 10 sec
dr_Li, dr_Ui = int(ti_DR//dt), int(20.0//dt)

'''u,v,w,p,q,r,xf,yf,zf,phi,theta,psi'''
labels = ['\u0394u','\u0394v','\u0394w','\u0394p','\u0394q','\u0394r','\u0394xf','\u0394yf','\u0394zf','\u0394\u03C6','\u0394\u03B8','\u0394\u03C8']
LP_all = np.zeros((12,len(time)))
SP_all = np.zeros((12,len(time)))
DR_all = np.zeros((12,len(time)))

SP_shift = 0.0
SP_p = damped_sinusoid(time, SP_amps[3], sigma[SP_index], omega_d[SP_index], SP_phases[3] + SP_shift, z = 0.0)
SP_q = damped_sinusoid(time, SP_amps[4], sigma[SP_index], omega_d[SP_index], SP_phases[4] + SP_shift, z = 0.0)
SP_r = damped_sinusoid(time, SP_amps[5], sigma[SP_index], omega_d[SP_index], SP_phases[5] + SP_shift, z = 0.0)

'''---------------PLOT LONG PERIOD---------------'''

'''Plot sinusoids of the phugoid eivenvect components'''
plt.figure(0)
for i in range(12):
    LP_all[i] = damped_sinusoid(time, LP_amps[i], sigma[LP_index], omega_d[LP_index], LP_phases[i], z = 0.0)
    plt.plot(time,LP_all[i])
plt.legend(labels,loc='right')
plt.title('Long-Period (Linear)')
plt.ylabel('Relative Amplitude')
plt.xlabel('Time [s]')
plt.show()

# LP_amp_1,_ = estimate_amp(time,alt)



'''SINUSOID FROM THE LINEAR METHOD'''
zf = damped_sinusoid(time, 1.0, sigma[LP_index], omega_d[LP_index], -np.pi/2, z = 0.0) #amplitude of 1

'''Normalize data'''
zf = normalize_sim_data(zf)
alt_norm = normalize_sim_data(alt)

'''Align first sinusoidal peak'''
i_zero = get_peak_align_index(alt_norm, zf)


t_start_i = i_zero
# i_init = i_zero
# i_final = int(50//dt)

# lin_max = max(abs(zf[i_init:i_final]))
# sim_max = max(abs(alt_norm[i_init:i_final]))
# ratio = lin_max/sim_max

# diff = lin_max - sim_max

'''Try Adding u-velo to the plot'''

u_lin = damped_sinusoid(time, 1.0, sigma[LP_index], omega_d[LP_index], -np.pi/2, z = 0.0) #amplitude of 1

'''Normalize data'''
u_lin = normalize_sim_data(u_lin)
u_norm = normalize_sim_data(u_velo)

'''Align first sinusoidal peak'''
# i_zero = get_peak_align_index(w_norm, w_lin)
# i_zero = 0

t_start_i = i_zero


'''KEEP PLAYING WITH THIS SCALING AND ALIGNING THING'''

plt.figure(1)
# plt.plot(time, zf/max(abs(zf)))
plt.plot(time, zf)
plt.plot(np.array(time[t_start_i:]) - time[t_start_i], (alt_norm[t_start_i:]))
plt.plot(time, np.ones(len(zf))*np.mean(alt_norm), color='r')
plt.plot(np.array(time[t_start_i:]) - time[t_start_i], (u_norm[t_start_i:]))
# plt.plot(np.array(time[t_start_i:]) - time[t_start_i], ratio*(alt_norm[t_start_i:] - diff))
plt.legend(['Linear','SIM'])
plt.ylabel('Normalized Altitude')
plt.show()

'''
Procedure:
    - generate linear data using the damped sinusoid function and linear omega and sigma
    - normalize flight sim data WRT difference between max and min values
    - find indices of the max ABS of lin and SIM data
    - shift the sim data using the difference in the previously determined indices (new zero of sim)
    
'''



'''SINUSOID FROM THE LINEAR METHOD'''
w_lin = damped_sinusoid(time, 1.0, sigma[SP_index], omega_d[SP_index], -np.pi/2, z = 0.0) #amplitude of 1

'''Normalize data'''
w_lin = normalize_sim_data(w_lin)
w_norm = normalize_sim_data(w_velo)

'''Align first sinusoidal peak'''
i_zero = get_peak_align_index(w_norm, w_lin)
# i_zero = 0

t_start_i = i_zero


plt.figure(2)
# plt.plot(time, zf/max(abs(zf)))
plt.plot(time, w_lin)
plt.plot(np.array(time[t_start_i:]) - time[t_start_i], (w_norm[t_start_i:]))
plt.plot(time, np.ones(len(w_lin))*np.mean(w_norm), color='r')
# plt.plot(np.array(time[t_start_i:]) - time[t_start_i], ratio*(alt_norm[t_start_i:] - diff))
plt.legend(['Linear','SIM'])
plt.ylabel('Normalized w-velocity')
plt.show()


'''----------TEST PLOTTING ROTATION RATES RELATIVE TO THE LARGEST VALUE----------'''

max_rot_rate = max(np.array([max(abs(p)),
                             max(abs(q)),
                             max(abs(r))]))

p_norm = normalize_sim_data(p, max_rot_rate)
q_norm = normalize_sim_data(q, max_rot_rate)
r_norm = normalize_sim_data(r, max_rot_rate)

plt.figure(3)
# plt.plot(time, zf/max(abs(zf)))
plt.plot(time, p_norm)
plt.plot(time, q_norm)
plt.plot(time, r_norm)
plt.legend(['p-norm','q-norm','r-norm'])
plt.ylabel('Normalized Relative Rotation Rates')
plt.show()

max_rot_rate = max(np.array([max(abs(SP_p)),
                             max(abs(SP_q)),
                             max(abs(SP_r))]))


SP_p_norm = normalize_sim_data(SP_p, max_rot_rate)
SP_q_norm = normalize_sim_data(SP_q, max_rot_rate)
SP_r_norm = normalize_sim_data(SP_r, max_rot_rate)

plt.figure(4)
# plt.plot(time, zf/max(abs(zf)))
plt.plot(time, SP_p_norm)
plt.plot(time, SP_q_norm)
plt.plot(time, SP_r_norm)
plt.legend(['p-norm','q-norm','r-norm'])
plt.ylabel('Normalized Relative Rotation Rates')
plt.show()


# '''PLOT SHORT PERIOD'''

# '''Plot sinusoids of the phugoid eivenvect components'''
# plt.figure(2)
# for i in range(12):
#     SP_all[i] = damped_sinusoid(time[:], SP_amps[i], sigma[SP_index], omega_d[SP_index], SP_phases[i], z = 0.0)
#     plt.plot(time[sp_Li:sp_Ui],SP_all[i][sp_Li:sp_Ui])
# plt.legend(labels,loc='right')
# plt.show()

# SP_amp_1,_ = estimate_amp(time,w_velo)

# '''w VELO FROM SIM DATA'''
# w_norm = ((w_velo - w_velo[0])/max(abs(w_velo - w_velo[0]))) #normalize relative to the maximum
# t_start_i = 100


# '''SINUSOID FROM THE LINEAR METHOD'''
# w = damped_sinusoid(time, 1.0, sigma[SP_index], omega_d[SP_index], -np.pi/2, z = 0.0) #amplitude of 1

# plt.figure(3)
# plt.plot(time[:sp_Ui], (w/max(abs(w)))[:sp_Ui])
# plt.plot(np.array(time[t_start_i:sp_Ui]) - time[t_start_i], w_norm[t_start_i:sp_Ui])
# plt.legend(['Linear','SIM'])
# plt.ylabel('Normalized w - velo')
# plt.show()


# '''PLOT DUTCH ROLL'''

# '''Plot sinusoids of the phugoid eivenvect components'''
# plt.figure(4)
# for i in range(12):
#     DR_all[i] = damped_sinusoid(time[:], DR_amps[i], sigma[DR_index], omega_d[DR_index], DR_phases[i], z = 0.0)
#     plt.plot(time[dr_Li:dr_Ui],DR_all[i][dr_Li:dr_Ui])
# plt.legend(labels,loc='right')
# plt.show()

# DR_amp_1,_ = estimate_amp(time,v_velo)

# '''v VELO FROM SIM DATA'''
# v_norm = ((v_velo - v_velo[0])/max(abs(v_velo - v_velo[0]))) #normalize relative to the maximum
# t_start_i = 100

# # 
# # '''SINUSOID FROM THE LINEAR METHOD'''
# v = damped_sinusoid(time, 1.0, sigma[DR_index], omega_d[DR_index], -np.pi/2, z = 0.0) #amplitude of 1

# plt.figure(5)
# plt.plot(time[:dr_Ui], (v/max(abs(v)))[:dr_Ui])
# plt.plot(np.array(time[t_start_i:dr_Ui]) - time[t_start_i], v_norm[t_start_i:dr_Ui])
# plt.legend(['Linear','SIM'])
# plt.ylabel('Normalized v - velo')
# plt.show()