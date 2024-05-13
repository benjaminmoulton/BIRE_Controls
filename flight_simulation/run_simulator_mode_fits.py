from simulator_physics import simulator
from fit_damped_sinusoid import *
import numpy as np
import json

''''''
ti_SP = 3.0
ti_LP = 20.0
ti_DR = 2.0

PHI = 0.0
de = 2.0
dr = 0.0

BIRE = False

'''REWRITE INPUT FILE FOR CURRENT CASE'''

json_vals=open("simulator_input.json").read()
input_dict = json.loads(json_vals)

input_dict["initial"]["trim"]["bank_angle[deg]"] = PHI
input_dict["perturbations"]["delta_de[deg]"] = de
input_dict["perturbations"]["delta_dr[deg]"] = dr
input_dict["initial"]["trim"]["type"] = "shss"
input_dict["aircraft"]["BIRE"] = BIRE
input_dict["aircraft"]["CG_shift[ft]"] = [1.0, 0.0, 0.0]

with open("simulator_input.json", "w") as outfile:
    json.dump(input_dict, outfile,indent=4)


simulator_class = simulator(init_filename = 'simulator_input.json')
simulator_class.run_sim()

time = simulator_class.time_plot
alpha = simulator_class.alpha_plot
airspeed = simulator_class.airspeed_plot
q = simulator_class.q_plot
alt = simulator_class.z_plot
theta = simulator_class.theta_plot

dt = simulator_class.dt
V0 = simulator_class.V0
alpha0 = simulator_class.alpha0
q0 = simulator_class.q0
alt0 = simulator_class.H0
theta0 = simulator_class.theta0


sp_Li, sp_Ui = int(ti_SP//dt), int(10.0//dt) # start at 1 sec, end at 10 sec
Lp_Li, Lp_Ui = int(ti_LP//dt), int(200.0//dt) # start at 1 sec, end at 100 sec

# print('\n------Short Period Estimate------\n')
# (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], alpha[sp_Li:sp_Ui] - alpha0, plot_results=True, ylabel = 'Alpha [deg]')
# (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], q[sp_Li:sp_Ui] - q0, plot_results=True, ylabel = 'q [deg/s]')
# print('\n------Phugoid Estimate------\n')
(A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=True, ylabel = 'Airspeed [ft/s]')
# (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(alt[Lp_Li:Lp_Ui]) - alt0, plot_results=True, plot_first = False, ylabel = 'Altitude [ft]')
# (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, plot_results=True, ylabel = 'Elevation Angle [deg]')

# params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, num_sine = 4, plot_results=True, ylabel = 'Airspeed [ft/s]')

# params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(alt[Lp_Li:Lp_Ui]) - alt0, num_sine = 4, plot_results=True, ylabel = 'Altitude [ft]')

# params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, num_sine = 4, plot_results=True, ylabel = 'Elevation Angle [deg]')



# '''DUTCH ROLL'''

# ti_SP = 1.0
# ti_LP = 1.1
# ti_DR = 2.0

# PHI = 0.0
# de = 0.0
# dr = 6.0

# '''REWRITE INPUT FILE FOR CURRENT CASE'''

# json_vals=open("simulator_input.json").read()
# input_dict = json.loads(json_vals)

# input_dict["initial"]["trim"]["bank_angle[deg]"] = PHI
# input_dict["perturbations"]["delta_de[deg]"] = de
# input_dict["perturbations"]["delta_dr[deg]"] = dr

# with open("simulator_input.json", "w") as outfile:
#     json.dump(input_dict, outfile,indent=4)

# # Reinitialize the simulator class (wasn't working without this)
# # and run the fit to Dutch roll related data using a rudder input
# simulator_class = simulator(init_filename = 'simulator_input.json')
# dt = simulator_class.dt
# simulator_class.run_sim()

# time = simulator_class.time_plot
# beta = simulator_class.beta_plot
# beta0 = simulator_class.beta0
# phi = simulator_class.phi_plot
# phi0 = simulator_class.phi0

# dr_Li, dr_Ui = int(ti_DR//dt), int(20.0//dt) # start at 1 sec, end at 20 sec
# print('\n------Dutch Roll Estimate------\n')
# # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], beta[dr_Li:dr_Ui] - beta0, plot_results=True, ylabel = 'Beta [deg]')

# # # print('\n------Dutch Roll Estimate------\n')
# # # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], phi[dr_Li:dr_Ui] - phi0, plot_results=True, ylabel = 'Phi [deg]')

# num_sine = 4
# params = fit_mult_sinusoid(time[dr_Li:dr_Ui], np.asarray(beta[dr_Li:dr_Ui]) - beta0, num_sine = num_sine, plot_results=True, ylabel = 'Beta [deg]')








# '''TEST HILBERT METHOD'''

# import scipy.signal as signal
# import scipy
# import matplotlib.pyplot as plt

# t = time[Lp_Li:Lp_Ui]
# data = np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0
# fs = len(data)
# analytic_signal = signal.hilbert(data)
# amplitude_envelope = np.abs(analytic_signal)
# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_frequency = (np.diff(instantaneous_phase) /
#                            (2.0*np.pi) * fs)



# fig, (ax0, ax1) = plt.subplots(nrows=2)
# ax0.plot(t, data, label='signal')
# ax0.plot(t, amplitude_envelope, label='envelope')
# ax0.set_xlabel("time in seconds")
# ax0.legend()
# ax1.plot(t[1:], instantaneous_frequency)
# ax1.set_xlabel("time in seconds")
# ax1.set_ylim(0.0, 120.0)
# fig.tight_layout()


# A1 = amplitude_envelope[6000]
# A2 = amplitude_envelope[20000]

# T1 = t[6000]
# T2 = t[20000]

# tau = T2 - T1

# B = (1/tau)*np.log(A1/A2)

# zeta = (np.log(A1/A2))/(np.sqrt((2*np.pi*w*tau)**2 + (np.log(A1/A2))**2))


# amp_slice = amplitude_envelope[6000:16000]

# t_slice = t[6000:16000]


# fit_data = np.log(amp_slice)

# res = scipy.stats.linregress(t_slice,fit_data)

# slope = res.slope
# intercept = res.intercept