import sys
import json

# sim_directory = '../'
# sys.path.insert(1, sim_directory)

# aero_directory = '../../aerodynamics_model/'
# sys.path.insert(1, aero_directory)

from simulator_physics import simulator
from fit_damped_sinusoid import *

t_total = 200
ti_LP = 10.0 # 10.0 skips the short period data


V = 634 #ft/s
H = 15000 #ft
phi = 0.0

trim_type = 'sct'
cg_shift = [1.0,0.0,0.0]

BIRE = False
STALL = False
COMP = False

def cubic_poly(x,a,b,c):
    
    y = a*x*x + b*x + c
    
    return y

ti_modes = 1.0
dt_array = [0.02, 0.03, 0.04, 0.05]

de_array = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0] # in degs

dr = 0.0

dq_array = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0] # in degs/s

de_sigma_array = []
de_period_array = []

dq_sigma_array = []
dq_period_array = []

for i in range(len(de_array[:])):
    
    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)
    
    input_dict["simulation"]["total_time[sec]"] = t_total
    
    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi
    input_dict["perturbations"]["delta_de[deg]"] = de_array[i]
    input_dict["perturbations"]["delta_dr[deg]"] = dr
    input_dict["perturbations"]["delta_q[deg/s]"] = 0.0
    input_dict["initial"]["airspeed[ft/s]"] = V
    input_dict["initial"]["altitude[ft]"] = H
    input_dict["initial"]["trim"]["type"] = trim_type
    
    input_dict["aircraft"]["BIRE"] = BIRE
    input_dict["aircraft"]["CG_shift[ft]"] = cg_shift

    input_dict["aerodynamics"]["stall_model"]["use_stall_model"] = STALL
    input_dict["aerodynamics"]["compressibility_model"]["use_comp_model"] = COMP
    
    with open("simulator_input.json", "w") as outfile:
        json.dump(input_dict, outfile,indent=4)
        
    simulator_class = simulator(init_filename = 'simulator_input.json')
    simulator_class.run_sim()
    
    time = simulator_class.time_plot
    airspeed = simulator_class.airspeed_plot
    u_velo = simulator_class.u_plot
    alt = simulator_class.z_plot
    theta = simulator_class.theta_plot
    xf = simulator_class.x_plot
    
    dt = simulator_class.dt
    V0 = simulator_class.V0
    alt0 = simulator_class.H0
    theta0 = simulator_class.theta0
    u0 = simulator_class.u0
    
    Lp_Li, Lp_Ui = int(ti_LP//dt), int(200.0//dt)
    
    print('\n------Phugoid Estimate------\n')
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=True, ylabel = 'Airspeed [ft/s]')
    
    de_sigma_array.append(a)
    de_period_array.append(2*np.pi/abs(w))


for i in range(len(dq_array[:])):
    
    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)
    
    input_dict["simulation"]["total_time[sec]"] = t_total
    
    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi
    input_dict["perturbations"]["delta_de[deg]"] = 0.0
    input_dict["perturbations"]["delta_q[deg/s]"] = dq_array[i]
    input_dict["perturbations"]["delta_dr[deg]"] = dr
    input_dict["initial"]["airspeed[ft/s]"] = V
    input_dict["initial"]["altitude[ft]"] = H
    input_dict["initial"]["trim"]["type"] = trim_type
    
    input_dict["aircraft"]["BIRE"] = BIRE
    input_dict["aircraft"]["CG_shift[ft]"] = cg_shift

    input_dict["aerodynamics"]["stall_model"]["use_stall_model"] = STALL
    input_dict["aerodynamics"]["compressibility_model"]["use_comp_model"] = COMP
    
    with open("simulator_input.json", "w") as outfile:
        json.dump(input_dict, outfile,indent=4)
        
    simulator_class = simulator(init_filename = 'simulator_input.json')
    simulator_class.run_sim()
    
    time = simulator_class.time_plot
    airspeed = simulator_class.airspeed_plot
    u_velo = simulator_class.u_plot
    alt = simulator_class.z_plot
    theta = simulator_class.theta_plot
    xf = simulator_class.x_plot
    
    dt = simulator_class.dt
    V0 = simulator_class.V0
    alt0 = simulator_class.H0
    theta0 = simulator_class.theta0
    u0 = simulator_class.u0
    
    print('\n------Phugoid Estimate------\n')
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=True, ylabel = 'Airspeed [ft/s]')
    
    dq_sigma_array.append(a)
    dq_period_array.append(2*np.pi/abs(w))
    

        

popt, pcov = optimize.curve_fit(cubic_poly, de_array, de_sigma_array, xtol=1e-6)
a,b,c = popt

sigma_de = cubic_poly(0.0,a,b,c)
# sigma_0_array.append(sigma_de)

plt.figure()
# plt.scatter(de_array, de_sigma_array, label = str(dt_array[j]))
plt.scatter(de_array, de_sigma_array)
plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
plt.ylim((0.99*sigma_de, 1.01*sigma_de))
plt.ylabel('sigma')
plt.xlabel('de [deg]')
plt.legend()
plt.tight_layout()
plt.show()

popt, pcov = optimize.curve_fit(cubic_poly, de_array, de_period_array, xtol=1e-6)
a,b,c = popt

period_de = cubic_poly(0.0,a,b,c)
# period_0_array.append(period_0)

plt.figure()
# plt.scatter(de_array, de_period_array, label = str(dt_array[j]))
plt.scatter(de_array, de_period_array)
plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
plt.ylim((0.99*period_de, 1.01*period_de))
plt.ylabel('period')
plt.xlabel('de [deg]')
plt.legend()
plt.tight_layout()
plt.show()

popt, pcov = optimize.curve_fit(cubic_poly, dq_array, dq_sigma_array, xtol=1e-6)
a,b,c = popt

sigma_dq = cubic_poly(0.0,a,b,c)
# sigma_0_array.append(sigma_de)

plt.figure()
# plt.scatter(de_array, de_sigma_array, label = str(dt_array[j]))
plt.scatter(dq_array, dq_sigma_array)
plt.plot(dq_array, cubic_poly(np.array(dq_array),a,b,c))
plt.ylim((0.99*sigma_dq, 1.01*sigma_dq))
plt.ylabel('sigma')
plt.xlabel('dq [deg/s]')
plt.legend()
plt.tight_layout()
plt.show()

popt, pcov = optimize.curve_fit(cubic_poly, dq_array, dq_period_array, xtol=1e-6)
a,b,c = popt

period_dq = cubic_poly(0.0,a,b,c)
# period_0_array.append(period_0)

plt.figure()
# plt.scatter(de_array, de_period_array, label = str(dt_array[j]))
plt.scatter(dq_array, dq_period_array)
plt.plot(dq_array, cubic_poly(np.array(dq_array),a,b,c))
plt.ylim((0.99*period_dq, 1.01*period_dq))
plt.ylabel('period')
plt.xlabel('dq [deg/s]')
plt.legend()
plt.tight_layout()
plt.show()