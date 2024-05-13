import sys

sim_directory = '../'
sys.path.insert(1, sim_directory)

from simulator_physics import simulator
from fit_damped_sinusoid import *
import json
import matplotlib
import matplotlib.pyplot as plt

def cubic_poly(x,a,b,c):
    
    y = a*x*x + b*x + c
    
    return y

ti_LP = 10.0 # 10.0 skips the short period data

V = 634 #ft/s
H = 15000 #ft
phi = 0.0

trim_type = 'sct'
cg_shift = [1.0,0.0,0.0]

de = 1.0
dr = 0.0

BIRE = False
STALL = False
COMP = False

t_total_range = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]

phugoid_V_damp = []
phugoid_V_period = []

phugoid_alt_damp = []
phugoid_alt_period = []

phugoid_theta_damp = []
phugoid_theta_period = []

phugoid_u_damp = []
phugoid_u_period = []

for i in range(len(t_total_range)):

    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)
    
    input_dict["simulation"]["total_time[sec]"] = t_total_range[i]
    
    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi
    input_dict["perturbations"]["delta_de[deg]"] = de
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
    
    Lp_Li, Lp_Ui = int(ti_LP//dt), int(t_total_range[i]//dt) # start at 1 sec, end at 100 sec
    Lp_Li_alt, Lp_Ui_alt = int(10.0//dt), int(t_total_range[i]//dt) # start at 1 sec, end at 100 sec
    

    # print('\n')
    print('\n------Phugoid Estimate------\n')
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=True, ylabel = 'Airspeed [ft/s]')
    phugoid_V_damp.append(a)
    phugoid_V_period.append(2*np.pi/abs(w))
    
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li_alt:Lp_Ui_alt], np.asarray(alt[Lp_Li_alt:Lp_Ui_alt]) - alt0, plot_results=True, ylabel = 'Altitude [ft]')
    phugoid_alt_damp.append(a)
    phugoid_alt_period.append(2*np.pi/abs(w))
    
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, plot_results=True, ylabel = 'Elevation Angle [deg]')
    phugoid_theta_damp.append(a)
    phugoid_theta_period.append(2*np.pi/abs(w))
    print('\n')

    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(u_velo[Lp_Li:Lp_Ui]) - u0, plot_results=True, ylabel = 'u - velocity [ft/s]')
    phugoid_u_damp.append(a)
    phugoid_u_period.append(2*np.pi/abs(w))
    print('\n')


popt, pcov = optimize.curve_fit(cubic_poly, t_total_range, phugoid_V_damp, xtol=1e-6)
a,b,c = popt

sigma_0 = cubic_poly(500.0,a,b,c)

plt.figure()
# plt.scatter(de_array, de_sigma_array, label = str(dt_array[j]))
plt.scatter(t_total_range, phugoid_V_damp, label='Airspeed')
plt.scatter(t_total_range, phugoid_alt_damp, label='Altitude')
plt.scatter(t_total_range, phugoid_theta_damp, label='Elevation Angle')
plt.scatter(t_total_range, phugoid_u_damp, label='u-velocity')
# plt.plot(t_total_range, cubic_poly(np.array(t_total_range),a,b,c))

plt.ylim((0.99*sigma_0, 1.01*sigma_0))
plt.ylabel('sigma')
plt.xlabel('t_total [s]')
plt.legend()
plt.tight_layout()
plt.show()


popt, pcov = optimize.curve_fit(cubic_poly, t_total_range, phugoid_V_period, xtol=1e-6)
a,b,c = popt

period_0 = cubic_poly(500.0,a,b,c)

plt.figure()
# plt.scatter(de_array, de_sigma_array, label = str(dt_array[j]))
plt.scatter(t_total_range, phugoid_V_period, label='Airspeed')
plt.scatter(t_total_range, phugoid_alt_period, label='Altitude')
plt.scatter(t_total_range, phugoid_theta_period, label='Elevation Angle')
plt.scatter(t_total_range, phugoid_u_period, label='u-velocity')
# plt.plot(t_total_range, cubic_poly(np.array(t_total_range),a,b,c))

plt.ylim((0.99*period_0, 1.01*period_0))
plt.ylabel('period')
plt.xlabel('t_total [s]')
plt.legend()
plt.tight_layout()
plt.show()

