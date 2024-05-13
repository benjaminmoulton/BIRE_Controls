from simulator_physics import simulator
from fit_damped_sinusoid import *
import json
import matplotlib
import matplotlib.pyplot as plt


t_total = 500
ti_LP = 10.0

V = 634 #ft/s
H = 15000 #ft
trim_type = 'sct'
cg_shift = [1.0,0.0,0.0]

de = 1.0
dr = 0.0
dq = 0.0 # deg/s

BIRE = False
STALL = False
COMP = False

PLOT_ITERM_RESULTS = False

phi_range = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0]
# phi_range = [30.0, 36.0, 42.0, 48.0, 54.0, 60.0]
# phi_range = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
# phi_range = [0.0, 10.0]

# phi_range = [0.0]

phugoid_V_damp = []
phugoid_V_period = []

phugoid_alt_damp = []
phugoid_alt_period = []

phugoid_elev_damp = []
phugoid_elev_period = []

phugoid_u_damp = []
phugoid_u_period = []

phugoid_xf_damp = []
phugoid_xf_period = []

full_results = []

for i in range(len(phi_range)):


    
    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)
    
    input_dict["simulation"]["total_time[sec]"] = t_total
    
    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi_range[i]
    input_dict["perturbations"]["delta_de[deg]"] = de
    input_dict["perturbations"]["delta_dr[deg]"] = dr
    input_dict["perturbations"]["delta_q[deg/s]"] = dq
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
    # simulator_class.dde = np.deg2rad(10.0)
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
    
    Lp_Li, Lp_Ui = int(ti_LP//dt), int(t_total//dt) # start at 1 sec, end at 100 sec
    Lp_Li_alt, Lp_Ui_alt = int(10.0//dt), int(t_total//dt) # start at 1 sec, end at 100 sec
    

    print('\n------Phugoid Estimate------\n')
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=PLOT_ITERM_RESULTS, ylabel = 'Airspeed [ft/s]')
    phugoid_V_damp.append(a)
    phugoid_V_period.append(2*np.pi/abs(w))
    
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li_alt:Lp_Ui_alt], np.asarray(alt[Lp_Li_alt:Lp_Ui_alt]) - alt0, plot_results=PLOT_ITERM_RESULTS, ylabel = 'Altitude [ft]')
    phugoid_alt_damp.append(a)
    phugoid_alt_period.append(2*np.pi/abs(w))
    
    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, plot_results=PLOT_ITERM_RESULTS, ylabel = 'Elevation Angle [deg]')
    phugoid_elev_damp.append(a)
    phugoid_elev_period.append(2*np.pi/abs(w))
    print('\n')

    (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(u_velo[Lp_Li:Lp_Ui]) - u0, plot_results=PLOT_ITERM_RESULTS, ylabel = 'u - velocity [ft/s]')
    phugoid_u_damp.append(a)
    phugoid_u_period.append(2*np.pi/abs(w))
    print('\n')
    
    
    # '''FIT THE PHUGOID DATA USING MULTIPLE SINE WAVES'''
    # print('\n')
    # print('\n------Phugoid Estimate - Multiple Sinusoids------\n')
    # num_sine = 4
    
    # '''This slice index is for fitting data starting at 1.0 second and going to 100 seconds'''
    # slice_ind = [(0,-1), (5,800), (3000,8000), (300,500)] # F-16 phi = 0.0, both airspeed and elevation angle
    
    # params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, slice_ind, num_sine = num_sine, plot_results=True, ylabel = 'Airspeed [ft/s]')
    # phugoid_V_damp.append(params[0,1])
    # phugoid_V_period.append(2*np.pi/abs(params[0,2]))
    
    # # params = fit_mult_sinusoid(time[Lp_Li_alt:Lp_Ui_alt], np.asarray(alt[Lp_Li_alt:Lp_Ui_alt]) - alt0, num_sine = num_sine, plot_results=True, ylabel = 'Altitude [ft]')
    # # phugoid_alt_damp.append(params[0,1])
    # # phugoid_alt_period.append(2*np.pi/abs(params[0,2]))

    # # params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, num_sine = num_sine, plot_results=True, ylabel = 'Elevation Angle [deg]')
    # # phugoid_elev_damp.append(params[0,1])
    # # phugoid_elev_period.append(2*np.pi/abs(params[0,2]))
    # # print('\n')

    # params = fit_mult_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(u_velo[Lp_Li:Lp_Ui]) - u0, slice_ind, num_sine = num_sine, plot_results=True, ylabel = 'u - Velocity [ft/s]')
    # phugoid_u_damp.append(params[0,1])
    # phugoid_u_period.append(2*np.pi/abs(params[0,2])) 
    
    # print('\n')
        
        
# np.save('phugoid_airspeed_properties_bank_F16.npy', np.asarray([phugoid_V_damp,phugoid_V_period]))        
# np.save('phugoid_altitude_properties_bank_F16.npy', np.asarray([phugoid_alt_damp,phugoid_alt_period]))
# np.save('phugoid_elevation_properties_bank_F16.npy', np.asarray([phugoid_elev_damp,phugoid_elev_period]))
# np.save('phugoid_u_velo_properties_bank_F16.npy', np.asarray([phugoid_u_damp,phugoid_u_period]))

# phu_air_data = np.load('phugoid_airspeed_properties_bank_F16.npy')
# phu_elev_data = np.load('phugoid_elevation_properties_bank_F16.npy')

phugoid_V_damp = np.array(phugoid_V_damp)       
phugoid_V_period = np.array(phugoid_V_period)

phugoid_alt_damp = np.array(phugoid_alt_damp)       
phugoid_alt_period = np.array(phugoid_alt_period)

# ind_del = np.argwhere(phugoid_alt_period[:] < 50.0)

phugoid_elev_damp = np.array(phugoid_elev_damp)       
phugoid_elev_period = np.array(phugoid_elev_period)

phugoid_u_damp = np.array(phugoid_u_damp)
phugoid_u_period = np.array(phugoid_u_period)


fig_size = (4,4)
size = 5
axes_linewidth = 0.6
label1 = 'Airspeed Fit'
label2 = 'Altitude Fit'
label3 = 'Elevation Fit'
label4 = 'u-velocity Fit'

xaxis_label = 'Real Component'
yaxis_label = 'Imaginary Component'

V_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='full')
ALT_marker = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
ELEV_marker = matplotlib.markers.MarkerStyle(marker='x', fillstyle='none')
u_velo_marker = matplotlib.markers.MarkerStyle(marker='^', fillstyle='none')


plt.figure(figsize=fig_size)

for i in range(len(phi_range[:])):
    # Phugoid
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
        labeld = label4
    else:
        labela = ''
        labelb = '' 
        labelc = ''
        labeld = ''

    plt.scatter(-phugoid_V_damp[i], 2*np.pi/phugoid_V_period[i], marker=V_marker, color='k', s = size, label=labela)
    
    # plt.scatter(-phu_air_data[0,i], 2*np.pi/phu_air_data[0,i], marker=V_marker, color='g', s = size, label=labela)
    
    if phugoid_alt_period[i] > 50.0:
        plt.scatter(-phugoid_alt_damp[i],2*np.pi/phugoid_alt_period[i], marker=ALT_marker, color='r', s = size,label=labelb)

    # plt.scatter(-phugoid_elev_damp[i], 2*np.pi/phugoid_elev_period[i], marker=ELEV_marker, color='b', s = size, label=labelc)
    
    plt.scatter(-phugoid_elev_damp[i], 2*np.pi/phugoid_elev_period[i], marker=V_marker, color='g', s = size, label=labelc)
        
    plt.scatter(-phugoid_u_damp[i], 2*np.pi/phugoid_u_period[i], marker=u_velo_marker, color='k', s = size, label=labeld)

    size += 3.5

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.0045, -0.0030, -0.0015, 0.000, 0.0015])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.ylim(0.06,0.12)
plt.xlim(-0.006,0.002)
plt.title('Phugoid')
plt.legend()
plt.tight_layout()
plt.show()









# fig_size = (4,4)
# size = 5
# axes_linewidth = 0.6
# label1 = 'u - Velocity Fit'

# xaxis_label = 'Real Component'
# yaxis_label = 'Imaginary Component'

# u_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='full')

# plt.figure(figsize=fig_size)

# for i in range(len(phi_range[:])):
#     # Phugoid
#     if i == 0:
#         labela = label1
#     else:
#         labela = ''

#     plt.scatter(-phugoid_u_damp[i], 2*np.pi/phugoid_u_period[i], marker=u_marker, color='k', s = size, label=labela)
#     size += 3.5

# plt.axhline(y=0, color='k',linewidth=axes_linewidth)
# plt.grid(visible=True)
# # plt.xticks([-0.0045, -0.0030, -0.0015, 0.000, 0.0015])
# plt.xlabel(xaxis_label)
# plt.ylabel(yaxis_label)
# plt.ylim(0.06,0.12)
# plt.xlim(-0.006,0.002)
# plt.title('Phugoid')
# plt.legend()
# plt.tight_layout()
# plt.show()

