from simulator_physics import simulator
from fit_damped_sinusoid import *
import json
import matplotlib
import matplotlib.pyplot as plt

ti_SP = 1.0
ti_LP = 1.1
ti_DR = 2.0

phi_range = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0]
# phi_range = [30.0, 36.0, 42.0, 48.0, 54.0, 60.0]
# phi_range = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

DR_beta_damp = []
DR_beta_period = []

DR_phi_damp = []
DR_phi_period = []

DR_v_damp = []
DR_v_period = []

DR_p_damp = []
DR_p_period = []

DR_r_damp = []
DR_r_period = []


full_results = []

for i in range(len(phi_range)):
    

    de = 0.0
    dr = 6.0
    
    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)
    
    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi_range[i]
    input_dict["perturbations"]["delta_de[deg]"] = de
    input_dict["perturbations"]["delta_dr[deg]"] = dr
    
    with open("simulator_input.json", "w") as outfile:
        json.dump(input_dict, outfile,indent=4)



    '''Dutch roll study'''

    simulator_class = simulator(init_filename = 'simulator_input.json')
    dt = simulator_class.dt
    simulator_class.run_sim()
    
    time = simulator_class.time_plot
    beta = simulator_class.beta_plot
    beta0 = simulator_class.beta0
    phi = simulator_class.phi_plot
    phi0 = simulator_class.phi0
    p0 = simulator_class.p0
    p = simulator_class.p_plot
    
    dr_Li, dr_Ui = int(ti_DR//dt), int(20.0//dt) 
    # print('\n')
    # print('\n------Dutch Roll Estimate------\n')
    # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], beta[dr_Li:dr_Ui] - beta0, plot_results=True, ylabel = 'Beta [deg]')
    
    # print('\n------Dutch Roll Estimate------\n')
    # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], phi[dr_Li:dr_Ui] - phi0, plot_results=True, ylabel = 'Phi [deg]')
    
    # print('\n------Dutch Roll Estimate------\n')
    # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], p[dr_Li:dr_Ui] - p0, plot_results=True, ylabel = 'p [deg/s]')
    
    # dutch_roll_damp.append(a)
    # dutch_roll_period.append(2*np.pi/abs(w))
    # print('\n')
    
    
    print('\n')
    print('\n------Dutch Roll Estimate------\n')
    num_sine = 4
    params = fit_mult_sinusoid(time[dr_Li:dr_Ui], np.asarray(beta[dr_Li:dr_Ui]) - beta0, num_sine = num_sine, plot_results=True, ylabel = 'Beta [deg]')
    DR_beta_damp.append(params[0,1])
    DR_beta_period.append(2*np.pi/abs(params[0,2]))
    
    params = fit_mult_sinusoid(time[dr_Li:dr_Ui], np.asarray(phi[dr_Li:dr_Ui]) - phi0, num_sine = num_sine, plot_results=True, ylabel = 'Phi [deg]')
    DR_phi_damp.append(params[0,1])
    DR_phi_period.append(2*np.pi/abs(params[0,2]))
    
    params = fit_mult_sinusoid(time[dr_Li:dr_Ui], np.asarray(p[dr_Li:dr_Ui]) - p0, num_sine = num_sine, plot_results=True, ylabel = 'p [deg/s]')
    DR_p_damp.append(params[0,1])
    DR_p_period.append(2*np.pi/abs(params[0,2]))
    # print('\n')

        
# np.save('phugoid_airspeed_properties_bank_F16.npy', np.asarray([phugoid_V_damp,phugoid_V_period]))        
# np.save('phugoid_elevation_properties_bank_F16.npy', np.asarray([phugoid_elev_damp,phugoid_elev_period])) 

# phu_air_data = np.load('phugoid_airspeed_properties_bank_F16.npy')
# phu_elev_data = np.load('phugoid_elevation_properties_bank_F16.npy')


DR_beta_damp = np.array(DR_beta_damp)
DR_beta_period = np.array(DR_beta_period)

DR_phi_damp = np.array(DR_phi_damp)
DR_phi_period = np.array(DR_phi_period)

DR_v_damp = np.array(DR_v_damp)
DR_v_period = np.array(DR_v_period)

DR_p_damp = np.array(DR_p_damp)
DR_p_period = np.array(DR_p_period)

DR_r_damp = np.array(DR_r_damp)
DR_r_period = np.array(DR_r_period)


fig_size = (4,4)
size = 5
axes_linewidth = 0.6
label1 = 'Sideslip Fit'
label2 = 'Bank Angle Fit'
label3 = 'Roll Rate Fit'

xaxis_label = 'Real Component'
yaxis_label = 'Imaginary Component'

beta_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='full')
phi_marker = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
p_marker = matplotlib.markers.MarkerStyle(marker='x', fillstyle='none')


plt.figure(figsize=fig_size)

for i in range(len(phi_range[:])):
    # Phugoid
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''

    plt.scatter(-DR_beta_damp[i], 2*np.pi/DR_beta_period[i], marker=beta_marker, color='k', s = size, label=labela)
    
    plt.scatter(-DR_phi_damp[i], 2*np.pi/DR_phi_period[i], marker=phi_marker, color='b', s = size, label=labelb)
    
    plt.scatter(-DR_p_damp[i], 2*np.pi/DR_p_period[i], marker=p_marker, color='g', s = size, label=labelc)
    
    # if phugoid_alt_period[i] > 50.0:
    #     plt.scatter(-phugoid_alt_damp[i],2*np.pi/phugoid_alt_period[i], marker=ALT_marker, color='r', s = size,label=labelb)

    # plt.scatter(-phugoid_elev_damp[i], 2*np.pi/phugoid_elev_period[i], marker=ELEV_marker, color='b', s = size, label=labelc)
    
    # plt.scatter(-phu_elev_data[1,i], 2*np.pi/phu_elev_data[1,i], marker=V_marker, color='g', s = size, label=labelc)


    size += 3.5

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.0045, -0.0030, -0.0015, 0.000, 0.0015])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
# plt.ylim(0.06,0.12)
# plt.xlim(-0.006,0.002)
plt.title('Dutch Roll')
plt.legend()
plt.tight_layout()
plt.show()

