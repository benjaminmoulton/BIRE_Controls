from dynamics_analysis import dynamicAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

directory = './Studies/'
save_figs = True

real_list_f16 = []
imag_list_f16 = []

real_list_BIRE = []
imag_list_BIRE = []

'''TEST BANK ANGLE STUDY'''

# V = 634 #ft/s
gamma = np.deg2rad(0.0) #rad
H = 15000. #ft
cg_shift = [1., 0., 0.] #ft

airspeed_array = np.array(np.linspace(400,900,10))
phi = 0.0
    
for i in range(len(airspeed_array)):

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(airspeed_array[i], H, dB = 0.0)
    
    case.solve_equilibrium_state(airspeed_array[i], H, gamma, phi, cg_shift)
    
    'My derivative method'
    dAlpha = np.deg2rad(0.25) #rad
    dBeta = np.deg2rad(0.25) #rad
    dp = 0.06; #rad/s
    dq = 0.5 * dp;
    dr = 0.5 * dp;
    case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    du = 1.0
    dv = 0.5 * du
    dw = du
    
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_f16.append(case.eigreal)
    imag_list_f16.append(case.eigimag)

for i in range(len(airspeed_array)):

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=True, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(airspeed_array[i], H, dB = 0.0)
    
    case.solve_equilibrium_state(airspeed_array[i], H, gamma, phi, cg_shift)
    
    'My derivative method'
    dAlpha = np.deg2rad(0.25) #rad
    dBeta = np.deg2rad(0.25) #rad
    dp = 0.06; #rad/s
    dq = 0.5 * dp;
    dr = 0.5 * dp;
    case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    du = 1.0
    dv = 0.5 * du
    dw = du
    
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_BIRE.append(case.eigreal)
    imag_list_BIRE.append(case.eigimag)

real_list_f16 = np.asarray(real_list_f16)
imag_list_f16 = np.asarray(imag_list_f16)

real_list_BIRE = np.asarray(real_list_BIRE)
imag_list_BIRE = np.asarray(imag_list_BIRE)



# tempF1 = np.copy(real_list_f16[0,8:10])
# tempF2 = np.copy(real_list_f16[0,7])

# real_list_f16[0,7:9] = tempF1
# real_list_f16[0,9] = tempF2

# tempF1 = np.copy(imag_list_f16[0,8:10])
# tempF2 = np.copy(imag_list_f16[0,7])

# imag_list_f16[0,7:9] = tempF1
# imag_list_f16[0,9] = tempF2

tempBR1 = np.copy(real_list_BIRE[0:2,10:12])
tempBR2 = np.copy(real_list_BIRE[0:2,9])

real_list_BIRE[0:2,9:11] = tempBR1
real_list_BIRE[0:2,11] = tempBR2

tempBR1 = np.copy(imag_list_BIRE[0:2,10:12])
tempBR2 = np.copy(imag_list_BIRE[0:2,9])

imag_list_BIRE[0:2,9:11] = tempBR1
imag_list_BIRE[0:2,11] = tempBR2

'''PLOTTING RESULTS'''
plt.close('all')
# set constant plotting parameters
fig_size = (4,4)

f_size = 10
plt.rcParams.update({'font.size': f_size})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['axes.axisbelow'] = True

f16_marker = matplotlib.markers.MarkerStyle(marker='^', fillstyle='none')
BIRE_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='none')

xaxis_label = 'Real Component'
yaxis_label = 'Imaginary Component'

axes_linewidth = 0.6

F16_LP_ind = [5,6]
BIRE_LP_ind = [5,6]

F16_SP_ind = [7,8]
BIRE_SP_ind = [9,10]

F16_DR_ind = [10,11]
BIRE_DR_ind = [7,8]

F16_SR_ind = 4
BIRE_SR_ind = 4

F16_RL_ind = 9
BIRE_RL_ind = 11

size = 5

plt.figure(0, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
    # Phugoid
    if i == 9:
        labelF = 'Baseline'
        labelB = 'BIRE'
    else:
        labelF = ''
        labelB = '' 

    plt.scatter(real_list_f16[i,F16_LP_ind[0]], imag_list_f16[i,F16_LP_ind[0]], marker=f16_marker, color='k', s = size,label=labelF)
    plt.scatter(real_list_f16[i,F16_LP_ind[1]], imag_list_f16[i,F16_LP_ind[1]], marker=f16_marker, color='k', s = size)
    plt.scatter(real_list_BIRE[i,BIRE_LP_ind[0]], imag_list_BIRE[i,BIRE_LP_ind[0]], marker=BIRE_marker, color='k', s = size,label=labelB)
    plt.scatter(real_list_BIRE[i,BIRE_LP_ind[1]], imag_list_BIRE[i,BIRE_LP_ind[1]], marker=BIRE_marker, color='k', s = size)

    
    size += 3.5

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.xticks([-0.0065, -0.0055, -0.0045, -0.0035])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'phugoid_airspeed.png', dpi=250)
    
size = 5.0

plt.figure(1, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
    # Short period
    if i == 9:
        labelF = 'Baseline'
        labelB = 'BIRE'
    else:
        labelF = ''
        labelB = ''  
    plt.scatter(real_list_f16[i,F16_SP_ind[0]], imag_list_f16[i,F16_SP_ind[0]], marker=f16_marker, color='k', s = size,label=labelF)
    plt.scatter(real_list_f16[i,F16_SP_ind[1]], imag_list_f16[i,F16_SP_ind[1]], marker=f16_marker, color='k', s = size)
    plt.scatter(real_list_BIRE[i,BIRE_SP_ind[0]], imag_list_BIRE[i,BIRE_SP_ind[0]], marker=BIRE_marker, color='k', s = size,label=labelB)
    plt.scatter(real_list_BIRE[i,BIRE_SP_ind[1]], imag_list_BIRE[i,BIRE_SP_ind[1]], marker=BIRE_marker, color='k', s = size)

    
    size += 3.5
    
plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.legend(loc='center right')
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'short_period_airspeed.png', dpi=250)
    
size = 5

plt.figure(2, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
    # Dutch roll
    if i == 9:
        labelF = 'Baseline'
        labelB = 'BIRE'
    else:
        labelF = ''
        labelB = ''  

    plt.scatter(real_list_f16[i,F16_DR_ind[0]], imag_list_f16[i,F16_DR_ind[0]], marker=f16_marker, color='k', s = size,label=labelF)
    plt.scatter(real_list_f16[i,F16_DR_ind[1]], imag_list_f16[i,F16_DR_ind[1]], marker=f16_marker, color='k', s = size)
    plt.scatter(real_list_BIRE[i,BIRE_DR_ind[0]], imag_list_BIRE[i,BIRE_DR_ind[0]], marker=BIRE_marker, color='k', s = size,label=labelB)
    plt.scatter(real_list_BIRE[i,BIRE_DR_ind[1]], imag_list_BIRE[i,BIRE_DR_ind[1]], marker=BIRE_marker, color='k', s = size)

    
    size += 3.5

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.axvline(x=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.91, -0.90, -0.89, -0.88, -0.87, -0.86])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'dutch_roll_airspeed.png', dpi=250)

size = 30

plt.figure(3, figsize=fig_size)
labelF = 'Baseline'
labelB = 'BIRE'
plt.scatter(airspeed_array, real_list_f16[:,F16_SR_ind], marker=f16_marker, color='k', s = size,label=labelF)
plt.scatter(airspeed_array, real_list_BIRE[:,BIRE_SR_ind], marker=BIRE_marker, color='k', s = size,label=labelB)

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.xlabel('Airspeed ($V_0$) [ft/s]')
plt.ylabel(xaxis_label)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'spiral_airspeed.png', dpi=250) 
 
plt.figure(4, figsize=fig_size)
labelF = 'Baseline'
labelB = 'BIRE'
plt.scatter(airspeed_array, real_list_f16[:,F16_RL_ind], marker=f16_marker, color='k', s = size,label=labelF)
plt.scatter(airspeed_array, real_list_BIRE[:,BIRE_RL_ind], marker=BIRE_marker, color='k', s = size,label=labelB)

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.ylabel(xaxis_label)
plt.xlabel('Airspeed ($V_0$) [ft/s]')
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'roll_airspeed.png', dpi=250)
