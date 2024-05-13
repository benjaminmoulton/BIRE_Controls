from dynamics_analysis import dynamicAnalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''STUDY OF MODE PROPERTIES USING PHILLIPS SYMMETRY AND COORDINATE SYSTEM APPROX.'''

directory = './Studies/'
save_figs = False

label1 = 'Eq. (26)'
label2 = 'Classical C.S.'
label3 = 'Symmetric Aircraft'

'''WITHOUT PHILLIPS SYMMETRIC APPROXIMATIONS'''
real_list_F16 = []
imag_list_F16 = []


'''OUR NEW METHOD'''

V = 634 #ft/s
gamma = np.deg2rad(0.0) #rad
H = 15000. #ft
cg_shift = [1., 0.0, 0.] #ft
SHSS = False
COMP = False
STALL = False
NUM_DERIVS = False
THRUST_SIMPLE = True
# bank_angle_array = np.array(np.linspace(0.0,60.0,10))

bank_angle_array = np.array(np.linspace(0.0,60.0,11))

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=SHSS, compressible=COMP,stall=STALL,
                            coords_approx= False, derivs_approx=False, cg_shift=cg_shift,
                            simple_thrust = THRUST_SIMPLE)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)    
    case.solve_derivatives(num_derivs=NUM_DERIVS)
    case.solve_dynamics_system()

    real_list_F16.append(case.eigreal)
    imag_list_F16.append(case.eigimag)

'''SORT EIGENVALUES FOR CONTINUITY'''
    
real_list_F16 = np.asarray(real_list_F16)
imag_list_F16 = np.asarray(imag_list_F16)


'''WITH PHILLIPS SYMMETRIC APPROXIMATIONS'''

real_list_F16_approx1 = []
imag_list_F16_approx1 = []

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=SHSS, compressible=COMP, coords_approx= False, derivs_approx=True,
                            stall=STALL, cg_shift=cg_shift, simple_thrust = THRUST_SIMPLE)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)    
    case.solve_derivatives(num_derivs=NUM_DERIVS)
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_F16_approx1.append(case.eigreal)
    imag_list_F16_approx1.append(case.eigimag)


real_list_F16_approx1 = np.asarray(real_list_F16_approx1)
imag_list_F16_approx1 = np.asarray(imag_list_F16_approx1)

'''WITH PHILLIPS COORDINATE SYSTEM APPROXIMATIONS'''

real_list_F16_approx2 = []
imag_list_F16_approx2 = []


for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=SHSS, compressible=COMP, coords_approx= True, derivs_approx=False,
                            stall=STALL, cg_shift=cg_shift, simple_thrust = THRUST_SIMPLE)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)    
    case.solve_derivatives(num_derivs=NUM_DERIVS)
    case.solve_dynamics_system()
    
    real_list_F16_approx2.append(case.eigreal)
    imag_list_F16_approx2.append(case.eigimag)



'''ADD ONE MORE SET OF DATA, USING BOTH THE COORDS AND DERIVS APPROXIMATION'''

'''SORT EIGENVALUES FOR CONTINUITY'''
    
real_list_F16_approx2 = np.asarray(real_list_F16_approx2)
imag_list_F16_approx2 = np.asarray(imag_list_F16_approx2)

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

F16_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='none')
# BIRE_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='full')
DERIVS_marker = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
COORD_marker = matplotlib.markers.MarkerStyle(marker='^', fillstyle='none')
SIM_marker = matplotlib.markers.MarkerStyle(marker='x', fillstyle='none')

xaxis_label = 'Real Component'
yaxis_label = 'Imaginary Component'

axes_linewidth = 0.6

# F16_LP_ind = [5,6]
F16_LP_ind = [5,6]

# F16_SP_ind = [7,8]
F16_SP_ind = [7,8]

# F16_DR_ind = [10,11]
F16_DR_ind = [10,11]

# F16_SR_ind = 4
F16_SR_ind = 4

# F16_RL_ind = 9
F16_RL_ind = 9

size = 5

plt.figure(0, figsize=fig_size)

for i in range(len(real_list_F16[:,5])):
    # Phugoid
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''

    '''Our method'''
    plt.scatter(real_list_F16[i,F16_LP_ind[0]], imag_list_F16[i,F16_LP_ind[0]], marker=F16_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_F16[i,F16_LP_ind[1]], imag_list_F16[i,F16_LP_ind[1]], marker=F16_marker, color='k', s = size)
    
    '''Symetric Aircraft Approximations'''
    plt.scatter(real_list_F16_approx1[i,F16_LP_ind[0]], imag_list_F16_approx1[i,F16_LP_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_F16_approx1[i,F16_LP_ind[1]], imag_list_F16_approx1[i,F16_LP_ind[1]], marker=DERIVS_marker, color='k', s = size)

    '''Coord System Alignment Approximation'''
    plt.scatter(real_list_F16_approx2[i,F16_LP_ind[0]], imag_list_F16_approx2[i,F16_LP_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_F16_approx2[i,F16_LP_ind[1]], imag_list_F16_approx2[i,F16_LP_ind[1]], marker=COORD_marker, color='k', s = size)    
    


    size += 3.5

size_new = [5., 8.5, 12., 15.5, 19., 22.5, 26.0, 29.5, 33.0, 36.5, 40.0]
LP_props = np.load('phugoid_airspeed_properties_bank_F16.npy')
sigma_LP_sim = LP_props[0,:]
period_LP_sim = LP_props[1,:]
SP_props = np.load('short_period_properties_bank_F16.npy')
sigma_SP_sim = SP_props[0,:]
period_SP_sim = SP_props[1,:]

DR_props = np.load('dutch_roll_properties_bank_F16.npy')
sigma_DR_sim = DR_props[0,:]
period_DR_sim = DR_props[1,:]

# sigma_DR_sim = np.array([0.06480872363152955, 0.06535719875694464, 0.0658350302895639, 0.06601462948641822, 0.06571765420602427, 0.06543288506297679, 0.06611161478809865, 0.06945709286528785, 0.07815781864875897, 0.09667037507748884, 0.13132948966084948])
# period_DR_sim = np.array([21.412413867716737, 21.22352104687895, 20.7015274220396, 19.912788320833982, 18.945280878901865, 17.8846380105116, 16.78717178174139, 15.671455019359728, 14.522750455359773, 13.288985764787114, 11.835511072879836])


for i in range(len(sigma_LP_sim)):
    plt.scatter(-sigma_LP_sim[i],2*np.pi/period_LP_sim[i], marker=SIM_marker, color='r', s = size_new[i])

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.0045, -0.0030, -0.0015, 0.000, 0.0015])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
# plt.ylim(0.06,0.12)
# plt.xlim(-0.005,0.003)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'phugoid_bank_angle_comparison.png', dpi=250)
    
size = 5.0

plt.figure(1, figsize=fig_size)

for i in range(len(real_list_F16[:,5])):
    # Short period
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''
        
    plt.scatter(real_list_F16[i,F16_SP_ind[0]], imag_list_F16[i,F16_SP_ind[0]], marker=F16_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_F16[i,F16_SP_ind[1]], imag_list_F16[i,F16_SP_ind[1]], marker=F16_marker, color='k', s = size)
    
    plt.scatter(real_list_F16_approx2[i,F16_SP_ind[0]], imag_list_F16_approx2[i,F16_SP_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_F16_approx2[i,F16_SP_ind[1]], imag_list_F16_approx2[i,F16_SP_ind[1]], marker=COORD_marker, color='k', s = size)
    
    plt.scatter(real_list_F16_approx1[i,F16_SP_ind[0]], imag_list_F16_approx1[i,F16_SP_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_F16_approx1[i,F16_SP_ind[1]], imag_list_F16_approx1[i,F16_SP_ind[1]], marker=DERIVS_marker, color='k', s = size)

    
    size += 3.5
for i in range(len(sigma_DR_sim)):
    plt.scatter(-sigma_SP_sim[i],2*np.pi/period_SP_sim[i], marker=SIM_marker, color='r', s = size_new[i])
    
plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.91, -0.90, -0.89, -0.88, -0.87, -0.86])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
# plt.ylim(1.58,1.66)
# plt.xlim(-0.87,-0.844)
# plt.legend(loc='center right')
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'short_period_bank_angle_comparison.png', dpi=250)
    
size = 5

plt.figure(2, figsize=fig_size)

for i in range(len(real_list_F16[:,5])):
    # Dutch roll
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''

    plt.scatter(real_list_F16[i,F16_DR_ind[0]], imag_list_F16[i,F16_DR_ind[0]], marker=F16_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_F16[i,F16_DR_ind[1]], imag_list_F16[i,F16_DR_ind[1]], marker=F16_marker, color='k', s = size)
    
    plt.scatter(real_list_F16_approx2[i,F16_DR_ind[0]], imag_list_F16_approx2[i,F16_DR_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_F16_approx2[i,F16_DR_ind[1]], imag_list_F16_approx2[i,F16_DR_ind[1]], marker=COORD_marker, color='k', s = size)
    
    plt.scatter(real_list_F16_approx1[i,F16_DR_ind[0]], imag_list_F16_approx1[i,F16_DR_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_F16_approx1[i,F16_DR_ind[1]], imag_list_F16_approx1[i,F16_DR_ind[1]], marker=DERIVS_marker, color='k', s = size)

    
    size += 3.5
    
for i in range(len(sigma_DR_sim)):
    plt.scatter(-sigma_DR_sim[i],2*np.pi/period_DR_sim[i], marker=SIM_marker, color='r', s = size_new[i])
    
plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.axvline(x=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.91, -0.90, -0.89, -0.88, -0.87, -0.86])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
# plt.ylim(3.33,3.7)
# plt.xlim(-0.28,-0.15)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'dutch_roll_bank_angle_comparison.png', dpi=250)

size = 30

plt.figure(3, figsize=fig_size)
'''SPIRAL'''
labela = label1
labelb = label2
labelc = label3

plt.scatter(bank_angle_array, real_list_F16[:,F16_SR_ind], marker=F16_marker, color='k', s = size,label=labela)

plt.scatter(bank_angle_array, real_list_F16_approx2[:,F16_SR_ind], marker=COORD_marker, color='k', s = size,label=labelb)

plt.scatter(bank_angle_array, real_list_F16_approx1[:,F16_SR_ind], marker=DERIVS_marker, color='k', s = size,label=labelc)

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.xlabel('Bank Angle ($\phi$) [deg]')
plt.ylabel(xaxis_label)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'spiral_bank_angle_comparison.png', dpi=250) 
 
plt.figure(4, figsize=fig_size)
labela = label1
labelb = label2
labelc = label3
# plt.scatter(bank_angle_array, real_list_f16[:,F16_RL_ind], marker=f16_marker, color='k', s = size,label=labelF)
plt.scatter(bank_angle_array, real_list_F16[:,F16_RL_ind], marker=F16_marker, color='k', s = size,label=labela)

plt.scatter(bank_angle_array, real_list_F16_approx1[:,F16_RL_ind], marker=COORD_marker, color='k', s = size,label=labelb)

plt.scatter(bank_angle_array, real_list_F16_approx1[:,F16_RL_ind], marker=DERIVS_marker, color='k', s = size,label=labelc)

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.ylabel(xaxis_label)
plt.xlabel('Bank Angle ($\phi$) [deg]')
# plt.ylim(-1.75,-2.25)

plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'roll_bank_angle_comparison.png', dpi=250)