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
real_list_f16 = []
imag_list_f16 = []

real_list_BIRE = []
imag_list_BIRE = []

'''OUR NEW METHOD'''

V = 634 #ft/s
gamma = np.deg2rad(0.0) #rad
H = 15000. #ft
cg_shift = [1., 0., 0.] #ft

bank_angle_array = np.array(np.linspace(0.0,60.0,10))

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_F16()
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_f16.append(case.eigreal)
    imag_list_f16.append(case.eigimag)

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=True, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_BIRE()    
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_BIRE.append(case.eigreal)
    imag_list_BIRE.append(case.eigimag)

'''SORT EIGENVALUES FOR CONTINUITY'''
    
real_list_f16 = np.asarray(real_list_f16)
imag_list_f16 = np.asarray(imag_list_f16)

real_list_BIRE = np.asarray(real_list_BIRE)
imag_list_BIRE = np.asarray(imag_list_BIRE)

'''WAS PREVIOUSLY -1 instead of 5:10 for all 3 cases'''

tempBR1 = np.copy(real_list_BIRE[5:10,10:12])
tempBR2 = np.copy(real_list_BIRE[5:10,9])

real_list_BIRE[5:10,9:11] = tempBR1
real_list_BIRE[5:10,11] = tempBR2

tempBR1 = np.copy(imag_list_BIRE[5:10,10:12])
tempBR2 = np.copy(imag_list_BIRE[5:10,9])

imag_list_BIRE[5:10,9:11] = tempBR1
imag_list_BIRE[5:10,11] = tempBR2




'''WITH PHILLIPS SYMMETRIC APPROXIMATIONS'''

real_list_f16_approx1 = []
imag_list_f16_approx1 = []

real_list_BIRE_approx1 = []
imag_list_BIRE_approx1 = []



for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=False, compressible=True,
                            stall=True, coords_approx=False,derivs_approx=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_F16()
    
    # case.set_phillips_approx(coords=False,derivs=True)
    
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_f16_approx1.append(case.eigreal)
    imag_list_f16_approx1.append(case.eigimag)

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=True, shss=False, compressible=True,
                            stall=True, coords_approx=False,derivs_approx=True,  cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_BIRE()

    # case.set_phillips_approx(coords=False,derivs=True)
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_BIRE_approx1.append(case.eigreal)
    imag_list_BIRE_approx1.append(case.eigimag)


'''SORT EIGENVALUES FOR CONTINUITY'''
    
real_list_f16_approx1 = np.asarray(real_list_f16_approx1)
imag_list_f16_approx1 = np.asarray(imag_list_f16_approx1)

real_list_BIRE_approx1 = np.asarray(real_list_BIRE_approx1)
imag_list_BIRE_approx1 = np.asarray(imag_list_BIRE_approx1)

tempBR1 = np.copy(real_list_BIRE_approx1[5:10,10:12])
tempBR2 = np.copy(real_list_BIRE_approx1[5:10,9])

real_list_BIRE_approx1[5:10,9:11] = tempBR1
real_list_BIRE_approx1[5:10,11] = tempBR2

tempBR1 = np.copy(imag_list_BIRE_approx1[5:10,10:12])
tempBR2 = np.copy(imag_list_BIRE_approx1[5:10,9])

imag_list_BIRE_approx1[5:10,9:11] = tempBR1
imag_list_BIRE_approx1[5:10,11] = tempBR2


'''WITH PHILLIPS COORDINATE SYSTEM APPROXIMATIONS'''

real_list_f16_approx2 = []
imag_list_f16_approx2 = []

real_list_BIRE_approx2 = []
imag_list_BIRE_approx2 = []



for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=False, compressible=True,
                            stall=True, coords_approx=True,derivs_approx=False,cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_F16()
    
    # case.set_phillips_approx(coords=True,derivs=False)
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_f16_approx2.append(case.eigreal)
    imag_list_f16_approx2.append(case.eigimag)

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=True, shss=False, compressible=True,
                            stall=True, coords_approx=True,derivs_approx=False, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    # 'My derivative method'
    # dAlpha = np.deg2rad(0.25) #rad
    # dBeta = np.deg2rad(0.25) #rad
    # dp = 0.06; #rad/s
    # dq = 0.5 * dp;
    # dr = 0.5 * dp;
    # case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    
    case.analytic_derivatives_BIRE()

    case.set_phillips_approx(coords=True,derivs=False)
    case.solve_dynamics_system()
    
    # case.plot_eigvals()
    
    real_list_BIRE_approx2.append(case.eigreal)
    imag_list_BIRE_approx2.append(case.eigimag)


'''SORT EIGENVALUES FOR CONTINUITY'''
    
real_list_f16_approx2 = np.asarray(real_list_f16_approx2)
imag_list_f16_approx2 = np.asarray(imag_list_f16_approx2)

real_list_BIRE_approx2 = np.asarray(real_list_BIRE_approx2)
imag_list_BIRE_approx2 = np.asarray(imag_list_BIRE_approx2)

tempBR1 = np.copy(real_list_BIRE_approx2[5:10,10:12])
tempBR2 = np.copy(real_list_BIRE_approx2[5:10,9])

real_list_BIRE_approx2[5:10,9:11] = tempBR1
real_list_BIRE_approx2[5:10,11] = tempBR2

tempBR1 = np.copy(imag_list_BIRE_approx2[5:10,10:12])
tempBR2 = np.copy(imag_list_BIRE_approx2[5:10,9])

imag_list_BIRE_approx2[5:10,9:11] = tempBR1
imag_list_BIRE_approx2[5:10,11] = tempBR2


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
BIRE_marker = matplotlib.markers.MarkerStyle(marker='s', fillstyle='full')
DERIVS_marker = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
COORD_marker = matplotlib.markers.MarkerStyle(marker='^', fillstyle='none')
SIM_marker = matplotlib.markers.MarkerStyle(marker='x', fillstyle='none')

xaxis_label = 'Real Component'
yaxis_label = 'Imaginary Component'

axes_linewidth = 0.6

# F16_LP_ind = [5,6]
BIRE_LP_ind = [5,6]

# F16_SP_ind = [7,8]
BIRE_SP_ind = [9,10]

# F16_DR_ind = [10,11]
BIRE_DR_ind = [7,8]

# F16_SR_ind = 4
BIRE_SR_ind = 4

# F16_RL_ind = 9
BIRE_RL_ind = 11



size = 5

plt.figure(0, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
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
    plt.scatter(real_list_BIRE[i,BIRE_LP_ind[0]], imag_list_BIRE[i,BIRE_LP_ind[0]], marker=BIRE_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_BIRE[i,BIRE_LP_ind[1]], imag_list_BIRE[i,BIRE_LP_ind[1]], marker=BIRE_marker, color='k', s = size)
    
    '''Symetric Aircraft Approximations'''
    plt.scatter(real_list_BIRE_approx2[i,BIRE_LP_ind[0]], imag_list_BIRE_approx2[i,BIRE_LP_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_BIRE_approx2[i,BIRE_LP_ind[1]], imag_list_BIRE_approx2[i,BIRE_LP_ind[1]], marker=COORD_marker, color='k', s = size)    
    
    '''Coord System Alignment Approximation'''
    plt.scatter(real_list_BIRE_approx1[i,BIRE_LP_ind[0]], imag_list_BIRE_approx1[i,BIRE_LP_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_BIRE_approx1[i,BIRE_LP_ind[1]], imag_list_BIRE_approx1[i,BIRE_LP_ind[1]], marker=DERIVS_marker, color='k', s = size)

    size += 3.5

size_new = [5., 8.5, 12., 15.5, 19., 22.5, 26.0, 29.5, 33.0, 36.5, 40.0]
LP_props = np.load('phugoid_properties_bank.npy')
sigma_LP_sim = LP_props[0,:]
period_LP_sim = LP_props[1,:]
SP_props = np.load('short_period_properties_bank.npy')
sigma_SP_sim = SP_props[0,:]
period_SP_sim = SP_props[1,:]

DR_props = np.load('dutch_roll_properties_bank.npy')
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
plt.ylim(0.04,0.105)
plt.xlim(-0.0325,0.005)
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'phugoid_bank_angle_comparison.png', dpi=250)
    
size = 5.0

plt.figure(1, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
    # Short period
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''
        
    plt.scatter(real_list_BIRE[i,BIRE_SP_ind[0]], imag_list_BIRE[i,BIRE_SP_ind[0]], marker=BIRE_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_BIRE[i,BIRE_SP_ind[1]], imag_list_BIRE[i,BIRE_SP_ind[1]], marker=BIRE_marker, color='k', s = size)
    
    plt.scatter(real_list_BIRE_approx2[i,BIRE_SP_ind[0]], imag_list_BIRE_approx2[i,BIRE_SP_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_BIRE_approx2[i,BIRE_SP_ind[1]], imag_list_BIRE_approx2[i,BIRE_SP_ind[1]], marker=COORD_marker, color='k', s = size)
    
    plt.scatter(real_list_BIRE_approx1[i,BIRE_SP_ind[0]], imag_list_BIRE_approx1[i,BIRE_SP_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_BIRE_approx1[i,BIRE_SP_ind[1]], imag_list_BIRE_approx1[i,BIRE_SP_ind[1]], marker=DERIVS_marker, color='k', s = size)

    
    size += 3.5
for i in range(len(sigma_DR_sim)):
    plt.scatter(-sigma_SP_sim[i],2*np.pi/period_SP_sim[i], marker=SIM_marker, color='r', s = size_new[i])
    
plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.91, -0.90, -0.89, -0.88, -0.87, -0.86])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.ylim(1.8,1.9)
plt.xlim(-0.86,-0.83)
# plt.legend(loc='center right')
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'short_period_bank_angle_comparison.png', dpi=250)
    
size = 5

plt.figure(2, figsize=fig_size)

for i in range(len(real_list_f16[:,5])):
    # Dutch roll
    if i == 9:
        labela = label1
        labelb = label2
        labelc = label3
    else:
        labela = ''
        labelb = '' 
        labelc = ''

    plt.scatter(real_list_BIRE[i,BIRE_DR_ind[0]], imag_list_BIRE[i,BIRE_DR_ind[0]], marker=BIRE_marker, color='k', s = size,label=labela)
    plt.scatter(real_list_BIRE[i,BIRE_DR_ind[1]], imag_list_BIRE[i,BIRE_DR_ind[1]], marker=BIRE_marker, color='k', s = size)
    
    plt.scatter(real_list_BIRE_approx2[i,BIRE_DR_ind[0]], imag_list_BIRE_approx2[i,BIRE_DR_ind[0]], marker=COORD_marker, color='k', s = size,label=labelb)
    plt.scatter(real_list_BIRE_approx2[i,BIRE_DR_ind[1]], imag_list_BIRE_approx2[i,BIRE_DR_ind[1]], marker=COORD_marker, color='k', s = size)
    
    plt.scatter(real_list_BIRE_approx1[i,BIRE_DR_ind[0]], imag_list_BIRE_approx1[i,BIRE_DR_ind[0]], marker=DERIVS_marker, color='k', s = size,label=labelc)
    plt.scatter(real_list_BIRE_approx1[i,BIRE_DR_ind[1]], imag_list_BIRE_approx1[i,BIRE_DR_ind[1]], marker=DERIVS_marker, color='k', s = size)

    
    size += 3.5
    
for i in range(len(sigma_DR_sim)):
    plt.scatter(-sigma_DR_sim[i],2*np.pi/period_DR_sim[i], marker=SIM_marker, color='r', s = size_new[i])
    
plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.axvline(x=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
# plt.xticks([-0.91, -0.90, -0.89, -0.88, -0.87, -0.86])
plt.xlabel(xaxis_label)
plt.ylabel(yaxis_label)
plt.ylim(0.25,0.6)
plt.xlim(-0.25,-0.05)
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

plt.scatter(bank_angle_array, real_list_BIRE[:,BIRE_SR_ind], marker=BIRE_marker, color='k', s = size,label=labela)

plt.scatter(bank_angle_array, real_list_BIRE_approx2[:,BIRE_SR_ind], marker=COORD_marker, color='k', s = size,label=labelb)

plt.scatter(bank_angle_array, real_list_BIRE_approx1[:,BIRE_SR_ind], marker=DERIVS_marker, color='k', s = size,label=labelc)

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
plt.scatter(bank_angle_array, real_list_BIRE[:,BIRE_RL_ind], marker=BIRE_marker, color='k', s = size,label=labela)

plt.scatter(bank_angle_array, real_list_BIRE_approx1[:,BIRE_RL_ind], marker=COORD_marker, color='k', s = size,label=labelb)

plt.scatter(bank_angle_array, real_list_BIRE_approx1[:,BIRE_RL_ind], marker=DERIVS_marker, color='k', s = size,label=labelc)

plt.axhline(y=0, color='k',linewidth=axes_linewidth)
plt.grid(visible=True)
plt.ylabel(xaxis_label)
plt.xlabel('Bank Angle ($\phi$) [deg]')
plt.legend()
plt.tight_layout()
plt.show()

if save_figs == True:
    plt.savefig(directory + 'roll_bank_angle_comparison.png', dpi=250)