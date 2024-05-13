from dynamics_analysis import dynamicAnalysis
import numpy as np
import matplotlib.pyplot as plt


sigma_list = []
wd_list = []

'''TEST BANK ANGLE STUDY'''

V = 634 #ft/s
gamma = np.deg2rad(0.0) #rad
H = 15000. #ft
cg_shift = [0., 0., 0.] #ft

bank_angle_array = np.array(np.linspace(0.0,60.0,10))

for i in range(len(bank_angle_array)):
    phi = np.deg2rad(bank_angle_array[i]) #rad

    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=False, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0)
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
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
    
    sigma_list.append(case.sigma)
    wd_list.append(case.omegad)

sigma_list = np.asarray(sigma_list)
wd_list = np.asarray(wd_list)

for i in range(len(sigma_list[0,:]) - 4):

    plt.figure(i, figsize=(5,4))
    plt.grid(visible=True)
    plt.xlabel('Bank Angle (deg)')
    plt.ylabel('Sigma')
    plt.scatter(bank_angle_array,sigma_list[:,4 + i], marker='o', color='k')
    plt.tight_layout()
    plt.show()
    
    plt.figure(20 + i, figsize=(5,4))
    plt.grid(visible=True)
    plt.xlabel('Bank Angle (deg)')
    plt.ylabel('Omega_d')
    plt.scatter(bank_angle_array,wd_list[:,4 + i], marker='x', color='k')
    plt.tight_layout()
    plt.show()

