from simulator_physics import simulator
from fit_damped_sinusoid import *
import json

ti_SP = 2.0
ti_LP = 10.0
ti_DR = 2.0

phi_range = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0]

studies = [0,1]

phugoid_damp = []
phugoid_period = []

short_damp = []
short_period = []

dutch_roll_damp = []
dutch_roll_period = []

full_results = []

BIRE = False

for i in range(len(phi_range)):
    
    json_vals=open("simulator_input.json").read()
    input_dict = json.loads(json_vals)

    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi_range[i]
    input_dict["aircraft"]["CG_shift[ft]"] = [1.0, 0.0, 0.0]
    input_dict["initial"]["trim"]["type"] = "sct"
    input_dict["aircraft"]["BIRE"] = BIRE

    with open("simulator_input.json", "w") as outfile:
        json.dump(input_dict, outfile,indent=4)


    if 0 in studies:
        '''Phugoid STUDY'''

        simulator_class = simulator(init_filename = 'simulator_input.json')
        simulator_class.dde = np.deg2rad(2.0)
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
        Lp_Li, Lp_Ui = int(ti_LP//dt), int(120.0//dt) # start at 1 sec, end at 100 sec
        
        print('\n')
        print('\n------Short Period Estimate------\n')
        (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], alpha[sp_Li:sp_Ui] - alpha0, plot_results=True, ylabel = 'Alpha [deg]')
        # (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], q[sp_Li:sp_Ui] - q0, plot_results=True, ylabel = 'q [deg/s]')
        # (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], np.asarray(theta[sp_Li:sp_Ui]) - theta0, plot_results=True, ylabel = 'Elevation Angle [deg]')
        short_damp.append(a)
        short_period.append(2*np.pi/abs(w))
        print('\n')
        print('\n------Phugoid Estimate------\n')
        (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=True, ylabel = 'Airspeed [ft/s]')
        # (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(alt[Lp_Li:Lp_Ui]) - alt0, plot_results=True, ylabel = 'Altitude [ft]')
        # (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(theta[Lp_Li:Lp_Ui]) - theta0, plot_results=True, ylabel = 'Elevation Angle [deg]')
        phugoid_damp.append(a)
        phugoid_period.append(2*np.pi/abs(w))
        print('\n')
        

    
    if 1 in studies:
        '''Dutch roll study'''

        # Reinitialize the simulator class (wasn't working without this)
        # and run the fit to Dutch roll related data using a rudder input
        simulator_class = simulator(init_filename = 'simulator_input.json')
        dt = simulator_class.dt
        simulator_class.dde = 0.0 
        simulator_class.ddr = np.deg2rad(8.0)
        simulator_class.run_sim()
        
        time = simulator_class.time_plot
        beta = simulator_class.beta_plot
        beta0 = simulator_class.beta0
        phi = simulator_class.phi_plot
        phi0 = simulator_class.phi0
        p0 = simulator_class.p0
        p = simulator_class.p_plot
        
        dr_Li, dr_Ui = int(ti_DR//dt), int(20.0//dt) 
        print('\n')
        print('\n------Dutch Roll Estimate------\n')
        (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], beta[dr_Li:dr_Ui] - beta0, plot_results=True, ylabel = 'Beta [deg]')
        
        # print('\n------Dutch Roll Estimate------\n')
        # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], phi[dr_Li:dr_Ui] - phi0, plot_results=True, ylabel = 'Phi [deg]')

        # print('\n------Dutch Roll Estimate------\n')
        # (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], p[dr_Li:dr_Ui] - p0, plot_results=True, ylabel = 'p [deg/s]')
        
        dutch_roll_damp.append(a)
        dutch_roll_period.append(2*np.pi/abs(w))
        print('\n')

if BIRE == True:
    label = '_BIRE'
else:
    label = '_F16'
if 0 in studies:
    np.save('phugoid_properties_bank' + label + '.npy', np.asarray([phugoid_damp,phugoid_period]))
    np.save('short_period_properties_bank' + label + '.npy', np.asarray([short_damp,short_period]))
if 1 in studies:
    np.save('dutch_roll_properties_bank' + label + '.npy', np.asarray([dutch_roll_damp,dutch_roll_period]))