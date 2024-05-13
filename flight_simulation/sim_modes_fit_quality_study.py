from simulator_physics import simulator
from fit_damped_sinusoid import *

ti_modes = 1.0

simulator_class = simulator(init_filename = 'simulator_input.json')
simulator_class.dde = np.deg2rad(10.0)
simulator_class.run_sim()

time = simulator_class.time_plot
alpha = simulator_class.alpha_plot
airspeed = simulator_class.airspeed_plot

dt = simulator_class.dt
V0 = simulator_class.V0
alpha0 = simulator_class.alpha0


ti_modes = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
error_array = []

for i in range(len(ti_modes[:])):
    sp_Li, sp_Ui = int(ti_modes[i]//dt), int((5.0+ti_modes[i])//dt) # start at 1 sec, end at 10 sec
    # Lp_Li, Lp_Ui = int(ti_modes[i]//dt), int(100.0//dt) # start at 1 sec, end at 100 sec
    
    print('\n------Short Period Estimate------\n')
    (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], alpha[sp_Li:sp_Ui] - alpha0, plot_results=True)
    error_array.append(fun)
    
    
'''STUDIES WE COULD DO
- START WITH A FEW CHECK POINTS ON THE BIRE METHODS PLOT, VERIFY THAT FLIGHT SIM MATCHES OUR METHOD BETTER THAN THE OTHERS (REMEMBER THE POINT OF THE PAPER)
- CHECK THE EFFECT OF DEFLECTION AMPLITUDE AND DURATIONS ON THE DYNAMIC MODE PROPERTIES MEASURED
- TEST PQR AS STATE VARIABLES TO FIT'''