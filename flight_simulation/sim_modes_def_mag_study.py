from simulator_physics import simulator
from fit_damped_sinusoid import *

MODE = 2 # 0 - Short period, 1 - Phugoid, 2 - Dutch Roll

def cubic_poly(x,a,b,c):
    
    y = a*x*x + b*x + c
    
    return y

ti_modes = 1.0
dt_array = [0.02, 0.03, 0.04, 0.05]

de_array = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

sigma_0_array = []
period_0_array = []


for j in range(len(dt_array[:])):
# for j in range(1):
    sigma_array = []
    period_array = []
    for i in range(len(de_array[:])):
        
        
        simulator_class = simulator(init_filename = 'simulator_input.json')
        simulator_class.tpert_dur = dt_array[j]
        if MODE == 0 or MODE == 1:
            simulator_class.dde = np.deg2rad(de_array[i])
        elif MODE == 2:
            simulator_class.ddr = np.deg2rad(de_array[i])

        simulator_class.run_sim()
    
        time = simulator_class.time_plot
        alpha = simulator_class.alpha_plot
        airspeed = simulator_class.airspeed_plot
        beta = simulator_class.beta_plot
    
        dt = simulator_class.dt
        V0 = simulator_class.V0
        alpha0 = simulator_class.alpha0
        beta0 = simulator_class.beta0
        
        sp_Li, sp_Ui = int(1.0//dt), int((5.0)//dt) # start at 1 sec, end at 10 sec
        Lp_Li, Lp_Ui = int(1.0//dt), int((100.0)//dt) # start at 1 sec, end at 10 sec
        dr_Li, dr_Ui = int(1.0//dt), int((10.0)//dt) # start at 1 sec, end at 10 sec

        if MODE == 0:
            print('\n------Short Period Estimate------\n')
            (A,a,w,T,z), fun = fit_sinusoid(time[sp_Li:sp_Ui], alpha[sp_Li:sp_Ui] - alpha0, plot_results=False)
        elif MODE == 1:
            print('\n------Phugoid Estimate------\n')
            (A,a,w,T,z), fun = fit_sinusoid(time[Lp_Li:Lp_Ui], np.asarray(airspeed[Lp_Li:Lp_Ui]) - V0, plot_results=False)
        elif MODE == 2:
            print('\n------Dutch Roll Estimate------\n')
            (A,a,w,T,z), fun = fit_sinusoid(time[dr_Li:dr_Ui], beta[dr_Li:dr_Ui] - beta0, plot_results=False)
        
        sigma_array.append(a)
        period_array.append(2*np.pi/w)
        

    popt, pcov = optimize.curve_fit(cubic_poly, de_array, sigma_array, xtol=1e-6)
    a,b,c = popt
    
    sigma_0 = cubic_poly(0.0,a,b,c)
    sigma_0_array.append(sigma_0)
    
    
    plt.figure(1)
    plt.scatter(de_array, sigma_array, label = str(dt_array[j]))
    plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
    plt.ylabel('sigma')
    plt.xlabel('de [deg]')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    popt, pcov = optimize.curve_fit(cubic_poly, de_array, period_array, xtol=1e-6)
    a,b,c = popt
    
    period_0 = cubic_poly(0.0,a,b,c)
    period_0_array.append(period_0)
    
    plt.figure(2)
    plt.scatter(de_array, period_array, label = str(dt_array[j]))
    plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
    plt.ylabel('period')
    plt.xlabel('de [deg]')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
'''STUDIES WE COULD DO
- START WITH A FEW CHECK POINTS ON THE BIRE METHODS PLOT, VERIFY THAT FLIGHT SIM MATCHES OUR METHOD BETTER THAN THE OTHERS (REMEMBER THE POINT OF THE PAPER)
- CHECK THE EFFECT OF DEFLECTION AMPLITUDE AND DURATIONS ON THE DYNAMIC MODE PROPERTIES MEASURED
- TEST PQR AS STATE VARIABLES TO FIT'''