from simulator_physics import simulator
from fit_damped_sinusoid import *

MODE = 1

dts_array = [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]

sigma_array = []
period_array = []

for j in range(len(dts_array[:])):

    simulator_class = simulator(init_filename = 'simulator_input.json')
    
    simulator_class.dt = dts_array[j]
    
    if MODE == 0 or MODE == 1:
        simulator_class.dde = np.deg2rad(10.0)
    elif MODE == 2:
        simulator_class.ddr = np.deg2rad(6.0)

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
        
    # popt, pcov = optimize.curve_fit(cubic_poly, de_array, sigma_array, xtol=1e-6)
    # a,b,c = popt
    
    # sigma_0 = cubic_poly(0.0,a,b,c)
    # sigma_0_array.append(sigma_0)
    
    
    
# import matplotlib.pyplot as plt
# MODE = 0

# NO BANK
if MODE == 0:
    sigma_variable = [0.8531937248028206,0.8531937248028206]
    period_variable = [3.357616988358205,3.357616988358205]
elif MODE == 1:
    sigma_variable = [0.004430978873140865,0.004430978873140865]
    period_variable = [96.77244864531976,96.77244864531976]
    
#   # 30 DEG BANK  
# if MODE == 0:
#     sigma_variable = [0.8518030139406286,0.8518030139406286]
#     period_variable = [3.366067517015762,3.366067517015762]
# elif MODE == 1:
#     sigma_variable = [0.00807281775731695,0.00807281775731695]
#     period_variable = [128.9294889748445,128.9294889748445]


dts = [0.0, 0.01]

'''SHORT PERIOD DATA FROM PREVIOUS RUN'''
# dts_array = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
# period_array = [3.360836013724525, 3.3605962343000497, 3.3602756551950477, 3.3599561094775114, 3.3595591865179153, 3.357409455182236, 3.3538442153608976, 3.3506980324605804, 3.346568311696519]
# sigma_array = [0.8614646765057165, 0.8613565761508936, 0.8611824348110116, 0.8610044644371533, 0.8608313922664389, 0.8597786058204019, 0.8580133643515928, 0.8561807538954315, 0.8544052491715389]


plt.figure(1)
plt.scatter(dts_array, sigma_array)
plt.plot(dts,sigma_variable, linestyle='--')
# plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
plt.ylabel('sigma')
plt.xlabel('dt [s]')
plt.legend(['RK4', 'SciPy ODE (Variable)'])
plt.tight_layout()
plt.show()

# popt, pcov = optimize.curve_fit(cubic_poly, de_array, period_array, xtol=1e-6)
# a,b,c = popt

# period_0 = cubic_poly(0.0,a,b,c)
# period_0_array.append(period_0)

plt.figure(2)
plt.scatter(dts_array, period_array)
plt.plot(dts,period_variable, linestyle='--')
# plt.plot(de_array, cubic_poly(np.array(de_array),a,b,c))
plt.ylabel('period')
plt.xlabel('dt [s]')
plt.legend(['RK4', 'SciPy ODE (Variable)'])
plt.tight_layout()
plt.show()