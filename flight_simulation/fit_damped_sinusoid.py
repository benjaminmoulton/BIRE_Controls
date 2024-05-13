import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

'''OVER DAMPED CURVE '''
def damped_curve(x, A, a, c1, c2):
    '''Equation of a damped sinusoid
    A: Amplitude
    a: damping ratio
    w: frequency
    z: offset'''
    x = np.array(x)
    y = A*(np.exp(-a*x))*(c1 + c2*x)
    return y

def RMSE_curve(params,x,y):
    '''Objective function
    RMSE between the fit sinusoid and true data'''
    A,a,c1,c2 = params
    N = len(x)
    y_fit = damped_curve(x,A,a,c1,c2)
    
    deviation = ((y_fit - y)**2)
    root_mean_square_error = np.sqrt(sum(deviation)/N)
    # root_mean_square_error = sum(deviation)
    # print(root_mean_square_error)
    
    return root_mean_square_error

def opt_damped_curve(x,y,Ai,ai,c1i,c2i,opt_type = 'SLSQP'):
    '''setup and run optimization of sinusoid using RMSE objective function'''
    # bounds = [(-np.inf,np.inf), (-np.inf,np.inf), (0.0, np.inf), (-np.inf,np.inf), (-np.inf,np.inf)]
    bounds = [(-np.inf,np.inf), (-np.inf,np.inf),(-np.inf,np.inf), (-np.inf,np.inf)]
    optimum = optimize.minimize(RMSE_curve, x0 = [Ai,ai,c1i,c2i], bounds = bounds, args = (x,y), method=opt_type,
                   options={ 'disp': True, 'ftol':1e-8, 'maxiter': 500, 'eps': 1.5e-06})
    return optimum.x, optimum.fun

'''DAMPED SINUSOID FUNCTIONS'''

def damped_sinusoid(x, A, a, w, T, z):
    '''Equation of a damped sinusoid
    A: Amplitude
    a: damping ratio
    w: frequency
    z: offset'''
    x = np.array(x)
    y = A*(np.exp(-a*x))*np.cos(w*x - T) + z
    return y

def mult_damped_sinusoid(params,x,n):
    A = params[0:n]
    a = params[n:2*n]
    w = params[2*n:3*n]
    T = params[3*n:4*n]
    z = params[4*n:5*n]
    
    N = len(x)
    y_total = np.zeros(N)
    
    for i in range(n):
        y_fit = damped_sinusoid(x,A[i],a[i],w[i],T[i],z[i])
        
        y_total += y_fit

    return y_total

def RMSE_sinusoid(params,x,y):
    '''Objective function
    RMSE between the fit sinusoid and true data'''
    A,a,w,T,z = params
    N = len(x)
    y_fit = damped_sinusoid(x,A,a,w,T,z)
    
    deviation = ((y_fit - y)**2)
    root_mean_square_error = np.sqrt(sum(deviation)/N)
    # root_mean_square_error = sum(deviation)
    # print(root_mean_square_error)
    
    return root_mean_square_error

def RMSE_mult_sinusoid(params,x,y,n):
    '''Objective function
    RMSE between the fit sinusoid and true data'''
    
    N = len(x)

    y_total = mult_damped_sinusoid(params,x,n)
    
    deviation = ((y_total - y)**2)*100000
    root_mean_square_error = np.sqrt(sum(deviation)/N)
    
    return root_mean_square_error

def opt_sinusoid(x,y,Ai,ai,wi,Ti,zi,opt_type = 'SLSQP'):
    '''setup and run optimization of sinusoid using RMSE objective function'''
    # bounds = [(-np.inf,np.inf), (-np.inf,np.inf), (0.0, np.inf), (-np.inf,np.inf), (-np.inf,np.inf)]
    bounds = [(-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf)]
    optimum = optimize.minimize(RMSE_sinusoid, x0 = [Ai,ai,wi,Ti,zi], bounds = bounds, args = (x,y), method=opt_type,
                   options={ 'disp': True, 'ftol':1e-8, 'maxiter': 500, 'eps': 1.5e-06})
    return optimum.x, optimum.fun

def opt_mult_sinusoid(x,y,paramsi,n,opt_type = 'SLSQP'):
    '''setup and run optimization of sinusoid using RMSE objective function'''
    # bounds = [(-np.inf,np.inf), (-np.inf,np.inf), (0.0, np.inf), (-np.inf,np.inf), (-np.inf,np.inf)]
    optimum = optimize.minimize(RMSE_mult_sinusoid, x0 = paramsi, args = (x,y,n), method=opt_type,
                   options={ 'disp': True, 'ftol':1e-8, 'maxiter': 500, 'eps': 1.5e-06})
    return optimum.x, optimum.fun

def signal_decomp(x,y):
    '''uses FFT to find the frequency of the largest component based on power spectrum'''
    N = len(y) # number of data points
    dt = (x[-1] - x[0])/N # average step size of the data
    num_freqs = N//2 # Nyquist frequency, highest frequency that can be measured in the given data
    fourier_transform = np.fft.fft(y) # discrete Fourier transform
    
    fourier_zero = fourier_transform[0]
    amp_zero = fourier_zero/N
    
    fourier_oneside = fourier_transform[1:num_freqs] # slice only the positive frequency terms, exclude the first zero freq term
    fourier_oneside[:] = 2*fourier_oneside # doubles the non zero frequency terms. I think this only matters for the power calculation?
    
    amp_spec_one = np.absolute(fourier_oneside)/N # this looks like its missing the sqrt(2) that his paper has?
    phase_spec_one = np.pi/2.0 + np.arctan2(fourier_oneside.imag,
                                            fourier_oneside.real) # why the pi/2.0?
    
    # phase_spec_one = np.arctan2(fourier_oneside.imag, fourier_oneside.real)
    
    amplitude = max(amp_spec_one).real
    freq_spec = np.fft.fftfreq(N, d=dt) # frequency bin centers in cycles per unit of the sample spacing (with zero at the start)
    freq_spec_one = freq_spec[1:num_freqs]*2.0*np.pi # radians per unit of sample spacing, ignore the zero frequency index
    power_spec = (np.abs(fourier_oneside)**2)*((dt)**2)
    
    highest_power_index = power_spec.argmax()
    frequency = freq_spec_one[highest_power_index]
    
    return frequency

def estimate_amp(x,y):
    '''estimates amplitude and requ3ency of an assumed sinusoid'''
    N = len(x)//2

    imax = np.argmax(y[:N])
    imin = np.argmin(y[:N])
    
    dT = abs(x[imax] - x[imin])*2
    f = 2*np.pi*1/dT
    
    A = (y[imax] - y[imin])/2
    
    if imax<imin:
        return A,f
    elif imax>imin:
        return -A,f

'''FIT MULTIPLE SINUSOIDS'''

def fit_mult_sinusoid(x,y,slice_ind,num_sine=1,plot_results=False,**kwargs):

    ylabel = kwargs.get('ylabel', 'y-data')
    plot_first = kwargs.get('plot_first', False)
    
    if ylabel == 'Altitude [ft]':
        Ti_a = [np.pi, 0.0, 0.0, 0.0]
        opt_type = 'Nelder-Mead'
    else:
        Ti_a = [0.0, 0.0, 0.0, 0.0]
        opt_type = 'SLSQP'
    
    # y_temp = y
    y_orig = y
    
    initial_params = []
        
    # slice_ind = [(0,-1), (1100,1600), (100,200),(300,1500)] # F-16 sideslip angle
    
    for i in range(num_sine):
        Ti = Ti_a[i]
        i1,i2 = slice_ind[i]
        
        y_temp = y[i1:i2]
        x_temp = x[i1:i2]
        
        A,f_simp = estimate_amp(x_temp,y_temp)
        f_fft = signal_decomp(x_temp,y_temp)
        
        (A,a,w,T,z), fun = opt_sinusoid(x_temp,y_temp,Ai=A,ai=0.0,wi=f_fft,Ti=Ti,zi=0.0,opt_type=opt_type)
        
        initial_params.append([A,a,w,T,z])
        
        y_fit = damped_sinusoid(x_temp,A,a,w,T,z)
        
        y_fit_full = damped_sinusoid(x,A,a,w,T,z)
        y = y - y_fit_full
        
        if plot_results == True:
            plt.figure()
            plt.plot(x_temp,y_temp,linewidth=3.0,color='k')
            plt.plot(x_temp,y_fit,linewidth=3.0,linestyle='-.',color='orange')
            plt.ylabel(ylabel)
            plt.xlabel('Time [s]')
            plt.legend(['data', 'fit'])
            plt.tight_layout()
            plt.show()
            
            
            plt.figure()
            plt.plot(x,y,linewidth=3.0,linestyle=':',color='r')
            plt.ylabel(ylabel)
            plt.xlabel('Time [s]')
            plt.legend(['delta'])
            plt.tight_layout()
            plt.show()
            
    initial_params = np.asarray(initial_params)
    paramsi = np.concatenate((initial_params[:,0],initial_params[:,1], initial_params[:,2], initial_params[:,3], initial_params[:,4]))
    
    paramsf, funf = opt_mult_sinusoid(x, y_orig, paramsi, num_sine, opt_type = opt_type)
    
    y_final = mult_damped_sinusoid(paramsf,x,num_sine)
    
    plt.figure()
    plt.plot(x,y_orig,linewidth=3.0,color='k')
    plt.plot(x,y_final,linewidth=3.0,linestyle='-.',color='orange')
    plt.ylabel(ylabel)
    plt.xlabel('Time [s]')
    plt.legend(['data', 'fit'])
    plt.tight_layout()
    plt.show()
    
    # print('\n------A,a,w,T,z: ', A, a, w, T, z)
    
    # print('Damping Rate: ', a)
    
    # period = 2*np.pi/abs(w)
    # print('Period[s]: ', period)
    
    # diff = y - y_fit
    

    
    return initial_params


def fit_sinusoid(x,y,plot_results=False,**kwargs):
    ylabel = kwargs.get('ylabel', 'y-data')
    plot_first = kwargs.get('plot_first', False)
    
    if ylabel == 'Altitude [ft]':
        Ti = np.pi
        opt_type = 'Nelder-Mead'
    else:
        Ti = 0.0
        opt_type = 'SLSQP'
    
    A,f_simp = estimate_amp(x,y)
    f_fft = signal_decomp(x,y)
    
    y_first = damped_sinusoid(x,A,a = 0.0,w = f_fft,T=Ti,z = 0.0)
    
    if plot_first == True:
        plt.figure()
        plt.plot(x,y,linewidth=3.0, color = 'k')
        plt.plot(x,y_first,linewidth=3.0,linestyle='-.', color = 'gray')
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.legend(['data', 'fit'])
        plt.tight_layout()
        plt.title('Initial Estimate')
        plt.grid(visible=True)
        plt.show()
    
    (A,a,w,T,z), fun = opt_sinusoid(x,y,Ai=A,ai=0.0,wi=f_fft,Ti=Ti,zi=0.0, opt_type=opt_type)
    
    y_fit = damped_sinusoid(x,A,a,w,T,z)
    
    print('\n------A,a,w,T,z: ', A, a, w, T, z)
    
    print('Damping Rate: ', a)
    
    period = 2*np.pi/abs(w)
    print('Period[s]: ', period)
    
    diff = y - y_fit
    
    fig_size = (4,4)
    if plot_results == True:
        plt.figure(figsize=fig_size)
        # plt.plot(x,diff,linewidth=3.0)
        plt.plot(x,y,linewidth=3.0, color = 'k')
        plt.plot(x,y_fit,linewidth=3.0,linestyle='-.', color = 'gray')
        plt.ylabel(ylabel)
        plt.xlabel('Time [s]')
        plt.legend(['data', 'fit'])
        plt.tight_layout()
        plt.grid(visible=True)
        plt.show()
    
    return (A,a,w,T,z), fun



'''--------------------NORMALIZING DATA AND COMPARISON FUNCTIONS--------------------'''

def normalize_sim_data(y, norm_value = 999.):
    # y_norm = (2*(y - np.mean(y))/(max(y) - min(y))) # normalize WRT Amax-Amin=
    # y_norm = ((y - y[0])/(max(abs(y - y[0]))))
    if norm_value == 999.:
        norm_value = max(abs(y - np.mean(y))) # normalize WRT Amax around the mean
    y_norm = ((y - np.mean(y))/(norm_value))
    return y_norm

def get_peak_align_index(y1,y2):
    i_max1 = np.argmax(abs(y1))
    i_max2 = np.argmax(abs(y2))
    
    i_shift = i_max1 - i_max2 #assumes i_max1 will be shift to the right of i_max2
    
    return i_shift