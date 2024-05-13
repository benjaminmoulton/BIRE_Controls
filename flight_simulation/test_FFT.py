from math import cos, sin, atan, atan2, asin, sqrt, exp, pi
import numpy as np
from scipy.integrate import ode, odeint
from scipy import optimize, fft
import matplotlib.pyplot as plt


def RMSE_sinusoid(params,x,y):
    A,a,w,T,z = params
    print(params)
    N = len(x)
    y_fit = damped_sinusoid(x,A,a,w,T,z)
    deviation = ((y_fit - y)**2)
    root_mean_square_error = np.sqrt(sum(deviation)/N)
    # root_mean_square_error = sum(deviation)
    print(root_mean_square_error)
    
    return root_mean_square_error

def opt_sinusoid(x,y,Ai,ai,wi,Ti,zi):
    
    # bounds = [(-4.0,4.0), (-10.0,10.0), (0.0,2*np.pi),(-5.0,5.0)]
    optimum = optimize.minimize(RMSE_sinusoid, x0 = [Ai,ai,wi,Ti,zi], args = (x,y), method='SLSQP', tol = 1e-8,
                       options={ 'disp': True, 'ftol':1e-8, 'maxiter': 250, 'eps': 1.4901161193847656e-06})
    return optimum.x

def damped_sinusoid(x, C, a, w, T, z):
    x = np.array(x)
    y = C*(np.exp(-a*x))*np.cos(w*x - T) + z
    return y

def signal_decomp(x,y,include_offset = False):
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
    
    plt.figure(0)
    plt.plot(x,y)
    plt.xlabel('time[s]')
    plt.ylabel('alpha[deg]')
    plt.show()
    
    
    N = len(y) # number of data points
    dt = (x[-1] - x[0])/N # average step size of the data
    num_freqs = N//2 # Nyquist frequency, highest frequency that can be measured in the given data
    fourier_transform = np.fft.fft(y) # discrete Fourier transform, returns 
    # get the one sided spectrum
    # IVE CHANGE THESE TWO LINES
    fourier_one = fourier_transform[:num_freqs] # slice only the positive frequency terms
    fourier_one[1:num_freqs] = 2*fourier_transform[1:num_freqs] # doubles the non zero frequency terms, because we have sliced the negative ones?
    
    sr = 1.0/0.01 # [#/s] sample rate, known from the simulator step size
    N = len(fourier_transform) # number of data points in the data, both given and FFT output
    n = np.arange(N) # array of data point integers
    T = N/sr # total time of data
    freq = n/T # frequency of the FFT output
    
    plt.figure(1)
    plt.plot(freq, np.abs(fourier_transform))
    plt.xlabel('Freq[Hz]')
    plt.ylabel('FFT Amplitude')
    plt.show()
    
    amp_spec_one = np.absolute(fourier_one)/N
    phase_spec_one = np.pi/2.0 + np.arctan2(fourier_one.imag,
                                            fourier_one.real)
    amplitude = max(amp_spec_one).real
    # freq_spec = np.fft.fftfreq(N)/dt # divide by dt because 
    freq_spec = np.fft.fftfreq(N, d=dt) # frequency bin centers in cycles per unit of the sample spacing (with zero at the start)
    freq_spec_one = freq_spec[:num_freqs]*2.0*np.pi # radians per unit of sampel spacing
    power_spec = (np.abs(fourier_one)**2)*((dt)**2)
    plt.figure(2)
    plt.plot(freq_spec, fourier_transform)
    plt.xlim((-.25, .25))
    plt.show()

    highest_power_index = power_spec.argmax()
    frequency = freq_spec_one[highest_power_index]
    phase = phase_spec_one[highest_power_index]
    
    if include_offset is True:
        offset = np.average(y)
        harmonic_params = np.array([amplitude, frequency, phase, offset])
    else:
        harmonic_params = np.array([amplitude, frequency, phase])
    return harmonic_params

def estimate_amp(x,y):
    
    N = len(x)//2

    imax = np.argmax(y[:N])
    imin = np.argmin(y[:N])
    
    dT = abs(x[imax] - x[imin])*2
    f = 2*pi*1/dT
    
    A = (y[imax] - y[imin])/2
    
    if imax<imin:
        return -A,f
    elif imax>imin:
        return A,f


data = np.genfromtxt('states.txt', skip_header=1)



time = data[110:,0] # sliced to exclude data before the perturbation
alpha = data[110:,10]

alpha_trim = alpha[0]
dalpha = alpha - alpha_trim

A_guess,f_guess = estimate_amp(time,dalpha)

f_fft = signal_decomp(x = time, y = dalpha, include_offset=True)

# A, w, T, z = params
a = 0.0
T = 0.0
z = 0.0

y_guess = damped_sinusoid(time,A_guess,a,f_guess,T,z)

Ai  = A_guess
wi = f_fft

x = opt_sinusoid(time,dalpha,Ai,a,wi,T,z)

y_fit = damped_sinusoid(time,x[0],x[1],x[2],x[3],x[4])


plt.figure(4)
plt.plot(time,dalpha)
plt.plot(time, y_guess)
plt.plot(time, y_fit)
plt.legend(['Measured','Guess', 'Fit'])
plt.show()

