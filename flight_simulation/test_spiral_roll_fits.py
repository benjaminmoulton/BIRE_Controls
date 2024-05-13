from fit_damped_sinusoid import *
import numpy as np


data = np.genfromtxt('states.txt', skip_header=1)



time = data[4000:,0] # sliced to exclude data before the perturbation
y = data[0:,8]
psi = data[4000:,13]

# alpha_trim = alpha[0]
# dalpha = alpha - alpha_trim

# A_guess,f_guess = estimate_amp(time,dalpha)

# f_fft = signal_decomp(x = time, y = dalpha, include_offset=True)

# # A, w, T, z = params
A = -0.5
a = -0.06
c1 = 0.0
c2 = 0.0

y_guess = damped_curve(time,A,a,c1,c2)

# Ai  = A_guess
# wi = f_fft

x, fun = opt_damped_curve(time,psi,A,a,c1,c2, opt_type='Nelder-Mead')

y_fit = damped_curve(time,x[0],x[1],x[2],x[3])


plt.figure(4)
plt.plot(time,psi)
plt.plot(time, y_guess)
plt.plot(time, y_fit)
plt.legend(['Measured','Guess', 'fit'])
plt.show()
