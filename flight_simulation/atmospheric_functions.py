from math import cos, sin, atan, atan2, asin, sqrt, exp, pi
import numpy as np

#-----------------------------------------------------------------------------#
#ATMOSPHERIC PROFILE FUNCTIONS

def gravity_si(H):
   return 9.806645*(6356766./(6356766.+H))**2

def gravity_english(H):
   return 9.806645*(6356766./(6356766. + H*0.3048))**2/0.3048

def statsi(h):
    Psa = np.zeros(9)
    zsa = [0,11000,20000,32000,47000,52000,61000,79000,90000]
    Tsa = [288.15,216.65,216.65,228.65,270.65,270.65,252.65,180.65,180.65]
    g0 = 9.806645
    R = 287.0528
    Re = 6356766
    gamma = 1.4
    
    Psa[0] = 101325
    z = Re*h/(Re+h)
    for i in range(1,9):
        Lt = -(Tsa[i]-Tsa[i-1])/(zsa[i]-zsa[i-1])
        if Lt == 0:
            if z <= zsa[i]:
                t = Tsa[i-1]
                p = Psa[i-1]*exp(-g0*(z-zsa[i-1])/R/Tsa[i-1])
                d = (p/R)/t
                a = sqrt(gamma*R*t)
                return (h,z,t,p,d,a)
            else:
                Psa[i] = Psa[i-1]*exp(-g0*(zsa[i]-zsa[i-1])/R/Tsa[i-1])
        else:
            ex = (g0/R)/Lt
            if z < zsa[i]:
               t = Tsa[i-1]-Lt*(z-zsa[i-1])
               p = Psa[i-1]*(t/Tsa[i-1])**ex
               d = (p/R)/t
               a = sqrt(gamma*R*t)
               return (h,z,t,p,d,a)
            else:
               Psa[i] = Psa[i-1]*(Tsa[i]/Tsa[i-1])**ex
    t = Tsa[8]
    p = 0.
    d = 0.
    a = sqrt(gamma*R*t)
    
    return (h,z,t,p,d,a)

def statee(h):
    #     h = geometric altitude, specified by user (ft)
    #     z = geopotential altitude, returned by subroutine (ft)
    #     t = temperature, returned by subroutine (R)
    #     p = pressure, returned by subroutine (lbf/ft**2)
    #     d = density, returned by subroutine (slugs/ft**3)
    hsi = h*0.3048
    hsi,zsi,tsi,psi,dsi,asi = statsi(hsi)
    z = zsi/0.3048
    t = tsi*1.8
    p = psi*0.020885434304801722
    d = dsi*0.00194032032363104
    a = asi/0.3048
    return (h,z,t,p,d,a)