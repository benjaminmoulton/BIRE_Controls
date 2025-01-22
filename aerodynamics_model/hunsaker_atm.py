import numpy as np
import math as m

#Atmospheric Routines ----------------------------------------------------
def gravity_si(H):
   return 9.806645*(6356766./(6356766.+H))**2

def gravity_english(H):
   return 9.806645*(6356766./(6356766. + H*0.3048))**2/0.3048
#   return 32.1741

def stdatm_si(H):
   levels = np.array([0.,11000.,20000.,32000.,47000.,52000.,61000.,79000.,90000])
   temps = np.array([288.15,216.65,216.65,228.65,270.65,270.65,252.65,180.65,180.65])
   tprime = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.0, -4.0, 0.0])
   tprime = tprime/1000.0
   go = 9.806645
   R = 287.0528
   RE = 6356766.
   po = 101325.
   gamma = 1.4
   
   Z=RE*H/(RE+H)

   Z = max(Z,0.0)

   if(Z > 90000):
      T = 180.650
      p = 0.0
   else:
      i = 0
      while (Z >= levels[i]):
         if(tprime[i] == 0):
            if(Z < levels[i+1]):
               T = temps[i]
               p = po*m.exp(-go*(Z-levels[i])/R/temps[i])
            else:
               po = po*m.exp(-go*(levels[i+1]-levels[i])/R/temps[i])
         else:
            if(Z < levels[i+1]):
               T = temps[i] + tprime[i]*(Z-levels[i])
               p = po*((temps[i] + tprime[i]*(Z-levels[i]))/temps[i])**(-go/R/tprime[i])
            else:
               po = po*((temps[i] + tprime[i]*(levels[i+1]-levels[i]))/temps[i])**(-go/R/tprime[i])
         i = i + 1
   rho = p/R/T
   a = m.sqrt(gamma*R*T)
   return Z,T,p,rho,a

def stdatm_english(H):
   H *= 0.3048
   Z,T,p,rho,a = stdatm_si(H)
   Z/=0.3048
   T*=1.8
   p*=0.020885434304801722
   rho*=0.00194032032363104
   a/=0.3048
   return Z,T,p,rho,a

def moulton_stdatm_derivative_si(H):
    """Given the geometric altitude (m), calculates and returns the
    derivative of the
    geopotential altitude (-), gravitational acceleration (1/s^2),
    temperature (K/m), pressure (N/m^3), density (kg/m^4), and 
    speed of sound (1/s) with respect to the geometric altitude.

    Parameters
    ----------
    H : float
        geometric altitude (m).
    
    Returns
    -------
    dZdH : float
        derivative of geopotential altitude wrt geometric altitude (-).
    
    dGdH : float
        derivative of gravitational acceleration wrt geometric altitude (1/s^2)
    
    dTdH : float
        derivative of temperature wrt geometric altitude (K/m).
    
    dPdH : float
        derivative of pressure wrt geometric altitude (N/m^3).
    
    dRdH : float
        derivative of density wrt geometric altitude (kg/m^4).
    
    dAdH : float
        derivative of speed of sound wrt geometric altitude (1/s).
    """
    # get current properties
    Z,T,P,R,A = stdatm_si(H)
    G = gravity_si(H)

    # geopotential altitude derivative
    dZdH = 40408473978756.0/(6356766.0+H)/(6356766.0+H)

    # gravitation acceleration derivative
    dGdH = -19.61329*dZdH/(6356766.0+H)

    # temperature derivative
    if   Z < 11000.0: dTdZ = -0.0065
    elif Z < 20000.0: dTdZ = +0.0000
    elif Z < 32000.0: dTdZ = +0.0010
    elif Z < 47000.0: dTdZ = +0.0028
    elif Z < 52000.0: dTdZ = +0.0000
    elif Z < 61000.0: dTdZ = -0.0020
    elif Z < 79000.0: dTdZ = -0.0040
    elif Z < 90000.0: dTdZ = +0.0000
    else            : dTdZ = +0.0000
    dTdH = dTdZ*dZdH

    # pressure derivative
    dPdH = -G*P/287.0528/T

    # density derivative
    dRdH = dPdH/287.0528/T - P/287.0528/T/T*dTdH

    # speed of sound derivative
    dAdH = 10.023396629885498*T**-0.5*dTdH
    
    return dZdH,dGdH,dTdH,dPdH,dRdH,dAdH


def moulton_stdatm_derivative_english(H):
    """Given the geometric altitude (ft), calculates and returns the
    derivative of the
    geopotential altitude (-), gravitational acceleration (1/s^2), 
    temperature (deg R/ft), pressure (lbf/ft^3), density (slugs/ft^4), and 
    speed of sound (1/s) with respect to the geometric altitude.

    Parameters
    ----------
    H : float
        geometric altitude (ft).
    
    Returns
    -------
    dZdH : float
        derivative of geopotential altitude wrt geometric altitude (-).
    
    dGdH : float
        derivative of gravitational acceleration wrt geometric altitude (1/s^2)
    
    dTdH : float
        derivative of temperature wrt geometric altitude (deg R/ft).
    
    dPdH : float
        derivative of pressure wrt geometric altitude (lbf/ft^3).
    
    dRdH : float
        derivative of density wrt geometric altitude (slugs/ft^4).
    
    dAdH : float
        derivative of speed of sound wrt geometric altitude (1/s).
    """

    # calculate values
    dZdH,dGdH,dTdH,dPdH,dRdH,dAdH = moulton_stdatm_derivative_si(H*0.3048)
    return dZdH, dGdH, dTdH*0.54864, dPdH*0.006365880376103566, dRdH*0.000591409634642741, dAdH


def atm_print():
   output = open('stdatmos_si.txt','w')
   output.write('     Geometric Altitude [m]   Geopotential Altitude [m]   Temperature [K]          Pressure [N/m^2]         Density [kg/m^3]         Speed of Sound [m/s]\n')
   for H in range(0,72000,2000):
      Z, T, p, rho, a = stdatm_si(H)
      output.write('{:25.11E}{:25.11E}{:25.11E}{:25.11E}{:25.11E}{:25.11E}\n'.format(H,Z,T,p,rho,a))

   output = open('stdatmos_english.txt','w')
   output.write('    Geometric Altitude [ft]  Geopotential Altitude [ft]   Temperature [R]          Pressure [lbf/ft^2]      Density [slugs/ft^3]     Speed of Sound [ft/s]\n')
   for H in range(0,180000,5000):
      Z, T, p, rho, a = stdatm_english(H)
      output.write('{:25.11E}{:25.11E}{:25.11E}{:25.11E}{:25.11E}{:25.11E}\n'.format(H,Z,T,p,rho,a))
