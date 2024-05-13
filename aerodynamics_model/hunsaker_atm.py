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
