from helper import *

class atmosphere:
   def __init__(self,atm_dict):

      self.properties = atm_dict["properties"]

      self.gust_type = atm_dict["gust_model"]["type"]
      self.gust_ramp_in = atm_dict["gust_model"]["ramp_in[sec]"]

      print("--------- Creating Gust Model ----------")
      print(" Gust Model Type : ",self.gust_type)

      if(self.gust_type == "von_karman"):
         # self.sigma_u = atm_dict["gust_model"]["sigma_u[ft/s]"]
         self.gust_number = atm_dict["gust_model"]["number_of_simultaneous_gusts"]
         bin_energy = 0.99/float(self.gust_number)
         print("Ignoring top 1 percent of gust energy")
         print("      bin_energy = ",bin_energy)

         self.Lu = 2500.0
         self.Lv = 2500.0
         self.Lw = 2500.0

         self.sigma_u = 1.0
         self.sigma_v = 1.0
         self.sigma_w = 1.0

         self.gust_u_amplitude = self.sigma_u*m.sqrt(2.0*bin_energy)
         self.gust_v_amplitude = self.sigma_v*m.sqrt(2.0*bin_energy)
         self.gust_w_amplitude = self.sigma_w*m.sqrt(2.0*bin_energy)

         # for i in range(1802):
         #    omega = 0.0001*float(i)
         #    print(i,self.u_von_karman_spectral_density(omega), self.v_von_karman_spectral_density(omega), self.w_von_karman_spectral_density(omega))

         self.gust_u_omega = np.zeros(self.gust_number)
         self.gust_v_omega = np.zeros(self.gust_number)
         self.gust_w_omega = np.zeros(self.gust_number)

         self.gust_u_phase = np.zeros(self.gust_number)
         self.gust_v_phase = np.zeros(self.gust_number)
         self.gust_w_phase = np.zeros(self.gust_number)

         self.gust_u_omega[0] = self.bounding_frequency_newton_solver(0.0,0.5*bin_energy, self.u_von_karman_spectral_density)
         self.gust_v_omega[0] = self.bounding_frequency_newton_solver(0.0,0.5*bin_energy, self.v_von_karman_spectral_density)
         self.gust_w_omega[0] = self.bounding_frequency_newton_solver(0.0,0.5*bin_energy, self.w_von_karman_spectral_density)

         for i in range(self.gust_number-1):
            self.gust_u_omega[i+1] = self.bounding_frequency_newton_solver(self.gust_u_omega[i],bin_energy, self.u_von_karman_spectral_density)
            self.gust_v_omega[i+1] = self.bounding_frequency_newton_solver(self.gust_v_omega[i],bin_energy, self.v_von_karman_spectral_density)
            self.gust_w_omega[i+1] = self.bounding_frequency_newton_solver(self.gust_w_omega[i],bin_energy, self.w_von_karman_spectral_density)

         for i in range(self.gust_number):
            print(i,self.gust_u_omega[i])
            print(i,self.gust_v_omega[i])
            print(i,self.gust_w_omega[i])
            self.gust_u_phase[i] = random.uniform(-pi,pi)
            self.gust_v_phase[i] = random.uniform(-pi,pi)
            self.gust_w_phase[i] = random.uniform(-pi,pi)

      if(self.gust_type == "database"):
         fn = atm_dict["gust_model"]["numpy_binary_database"]
         scale = get_units_to_ft(atm_dict["gust_model"]["database_units"])

         with np.load(fn) as npzdata:
            xls = npzdata['xls']*scale
            yls = npzdata['yls']*scale
            zls = npzdata['zls']*scale
            tls = npzdata['tls']
            gVx = npzdata['gVx']*scale
            gVy = npzdata['gVy']*scale
            gVz = npzdata['gVz']*scale

            print(" Numpy Binary Database : ",fn)
            print("        Database Units : ",atm_dict["gust_model"]["database_units"])
            print("         x limits [ft] : ",min(xls), " to ", max(xls))
            print("         y limits [ft] : ",min(yls), " to ", max(yls))
            print("         z limits [ft] : ",min(zls), " to ", max(zls))
         
         self.gust_interp_Vx = RegularGridInterpolator((xls, yls, zls, tls), gVx, method='linear', bounds_error = False, fill_value = None)
         self.gust_interp_Vy = RegularGridInterpolator((xls, yls, zls, tls), gVy, method='linear', bounds_error = False, fill_value = None)
         self.gust_interp_Vz = RegularGridInterpolator((xls, yls, zls, tls), gVz, method='linear', bounds_error = False, fill_value = None)


      if("sample" in atm_dict["gust_model"]):
         fn = atm_dict["gust_model"]["sample"]["save_filename"]
         npoints = atm_dict["gust_model"]["sample"]["number_of_points"]
         [t0, t1] = np.array(atm_dict["gust_model"]["sample"]["time[s]"])
         [x0, x1] = np.array(atm_dict["gust_model"]["sample"]["x[ft]"])
         [y0, y1] = np.array(atm_dict["gust_model"]["sample"]["y[ft]"])
         [z0, z1] = np.array(atm_dict["gust_model"]["sample"]["z[ft]"])

         t = np.zeros(npoints)
         x = np.zeros(npoints)
         y = np.zeros(npoints)
         z = np.zeros(npoints)
         Vx = np.zeros(npoints)
         Vy = np.zeros(npoints)
         Vz = np.zeros(npoints)
         for i in range(npoints):
            percent = float(i)/float(npoints-1)
            t[i] = percent*(t1-t0) + t0
            x[i] = percent*(x1-x0) + x0
            y[i] = percent*(y1-y0) + y0
            z[i] = percent*(z1-z0) + z0

            nplocation = np.array([x[i],y[i],z[i],t[i]])

            Vx[i] = self.gust_interp_Vx(nplocation)
            Vy[i] = self.gust_interp_Vy(nplocation)
            Vz[i] = self.gust_interp_Vz(nplocation)

         save_data = np.transpose([t,x,y,z,Vx,Vy,Vz])
         np.savetxt(fn, save_data, header="time[s],x[ft],y[ft],z[ft],Vx[ft/s],Vy[ft/s],Vz[ft/s]",comments="",delimiter=",")

   def bounding_frequency_newton_solver(self,omega_start,target_value,f):
      max_error = 1.0e-12
      max_iter = 10000
      relaxation = 0.1
      
      omega_guess = omega_start #max(0.1*omega_start,0.000001)
      diff_delta = max(0.01*omega_start,1.0e-6)
      
      
      error = 1.0
      iter = 0
      while abs(error) > max_error:
         ans   = self.rk4(omega_start, omega_guess,           f)
         ans_p = self.rk4(omega_start, omega_guess+diff_delta,f)
         ans_n = self.rk4(omega_start, omega_guess-diff_delta,f)
         deriv = (ans_p - ans_n)/2.0/diff_delta

         omega_guess = omega_guess - relaxation*(ans-target_value)/deriv
         ans = self.rk4(omega_start, omega_guess,f)
         error = abs(ans - target_value)
         iter += 1

         if(iter > max_iter):
            print("Maximum iterations reached in bounding_frequency_newton_solver.")
            print(omega_guess,error)
            return 0.002

      return omega_start + omega_guess

   def rk4(self,t0, dt, f):
      k1 = f(t0)
      k2 = f(t0+0.5*dt)
      k3 = f(t0+0.5*dt)
      k4 = f(t0+dt)
      return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)

   def u_von_karman_spectral_density(self,omega):
      return self.sigma_u**2 * 2.0*self.Lu/pi/((1.0+(1.339*self.Lu*omega)**2)**(5.0/6.0))

   def v_von_karman_spectral_density(self,omega):
      return self.sigma_v**2 * self.Lv/pi*(1.0+(8.0/3.0)*(1.339*self.Lv*omega)**2)/((1.0+(1.339*self.Lv*omega)**2)**(11.0/6.0))

   def w_von_karman_spectral_density(self,omega):
      return self.sigma_w**2 * self.Lw/pi*(1.0+(8.0/3.0)*(1.339*self.Lw*omega)**2)/((1.0+(1.339*self.Lw*omega)**2)**(11.0/6.0))

   def get_body_fixed_gust(self,t,y,earth_location):
      ship_location = earth_2_ship(earth_location)
      gust = self.get_ship_fixed_gust(t,y,ship_location)
      gust = ship_2_earth(gust)
      gust = earth_2_body(gust,y[9:13])
      return gust


   def get_ship_fixed_gust(self,t,y,ship_location):
      gust = np.zeros(3)

      if(self.gust_type == 'von_karman'):
         V = m.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
         x = V*t
         for i in range(self.gust_number):
            gust[0] += m.sin(self.gust_u_omega[i]*x + self.gust_u_phase[i])
            gust[1] += m.sin(self.gust_v_omega[i]*x + self.gust_v_phase[i])
            gust[2] += m.sin(self.gust_w_omega[i]*x + self.gust_w_phase[i])

         gust[0] = gust[0]*self.gust_u_amplitude
         gust[1] = gust[1]*self.gust_v_amplitude
         gust[2] = gust[2]*self.gust_w_amplitude

      if(self.gust_type == "database"):
         nplocation = np.array([ship_location[0],ship_location[1],ship_location[2],t])

         gust[0] = self.gust_interp_Vx(nplocation)
         gust[1] = self.gust_interp_Vy(nplocation)
         gust[2] = self.gust_interp_Vz(nplocation)

      if(t < self.gust_ramp_in):
         gust[:] = t/self.gust_ramp_in*gust[:]
         
      return gust

   def gravity_si(self,H):
      if(self.properties != "standard"):
         H = 0.3048*self.properties #assumes altitude is specified in feet
      return 9.806645*(6356766./(6356766.+H))**2

   def gravity_english(self,H):
      if(self.properties != "standard"):
         H = self.properties #assumes altitude is specified in feet
      return 9.806645*(6356766./(6356766. + H*0.3048))**2/0.3048
   #   return 32.1741

   def stdatm_si(self,H):
      levels = np.array([0.,11000.,20000.,32000.,47000.,52000.,61000.,79000.,90000])
      temps = np.array([288.15,216.65,216.65,228.65,270.65,270.65,252.65,180.65,180.65])
      tprime = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.0, -4.0, 0.0])
      tprime = tprime/1000.0
      go = 9.806645
      R = 287.0528
      RE = 6356766.
      po = 101325.
      gamma = 1.4
      
      if(self.properties != "standard"):
         H = 0.3048*self.properties #assumes altitude is specified in feet

      Z=RE*H/(RE+H)

      Z = max(Z,0.0)
      # print(Z)
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

      #Dynamic Viscosity https://www.cfd-online.com/Wiki/Sutherland%27s_law
      T0 = 273.15 #Kelvin
      mu0 = 0.00001716 #kg/m-s
      C = 110.4 #Sutherland's Constant for air
      mu = mu0*(T0+C)/(T+C)*(T/T0)**1.5
      return Z,T,p,rho,a,mu

   def stdatm_english(self,H):
      Z,T,p,rho,a,mu = self.stdatm_si(H*0.3048)
      Z/=0.3048
      T*=1.8
      p*=0.020885434304801722
      rho*=0.00194032032363104
      a/=0.3048
      mu/=47.88025898 #slugs/(ft-s)
      return Z,T,p,rho,a,mu

   def atm_print(self):
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
