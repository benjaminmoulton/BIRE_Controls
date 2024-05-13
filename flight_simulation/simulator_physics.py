from math import cos, sin, atan, atan2, asin, sqrt, exp, pi
import numpy as np
from scipy.integrate import ode, odeint
from scipy import optimize
import matplotlib.pyplot as plt
import json
from fit_damped_sinusoid import *

from atmospheric_functions import statsi, statee, gravity_si, gravity_english
from quaternion_functions import NormQuat2, Euler2Quat, Quat2Euler, Body2Fixed
from trim_functions import solve_trim

import sys

aero_directory = '../aerodynamics_model/'
sys.path.insert(1, aero_directory)

from f16_aero import F16Aero
from bire_aero import BIREAero
from aircraft_properties import AircraftProperties

def rewrite_input_file(t_total, phi, V, H, trim_type, de, dr, dq, BIRE,
                       cg_shift, STALL, COMP, SIMPLE_THRUST,
                       filename='simulator_input.json'):
    
    '''REWRITE INPUT FILE FOR CURRENT CASE'''
    
    json_vals=open(filename).read()
    input_dict = json.loads(json_vals)
    
    input_dict["simulation"]["total_time[sec]"] = t_total

    input_dict["initial"]["trim"]["bank_angle[deg]"] = phi
    input_dict["initial"]["airspeed[ft/s]"] = V
    input_dict["initial"]["altitude[ft]"] = H
    input_dict["initial"]["trim"]["type"] = trim_type
    
    input_dict["perturbations"]["delta_de[deg]"] = de
    input_dict["perturbations"]["delta_dr[deg]"] = dr
    input_dict["perturbations"]["delta_q[deg/s]"] = dq
    
    input_dict["aircraft"]["BIRE"] = BIRE
    input_dict["aircraft"]["CG_shift[ft]"] = cg_shift
    input_dict["aircraft"]["prop_fit_model"] = not SIMPLE_THRUST

    input_dict["aerodynamics"]["stall_model"]["use_stall_model"] = STALL
    input_dict["aerodynamics"]["compressibility_model"]["use_comp_model"] = COMP
    
    with open("simulator_input.json", "w") as outfile:
        json.dump(input_dict, outfile,indent=4)

class BIREAero(BIREAero):
    '''Inhert BIREAero class and redefine the Cn beta function to address
    solution stability issues.
    '''
    def _Cn_beta(self, d_B):
        '''ONLY HERE FOR TESTING FLIGHT SIM ISSUES'''
        return -0.01
    #     return 0.2426 #F-16 value
    
# class F16Aero(F16Aero):
    # def _Cn_beta(self, d_B):
    #     '''ONLY HERE FOR TESTING FLIGHT SIM ISSUES'''
    #     return -0.01
    #     return 0.2426 #F-16 value

class simulator:

    """Flight Simulator Class

    Methods of significance
    -----------------------
    load_file():
              Function that loads simulation, aircraft, and reference state values.
    init_states():
              Initializes aircraft coefficients for drag and reference state
              thrust and lift.
    trim_func():
              Solves the system of equations for trim values at the flight
              conditions specified through json input. Initializes the simulator
              state values at these trim conditions.
    run_sim():
              Updates flight simulator state values at each timestep over the
              time interval provided.
    aero():
              Solves for aerodynamic forces and moments at each timestep (RK4 step)

    Notes
    ------
    There is a significant number of additonal methods included in this class.
    These methods are organized by application and include their own descriptions.
    """

#-----------------------------------------------------------------------------#
#SIMULATOR INITIALIZATION FUNCTIONS

    def __init__(self, init_filename = 'simulator_input.json'):
        
        filename = init_filename
        
        json_vals=open(filename).read()
        input_dict = json.loads(json_vals)

        # general simulation settings
        self.constant_density = input_dict["simulation"]["constant_density"]
        self.dt = input_dict["simulation"]["time_step[sec]"]
        self.ti = 0.0
        self.t_total = input_dict["simulation"]["total_time[sec]"]
        self.rk4_integrate = input_dict["simulation"]["rk4_integration"] # specify RK4 or scipy ODE integration
        
        #perturbation study inputs
        self.tpert = input_dict["perturbations"]["time_start[sec]"]
        self.tpert_dur = input_dict["perturbations"]["time_duration[sec]"]
        self.dda = np.deg2rad(input_dict["perturbations"]["delta_da[deg]"])
        self.dde = np.deg2rad(input_dict["perturbations"]["delta_de[deg]"])
        self.ddr = np.deg2rad(input_dict["perturbations"]["delta_dr[deg]"])
        self.ddq = np.deg2rad(input_dict["perturbations"]["delta_q[deg/s]"])
        
        # aircraft specific inputs
        self.BIRE = input_dict["aircraft"]["BIRE"]
        prop_fit_model = input_dict["aircraft"]["prop_fit_model"]
        if self.BIRE == True:
            self.AeroModel = BIREAero(aero_directory, use_fitted_thrust_model=prop_fit_model)
        else:
            self.AeroModel = F16Aero(aero_directory, use_fitted_thrust_model=prop_fit_model)
        self.cg_shift = input_dict["aircraft"]["CG_shift[ft]"]
        
        # initial simulation values
        self.V = input_dict["initial"]["airspeed[ft/s]"]
        self.long0 = input_dict["initial"]["longitude[deg]"]
        self.lat0 = input_dict["initial"]["latitude[deg]"]
        self.H = input_dict["initial"]["altitude[ft]"]
        self.psi = input_dict["initial"]["heading[deg]"]
        self.psiprev = self.psi
        self.zf = -self.H
        self.xf = 0.0
        self.yf = 0.0
        self.z0 = self.zf
                
        # initialize aircraft properties
        self.AircraftProps = AircraftProperties(V = self.V, H = self.H, bire=self.BIRE)
        self.AircraftProps.hx = 0.0 # FOR TESTING ONLY

        # geometric properties
        self.Sw = self.AircraftProps.S_w
        self.bw = self.AircraftProps.b_w
        self.cw = self.AircraftProps.c_w
        self.g = self.AircraftProps.g
        self.W = self.AircraftProps.W
        # flight condition and atmospheric properties
        self.rho_0 = self.AircraftProps.rho_0
        self.rho = self.AircraftProps.rho
        self.a_0 = self.AircraftProps.a_0
        self.M = self.AircraftProps.M
        
        # aerodynamic model settings
        self.compressible = input_dict["aerodynamics"]["compressibility_model"]["use_comp_model"]
        self.stall = input_dict["aerodynamics"]["stall_model"]["use_stall_model"]

        # trim or state inputs
        init_type = input_dict["initial"]["type"]
        if init_type == "state":
            self.theta = np.deg2rad(input_dict["initial"]["state"]["elevation_angle[deg]"])
            self.phi = np.deg2rad(input_dict["initial"]["state"]["bank_angle[deg]"])
            self.alpha = np.deg2rad(input_dict["initial"]["state"]["alpha[deg]"])
            self.beta = np.deg2rad(input_dict["initial"]["state"]["beta[deg]"])
            self.p = np.deg2rad(input_dict["initial"]["state"]["p[deg/s]"])
            self.q = np.deg2rad(input_dict["initial"]["state"]["q[deg/s]"])
            self.r = np.deg2rad(input_dict["initial"]["state"]["r[deg/s]"])
            self.da0 = np.deg2rad(input_dict["initial"]["state"]["aileron[deg]"])
            self.de0 = np.deg2rad(input_dict["initial"]["state"]["elevator[deg]"])
            self.dr0 = np.deg2rad(input_dict["initial"]["state"]["rudder[deg]"])
            self.tau0 = input_dict["initial"]["state"]["throttle"]
 
            self.u = self.V*cos(self.alpha)*cos(self.beta)
            self.v = self.V*sin(self.beta)
            self.w = self.V*sin(self.alpha)*cos(self.beta)
            
        elif init_type == "trim":
            
            # solves for trim solution of the specified type
            self.trim_type = input_dict["initial"]["trim"]["type"]
            # there are some options that could be added here based on input
            # but i will come back to that
            self.phi = np.deg2rad(input_dict["initial"]["trim"]["bank_angle[deg]"])
            self.gamma = np.deg2rad(input_dict["initial"]["trim"]["climb_angle[deg]"])
                
            # self.theta0 = input_dict["trim"]["___elevation_angle[deg]"]
            # self.beta0 = input_dict["trim"]["___sideslip_angle[deg]"]
            
            self.df = input_dict["initial"]["trim"]["solver"]["finite_difference_step_size"] #needed?
            Gamma = input_dict["initial"]["trim"]["solver"]["relaxation_factor"]
            self.AircraftProps.Gamma = Gamma
            tol = input_dict["initial"]["trim"]["solver"]["tolerance"]
            
            if self.trim_type == "shss":
                SHSS = True
            else:
                SHSS = False

            self.solution = solve_trim(aero_model = self.AeroModel, aircraft_props = self.AircraftProps,
                                  gamma = self.gamma, phi = self.phi, cg_shift=self.cg_shift,
                                  shss = SHSS, compressible = self.compressible, stall = self.stall,
                                  tol = tol, Gamma = Gamma)
            
            # pull necessary states from trim solution
            self.FM = self.solution.FM # forces and moments
            self.FM_dim = self.solution.FM_dim # dimensional body fixed forces and moments
            self.p, self.q, self.r = self.solution.rates # rotation rates
            self.u, self.v, self.w = self.solution.velocity # u v w
            self.tau0, self.alpha, self.beta, self.da0, self.de0, self.dr0 = self.solution.x # trim solution [tau, alpha, beta, da, de, dr(dB)]
            self.phi, self.theta, self.psi = self.solution.orient # phi theta psi
        #initiallize quaternion state
        self.e0, self.ex, self.ey, self.ez = Euler2Quat([self.phi,self.theta,self.psi])
        #update mass properties following trim solution
        self.dr = self.dr0
        self.mass()
        
        # for testing/checking stability values of interest
        if self.BIRE == True:
            print('Cm,alpha: ', self.AeroModel._Cm_alpha(self.dr0))
            print('Cl,beta: ', self.AeroModel._Cl_beta(self.dr0))
            print('Cn,beta: ', self.AeroModel._Cn_beta(self.dr0))
        else:
            print('Cm,alpha: ', self.AeroModel.Cma)
            print('Cl,beta: ', self.AeroModel.Clb)
            print('Cn,beta: ', self.AeroModel.Cnb)
        
        # sets the integration method as either RK4 or scipy ODE
        if self.rk4_integrate == True:
            self.int_method = self.rnkta4
        else:
            # initialize scipy ODE integration function
            atol=1e-12;rtol=1e-6
            self.integrator = ode(self.derivs)
            self.integrator.set_integrator("dopri5",atol=atol,rtol=rtol)
            y = np.array([self.u, self.v, self.w, self.p, self.q, 
                                self.r, self.xf, self.yf, self.zf, self.e0,
                                self.ex, self.ey, self.ez])
            self.integrator.set_initial_value(y * 1., 0.0)
        
            self.int_method = self.ode_integrate

        # set some initial values to be referenced and used later
        self.V0 = self.V
        self.H0 = self.H
        self.alpha0 = self.alpha*180./np.pi
        self.beta0 = self.beta*180./np.pi
        self.theta0 = self.theta*180./np.pi
        self.phi0 = self.phi*180./np.pi
        self.u0, self.v0, self.w0 = self.u, self.v, self.w
        self.p0, self.q0, self.r0 = self.p,self.q,self.r

#-----------------------------------------------------------------------------#
#MAIN SIMULATOR FUNCTION
    def run_sim(self, state_fn='states.txt', forces_fn='forces.txt', plot_results = False):
        # runs sim for specified time, writes and plots states
        self.plot_flag = plot_results
        with(open(state_fn, 'w+', encoding='utf-8')) as self.f:
            with(open(forces_fn, 'w+', encoding='utf-8')) as self.f_Fb:
                #initialize result writing and plotting, comments out to stop
                self.plot_results(0)
                self.write_results(0)
                self.write_forces(0)
                while self.ti < self.t_total:
                    self.iter_sim()
                    if -self.zf < 0.0:
                        break
        if self.plot_flag == True:
            self.plot_results(2)
        self.update_final_state_data()

    def iter_sim(self):

        '''Run the flight simulator for current iteration. Calls all necessary 
        member functions to update the current aircraft simulator state.
        Also prints/plots/ and writes state values if desired.
        '''
        # update state array to most recent values
        self.y = np.array([self.u, self.v, self.w, self.p, self.q, 
                           self.r, self.xf, self.yf, self.zf, self.e0,
                           self.ex, self.ey, self.ez])
        
        # integrates states forward        
        sol = self.int_method(self.ti, self.y, self.dt, self.derivs)
        
        #update time with time step
        self.ti = self.ti + self.dt
        
        #uses RK4 solution to update current state values
        
        #state velocities
        self.u = sol[0]
        self.v = sol[1]
        self.w = sol[2]
        
        #state rotation rates
        self.p = sol[3]
        self.q = sol[4]
        self.r = sol[5]
        
        #state positions
        self.xf = sol[6]
        self.yf = sol[7]
        self.zf = sol[8]
        
        #state quaternions
        self.e0, self.ex, self.ey, self.ez = NormQuat2([sol[9],
                                                        sol[10], 
                                                        sol[11], 
                                                        sol[12]])
        
        #state euler angles
        self.phi, self.theta, self.psi = Quat2Euler([self.e0, 
                                                    self.ex, 
                                                    self.ey,
                                                    self.ez],
                                                    self.psiprev)
        self.heading = self.fix_heading(self.psi)
        
        #state airspeed
        self.V = np.sqrt((self.u*self.u) + (self.v*self.v) + 
                         (self.w*self.w))
        
        #state angle of attack and sideslip angle
        self.alpha = atan(self.w/self.u)
        self.beta = asin(self.v/self.V)
        
        self.psiprev = self.psi
        
        #earth fixed velocities
        self.earth_vel()
        self.gnd_V = sqrt(self.Vxf*self.Vxf + self.Vyf*self.Vyf)
        
        #state densitity, constant or updated with alititude
        h,z,t,p,d,a = statee(-self.zf)
        
        if self.constant_density != True:
            self.rho = d
            self.M = self.V/a      
            self.H = -self.zf
            
        # #finalize states plot and write
        self.write_results(1)
        self.write_forces(1)
        self.plot_results(1)

#-----------------------------------------------------------------------------#
#REAL TIME STATE UPDATE FUNCTIONS

    def mass(self):

        '''Updates aircraft mass properties based on fuel and payload loss.
        Inverts the moment of inertia matrix.

        Parameters
        -----------
        '''

        #aircraft mass properties, including inverse of interia tensor
        if self.BIRE == False:
            self.Ixx = self.AircraftProps.Ixx
            self.Iyy = self.AircraftProps.Iyy
            self.Izz = self.AircraftProps.Izz
            self.Ixy = self.AircraftProps.Ixy
            self.Ixz = self.AircraftProps.Ixz
            self.Iyz = self.AircraftProps.Iyz
        else:
            self.AircraftProps.calc_BIRE_inertia(self.dr)
            self.Ixx = self.AircraftProps.Ixx
            self.Iyy = self.AircraftProps.Iyy
            self.Izz = self.AircraftProps.Izz
            self.Ixy = self.AircraftProps.Ixy
            self.Ixz = self.AircraftProps.Ixz
            self.Iyz = self.AircraftProps.Iyz

        self.Iinv = np.linalg.inv([[self.Ixx, -self.Ixy, -self.Ixz], 
                                   [-self.Ixy, self.Iyy, -self.Iyz],
                                   [-self.Ixz, -self.Iyz, self.Izz]])
            
        self.hx = self.AircraftProps.hx
        self.hy = self.AircraftProps.hy
        self.hz = self.AircraftProps.hz
        
    def wind(self):

        '''Updates wind velocities of the simulator based on time and position.

        Parameters
        -----------
        t: integer or float
            current time step
        pos: array of floats
            current position array
        '''

        self.Vwxf = 0.
        self.Vwyf = 0.
        self.Vwzf = 0.

    def control(self):

        '''Updates aircraft control inputs.

        Parameters
        -----------
        '''
        if (self.ti > self.tpert)*(self.ti < (self.tpert + self.tpert_dur)):
            
            #throttle setting
            self.tau = self.tau0

            #aileron input
            self.da = self.da0 + self.dda

            #elevator input
            self.de = self.de0 + self.dde

            #rudder input
            self.dr = self.dr0 + self.ddr
        else:
            #throttle setting
            self.tau = self.tau0
            #aileron input
            self.da = self.da0
            #elevator input
            self.de = self.de0
            #rudder input
            self.dr = self.dr0
        

    def aero(self,p,q,r):
        
        if (self.ti > self.tpert)*(self.ti < (self.tpert + self.tpert_dur)):
            
            #throttle setting
            q = q + self.ddq
            
        pbar = p*self.bw/(2.*self.V)
        qbar = q*self.cw/(2.*self.V)
        rbar = r*self.bw/(2.*self.V)
        
        FM_dim = self.AeroModel.aero_CG_offset_results(self.alpha, self.beta, pbar, qbar, rbar, self.da, self.de, self.dr, self.tau,
                                                   self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                   M = self.M, use_Anderson = True, enforce_stall=self.stall)
        
        self.Fxb,self.Fyb,self.Fzb,self.Mxb,self.Myb,self.Mzb = FM_dim
        
        # print('Forces at iter: ', FM)

#-----------------------------------------------------------------------------#
#INTEGRATION FUNCTIONS

    def derivs(self, t0, y0):

        '''Differential equations of rigid-body motion using the quaternion
        formulation. Calculates the derivatives of aircraft motion that are
        used in the Runge-Kutta 4 formulation. Defined in Mechanics of Flight,
        Second Edition, by Warren F. Phillips, Eq. (11.11.1) - (11.11.4).

        Parameters
        -----------
        t0: integer or float
            current time step
        y0: array
            array of current state values

        Returns
        -------
        sol: array
            array of state derivatives
        '''

        #current state values
        u, v, w, p, q, r, xf, yf, zf, e0, ex, ey, ez = y0

        #current airspeed
        self.V = np.sqrt((u*u) + (v*v) + (w*w))

        #function calls to state inputs, provides force, moment, and control
        #input values at current state
        self.wind()
        self.control()
        self.mass()
        self.aero(p,q,r)

        #velocity derivatives
        udot = (self.g*2*(ex*ez - ey*e0) + (self.g/self.W)*(self.Fxb) 
                + r*v - q*w)
        vdot = (self.g*2*(ey*ez + ex*e0) + (self.g/self.W)*(self.Fyb) + 
                p*w - r*u)
        wdot = (self.g*(ez*ez + e0*e0 - ex*ex - ey*ey) + 
                (self.g/self.W)*(self.Fzb) + q*u - p*v)

        #variables for rotation rate derivatives
        t1 = (-self.hz*q + self.hy*r + self.Mxb + (self.Iyy - self.Izz)*q*r + 
                self.Iyz*(q*q - r*r) + self.Ixz*p*q - self.Ixy*p*r)

        t2 = (self.hz*p - self.hx*r + self.Myb + (self.Izz - self.Ixx)*p*r + 
              self.Ixz*(r*r - p*p) + self.Ixy*q*r - self.Iyz*p*q)

        t3 = (-self.hy*p + self.hx*q + self.Mzb + (self.Ixx - self.Iyy)*p*q + 
              self.Ixy*(p*p - q*q) + self.Iyz*p*r - self.Ixz*q*r)

        #rotation rate derivatives
        pdot = self.Iinv[0,0]*t1 + self.Iinv[0,1]*t2 + self.Iinv[0,2]*t3
        qdot = self.Iinv[1,0]*t1 + self.Iinv[1,1]*t2 + self.Iinv[1,2]*t3
        rdot = self.Iinv[2,0]*t1 + self.Iinv[2,1]*t2 + self.Iinv[2,2]*t3

        #position derivatives
        xdot, ydot, zdot = Body2Fixed([u, v, w], [e0, ex, ey, ez])
        xdot = xdot + self.Vwxf
        ydot = ydot + self.Vwyf
        zdot = zdot + self.Vwzf

        #quaternion derivatives
        e0dot = 0.5*(-ex*p - ey*q - ez*r)
        exdot = 0.5*(e0*p - ez*q + ey*r)
        eydot = 0.5*(ez*p + e0*q - ex*r)
        ezdot = 0.5*(-ey*p + ex*q + e0*r)

        #array of all state derivatives
        sol = np.array([udot, vdot, wdot, pdot, qdot, rdot,
                        xdot, ydot, zdot, e0dot, exdot, eydot, ezdot])
        return sol

    def ode_integrate(self,t0,y0,dt,f):
        
        y = self.integrator.integrate(self.integrator.t+dt)
        
        return y
        
    def rnkta4(self,t0,y0,dt,f,n=13):

        '''Updates aerodynamic forces and moments at each timestep and RK4 call

        Parameters
        -----------
        t0: integer or float
            timestep initial condition
        y0: array
            array of current state conditions
        dt: interger or float
            timestep size
        f: function
            derivative function of values being integrated forward

        Returns
        -------
        y: array
            Next step of state values, predicted through Runge-Kutta formulation
        '''

        # pre-allocate arrays
        k1i = np.zeros(n)
        k2i = np.zeros(n)
        k3i = np.zeros(n)
        k4i = np.zeros(n)

        # all array operations for n # of variables
        # Runge Kutta 4 formulation
        k1i = f(t0, y0)
        yi = y0 + 0.5*k1i*dt

        k2i = f(t0 + 0.5*dt, yi)
        yi = y0 + 0.5*k2i*dt

        k3i = f(t0 + 0.5*dt, yi)
        yi = y0 + k3i*dt

        k4i = f(t0 + dt, yi)

        #final Runge-Kutta estimate
        y = y0 + (1/6)*(k1i + 2*k2i + 2*k3i + k4i)*dt

        return y

 #-----------------------------------------------------------------------------#
#Equilibrium State Functions

    def fix_heading(self,psi):
        '''maintains heading as a value between 0 and 2pi'''
        if psi < 0:
            h = 2*pi + psi
        else:
            h = psi
        return h

    def earth_vel(self):
        CT = cos(self.theta)
        CPS = cos(self.psi)
        CPH = cos(self.phi)
        ST = sin(self.theta)
        SPS = sin(self.psi)
        SPH = sin(self.phi)
        self.Vxf = CT*CPS*self.u + (SPH*ST*CPS - CPH*SPS)*self.v + (CPH*ST*CPS + SPH*SPS)*self.w
        self.Vyf = CT*SPS*self.u + (SPH*ST*SPS + CPH*CPS)*self.v + (CPH*ST*SPS - SPH*CPS)*self.w
        self.Vzf = -ST*self.u + SPH*CT*self.v + CPH*CT*self.w

    def update_vel(self,V):

        '''finds xb, yb, and zb components of airspeed

        Parameters
        -----------
        V: float
            current airspeed
        '''

        denom = sqrt(1-sin(self.alpha)*sin(self.alpha)*sin(self.beta)*sin(self.beta))

        self.uo = V*(cos(self.alpha)*cos(self.beta))/denom
        self.vo = V*(cos(self.alpha)*sin(self.beta))/denom
        self.wo = V*(sin(self.alpha)*cos(self.beta))/denom

#-----------------------------------------------------------------------------#
#Result plotting and writing functions
        
    def write_results(self, i):

        '''writes results to txt file

        Parameters
        ----------
        i: integer
            i= 0 is header, i = 1 writes values
        '''

        if i == 0:
            self.f.truncate(0)
            self.f.write('{0:<17} {1:<17} {2:<17} {3:<17} {4:<17} {5:<17} {6:<17} {7:<17} {8:<17} {9:<17} {10:<17} {11:<17} {12:<17} {13:<17} {14:<17} {15:<17}'.format('   Time[sec]', '   u[ft/s]', '   v[ft/s]', '   w[ft/s]', '   p[rad/s]', '   q[rad/s]', '   r[rad/s]', '   x[ft]', '   y[ft]', '   z[ft]', '   alpha[deg]', '   beta[deg]', '   phi[deg]   ', '   psi[deg]   ', '   eo', '   e1', '   e2', '   e3')+'\n')
        elif i == 1:
            self.f.write('{0:>18.9E} {1:>17.9E} {2:>17.9E} {3:>17.9E} {4:>17.9E} {5:>17.9E} {6:>17.9E} {7:>17.9E} {8:>17.9E} {9:>17.9E} {10:>17.9E} {11:>17.9E} {12:>17.9E} {13:>17.9E} {14:>17.9E} {15:>17.9E}'.format(self.ti, self.u, self.v, self.w, self.p, self.q, self.r, self.xf, self.yf, self.zf, self.alpha*180/np.pi, self.beta*180/np.pi, self.phi*180/np.pi, self.psi*180/np.pi, self.e0, self.ex,self.ey, self.ez)+'\n')
    
    def write_forces(self, i):

        '''writes results to txt file

        Parameters
        ----------
        i: integer
            i= 0 is header, i = 1 writes values
        '''

        if i == 0:
            self.f_Fb.truncate(0)
            self.f_Fb.write('{0:<17} {1:<17} {2:<17} {3:<17} {4:<17} {5:<17} {6:<17}'.format('   Time[sec]', '   Fxb   ', '   Fyb   ', '   Fzb   ', '   Mxb   ', '   Myb   ', '   Mzb   ')+'\n')
        elif i == 1:
            self.f_Fb.write('{0:>18.9E} {1:>17.9E} {2:>17.9E} {3:>17.9E} {4:>17.9E} {5:>17.9E} {6:>17.9E}'.format(self.ti, self.Fxb, self.Fyb, self.Fzb, self.Mxb, self.Myb, self.Mzb)+'\n')

    def normalize_states(self):
        
        self.x_plot = np.array(self.x_plot)
        self.y_plot = np.array(self.y_plot)
        self.z_plot = np.array(self.z_plot)
        self.u_plot = np.array(self.u_plot)
        self.v_plot = np.array(self.v_plot)
        self.w_plot = np.array(self.w_plot)
        self.p_plot = np.array(self.p_plot)
        self.q_plot = np.array(self.q_plot)
        self.r_plot = np.array(self.r_plot)
        
        self.u_norm = ((self.u_plot - self.u0)/max(abs(self.u_plot - self.u0)))
        self.v_norm = ((self.v_plot - self.v0)/max(abs(self.v_plot - self.v0)))
        self.w_norm = ((self.w_plot - self.w0)/max(abs(self.w_plot - self.w0)))
        self.p_norm = ((self.p_plot - self.p0)/max(abs(self.p_plot - self.p0)))
        self.q_norm = ((self.q_plot - self.q0)/max(abs(self.q_plot - self.q0)))
        self.r_norm = ((self.r_plot - self.r0)/max(abs(self.r_plot - self.r0)))
        self.x_norm = ((self.x_plot)/max(abs(self.x_plot)))
        self.y_norm = ((self.y_plot)/max(abs(self.y_plot)))
        self.z_norm = ((self.z_plot - self.z0)/max(abs(self.z_plot - self.z0)))
        # self.phi_norm = ((self.phi_plot - self.phi0)/max(abs(self.phi_plot - self.phi0)))
        # self.theta_norm = ((self.theta_plot - self.theta0)/max(abs(self.theta_plot - self.theta0)))
        # self.psi_norm = ((self.psi_plot)/max(abs(self.psi_plot)))
        
        
        plt.figure()
        # plt.plot(self.u_norm)
        # plt.plot(self.v_norm)
        # plt.plot(self.w_norm)
        plt.show()
        
    def update_final_state_data(self):
        self.time_plot = np.array(self.time_plot)
        self.airspeed_plot = np.array(self.airspeed_plot)
        self.x_plot = np.array(self.x_plot)
        self.y_plot = np.array(self.y_plot)
        self.z_plot = np.array(self.z_plot)
        self.u_plot = np.array(self.u_plot)
        self.v_plot = np.array(self.v_plot)
        self.w_plot = np.array(self.w_plot)
        self.p_plot = np.array(self.p_plot)
        self.q_plot = np.array(self.q_plot)
        self.r_plot = np.array(self.r_plot)
        self.phi_plot = np.array(self.phi_plot)
        self.theta_plot = np.array(self.theta_plot)
        self.psi_plot = np.array(self.psi_plot)
        self.alpha_plot = np.array(self.alpha_plot)
        self.beta_plot = np.array(self.beta_plot)
        
    def plot_results(self, i):

        '''plots simulator results

        Parameters
        ----------
        i: integer
            i= 0 initializes lists, i = 1 appends lists, i = 2 plots results
        '''

        #appends and stores values for plotting, i = 2 then plots results
        if i == 0:
            self.time_plot = []
            self.range_plot = []
            self.airspeed_plot = []
            self.x_plot = []
            self.y_plot = []
            self.z_plot = []
            self.u_plot = []
            self.v_plot = []
            self.w_plot = []
            self.p_plot = []
            self.q_plot = []
            self.r_plot = []
            self.phi_plot = []
            self.theta_plot = []
            self.psi_plot = []
            self.alpha_plot = []
            self.beta_plot = []
        elif i == 1:
            self.time_plot.append(self.ti)
#            self.range_plot.append(self.range_check)
            self.airspeed_plot.append(self.V)
            self.x_plot.append(self.xf)
            self.y_plot.append(self.yf)
            self.z_plot.append(-self.zf)
            self.u_plot.append(self.u)
            self.v_plot.append(self.v)
            self.w_plot.append(self.w)
            self.p_plot.append(self.p*(180/pi))
            self.q_plot.append(self.q*(180/pi))
            self.r_plot.append(self.r*(180/pi))
            self.phi_plot.append(self.phi*(180/pi))
            self.theta_plot.append(self.theta*(180/pi))
            self.psi_plot.append(self.psi*(180/pi))
            self.alpha_plot.append(self.alpha*(180/pi))
            self.beta_plot.append(self.beta*(180/pi))

        elif i == 2:
            
            plt.figure(1)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.theta_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Elevation Angle (degrees)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()

            plt.figure(2)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.phi_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Bank Angle (degrees)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()

            plt.figure(3)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.alpha_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Angle of Attack (degrees)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()
            
            plt.figure(4)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.airspeed_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Airspeed (ft/s)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()
            
            plt.figure(5)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.x_plot)
            plt.xlabel('Time, s')
            plt.ylabel('x [ft]')
            plt.tight_layout()
            plt.show()
            
            plt.figure(6)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.y_plot)
            plt.xlabel('Time, s')
            plt.ylabel('y [ft]')
            plt.tight_layout()
            plt.show()
            
            plt.figure(7)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.z_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Height [ft]')
#            plt.xlim([0,100])
            plt.ylim(min(self.z_plot) - 10, max(self.z_plot) + 10)
            plt.tight_layout()
            plt.show()
            
            plt.figure(8)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.u_plot)
            plt.plot(self.time_plot, self.v_plot)
            plt.plot(self.time_plot, self.w_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Body Fixed Velocities [ft/s]')
            plt.legend(['u','v','w'])
            plt.tight_layout()
            plt.show()
            
            plt.figure(9)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.psi_plot)
            plt.xlabel('Time, s')
            plt.ylabel('Heading (degrees)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()
            
            plt.figure(10)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.p_plot)
            plt.xlabel('Time, s')
            plt.ylabel('p (deg/s)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()

            plt.figure(11)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.q_plot)
            plt.xlabel('Time, s')
            plt.ylabel('q (deg/s)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()
            
            plt.figure(12)
            plt.grid(visible=True)
            plt.plot(self.time_plot, self.r_plot)
            plt.xlabel('Time, s')
            plt.ylabel('r (deg/s)')
#            plt.xlim([0,100])
            plt.tight_layout()
            plt.show()

            fig = plt.figure(figsize = (10,10))
            ax = fig.add_subplot(111, projection = '3d')
            ax.plot(self.y_plot, self.x_plot, self.z_plot)
            # ax.set_zlim3d(bottom=self.H0-500, top=self.H0+500)
            plt.show()
