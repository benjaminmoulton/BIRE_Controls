import sys
import numpy as np
import json
from scipy.linalg import eig
import matplotlib.pyplot as plt

# add the BIRE specific directorys to the system path
aero_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/aerodynamics_model/'
trim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/trim/'
sim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/flight_simulation/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, trim_directory)
sys.path.insert(1, sim_directory)

from f16_aero import F16Aero
from bire_aero import BIREAero
from aero_trim import trim
from aircraft_properties import AircraftProperties
from trim_functions import solve_trim
from thrust import Propulsion
from dynamic_derivatives import solveDerivatives, test_derivs
from fit_damped_sinusoid import damped_sinusoid

class BIREAero(BIREAero):
    '''Inhert BIREAero class and redefine the Cn beta function to address
    solution stability issues.
    '''
    def _Cn_beta(self, d_B):
        '''ONLY HERE FOR TESTING FLIGHT SIM ISSUES'''
        return -0.01

class dynamicAnalysis:
            
    def __init__(self, path='./', write_output = False, output_filename = 'dynamic_output.txt',
                 BIRE=False, shss=False, compressible=False, stall=False, coords_approx = False, derivs_approx = False,
                 cg_shift=[0.0, 0.0, 0.0], JSON_input = False, simple_thrust = True):
        '''
        Load all the non-changing values related to either aircraft.
        props for loads type data
        '''
        self.write_output = write_output
        self.output_filename = output_filename
        
        if JSON_input == False:
            self.BIRE = BIRE
            self.cg_shift = cg_shift
            self.shss = shss
            self.compressible = compressible
            self.stall = stall
            self.coords_approx = coords_approx
            self.derivs_approx = derivs_approx
            self.simple_thrust = simple_thrust
            
            self.Gamma = 0.5 # relaxation factor for the trim algorithm, may not need to ever update this
            
            # initialize the aerodynamic model
            if self.BIRE == False:
                self.aeroModel = F16Aero(inp_dir=aero_directory)
            else:
                self.aeroModel = BIREAero(inp_dir=aero_directory)
                
            self.thrust_model = Propulsion(inp_dir=aero_directory)
        else:
            derivs = test_derivs()
            derivs.get_JSON_inputs('9.8.1.json')
            
            deriv_solution = derivs.solve_derivs()
            self.set_deriv_solution(deriv_solution)
            
            self.V = derivs.V
            self.W = derivs.W
            self.g = derivs.g
            self.Sw = derivs.Sw
            self.bw = derivs.bw
            self.cw = derivs.cw
            self.rho = derivs.rho
            self.eq_euler = derivs.eq_euler
            self.eq_velo = derivs.eq_vel
            self.eq_rot = derivs.eq_rot
            self.Ixxb = derivs.Ixxb
            self.Iyyb = derivs.Iyyb
            self.Izzb = derivs.Izzb
            self.Ixyb = derivs.Ixyb
            self.Ixzb = derivs.Ixzb
            self.Iyzb = derivs.Iyzb
            
            self.hxb = derivs.hxb
            self.hyb = derivs.hyb
            self.hzb = derivs.hzb
        
            
    def update_aircraft_properties(self, V, H, dB = 0.0):
                
        self.aircraft_properties = AircraftProperties(V, H, self.Gamma, bire=self.BIRE)
        
        # update operating properties
        self.V = self.aircraft_properties.V
        self.H =self.aircraft_properties.H
        self.g = self.aircraft_properties.g
        self.nondim_const = self.aircraft_properties.nondim_const
        self.rho = self.aircraft_properties.rho
        self.rho_0 = self.aircraft_properties.rho_0
        self.M = self.aircraft_properties.M
        self.a = self.aircraft_properties.a
        self.a_0 = self.aircraft_properties.a_0
        
        # update aircraft geometric properties
        self.bw = self.aircraft_properties.b_w
        self.cw = self.aircraft_properties.c_w
        self.Sw = self.aircraft_properties.S_w
        
        self.Sh = 63.675 #NEEDS TO BE UPDATED TO PULL FROM THE JSON PROPERTIES
        
        # BORROWED FROM AUSTINS CODE
        xbwt = -7.358
        self.xbh = -13.13
        self.lwt = 1.1 * (xbwt - self.xbh)       
        self.CLw_a = 3.3775691217788646
        
        # update aircraft mass and inertia properties
        self.W = self.aircraft_properties.W

        if self.BIRE == False:
            '''F-16'''
            self.Ixxb = self.aircraft_properties.Ixx
            self.Iyyb = self.aircraft_properties.Iyy
            self.Izzb = self.aircraft_properties.Izz
            self.Ixyb = self.aircraft_properties.Ixy
            self.Ixzb = self.aircraft_properties.Ixz
            self.Iyzb = self.aircraft_properties.Iyz
            self.CLh_a = 1.3657050471586294
            self.aircraft_properties.CLh_a = self.CLh_a
        else:
            '''BIRE'''
            self.aircraft_properties.calc_BIRE_inertia(dB)
            self.Ixxb = self.aircraft_properties.Ixx
            self.Iyyb = self.aircraft_properties.Iyy
            self.Izzb = self.aircraft_properties.Izz
            self.Ixyb = self.aircraft_properties.Ixy
            self.Ixzb = self.aircraft_properties.Ixz
            self.Iyzb = self.aircraft_properties.Iyz

            #BORROWED FROM AUSTINS CODE
            self.CLh_a = 1.3858047943592773 * np.abs(np.cos(dB))
            self.aircraft_properties.CLh_a = self.CLh_a
        
        self.aircraft_properties.hx = 0.0 # for testing
        self.hxb = self.aircraft_properties.hx
        self.hyb = self.aircraft_properties.hy
        self.hzb = self.aircraft_properties.hz
        
    def solve_equilibrium_state(self, V, H, gamma, phi, cg_shift):
        
        '''
        Given airspeed, altitude, CG shift, climb angle,
        and bank angle, solve equilibrium trim condition.
        '''
        self.update_aircraft_properties(V, H, dB = 0.0)

        self.solution = solve_trim(aero_model = self.aeroModel, aircraft_props=self.aircraft_properties,
                              gamma = gamma, phi = phi, cg_shift=self.cg_shift,
                              shss = self.shss, compressible = self.compressible, stall = self.stall)
        
        tau, alpha, beta, da, de, dr = self.solution.x
        u, v, w, p, q, r, phi, theta = self.solution.states
        
        print('u, v, w, p, q, r:', u, v, w, p, q, r)
        
        FX, FY, FZ, Mx, My, Mz = self.solution.FM_dim
        [CL, CS, CD, Cl, Cm, Cn] = self.solution.FM
        
        
        print('\nBody Force/Moment from trim solution:')
        print(FX, FY, FZ, Mx, My, Mz)
        print('\n')
        
        pbar = p*self.bw/(2.*V)
        qbar = q*self.cw/(2.*V)
        rbar = r*self.bw/(2.*V)
        
        # print('\nBody Force/Moment from CS change:')
        # print(FX, FY, FZ, Mx, My, Mz)
        
        if self.BIRE == True:
            self.update_aircraft_properties(V, H, dB = dr)
        
        self.eq_velo = np.array([u,v,w])
        self.eq_rot = np.array([p,q,r])
        self.eq_euler = np.array([phi,theta])
        self.eq_inputs = np.array([tau, alpha, beta, da, de, dr])
        self.eq_FM_wind = np.array([CL, CS, CD, Cl, Cm, Cn])
        self.eq_FM = np.array([FX, FY, FZ, Mx, My, Mz])
        
        self.alpha = alpha
        self.beta = beta
        
        # return FX, FY, FZ, Mx, My, Mz
    
    def solve_derivatives(self, num_derivs = False):
        
        print('\nSolving new derivatives...')
        
        derivs = solveDerivatives(aeroModel = self.aeroModel, aircraft_properties = self.aircraft_properties, cg_shift = self.cg_shift,
                    trim_solution = self.solution, numerical_derivs = num_derivs, compressible = self.compressible, stall = self.stall,
                    coords_approx = self.coords_approx, derivs_approx = self.derivs_approx, simple_thrust = self.simple_thrust)
        
        deriv_solution = derivs.solve_derivs()
        
        self.set_deriv_solution(deriv_solution)
        
        derivs.print_derivs()
        
    def set_deriv_solution(self,deriv_solution):
        multiplier = 1.0
        self.Fxb_u = deriv_solution.Fxb_u
        self.Fxb_v = deriv_solution.Fxb_v*multiplier
        self.Fxb_w = deriv_solution.Fxb_w
        self.Fxb_p = deriv_solution.Fxb_p*multiplier
        self.Fxb_q = deriv_solution.Fxb_q
        self.Fxb_r = deriv_solution.Fxb_r*multiplier
        
        
        self.Fyb_u = deriv_solution.Fyb_u*multiplier
        self.Fyb_v = deriv_solution.Fyb_v
        self.Fyb_w = deriv_solution.Fyb_w*multiplier
        self.Fyb_p = deriv_solution.Fyb_p
        self.Fyb_q = deriv_solution.Fyb_q*multiplier
        self.Fyb_r = deriv_solution.Fyb_r
        
        self.Fzb_u = deriv_solution.Fzb_u
        self.Fzb_v = deriv_solution.Fzb_v*multiplier 
        self.Fzb_w = deriv_solution.Fzb_w
        self.Fzb_p = deriv_solution.Fzb_p*multiplier
        self.Fzb_q = deriv_solution.Fzb_q
        self.Fzb_r = deriv_solution.Fzb_r*multiplier
                
        self.Mxb_u = deriv_solution.Mxb_u*multiplier
        self.Mxb_v = deriv_solution.Mxb_v 
        self.Mxb_w = deriv_solution.Mxb_w*multiplier
        self.Mxb_p = deriv_solution.Mxb_p
        self.Mxb_q = deriv_solution.Mxb_q*multiplier
        self.Mxb_r = deriv_solution.Mxb_r 
        
        self.Myb_u = deriv_solution.Myb_u
        self.Myb_v = deriv_solution.Myb_v*multiplier
        self.Myb_w = deriv_solution.Myb_w
        self.Myb_p = deriv_solution.Myb_p*multiplier
        self.Myb_q = deriv_solution.Myb_q
        self.Myb_r = deriv_solution.Myb_r*multiplier
        
        self.Mzb_u = deriv_solution.Mzb_u*multiplier
        self.Mzb_v = deriv_solution.Mzb_v
        self.Mzb_w = deriv_solution.Mzb_w*multiplier
        self.Mzb_p = deriv_solution.Mzb_p
        self.Mzb_q = deriv_solution.Mzb_q*multiplier
        self.Mzb_r = deriv_solution.Mzb_r
        
        self.Fzb_wdot = deriv_solution.Fzb_wdot
        self.Myb_wdot = deriv_solution.Myb_wdot

        self.Fxb_udot = deriv_solution.Fxb_udot
        self.Fxb_vdot = deriv_solution.Fxb_vdot
        self.Fxb_wdot = deriv_solution.Fxb_wdot
        self.Fyb_udot = deriv_solution.Fyb_udot
        self.Fyb_vdot = deriv_solution.Fyb_vdot
        self.Fyb_wdot = deriv_solution.Fyb_wdot
        self.Fzb_vdot = deriv_solution.Fzb_vdot
        
        self.Fzb_udot = deriv_solution.Fzb_udot
        self.Myb_udot = deriv_solution.Myb_udot
    
    def solve_dynamics_system(self):
                        
        t_o = 0.0
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]        
        
        phi_o = (self.eq_euler[0])
        theta_o = (self.eq_euler[1])
        
        # print('phi_o:', phi_o*180/np.pi)
        # print('theta_o:', theta_o*180/np.pi)
        
        
        '''INTERNAL CALCULATIONS'''
        
        ST_o = np.sin(theta_o)
        CT_o = np.cos(theta_o)
        TT_o = np.tan(theta_o)
        
        SP_o = np.sin(phi_o)
        CP_o = np.cos(phi_o)
        
        Omega = (self.g*SP_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o) #radians/s
        # print('Omega [deg/s]:', np.rad2deg(Omega))
        
        SO_o = np.sin(Omega*t_o)
        CO_o = np.cos(Omega*t_o)
        
        
        # p_o = (-self.g*SP_o*ST_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        # q_o = (self.g*SP_o*SP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        # r_o = (self.g*SP_o*CP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        
        '''UPDATED THIS SO THAT THESE WOULD ACTUALLY BE ZERO FOR A SHSS CONDITION'''
        p_o = self.eq_rot[0]
        q_o = self.eq_rot[1]
        r_o = self.eq_rot[2]    
        
        print('u, v, w, p, q, r:', u_o, v_o, w_o, p_o, q_o, r_o)
        print('\n')
        
        W_g = self.W/self.g
        
        # print(self.Fxb_q)
        # print(W_g*w_o)
        # print(self.Fxb_q - W_g*w_o)
                
        B_matrix = np.array([[W_g - self.Fxb_udot, 0.0,      -self.Fxb_wdot,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, W_g,                 0.0,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [     -self.Fzb_udot, 0.0, W_g - self.Fzb_wdot,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,  self.Ixxb, -self.Ixyb, -self.Ixzb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [     -self.Myb_udot, 0.0,      -self.Myb_wdot, -self.Ixyb,  self.Iyyb, -self.Iyzb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0, -self.Ixzb, -self.Iyzb,  self.Izzb, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [                0.0, 0.0,                 0.0,        0.0,        0.0,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        AMxp = self.Mxb_p + self.Ixzb*q_o - self.Ixyb*r_o
        AMxq = self.Mxb_q - self.hzb + (self.Iyyb - self.Izzb)*r_o + 2*self.Iyzb*q_o + self.Ixzb*p_o
        AMxr = self.Mxb_r + self.hyb + (self.Iyyb - self.Izzb)*q_o - 2*self.Iyzb*r_o - self.Ixyb*p_o
        AMyp = self.Myb_p + self.hzb + (self.Izzb - self.Ixxb)*r_o - 2*self.Ixzb*p_o - self.Iyzb*q_o
        AMyq = self.Myb_q + self.Ixyb*r_o - self.Iyzb*p_o
        AMyr = self.Myb_r - self.hxb + (self.Izzb - self.Ixxb)*p_o + 2*self.Ixzb*r_o + self.Ixyb*q_o
        AMzp = self.Mzb_p - self.hyb + (self.Ixxb - self.Iyyb)*q_o + 2*self.Ixyb*p_o + self.Iyzb*r_o
        AMzq = self.Mzb_q + self.hxb + (self.Ixxb - self.Iyyb)*p_o - 2*self.Ixyb*q_o - self.Ixzb*r_o
        AMzr = self.Mzb_r + self.Iyzb*p_o - self.Ixzb*q_o
        
        AxP = v_o*(CP_o*ST_o*CO_o + SP_o*SO_o) + w_o*(-SP_o*ST_o*CO_o + CP_o*SO_o)
        AxT = -u_o*ST_o*CO_o + v_o*SP_o*CT_o*CO_o + w_o*CP_o*CT_o*CO_o
        AxO = -u_o*CT_o*SO_o - v_o*(SP_o*ST_o*SO_o + CP_o*CO_o) + w_o*(-CP_o*ST_o*SO_o + SP_o*CO_o)
        AyP = v_o*(CP_o*ST_o*SO_o - SP_o*CO_o) - w_o*(SP_o*ST_o*SO_o + CP_o*CO_o)
        AyT = -u_o*ST_o*SO_o + v_o*SP_o*CT_o*SO_o + w_o*CP_o*CT_o*SO_o
        AyO = u_o*CT_o*CO_o + v_o*(SP_o*ST_o*CO_o - CP_o*SO_o) + w_o*(CP_o*ST_o*CO_o + SP_o*SO_o)
        AzP = v_o*CP_o*CT_o - w_o*SP_o*CT_o
        AzT = -u_o*CT_o - v_o*SP_o*ST_o - w_o*CP_o*ST_o
        AzO = 0.0
        
        # A441 = (self.g*SP_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        # A442 = (-self.g*SP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        # A443 = (self.g*SP_o*CT_o*TT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        A441 = (q_o*CP_o - r_o*SP_o)*TT_o
        A442 = (q_o*SP_o + r_o*CP_o)*(1/(CT_o*CT_o))
        A443 = (-q_o*SP_o - r_o*CP_o)
        A444 = (q_o*CP_o - r_o*SP_o)*(1/CT_o)
        A445 = (q_o*SP_o + r_o*CP_o)*(TT_o/CT_o)
        
        A_matrix = np.array([[          self.Fxb_u,       self.Fxb_v + W_g*r_o,       self.Fxb_w - W_g*q_o,           self.Fxb_p, self.Fxb_q - W_g*w_o, self.Fxb_r + W_g*v_o,       0.0, 0.0, 0.0,               0.0,      -self.W*CT_o, 0.0],
                             [self.Fyb_u - W_g*r_o,                 self.Fyb_v,       self.Fyb_w + W_g*p_o, self.Fyb_p + W_g*w_o,           self.Fyb_q, self.Fyb_r - W_g*u_o,       0.0, 0.0, 0.0,  self.W*CP_o*CT_o, -self.W*SP_o*ST_o, 0.0],
                             [self.Fzb_u + W_g*q_o,       self.Fzb_v - W_g*p_o,                 self.Fzb_w, self.Fzb_p - W_g*v_o, self.Fzb_q + W_g*u_o,           self.Fzb_r,       0.0, 0.0, 0.0, -self.W*SP_o*CT_o, -self.W*CP_o*ST_o, 0.0],
                             [          self.Mxb_u,                 self.Mxb_v,                 self.Mxb_w,                 AMxp,                 AMxq,                 AMxr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [          self.Myb_u,                 self.Myb_v,                 self.Myb_w,                 AMyp,                 AMyq,                 AMyr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [          self.Mzb_u,                 self.Mzb_v,                 self.Mzb_w,                 AMzp,                 AMzq,                 AMzr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [           CT_o*CO_o, SP_o*ST_o*CO_o - CP_o*SO_o, CP_o*ST_o*CO_o + SP_o*SO_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AxP,               AxT, AxO],
                             [           CT_o*SO_o, SP_o*ST_o*SO_o + CP_o*CO_o, CP_o*ST_o*SO_o - SP_o*CO_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AyP,               AyT, AyO],
                             [               -ST_o,                  SP_o*CT_o,                  CP_o*CT_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AzP,               AzT, AzO],
                             [                 0.0,                        0.0,                        0.0,                  1.0,            SP_o*TT_o,            CP_o*TT_o,       0.0, 0.0, 0.0,              A441,              A442, 0.0],
                             [                 0.0,                        0.0,                        0.0,                  0.0,                 CP_o,                -SP_o,       0.0, 0.0, 0.0,              A443,               0.0, 0.0],
                             [                 0.0,                        0.0,                        0.0,                  0.0,            SP_o/CT_o,            CP_o/CT_o,       0.0, 0.0, 0.0,              A444,              A445, 0.0]])


        # calculate C matrix
        C_matrix = np.matmul(np.linalg.inv(B_matrix),A_matrix)
        self.eigvals, self.eigvecs = eig(C_matrix)
        
        print('A-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in A_matrix]))
        
        print('\n')
        print('B-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in B_matrix]))
        
        print('\n')
        print('C-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in C_matrix]))    

        #normalize eigenvectors relative to the largest in each array
        for i in range(12):
            index_max = np.argmax(np.abs(self.eigvecs[:,i]))
            
            cc = np.conj(self.eigvecs[index_max,i])
            
            new_vec = cc*self.eigvecs[:,i]
            
            new_vec = new_vec / np.sqrt(np.sum(np.square(np.abs(new_vec))))
            
            self.eigvecs[:,i] = new_vec
        
        i_sort = np.argsort(np.sqrt(np.real(self.eigvals)**2 + np.imag(self.eigvals)**2))
        
        self.eigvals = self.eigvals[i_sort]
        
        self.eigvecs = self.eigvecs[:,i_sort]
        
        self.amps = np.abs(self.eigvecs)
        self.phase = np.rad2deg(np.arctan2(np.imag(self.eigvecs),np.real(self.eigvecs)))
        
        print('\nEigenvalues') 
        print('\n'.join('{:>32.12f}'.format(item) for item in self.eigvals))
                
        if self.write_output != False:
            with open(self.output_filename, 'w') as export_handle:
                for i in range(len(self.eigvals)):
                    print('{:<16.12f}{:<16.12f}'.format(np.real(self.eigvals[i]), np.imag(self.eigvals[i])), file=export_handle)

        print('\nEigenvectors (1-6)') 
        print('\n'.join([''.join(['{:>26.8f}'.format(item) for item in row]) 
                for row in self.eigvecs[:,:6]]))
        
        print('\nEigenvectors (7-12)') 
        print('\n'.join([''.join(['{:>26.8f}'.format(item) for item in row]) 
                for row in self.eigvecs[:,6:]]))
        
        self.eigreal = np.real(self.eigvals[:])
        self.eigimag = np.imag(self.eigvals[:])
        self.sigma = -np.real(self.eigvals[:])
        self.omegad = np.abs(np.imag(self.eigvals))
        self.period = 2.0*np.pi/self.omegad
        
        for i in range(12):
            
            print('\n---------------------------------------------------------------------------')
            print('{:<24}'.format('Eigenvalue'), '{:^24}'.format('Period'), '{:^24}'.format('Damping'))
            print('{:<18.12f} {:^18.12f} {:^18.12f}'.format(self.eigvals[i], self.period[i], self.sigma[i]))
            
            print("\nEigenvectors:")
            print('{:>4}'.format('\u0394u:'), '{:>28.10f}'.format(self.eigvecs[0,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[0,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[0,i])) #mu
            print('{:>4}'.format('\u0394v:'), '{:>28.10f}'.format(self.eigvecs[1,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[1,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[1,i])) #beta
            print('{:>4}'.format('\u0394w:'), '{:>28.10f}'.format(self.eigvecs[2,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[2,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[2,i])) #alpha
            print('{:>4}'.format('\u0394p:'), '{:>28.10f}'.format(self.eigvecs[3,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[3,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[3,i])) #phat
            print('{:>4}'.format('\u0394q:'), '{:>28.10f}'.format(self.eigvecs[4,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[4,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[4,i])) #qhat
            print('{:>4}'.format('\u0394r:'), '{:>28.10f}'.format(self.eigvecs[5,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[5,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[5,i])) #rhat
            print('{:>4}'.format('\u0394xf:'), '{:>28.10f}'.format(self.eigvecs[6,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[6,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[6,i])) #sigmax
            print('{:>4}'.format('\u0394yf:'), '{:>28.10f}'.format(self.eigvecs[7,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[7,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[7,i])) #sigmay
            print('{:>4}'.format('\u0394zf:'), '{:>28.10f}'.format(self.eigvecs[8,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[8,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[8,i])) #sigmaz
            print('{:>4}'.format('\u0394\u03C6:'), '{:>28.10f}'.format(self.eigvecs[9,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[9,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[9,i])) #PHI
            print('{:>4}'.format('\u0394\u03B8:'), '{:>28.10f}'.format(self.eigvecs[10,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[10,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[10,i])) #THETA       
            print('{:>4}'.format('\u0394\u03C8:'), '{:>28.10f}'.format(self.eigvecs[11,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(self.phase[11,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(self.amps[11,i])) #PSI
        
    def plot_eigvals(self):
        
        markers = ['1','o','o','>','<','x','s','s']
        
        plt.figure(0, figsize=(5,4))
        plt.grid(visible=True)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        
        for i in range(len(self.eigvals[4:])):
            plt.scatter(np.real(self.eigvals[i+4]),np.imag(self.eigvals[i+4]), marker=markers[i], color='k')
        plt.tight_layout()
        plt.show()


    def plot_eigenvectors(self,eigvecs,inds,omegad,sigma,time):
        
        fig_size = (6,4)
        f_size = 10
        plt.rcParams.update({'font.size': f_size})
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        # plt.rcParams['axes.axisbelow'] = True
        
        labels = ['u','v','w','p','q','r','xf','yf','zf','\u03C6','\u03B8','\u03C8']
        colors = ['b','g','r','c','m','y','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','k']
        
        amps = np.abs(eigvecs)
        phase = np.rad2deg(np.arctan2(np.imag(eigvecs),np.real(eigvecs)))
        vecs_all = np.zeros((len(inds),len(time)))
            
        plt.figure(figsize=fig_size)

        for i in range(len(inds)):
            vecs_all[i,:] = damped_sinusoid(time, amps[inds[i]], sigma, omegad, phase[inds[i]], z = 0.0)
            plt.plot(time,vecs_all[i], label=labels[inds[i]], color=colors[inds[i]])
        plt.xlabel('Time [s]')
        plt.ylabel('Relative Amplitude')
        plt.xlim(0,max(time))
        plt.legend(loc='right')
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    
    '''INPUTS'''
    bire = False
    V = 634 #ft/s
    gamma = np.deg2rad(0.0) #rad
    phi = np.deg2rad(0.0) #rad
    H = 15000. #ft
    cg_shift = [1.0, 0.0, 0.0] #ft
    SIMPLE_THRUST = True
    
    '''RUN CASE'''
    case = dynamicAnalysis(path='./', write_output=False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=bire, shss=False, compressible=False, coords_approx=False, derivs_approx=False,
                            stall=False, cg_shift=cg_shift, simple_thrust = SIMPLE_THRUST)
    
    # case.update_aircraft_properties(V, H, dB = 0.0) # initializes the aircraftProperties object, might be able to internalize 
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    case.solve_derivatives(num_derivs = False)
    case.solve_dynamics_system()
    case.plot_eigvals()    
    
    eig_vecs = case.eigvecs
    omegads = case.omegad
    sigmas = case.sigma
    
    time = np.linspace(0,200,1000)
    
    eigval_index = 10
    # eigvecs_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
    # eigvecs_indices = [0,1,2]
    eigvecs_indices = [3,4,5]
    # eigvecs_indices = [9,10,11]
    # eigvecs_indices = [0,2,6,8] #phugoid
    # eigvecs_indices = [0,6,7,8,9] #phugoid
    # eigvecs_indices = [6,8]
    # eigvecs_indices = [3,4,5]
    # eigvecs_indices = [0,2,8] #short period
    # eigvecs_indices = [3,5,7,9] #dutch roll
    
    case.plot_eigenvectors(eigvecs=eig_vecs[:,eigval_index], inds=eigvecs_indices, omegad=omegads[eigval_index], sigma=sigmas[eigval_index], time=time)
    
    
    # case.solve_derivatives(num_derivs = True)

    # case.print_derivatives() 

    # case.solve_derivatives_new()
 
    
    # case = dynamicAnalysis(path='./', write_output=False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
    #                         BIRE=bire, shss=False, compressible=True, coords_approx=False, derivs_approx=True,
    #                         stall=True, cg_shift=cg_shift)
    
    # case.update_aircraft_properties(V, H, dB = 0.0) # initializes the aircraftProperties object, might be able to internalize 
    # case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    # case.solve_derivatives()
    # case.print_derivatives()  
    
    # case = dynamicAnalysis(path='./', write_output=False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
    #                         BIRE=bire, shss=True, compressible=False, coords_approx=False, derivs_approx=False,
    #                         stall=False, cg_shift=cg_shift, JSON_input=False)
    
    
    # case.solve_dynamics_system()
    
    '''OUTPUT RESULTS'''
    # case.plot_eigvals()
    


    ''' TEST EXAMPLE 9.8.1'''
    # derivs = test_derivs()
    # derivs.get_JSON_inputs('9.8.1.json')
    # test_9_8_1 = derivs.solve_derivs()
    
    
    # case = dynamicAnalysis(path='./', write_output=False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
    #                         BIRE=bire, shss=False, compressible=False, coords_approx=False, derivs_approx=False,
    #                         stall=True, cg_shift=cg_shift, JSON_input = True)
    # case.solve_dynamics_system()