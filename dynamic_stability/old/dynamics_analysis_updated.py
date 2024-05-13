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
from aero_trim import trim, AircraftProperties

from trim_functions import solve_trim

class dynamicAnalysis:
                
    def __init__(self, aeroModel, path='./', write_output = False, output_filename = 'dynamic_output.txt',
                 BIRE=False, shss=False, compressible=False, stall=False,
                 cg_shift=[0.0, 0.0, 0.0], JSON_input = False):
        '''
        Load all the non changing values related to either aircraft.
        props for loads type data
        '''
        self.write_output = write_output
        self.output_filename = output_filename
        
        self.BIRE = BIRE
        self.cg_shift = cg_shift
        self.shss = shss
        self.compressible = compressible
        self.stall = stall
        self.Gamma = 0.5 # relaxation factor for the trim algorithm, may not need to ever update this
        
        # initialize the aerodynamic model
        self.aeroModel = aeroModel
            
    def update_aircraft_properties(self, V, H, dB = 0.0):
        
        properties_dir = '../trim/'
        
        aircraft_properties = AircraftProperties(V, H, self.Gamma, path=properties_dir, bire=self.BIRE)

        # update operating properties
        self.V = aircraft_properties.V
        self.g = aircraft_properties.g
        self.nondim_const = aircraft_properties.nondim_const
        self.rho = aircraft_properties.rho
        self.rho_0 = aircraft_properties.rho_0
        self.M = aircraft_properties.M
        self.a = aircraft_properties.a
        self.a_0 = aircraft_properties.a_0
        
        # update aircraft geometric properties
        self.bw = aircraft_properties.b_w
        self.cw = aircraft_properties.c_w
        self.Sw = aircraft_properties.S_w
        
        self.Sh = 63.675 #NEEDS TO BE UPDATED TO PULL FROM THE JSON PROPERTIES
        
        # BORROWED FROM AUSTINS CODE
        xbwt = -7.358
        self.xbh = -13.13
        self.lwt = 1.1 * (xbwt - self.xbh)       
        self.CLw_a = 3.3775691217788646
        self.CLh_a = 1.3657050471586294
        
        # update aircraft mass and inertia properties
        self.W = aircraft_properties.W

        if self.BIRE == False:
            self.Ixxb = aircraft_properties.Ixx
            self.Iyyb = aircraft_properties.Iyy
            self.Izzb = aircraft_properties.Izz
            self.Ixyb = aircraft_properties.Ixy
            self.Ixzb = aircraft_properties.Ixz
            self.Iyzb = aircraft_properties.Iyz
        else:
            aircraft_properties.calc_BIRE_inertia(dB)
            self.Ixxb = aircraft_properties.Ixx
            self.Iyyb = aircraft_properties.Iyy
            self.Izzb = aircraft_properties.Izz
            self.Ixyb = aircraft_properties.Ixy
            self.Ixzb = aircraft_properties.Ixz
            self.Iyzb = aircraft_properties.Iyz
            
            #BORROWED FROM AUSTINS CODE
            self.CLh_a = 1.3858047943592773 * np.abs(np.cos(dB))
        
        self.hxb = aircraft_properties.hx
        self.hyb = aircraft_properties.hy
        self.hzb = aircraft_properties.hz
        
    def solve_equilibrium_state(self, V, H, gamma, phi, cg_shift):
        
        '''
        Given airspeed, altitude, CG shift, climb angle,
        and bank angle, solve equilibrium trim condition.
        '''

        solution1 = trim(V, H, gamma, phi, Gamma = self.Gamma, shss=self.shss,
                        bire=self.BIRE, cg_shift=self.cg_shift, fixed_point=False,
                        compressible=self.compressible, stall=self.stall, aero_dir= aero_directory,
                        trim_dir = trim_directory)
        
        solution2 = solve_trim(aero_model = self.aeroModel, aircraft_props=self.aicraft_properties,
                              gamma = gamma, phi = phi, cg_shift=self.cg_shift,
                              shss = self.shss, compressible = self.compressible, stall = self.stall)

        # solution = solve_trim(V, H, gamma, phi, Gamma = self.Gamma, shss=self.shss,
        #                 bire=self.BIRE, cg_shift=self.cg_shift,
        #                 compressible=self.compressible, stall=self.stall, aero_dir= aero_directory,
        #                 trim_dir = trim_directory)
        
        tau, alpha, beta, da, de, dr = solution2.x
        u, v, w, p, q, r, phi, theta = solution2.states
        
        print('u, v, w, p, q, r:', u, v, w, p, q, r)
        
        FX, FY, FZ, Mx, My, Mz = solution2.FM_dim
        
        print('\nBody Force/Moment from trim solution:')
        print(FX, FY, FZ, Mx, My, Mz)
        print('\n')
        
        # pbar = p*self.bw/(2.*V)
        # qbar = q*self.cw/(2.*V)
        # rbar = r*self.bw/(2.*V)
        
        # print('\nBody Force/Moment from CS change:')
        # print(FX, FY, FZ, Mx, My, Mz)
        
        # FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha, beta, pbar, qbar, rbar, da, de, dr]),tau,V)
        
        if self.BIRE == True:
            self.update_aircraft_properties(V, H, dB = dr)
        
        self.eq_velo = np.array([u,v,w])
        self.eq_rot = np.array([p,q,r])
        self.eq_euler = np.array([phi,theta])
        self.eq_inputs = np.array([tau, alpha, beta, da, de, dr])
        self.eq_FM = np.array([FX, FY, FZ, Mx, My, Mz])
        
        self.alpha = alpha
        self.beta = beta  
        
    def solve_derivatives(self):
        

    def convert_all_bf2wind(self):
        
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        cb = np.cos(self.beta)
        sb = np.sin(self.beta)
        
        bdy2wd = np.array([[ca*cb, sb, sa*cb],
                          [-ca*sb, cb, -sa*sb],
                          [-sa, 0.0, ca]])
        
        wd2bdy = np.array([[ca*cb, -ca*sb, -sa],
                          [sb, cb, 0.0],
                          [sa*cb, -sa*sb, ca]])
        
        
        Ibdy = np.array([[self.Ixxb, -self.Ixyb, -self.Ixzb],[-self.Ixyb,  self.Iyyb, -self.Iyzb],[-self.Ixzb, -self.Iyzb,  self.Izzb]])
        # trace1 = self.Ixxb + self.Iyyb + self.Izzb
        Iwd = np.matmul(bdy2wd,np.matmul(Ibdy,wd2bdy))
        self.Ixxb = Iwd[0,0]
        self.Ixyb = -Iwd[0,1]
        self.Ixzb = -Iwd[0,2]
        self.Ixyb = -Iwd[1,0]
        self.Iyyb = Iwd[1,1]
        self.Iyzb = -Iwd[1,2]
        self.Ixzb = -Iwd[2,0]
        self.Iyzb = -Iwd[2,1]
        self.Izzb = Iwd[2,2]
        # trace2 = self.Ixxb + self.Iyyb + self.Izzb
        
        Fbuvw = np.matmul(bdy2wd,np.matmul(np.array([[self.Fxb_u, self.Fxb_v, self.Fxb_w],[self.Fyb_u, self.Fyb_v, self.Fyb_w],[self.Fzb_u, self.Fzb_v, self.Fzb_w]]),wd2bdy))
        self.Fxb_u = Fbuvw[0,0]
        self.Fxb_v = Fbuvw[0,1]
        self.Fxb_w = Fbuvw[0,2]
        self.Fyb_u = Fbuvw[1,0]
        self.Fyb_v = Fbuvw[1,1]
        self.Fyb_w = Fbuvw[1,2]
        self.Fzb_u = Fbuvw[2,0]
        self.Fzb_v = Fbuvw[2,1]
        self.Fzb_w = Fbuvw[2,2]
        Fbpqr = np.matmul(bdy2wd,np.matmul(np.array([[self.Fxb_p, self.Fxb_q, self.Fxb_r],[self.Fyb_p, self.Fyb_q, self.Fyb_r],[self.Fzb_p, self.Fzb_q, self.Fzb_r]]),wd2bdy))
        self.Fxb_p = Fbpqr[0,0]
        self.Fxb_q = Fbpqr[0,1]
        self.Fxb_r = Fbpqr[0,2]
        self.Fyb_p = Fbpqr[1,0]
        self.Fyb_q = Fbpqr[1,1]
        self.Fyb_r = Fbpqr[1,2]
        self.Fzb_p = Fbpqr[2,0]
        self.Fzb_q = Fbpqr[2,1]
        self.Fzb_r = Fbpqr[2,2]
        Fbuvwdot = np.matmul(bdy2wd,np.matmul(np.array([[self.Fxb_udot, self.Fxb_vdot, self.Fxb_wdot],[self.Fyb_udot, self.Fyb_vdot, self.Fyb_wdot],[self.Fzb_udot, self.Fzb_vdot, self.Fzb_wdot]]),wd2bdy))
        self.Fxb_udot = Fbuvwdot[0,0]
        self.Fxb_vdot = Fbuvwdot[0,1]
        self.Fxb_wdot = Fbuvwdot[0,2]
        self.Fyb_udot = Fbuvwdot[1,0]
        self.Fyb_vdot = Fbuvwdot[1,1]
        self.Fyb_wdot = Fbuvwdot[1,2]
        self.Fzb_udot = Fbuvwdot[2,0]
        self.Fzb_vdot = Fbuvwdot[2,1]
        self.Fzb_wdot = Fbuvwdot[2,2]
        
        Mbuvw = np.matmul(bdy2wd,np.matmul(np.array([[self.Mxb_u, self.Mxb_v, self.Mxb_w],[self.Myb_u, self.Myb_v, self.Myb_w],[self.Mzb_u, self.Mzb_v, self.Mzb_w]]),wd2bdy))
        self.Mxb_u = Mbuvw[0,0]
        self.Mxb_v = Mbuvw[0,1]
        self.Mxb_w = Mbuvw[0,2]
        self.Myb_u = Mbuvw[1,0]
        self.Myb_v = Mbuvw[1,1]
        self.Myb_w = Mbuvw[1,2]
        self.Mzb_u = Mbuvw[2,0]
        self.Mzb_v = Mbuvw[2,1]
        self.Mzb_w = Mbuvw[2,2]
        Mbpqr = np.matmul(bdy2wd,np.matmul(np.array([[self.Mxb_p, self.Mxb_q, self.Mxb_r],[self.Myb_p, self.Myb_q, self.Myb_r],[self.Mzb_p, self.Mzb_q, self.Mzb_r]]),wd2bdy))
        self.Mxb_p = Mbpqr[0,0]
        self.Mxb_q = Mbpqr[0,1]
        self.Mxb_r = Mbpqr[0,2]
        self.Myb_p = Mbpqr[1,0]
        self.Myb_q = Mbpqr[1,1]
        self.Myb_r = Mbpqr[1,2]
        self.Mzb_p = Mbpqr[2,0]
        self.Mzb_q = Mbpqr[2,1]
        self.Mzb_r = Mbpqr[2,2]
        Mbuvwdot = np.matmul(bdy2wd,np.matmul(np.array([[0.0, 0.0, 0.0],[self.Myb_udot, 0.0, self.Myb_wdot],[0.0, 0.0, 0.0]]),wd2bdy))
        self.Myb_udot = Mbuvwdot[1,0]
        self.Myb_wdot = Mbuvwdot[1,2]

    def print_derivatives(self):
        
        print('Fxb,u, Fyb,u, Fzb,u, Mxb,u, Myb,u, Myb,u')
        print('Fxb,v, Fyb,v, Fzb,v, Mxb,v, Myb,v, Myb,v')
        print('Fxb,w, Fyb,w, Fzb,w, Mxb,w, Myb,w, Myb,w')
        print('Fxb,p, Fyb,p, Fzb,p, Mxb,p, Myb,p, Myb,p')
        print('Fxb,w, Fyb,q, Fzb,q, Mxb,q, Myb,q, Myb,q')
        print('Fxb,r, Fyb,r, Fzb,r, Mxb,r, Myb,r, Myb,r')
        
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Fxb_u,self.Fxb_v,self.Fxb_w,self.Fxb_p,self.Fxb_q,self.Fxb_r))
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Fyb_u,self.Fyb_v,self.Fyb_w,self.Fyb_p,self.Fyb_q,self.Fyb_r))
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Fzb_u,self.Fzb_v,self.Fzb_w,self.Fzb_p,self.Fzb_q,self.Fzb_r))
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Mxb_u,self.Mxb_v,self.Mxb_w,self.Mxb_p,self.Mxb_q,self.Mxb_r))
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Myb_u,self.Myb_v,self.Myb_w,self.Myb_p,self.Myb_q,self.Myb_r))
        print('{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format(self.Mzb_u,self.Mzb_v,self.Mzb_w,self.Mzb_p,self.Mzb_q,self.Mzb_r))
        print('\n')
        
    def set_phillips_approx(self, coords = False, derivs = False):
        
        if coords == True:
            self.convert_all_bf2wind()
            
            self.eq_velo = np.array([self.V,0.0,0.0])
        
        if derivs == True:
        
            self.Fxb_v = 0.0
            
            self.Fyb_u = 0.0
            self.Fyb_w = 0.0
            
            self.Fzb_v = 0.0
            
            self.Fxb_p = 0.0
            self.Fxb_r = 0.0
            self.Fyb_q = 0.0
            self.Fzb_p = 0.0
            self.Fzb_r = 0.0
            
            self.Mxb_u = 0.0
            self.Mzb_u = 0.0
            
            self.Mxb_q = 0.0
            self.Myb_p = 0.0
            self.Myb_r = 0.0
            self.Mzb_q = 0.0

    def solve_dynamics_system(self):
        
        self.print_derivatives()
        
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
        
        
        p_o = (-self.g*SP_o*ST_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        q_o = (self.g*SP_o*SP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        r_o = (self.g*SP_o*CP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        
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
        
        A441 = (self.g*SP_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        A442 = (-self.g*SP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        A443 = (self.g*SP_o*CT_o*TT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        
        A_matrix = np.array([[          self.Fxb_u,       self.Fxb_v + W_g*r_o,       self.Fxb_w - W_g*q_o,           self.Fxb_p, self.Fxb_q - W_g*w_o, self.Fxb_r + W_g*v_o,       0.0, 0.0, 0.0,               0.0,      -self.W*CT_o, 0.0],
                             [self.Fyb_u - W_g*r_o,                 self.Fyb_v,       self.Fyb_w + W_g*p_o, self.Fyb_p + W_g*w_o,           self.Fyb_q, self.Fyb_r - W_g*u_o,       0.0, 0.0, 0.0,  self.W*CP_o*CT_o, -self.W*SP_o*ST_o, 0.0],
                             [self.Fzb_u + W_g*q_o,       self.Fzb_v - W_g*p_o,                 self.Fzb_w, self.Fzb_p - W_g*v_o, self.Fzb_q + W_g*u_o,           self.Fzb_r,       0.0, 0.0, 0.0, -self.W*SP_o*CT_o, -self.W*CP_o*ST_o, 0.0],
                             [          self.Mxb_u,                 self.Mxb_v,                 self.Mxb_w,                 AMxp,                 AMxq,                 AMxr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [          self.Myb_u,                 self.Myb_v,                 self.Myb_w,                 AMyp,                 AMyq,                 AMyr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [          self.Mzb_u,                 self.Mzb_v,                 self.Mzb_w,                 AMzp,                 AMzq,                 AMzr,       0.0, 0.0, 0.0,               0.0,               0.0, 0.0],
                             [           CT_o*CO_o, SP_o*ST_o*CO_o - CP_o*SO_o, CP_o*ST_o*CO_o + SP_o*SO_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AxP,               AxT, AxO],
                             [           CT_o*SO_o, SP_o*ST_o*SO_o + CP_o*CO_o, CP_o*ST_o*SO_o - SP_o*CO_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AyP,               AyT, AyO],
                             [               -ST_o,                  SP_o*CT_o,                  CP_o*CT_o,                  0.0,                  0.0,                  0.0,       0.0, 0.0, 0.0,               AzP,               AzT, AzO],
                             [                 0.0,                        0.0,                        0.0,                  1.0,            SP_o*TT_o,            CP_o*TT_o,       0.0, 0.0, 0.0,               0.0,              A441, 0.0],
                             [                 0.0,                        0.0,                        0.0,                  0.0,                 CP_o,                -SP_o,       0.0, 0.0, 0.0,              A442,               0.0, 0.0],
                             [                 0.0,                        0.0,                        0.0,                  0.0,            SP_o/CT_o,            CP_o/CT_o,       0.0, 0.0, 0.0,               0.0,              A443, 0.0]])


        # calculate C matrix
        C_matrix = np.matmul(np.linalg.inv(B_matrix),A_matrix)
        self.eigvals, self.eigvecs = eig(C_matrix)
        
        # print('\n------------------------------------- Longitudinal Analysis -------------------------------------\n')
        print('A-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in A_matrix]))
        
        # for i in range(12):
        #     index_max = np.argmax(np.abs(A_matrix[i,:]))
            
        #     new_row = A_matrix[i,:]
            
        #     new_row = new_row / np.sqrt(np.sum(np.square(np.abs(new_row))))
            
        #     A_matrix[i,:] = new_row
        
        # print('A-Matrix')
        # print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
        #         for row in A_matrix]))
        
        print('\n')
        print('B-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in B_matrix]))
        
        print('\n')
        print('C-Matrix')
        print('\n'.join([''.join(['{:>16.6f}'.format(item) for item in row]) 
                for row in C_matrix]))
        
        # np.save('A_matrix_maustin', A_matrix)
        # np.save('B_matrix_maustin', B_matrix)
        

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
        
        amps = np.abs(self.eigvecs)
        phase = np.rad2deg(np.arctan2(np.imag(self.eigvecs),np.real(self.eigvecs)))
        
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
            print('{:>4}'.format('\u0394u:'), '{:>28.10f}'.format(self.eigvecs[0,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[0,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[0,i])) #mu
            print('{:>4}'.format('\u0394v:'), '{:>28.10f}'.format(self.eigvecs[1,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[1,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[1,i])) #beta
            print('{:>4}'.format('\u0394w:'), '{:>28.10f}'.format(self.eigvecs[2,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[2,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[2,i])) #alpha
            print('{:>4}'.format('\u0394p:'), '{:>28.10f}'.format(self.eigvecs[3,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[3,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[3,i])) #phat
            print('{:>4}'.format('\u0394q:'), '{:>28.10f}'.format(self.eigvecs[4,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[4,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[4,i])) #qhat
            print('{:>4}'.format('\u0394r:'), '{:>28.10f}'.format(self.eigvecs[5,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[5,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[5,i])) #rhat
            print('{:>4}'.format('\u0394xf:'), '{:>28.10f}'.format(self.eigvecs[6,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[6,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[6,i])) #sigmax
            print('{:>4}'.format('\u0394yf:'), '{:>28.10f}'.format(self.eigvecs[7,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[7,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[7,i])) #sigmay
            print('{:>4}'.format('\u0394zf:'), '{:>28.10f}'.format(self.eigvecs[8,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[8,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[8,i])) #sigmaz
            print('{:>4}'.format('\u0394\u03C6:'), '{:>28.10f}'.format(self.eigvecs[9,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[9,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[9,i])) #PHI
            print('{:>4}'.format('\u0394\u03B8:'), '{:>28.10f}'.format(self.eigvecs[10,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[10,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[10,i])) #THETA       
            print('{:>4}'.format('\u0394\u03C8:'), '{:>28.10f}'.format(self.eigvecs[11,i]),'{:>6}'.format('Phase:'), '{:>20.12f}'.format(phase[11,i]),'{:>12}'.format('Amplitude:'), '{:>18.12f}'.format(amps[11,i])) #PSI
        
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
            
        
    def wind_2_body_forces(self, forces, alpha, beta):
        
        CL, CS, CD = forces
        
        CXb = -(CD*np.cos(alpha)*np.cos(beta) + CS*np.cos(alpha)*np.sin(beta) - CL*np.sin(alpha))
        CYb = CS*np.cos(beta) - CD*np.sin(beta)
        CZb = -(CD*np.sin(alpha)*np.cos(beta) + CS*np.sin(alpha)*np.sin(beta) + CL*np.cos(alpha))
            
        return np.array([CXb, CYb, CZb])

    def wind_2_body_vector(self, vector, alpha, beta):
        
        x, y, z = vector
        
        Xb = (x*np.cos(alpha)*np.cos(beta) - y*np.cos(alpha)*np.sin(beta) - z*np.sin(alpha))
        Yb = y*np.cos(beta) + x*np.sin(beta)
        Zb = (x*np.sin(alpha)*np.cos(beta) - y*np.sin(alpha)*np.sin(beta) + z*np.cos(alpha))
            
        return np.array([Xb, Yb, Zb])  
    
    def find_body_velocity(self, alpha, beta):
        '''Find body-fixed velocities from alpha, beta, total airspeed'''
        self.alpha = alpha
        self.beta = beta
        u = self.V*np.cos(self.alpha)*np.cos(self.beta)
        v = self.V*np.sin(self.beta)
        w = self.V*np.sin(self.alpha)*np.cos(self.beta)
        
        self.eq_velo = np.array([u, v, w])

if __name__ == "__main__":
    
    phillips_approx = True

    V = 634 #ft/s
    gamma = np.deg2rad(0.0) #rad
    phi = np.deg2rad(0.0) #rad
    H = 15000. #ft
    cg_shift = [1., 0., 0.] #ft
    
    case = dynamicAnalysis(path='./', write_output = False, output_filename = 'eig_vals_BIRE_60deg_bank_cg_shift.txt',
                            BIRE=True, shss=False, compressible=True,
                            stall=True, cg_shift=cg_shift)
    
    case.update_aircraft_properties(V, H, dB = 0.0) # is this doing anything?
    
    case.solve_equilibrium_state(V, H, gamma, phi, cg_shift)
    
    'My derivative method'
    dAlpha = np.deg2rad(0.25) #rad
    dBeta = np.deg2rad(0.25) #rad
    dp = 0.06; #rad/s
    dq = 0.5 * dp;
    dr = 0.5 * dp;
    case.solve_derivatives(dAlpha, dBeta, dp, dq, dr)
    # case.print_derivatives()
    # case.convert_all_bf2wind()
    # case.print_derivatives()
    
    if phillips_approx == True:
        
        case.set_phillips_approx()
    
    case.solve_dynamics_system()
    
    case.plot_eigvals()
        
    
    '''TEST JSON INPUTS - 9.8.1'''
    # case = dynamicAnalysis(path='./', write_output = True, output_filename = '9.8.1_new.txt',
    #                        BIRE=False, shss=False, compressible=False, stall=False, JSON_input=True)
    
    # case.get_JSON_inputs(filename='9.8.1.json')
    # case.find_body_velocity(0.0, 0.0)
    # case.solve_dynamics_system()
