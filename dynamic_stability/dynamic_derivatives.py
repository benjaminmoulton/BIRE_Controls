import numpy as np
import json
import sys

# add the BIRE specific directorys to the system path
aero_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/aerodynamics_model/'
sys.path.insert(1, aero_directory)

from thrust import Propulsion

class derivs:
    '''This is a stand alone class for storing the necessary derivatives for 
    the linearized dynamic analysis code.'''
    def __init__(self):
        self.Fxb_u = 0.0
        self.Fxb_v = 0.0
        self.Fxb_w = 0.0
        self.Fxb_p = 0.0
        self.Fxb_q = 0.0
        self.Fxb_r = 0.0
        
        self.Fyb_u = 0.0
        self.Fyb_v = 0.0
        self.Fyb_w = 0.0
        self.Fyb_p = 0.0
        self.Fyb_q = 0.0
        self.Fyb_r = 0.0
        
        self.Fzb_u = 0.0
        self.Fzb_v = 0.0
        self.Fzb_w = 0.0
        self.Fzb_p = 0.0
        self.Fzb_q = 0.0
        self.Fzb_r = 0.0
                
        self.Mxb_u = 0.0
        self.Mxb_v = 0.0
        self.Mxb_w = 0.0
        self.Mxb_p = 0.0
        self.Mxb_q = 0.0
        self.Mxb_r = 0.0
        
        self.Myb_u = 0.0
        self.Myb_v = 0.0
        self.Myb_w = 0.0
        self.Myb_p = 0.0
        self.Myb_q = 0.0
        self.Myb_r = 0.0
        
        self.Mzb_u = 0.0
        self.Mzb_v = 0.0
        self.Mzb_w = 0.0
        self.Mzb_p = 0.0
        self.Mzb_q = 0.0
        self.Mzb_r = 0.0
        
        '''These two have approximations - Phillips'''
        self.Fzb_wdot = 0.0
        self.Myb_wdot = 0.0

        self.Fxb_udot = 0.0
        self.Fxb_vdot = 0.0
        self.Fxb_wdot = 0.0
        self.Fyb_udot = 0.0
        self.Fyb_vdot = 0.0
        self.Fyb_wdot = 0.0
        self.Fzb_udot = 0.0
        self.Fzb_vdot = 0.0
        self.Myb_udot = 0.0

class solveDerivatives:
    
    def __init__(self, aeroModel, aircraft_properties, cg_shift, trim_solution, 
                 numerical_derivs, compressible, stall, coords_approx = False,
                 derivs_approx = False, simple_thrust = True):
        '''
        Parameters
        -----------
        aeroModel: object
        aircraft_properties: object
        trim_solution: object
        cg_shift: array [X.X, X.X, X.X]
        
        Flags
        ----------
        numerical_derivs: boolean
        compressible: boolean
        stall: boolean
        coords_approx: boolean
        derivs_approx: boolean
        '''

        self.derivs_sol = derivs() # initialize the derivatives variable for storing solutions
        
        '''Store the necessary input options'''
        self.BIRE = aircraft_properties.BIRE
        self.numerical_derivs = numerical_derivs
        self.simple_thrust = simple_thrust
        self.compressible = compressible
        self.stall = stall
        self.cg_shift = cg_shift
        self.derivs_approx = derivs_approx
        self.coords_approx = coords_approx
        
        '''Assign the aeroModel, aircraft properties, and thrust model to 
        member variables'''
        self.aeroModel = aeroModel
        self.aircraft_properties = aircraft_properties
        self.thrust_model = Propulsion(inp_dir=aero_directory)
        
        # store operating parameters
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
        self.CLh_a = 1.3657050471586294
        
        # update aircraft mass and inertia properties
        self.W = self.aircraft_properties.W

        self.Ixxb = self.aircraft_properties.Ixx
        self.Iyyb = self.aircraft_properties.Iyy
        self.Izzb = self.aircraft_properties.Izz
        self.Ixyb = self.aircraft_properties.Ixy
        self.Ixzb = self.aircraft_properties.Ixz
        self.Iyzb = self.aircraft_properties.Iyz
    
        #BORROWED FROM AUSTINS CODE
        self.CLh_a = self.aircraft_properties.CLh_a
        # self.CLh_a = 1.3858047943592773 * np.abs(np.cos(dB))
        
        self.hxb = self.aircraft_properties.hx
        self.hyb = self.aircraft_properties.hy
        self.hzb = self.aircraft_properties.hz
        
        # EXTRACT TRIM SOLUTION VALUES
        tau, alpha, beta, da, de, dr = trim_solution.x
        u, v, w, p, q, r, phi, theta = trim_solution.states
        
        self.alpha = alpha
        self.beta = beta
        
        self.eq_velo = np.array([u,v,w])
        self.eq_rot = np.array([p,q,r])
        self.eq_euler = np.array([phi,theta])
        self.eq_inputs = np.array([tau, alpha, beta, da, de, dr])
        
        self.eq_FM = trim_solution.FM_dim
        self.eq_FM_wind = trim_solution.FM

    def solve_derivs(self):

        if self.numerical_derivs == True:
            dAlpha = np.deg2rad(0.25) #rad
            dBeta = np.deg2rad(0.25) #rad
            dp = 0.06; #rad/s
            dq = 0.5 * dp;
            dr = 0.5 * dp;
            self.solve_numeric_derivatives(dAlpha, dBeta, dp, dq, dr)
        else:
            if self.BIRE == True:
                self.analytic_derivatives_BIRE()
            elif self.BIRE ==  False:
                self.analytic_derivatives_F16()
                
        if self.coords_approx == True:
            self.set_phillips_approx(coords=True,derivs=False)
        
        self.set_deriv_solution()
    
        return self.derivs_sol
            
    def acceleration_derivatives(self):
        # self.Fzb_wdot = -((self.rho*self.Sw*self.Sh*self.lwt)/(np.pi*self.bw*self.bw))*self.CLw_a*self.CLh_a
        # self.Myb_wdot = -self.xbh*self.Fzb_wdot
        self.Fzb_wdot = 0.0
        self.Myb_wdot = 0.0
        
        self.Fxb_udot = 0.0
        self.Fxb_vdot = 0.0
        self.Fxb_wdot = 0.0
        self.Fyb_udot = 0.0
        self.Fyb_vdot = 0.0
        self.Fyb_wdot = 0.0
        self.Fzb_vdot = 0.0
        
        self.Fzb_udot = 0.0
        self.Myb_udot = 0.0
        
    def analytic_derivatives_F16(self):
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        p_o = self.eq_rot[0]
        q_o = self.eq_rot[1]
        r_o = self.eq_rot[2]
        
        pbar_o = p_o*self.bw/(2.*self.V)
        qbar_o = q_o*self.cw/(2.*self.V)
        rbar_o = r_o*self.bw/(2.*self.V)
                
        tau_o = self.eq_inputs[0]
        alpha_o = self.eq_inputs[1]
        beta_o = self.eq_inputs[2]
        da_o = self.eq_inputs[3]
        de_o = self.eq_inputs[4]
        dr_o = self.eq_inputs[5]
        
        Ca = np.cos(alpha_o)
        Sa = np.sin(alpha_o)
        Cb = np.cos(beta_o)
        Sb = np.sin(beta_o)
        
        a_u = - (w_o/(u_o*u_o + w_o*w_o))
        a_w = (u_o/(u_o*u_o + w_o*w_o))
        
        b_u = - ((u_o*v_o)/(self.V*self.V*np.sqrt(u_o*u_o + w_o*w_o)))
        b_v = np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V)
        b_w = - ((v_o*w_o)/(self.V*self.V*np.sqrt(u_o*u_o + w_o*w_o)))
        
        pbar_u, qbar_u, rbar_u = -(u_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        pbar_v, qbar_v, rbar_v = -(v_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        pbar_w, qbar_w, rbar_w = -(w_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        
        pbar_p = self.bw/(2*self.V)
        rbar_r = pbar_p
        qbar_q = self.cw/(2*self.V)
        
        delta_x, delta_y, delta_z = self.cg_shift

        # dT_dV = tau_o*self.dthrust_dV(self.V, self.rho, self.rho_0)
        if self.simple_thrust == True:
            dT_dV = self.thrust_derivatives_simple(self.V)
        else:
            dT_dV = self.thrust_derivatives(tau_o,self.V,self.H)
        
        print('dT_dV: ', dT_dV)

        # Mach = self.V/self.a
        C_nonDim = 0.5*self.rho*self.V*self.V*self.Sw
        
        CL, CS, CD, Cl, Cm, Cn = self.eq_FM_wind
        
        print('\nBody Force/Moment Coeffs from trim solution:')
        print(CL, CS, CD, Cl, Cm, Cn)
        
        FX, FY, FZ, Mx, My, Mz = self.eq_FM
        print('\nBody Force/Moment Coeffs from trim solution:')
        print(FX, FY, FZ, Mx, My, Mz)
        
        CL1 = self.aeroModel._CL(alpha_o, 0., 0., 0., 0., 0., 0., 0.)
        CLa = self.aeroModel.CLa
        CLq = self.aeroModel.CLq
        
        CS1 = self.aeroModel._CS(0., beta_o, 0., 0., 0., 0., 0., 0.)
        CSb = self.aeroModel.CSb
        CSLp = self.aeroModel.CSLp
        CSp = self.aeroModel.CSp
        CSr = self.aeroModel.CSr
        
        CDL = self.aeroModel.CDL
        CDL2 = self.aeroModel.CDL2
        CDLq = self.aeroModel.CDLq
        CDL2q = self.aeroModel.CDL2q
        CDLde = self.aeroModel.CDLde
        CDS2 = self.aeroModel.CDS2
        CDSp = self.aeroModel.CDSp
        CDSr = self.aeroModel.CDSr
        CDSda = self.aeroModel.CDSda
        CDSdr = self.aeroModel.CDSdr
        CDq = self.aeroModel.CDq
        
        Clb = self.aeroModel.Clb
        Clp = self.aeroModel.Clp
        Clr = self.aeroModel.Clr
        ClLr = self.aeroModel.ClLr
        
        Cma = self.aeroModel.Cma
        Cmq = self.aeroModel.Cmq
        
        Cnb = self.aeroModel.Cnb
        CnLp = self.aeroModel.CnLp
        CnLda = self.aeroModel.CnLda
        Cnp = self.aeroModel.Cnp
        CnLp = self.aeroModel.CnLp
        Cnr = self.aeroModel.Cnr
        
        CL1_u = CLa*a_u
        CL1_w = CLa*a_w
        
        CS1_u = CSb*b_u
        CS1_v = CSb*b_v
        CS1_w = CSb*b_w

        '''wind force and moment coefficient derivatives'''
        CL_u = CLa*a_u + CLq*qbar_u
        CL_v = CLq*qbar_v
        CL_w = CLa*a_w + CLq*qbar_w
        CL_p = 0.0
        CL_q = CLq*qbar_q
        CL_r = 0.0
        
        CS_u = CSb*b_u + CSLp*CL1_u*pbar_o + (CSp + CSLp*CL1)*pbar_u + CSr*rbar_u
        CS_v = CSb*b_v + (CSp + CSLp*CL1)*pbar_v + CSr*rbar_v
        CS_w = CSb*b_w + CSLp*CL1_w*pbar_o + (CSp + CSLp*CL1)*pbar_w + CSr*rbar_w
        CS_p = (CSp + CSLp*CL1)*pbar_p
        CS_q = 0.0
        CS_r = CSr*rbar_r
        
        CD_u = (CDL + 2*CDL2*CL1 + (CDLq + 2*CDL2q*CL1)*qbar_o + CDLde*de_o)*CL1_u + (2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o + CDSdr*dr_o)*CS1_u + CDSp*CS1*pbar_u + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_u + CDSr*CS1*rbar_u
        CD_v = (2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o + CDSdr*dr_o)*CS1_v + CDSp*CS1*pbar_v + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_v + CDSr*CS1*rbar_v
        CD_w = (CDL + 2*CDL2*CL1 + (CDLq + 2*CDL2q*CL1)*qbar_o + CDLde*de_o)*CL1_w + (2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o + CDSdr*dr_o)*CS1_w + CDSp*CS1*pbar_w + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_w + CDSr*CS1*rbar_w
        CD_p = CDSp*CS1*pbar_p
        CD_q = (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_q
        CD_r = CDSr*CS1*rbar_r
        
        Cl_u = Clb*b_u + Clp*pbar_u + ClLr*CL1_u*rbar_o + (Clr + ClLr*CL1)*rbar_u
        Cl_v = Clb*b_v + Clp*pbar_v + (Clr + ClLr*CL1)*rbar_v
        Cl_w = Clb*b_w + Clp*pbar_w + ClLr*CL1_w*rbar_o + (Clr + ClLr*CL1)*rbar_w
        Cl_p = Clp*pbar_p
        Cl_q = 0.0
        Cl_r = (Clr + ClLr*CL1)*rbar_r
        
        Cm_u = Cma*a_u + Cmq*qbar_u
        Cm_v = Cmq*qbar_v
        Cm_w = Cma*a_w + Cmq*qbar_w
        Cm_p = 0.0
        Cm_q = Cmq*qbar_q
        Cm_r = 0.0
        
        Cn_u = Cnb*b_u + (CnLp*pbar_o + CnLda*da_o)*CL1_u + (Cnp + CnLp*CL1)*pbar_u + Cnr*rbar_u
        Cn_v = Cnb*b_v + (Cnp + CnLp*CL1)*pbar_v + Cnr*rbar_v
        Cn_w = Cnb*b_w + (CnLp*pbar_o + CnLda*da_o)*CL1_w + (Cnp + CnLp*CL1)*pbar_w + Cnr*rbar_w
        Cn_p = (Cnp + CnLp*CL1)*pbar_p
        Cn_q = 0.0
        Cn_r = Cnr*rbar_r
                
        '''Body-fixed force and moment derivatives'''
        self.Fxb_u = self.rho*self.Sw*u_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL*Ca*a_u + CL_u*Sa + CS*Sa*Sb*a_u - CS*Ca*Cb*b_u - CS_u*Ca*Sb + CD*Sa*Cb*a_u + CD*Ca*Sb*b_u - CD_u*Ca*Cb) + dT_dV*u_o/self.V
        self.Fxb_v = self.rho*self.Sw*v_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL_v*Sa - CS*Ca*Cb*b_v - CS_v*Ca*Sb + CD*Ca*Sb*b_v - CD_v*Ca*Cb) + dT_dV*v_o/self.V
        self.Fxb_w = self.rho*self.Sw*w_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL*Ca*a_w + CL_w*Sa + CS*Sa*Sb*a_w - CS*Ca*Cb*b_w - CS_w*Ca*Sb + CD*Sa*Cb*a_w + CD*Ca*Sb*b_w - CD_w*Ca*Cb) + dT_dV*w_o/self.V
        self.Fxb_p = C_nonDim*(-CD_p*Ca*Cb - CS_p*Ca*Sb + CL_p*Sa)
        self.Fxb_q = C_nonDim*(-CD_q*Ca*Cb - CS_q*Ca*Sb + CL_q*Sa)
        self.Fxb_r = C_nonDim*(-CD_r*Ca*Cb - CS_r*Ca*Sb + CL_r*Sa)
        
        
        self.Fyb_u = self.rho*self.Sw*u_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_u*Cb - CS*Sb*b_u - CD_u*Sb - CD*Cb*b_u)
        self.Fyb_v = self.rho*self.Sw*v_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_v*Cb - CS*Sb*b_v - CD_v*Sb - CD*Cb*b_v)
        self.Fyb_w = self.rho*self.Sw*w_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_w*Cb - CS*Sb*b_w - CD_w*Sb - CD*Cb*b_w)
        self.Fyb_p = C_nonDim*(CS_p*Cb - CD_p*Sb)
        self.Fyb_q = C_nonDim*(CS_q*Cb - CD_q*Sb)
        self.Fyb_r = C_nonDim*(CS_r*Cb - CD_r*Sb)
        
        self.Fzb_u = self.rho*self.Sw*u_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_u*Sa*Cb - CD*Ca*Cb*a_u + CD*Sa*Sb*b_u - CS_u*Sa*Sb - CS*Ca*Sb*a_u - CS*Sa*Cb*b_u - CL_u*Ca + CL*Sa*a_u)
        self.Fzb_v = self.rho*self.Sw*v_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_v*Sa*Cb + CD*Sa*Sb*b_v - CS_v*Sa*Sb - CS*Sa*Cb*b_v - CL_v*Ca)
        self.Fzb_w = self.rho*self.Sw*w_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_w*Sa*Cb - CD*Ca*Cb*a_w + CD*Sa*Sb*b_w - CS_w*Sa*Sb - CS*Ca*Sb*a_w - CS*Sa*Cb*b_w - CL_w*Ca + CL*Sa*a_w)
        self.Fzb_p = C_nonDim*(-CD_p*Sa*Cb - CS_p*Sa*Sb - CL_p*Ca)
        self.Fzb_q = C_nonDim*(-CD_q*Sa*Cb - CS_q*Sa*Sb - CL_q*Ca)
        self.Fzb_r = C_nonDim*(-CD_r*Sa*Cb - CS_r*Sa*Sb - CL_r*Ca)
        
        if self.derivs_approx == True:
            self.set_phillips_approx(coords=False,derivs=True)
        
        self.Mxb_u = C_nonDim*self.bw*Cl_u + self.rho*self.Sw*self.bw*u_o*Cl - delta_y*self.Fzb_u + delta_z*self.Fyb_u
        self.Mxb_v = C_nonDim*self.bw*Cl_v + self.rho*self.Sw*self.bw*v_o*Cl - delta_y*self.Fzb_v + delta_z*self.Fyb_v
        self.Mxb_w = C_nonDim*self.bw*Cl_w + self.rho*self.Sw*self.bw*w_o*Cl - delta_y*self.Fzb_w + delta_z*self.Fyb_w
        self.Mxb_p = C_nonDim*self.bw*Cl_p - delta_y*self.Fzb_p + delta_z*self.Fyb_p
        self.Mxb_q = C_nonDim*self.bw*Cl_q - delta_y*self.Fzb_q + delta_z*self.Fyb_q
        self.Mxb_r = C_nonDim*self.bw*Cl_r - delta_y*self.Fzb_r + delta_z*self.Fyb_r
        
        self.Myb_u = C_nonDim*self.cw*Cm_u + self.rho*self.Sw*self.cw*u_o*Cm - delta_z*self.Fxb_u + delta_x*self.Fzb_u
        self.Myb_v = C_nonDim*self.cw*Cm_v + self.rho*self.Sw*self.cw*v_o*Cm - delta_z*self.Fxb_v + delta_x*self.Fzb_v
        self.Myb_w = C_nonDim*self.cw*Cm_w + self.rho*self.Sw*self.cw*w_o*Cm - delta_z*self.Fxb_w + delta_x*self.Fzb_w
        self.Myb_p = C_nonDim*self.cw*Cm_p - delta_z*self.Fxb_p + delta_x*self.Fzb_p
        self.Myb_q = C_nonDim*self.cw*Cm_q - delta_z*self.Fxb_q + delta_x*self.Fzb_q
        self.Myb_r = C_nonDim*self.cw*Cm_r - delta_z*self.Fxb_r + delta_x*self.Fzb_r
        
        self.Mzb_u = C_nonDim*self.bw*Cn_u + self.rho*self.Sw*self.bw*u_o*Cn + delta_y*self.Fxb_u - delta_x*self.Fyb_u
        self.Mzb_v = C_nonDim*self.bw*Cn_v + self.rho*self.Sw*self.bw*v_o*Cn + delta_y*self.Fxb_v - delta_x*self.Fyb_v
        self.Mzb_w = C_nonDim*self.bw*Cn_w + self.rho*self.Sw*self.bw*w_o*Cn + delta_y*self.Fxb_w - delta_x*self.Fyb_w
        self.Mzb_p = C_nonDim*self.bw*Cn_p + delta_y*self.Fxb_p - delta_x*self.Fyb_p
        print(self.Fxb_p)
        print(self.Fyb_p)
        self.Mzb_q = C_nonDim*self.bw*Cn_q + delta_y*self.Fxb_q - delta_x*self.Fyb_q
        self.Mzb_r = C_nonDim*self.bw*Cn_r + delta_y*self.Fxb_r - delta_x*self.Fyb_r
    
        self.acceleration_derivatives()

    def analytic_derivatives_BIRE(self):

        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        p_o = (self.eq_rot[0])
        q_o = (self.eq_rot[1])
        r_o = (self.eq_rot[2])
        
        pbar_o = p_o*self.bw/(2.*self.V)
        qbar_o = q_o*self.cw/(2.*self.V)
        rbar_o = r_o*self.bw/(2.*self.V)
                
        tau_o = self.eq_inputs[0]
        alpha_o = (self.eq_inputs[1])
        beta_o = (self.eq_inputs[2])
        da_o = (self.eq_inputs[3])
        de_o = (self.eq_inputs[4])
        dr_o = (self.eq_inputs[5])
        
        Ca = np.cos(alpha_o)
        Sa = np.sin(alpha_o)
        Cb = np.cos(beta_o)
        Sb = np.sin(beta_o)
        
        a_u = - (w_o/(u_o*u_o + w_o*w_o))
        a_w = (u_o/(u_o*u_o + w_o*w_o))
        
        b_u = - ((u_o*v_o)/(self.V*self.V*np.sqrt(u_o*u_o + w_o*w_o)))
        b_v = np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V)
        b_w = - ((v_o*w_o)/(self.V*self.V*np.sqrt(u_o*u_o + w_o*w_o)))
        
        pbar_u, qbar_u, rbar_u = -(u_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        pbar_v, qbar_v, rbar_v = -(v_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        pbar_w, qbar_w, rbar_w = -(w_o/(self.V*self.V))*np.array([pbar_o,qbar_o,rbar_o])
        
        pbar_p = self.bw/(2*self.V)
        rbar_r = pbar_p
        qbar_q = self.cw/(2*self.V)
        
        delta_x, delta_y, delta_z = self.cg_shift

        # dT_dV = tau_o*self.dthrust_dV(self.V, self.rho, self.rho_0)
        if self.simple_thrust == True:
            dT_dV = self.thrust_derivatives_simple(self.V)
        else:
            dT_dV = self.thrust_derivatives(tau_o,self.V,self.H)
        
        print('dT_dV: ', dT_dV)

        # Mach = self.V/self.a
        C_nonDim = 0.5*self.rho*self.V*self.V*self.Sw

        CL, CS, CD, Cl, Cm, Cn = self.eq_FM_wind
        
        print('\nWind Force/Moment Coeffs from trim solution:')
        print(CL, CS, CD, Cl, Cm, Cn)
        
        FX, FY, FZ, Mx, My, Mz = self.eq_FM
        print('\nBody Force/Moment Coeffs from trim solution:')
        print(FX, FY, FZ, Mx, My, Mz)
        
        self.aeroModel.evaluate_coeffs(dr_o)
        
        CL1 = self.aeroModel._CL(alpha_o, 0., 0., 0., 0., 0., 0., 0.)
        CLa = self.aeroModel.CLa
        CLb = self.aeroModel.CLb
        CLp = self.aeroModel.CLp
        CLr = self.aeroModel.CLr
        CLq = self.aeroModel.CLq
        
        CS1 = self.aeroModel._CS(0., beta_o, 0., 0., 0., 0., 0., 0.)
        CSa = self.aeroModel.CSa
        CSb = self.aeroModel.CSb
        CSLp = self.aeroModel.CSLp
        CSp = self.aeroModel.CSp
        CSr = self.aeroModel.CSr
        CSq = self.aeroModel.CSq
        
        CDL = self.aeroModel.CDL
        CDL2 = self.aeroModel.CDL2
        CDLq = self.aeroModel.CDLq
        CDL2q = self.aeroModel.CDL2q
        CDLde = self.aeroModel.CDLde
        CDS = self.aeroModel.CDS
        CDS2 = self.aeroModel.CDS2
        CDp = self.aeroModel.CDp
        CDSp = self.aeroModel.CDSp
        CDr = self.aeroModel.CDr
        CDSr = self.aeroModel.CDSr
        CDSda = self.aeroModel.CDSda

        CDq = self.aeroModel.CDq
        
        Cla = self.aeroModel.Cla
        Clb = self.aeroModel.Clb
        Clp = self.aeroModel.Clp
        Clq = self.aeroModel.Clq
        Clr = self.aeroModel.Clr
        ClLr = self.aeroModel.ClLr
        
        Cma = self.aeroModel.Cma
        Cmb = self.aeroModel.Cmb
        Cmp = self.aeroModel.Cmp
        Cmq = self.aeroModel.Cmq
        Cmr = self.aeroModel.Cmr
        
        Cna = self.aeroModel.Cna
        Cnb = self.aeroModel.Cnb
        CnLp = self.aeroModel.CnLp
        CnLda = self.aeroModel.CnLda
        Cnp = self.aeroModel.Cnp
        CnLp = self.aeroModel.CnLp
        Cnq = self.aeroModel.Cnq
        Cnr = self.aeroModel.Cnr
        
        CL1_u = CLa*a_u
        CL1_w = CLa*a_w
        
        CS1_u = CSb*b_u
        CS1_v = CSb*b_v
        CS1_w = CSb*b_w

        '''wind force and moment coefficient derivatives'''
        CL_u = CLa*a_u + CLb*b_u + CLp*pbar_u + CLq*qbar_u + CLr*rbar_u
        CL_v = CLb*b_v + CLp*pbar_v + CLq*qbar_v + CLr*rbar_v
        CL_w = CLa*a_w + CLb*b_w + CLp*pbar_w + CLq*qbar_w + CLr*rbar_w
        CL_p = CLp*pbar_p
        CL_q = CLq*qbar_q
        CL_r = CLr*rbar_r
        
        CS_u = CSa*a_u + CSb*b_u + CSLp*CL1_u*pbar_o + (CSp + CSLp*CL1)*pbar_u + CSq*qbar_u + CSr*rbar_u
        CS_v = CSb*b_v + (CSp + CSLp*CL1)*pbar_v + CSq*qbar_v + CSr*rbar_v
        CS_w = CSa*a_w + CSb*b_w + CSLp*CL1_w*pbar_o + (CSp + CSLp*CL1)*pbar_w + CSq*qbar_w + CSr*rbar_w
        CS_p = (CSp + CSLp*CL1)*pbar_p
        CS_q = CSq*qbar_q
        CS_r = CSr*rbar_r
        
        CD_u = (CDL + 2*CDL2*CL1 + (CDLq + 2*CDL2q*CL1)*qbar_o + CDLde*de_o)*CL1_u + (CDS + 2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o)*CS1_u + (CDp + CDSp*CS1)*pbar_u + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_u + (CDr + CDSr*CS1)*rbar_u
        CD_v = (CDS + 2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o)*CS1_v + (CDp + CDSp*CS1)*pbar_v + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_v + (CDr + CDSr*CS1)*rbar_v
        CD_w = (CDL + 2*CDL2*CL1 + (CDLq + 2*CDL2q*CL1)*qbar_o + CDLde*de_o)*CL1_w + (CDS + 2*CDS2*CS1 + CDSp*pbar_o + CDSr*rbar_o + CDSda*da_o)*CS1_w + (CDp + CDSp*CS1)*pbar_w + (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_w + (CDr + CDSr*CS1)*rbar_w
        CD_p = (CDp + CDSp*CS1)*pbar_p
        CD_q = (CDq + CDLq*CL1 + CDL2q*CL1*CL1)*qbar_q
        CD_r = (CDr + CDSr*CS1)*rbar_r
        
        Cl_u = Cla*a_u + Clb*b_u + Clp*pbar_u + Clq*qbar_u + ClLr*CL1_u*rbar_o + (Clr + ClLr*CL1)*rbar_u
        Cl_v = Clb*b_v + Clp*pbar_v + (Clr + ClLr*CL1)*rbar_v
        Cl_w = Cla*a_w + Clb*b_w + Clp*pbar_w + Clq*qbar_w + ClLr*CL1_w*rbar_o + (Clr + ClLr*CL1)*rbar_w
        Cl_p = Clp*pbar_p
        Cl_q = Clq*qbar_q
        Cl_r = (Clr + ClLr*CL1)*rbar_r
        
        Cm_u = Cma*a_u + Cmb*b_u + Cmp*pbar_u + Cmq*qbar_u + Cmr*rbar_u
        Cm_v = Cmb*b_v + Cmp*pbar_v + Cmq*qbar_v + Cmr*rbar_v
        Cm_w = Cma*a_w + Cmb*b_w + Cmp*pbar_w + Cmq*qbar_w + Cmr*rbar_w
        Cm_p = Cmp*pbar_p
        Cm_q = Cmq*qbar_q
        Cm_r = Cmr*rbar_r
        
        Cn_u = Cna*a_u + Cnb*b_u + (CnLp*pbar_o + CnLda*da_o)*CL1_u + (Cnp + CnLp*CL1)*pbar_u + Cnq*qbar_u + Cnr*rbar_u
        Cn_v = Cnb*b_v + (Cnp + CnLp*CL1)*pbar_v + Cnr*rbar_v
        Cn_w = Cna*a_w + Cnb*b_w + (CnLp*pbar_o + CnLda*da_o)*CL1_w + (Cnp + CnLp*CL1)*pbar_w  + Cnq*qbar_w + Cnr*rbar_w
        Cn_p = (Cnp + CnLp*CL1)*pbar_p
        Cn_q = Cnq*qbar_q
        Cn_r = Cnr*rbar_r
                
        '''Body-fixed force and moment derivatives'''
        self.Fxb_u = self.rho*self.Sw*u_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL*Ca*a_u + CL_u*Sa + CS*Sa*Sb*a_u - CS*Ca*Cb*b_u - CS_u*Ca*Sb + CD*Sa*Cb*a_u + CD*Ca*Sb*b_u - CD_u*Ca*Cb) + dT_dV*u_o/self.V
        self.Fxb_v = self.rho*self.Sw*v_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL_v*Sa - CS*Ca*Cb*b_v - CS_v*Ca*Sb + CD*Ca*Sb*b_v - CD_v*Ca*Cb) + dT_dV*v_o/self.V
        self.Fxb_w = self.rho*self.Sw*w_o*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + C_nonDim*(CL*Ca*a_w + CL_w*Sa + CS*Sa*Sb*a_w - CS*Ca*Cb*b_w - CS_w*Ca*Sb + CD*Sa*Cb*a_w + CD*Ca*Sb*b_w - CD_w*Ca*Cb) + dT_dV*w_o/self.V
        self.Fxb_p = C_nonDim*(-CD_p*Ca*Cb - CS_p*Ca*Sb + CL_p*Sa)
        self.Fxb_q = C_nonDim*(-CD_q*Ca*Cb - CS_q*Ca*Sb + CL_q*Sa)
        self.Fxb_r = C_nonDim*(-CD_r*Ca*Cb - CS_r*Ca*Sb + CL_r*Sa)
        
        
        self.Fyb_u = self.rho*self.Sw*u_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_u*Cb - CS*Sb*b_u - CD_u*Sb - CD*Cb*b_u)
        self.Fyb_v = self.rho*self.Sw*v_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_v*Cb - CS*Sb*b_v - CD_v*Sb - CD*Cb*b_v)
        self.Fyb_w = self.rho*self.Sw*w_o*(CS*Cb - CD*Sb) + C_nonDim*(CS_w*Cb - CS*Sb*b_w - CD_w*Sb - CD*Cb*b_w)
        self.Fyb_p = C_nonDim*(CS_p*Cb - CD_p*Sb)
        self.Fyb_q = C_nonDim*(CS_q*Cb - CD_q*Sb)
        self.Fyb_r = C_nonDim*(CS_r*Cb - CD_r*Sb)
        
        self.Fzb_u = self.rho*self.Sw*u_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_u*Sa*Cb - CD*Ca*Cb*a_u + CD*Sa*Sb*b_u - CS_u*Sa*Sb - CS*Ca*Sb*a_u - CS*Sa*Cb*b_u - CL_u*Ca + CL*Sa*a_u)
        self.Fzb_v = self.rho*self.Sw*v_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_v*Sa*Cb + CD*Sa*Sb*b_v - CS_v*Sa*Sb - CS*Sa*Cb*b_v - CL_v*Ca)
        self.Fzb_w = self.rho*self.Sw*w_o*(-CD*Sa*Cb - CS*Sa*Sb - CL*Ca) + C_nonDim*(-CD_w*Sa*Cb - CD*Ca*Cb*a_w + CD*Sa*Sb*b_w - CS_w*Sa*Sb - CS*Ca*Sb*a_w - CS*Sa*Cb*b_w - CL_w*Ca + CL*Sa*a_w)
        self.Fzb_p = C_nonDim*(-CD_p*Sa*Cb - CS_p*Sa*Sb - CL_p*Ca)
        self.Fzb_q = C_nonDim*(-CD_q*Sa*Cb - CS_q*Sa*Sb - CL_q*Ca)
        self.Fzb_r = C_nonDim*(-CD_r*Sa*Cb - CS_r*Sa*Sb - CL_r*Ca)
        
        if self.derivs_approx == True:
            self.set_phillips_approx(coords=False,derivs=True)
        
        self.Mxb_u = C_nonDim*self.bw*Cl_u + self.rho*self.Sw*self.bw*u_o*Cl - delta_y*self.Fzb_u + delta_z*self.Fyb_u
        self.Mxb_v = C_nonDim*self.bw*Cl_v + self.rho*self.Sw*self.bw*v_o*Cl - delta_y*self.Fzb_v + delta_z*self.Fyb_v
        self.Mxb_w = C_nonDim*self.bw*Cl_w + self.rho*self.Sw*self.bw*w_o*Cl - delta_y*self.Fzb_w + delta_z*self.Fyb_w
        self.Mxb_p = C_nonDim*self.bw*Cl_p - delta_y*self.Fzb_p + delta_z*self.Fyb_p
        self.Mxb_q = C_nonDim*self.bw*Cl_q - delta_y*self.Fzb_q + delta_z*self.Fyb_q
        self.Mxb_r = C_nonDim*self.bw*Cl_r - delta_y*self.Fzb_r + delta_z*self.Fyb_r
        
        self.Myb_u = C_nonDim*self.cw*Cm_u + self.rho*self.Sw*self.cw*u_o*Cm - delta_z*self.Fxb_u + delta_x*self.Fzb_u
        self.Myb_v = C_nonDim*self.cw*Cm_v + self.rho*self.Sw*self.cw*v_o*Cm - delta_z*self.Fxb_v + delta_x*self.Fzb_v
        self.Myb_w = C_nonDim*self.cw*Cm_w + self.rho*self.Sw*self.cw*w_o*Cm - delta_z*self.Fxb_w + delta_x*self.Fzb_w
        self.Myb_p = C_nonDim*self.cw*Cm_p - delta_z*self.Fxb_p + delta_x*self.Fzb_p
        self.Myb_q = C_nonDim*self.cw*Cm_q - delta_z*self.Fxb_q + delta_x*self.Fzb_q
        self.Myb_r = C_nonDim*self.cw*Cm_r - delta_z*self.Fxb_r + delta_x*self.Fzb_r
        
        self.Mzb_u = C_nonDim*self.bw*Cn_u + self.rho*self.Sw*self.bw*u_o*Cn + delta_y*self.Fxb_u - delta_x*self.Fyb_u
        self.Mzb_v = C_nonDim*self.bw*Cn_v + self.rho*self.Sw*self.bw*v_o*Cn + delta_y*self.Fxb_v - delta_x*self.Fyb_v
        self.Mzb_w = C_nonDim*self.bw*Cn_w + self.rho*self.Sw*self.bw*w_o*Cn + delta_y*self.Fxb_w - delta_x*self.Fyb_w
        self.Mzb_p = C_nonDim*self.bw*Cn_p + delta_y*self.Fxb_p - delta_x*self.Fyb_p
        self.Mzb_q = C_nonDim*self.bw*Cn_q + delta_y*self.Fxb_q - delta_x*self.Fyb_q
        self.Mzb_r = C_nonDim*self.bw*Cn_r + delta_y*self.Fxb_r - delta_x*self.Fyb_r
    
        self.acceleration_derivatives()
    
    def thrust_derivatives_simple(self, V):
        # thrust model for the F-16/BIRE as a function of velocity
        T0 = 29550.
        T1 = 0.
        T2 = 0.
        a = 0.84
        C1 = (self.rho/self.rho_0)**a
        C2 = T1 + 2*T2*V
        dT = C1*C2
        return dT

    def thrust_derivatives(self,tau,V,H):
        
        if tau <= 0.:
            P1 = 0.
        elif tau <= 0.77:
            P1 = 64.94*tau
        elif tau <= 1.:
            P1 = 217.38*tau - 117.38
        else:
            P1 = 100.
        
        a_idle , T0_idle, T1_idle, T2_idle = self.thrust_model.idle_coefs(H)
        a_mil , T0_mil, T1_mil, T2_mil = self.thrust_model.mil_coefs(H)
        a_max , T0_max, T1_max, T2_max = self.thrust_model.max_coefs(H)
        
        T_V_idle = ((self.rho/self.rho_0)**a_idle)*(T1_idle + 2*T2_idle*V)
        T_V_mil = ((self.rho/self.rho_0)**a_mil)*(T1_mil + 2*T2_mil*V)
        T_V_max = ((self.rho/self.rho_0)**a_max)*(T1_max + 2*T2_max*V)
        
        if P1 < 50:
            T_V = T_V_idle + (T_V_mil - T_V_idle)*(P1/50)
        elif P1 >= 50:
            T_V = T_V_mil + (T_V_max - T_V_mil)*((P1 - 50)/50)  
        
        return T_V

    def force_derivs(self, Fm2, Fm1, Fp1, Fp2, delta):
        '''fourth order central difference'''
        df_prime = (-Fp2 + 8*Fp1 - 8*Fm1 + Fm2)/(12*delta)
        return df_prime

    def solve_numeric_derivatives(self, dAlpha, dBeta, dp, dq, dr):
        
        u_o = self.eq_velo[0]
        v_o = self.eq_velo[1]
        w_o = self.eq_velo[2]
        
        p_o = (self.eq_rot[0])
        q_o = (self.eq_rot[1])
        r_o = (self.eq_rot[2])
        
        pbar_o = p_o*self.bw/(2.*self.V)
        qbar_o = q_o*self.cw/(2.*self.V)
        rbar_o = r_o*self.bw/(2.*self.V)
        
        # phi_o = self.eq_euler[0]
        # theta_o = self.eq_euler[1]
        
        tau_o = self.eq_inputs[0]
        alpha_o = (self.eq_inputs[1])
        beta_o = (self.eq_inputs[2])
        da_o = (self.eq_inputs[3])
        de_o = (self.eq_inputs[4])
        dr_o = (self.eq_inputs[5])
        
        FM_dim = self.aeroModel.aero_CG_offset_results(alpha_o, beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o, 0.0,
                                                   self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                   M = self.M, use_Anderson = True, enforce_stall=self.stall, thrust_off = True)
        
        FX0, FY0, FZ0, Mx0, My0, Mz0 = FM_dim
        
        print('\nBody Force/Moment Coeffs from trim solution:')
        print(FX0, FY0, FZ0, Mx0, My0, Mz0)
                        
        '''dAlpha Data'''
        
        dAlpha_array = np.array([alpha_o - 2*dAlpha, alpha_o - dAlpha, alpha_o + dAlpha, alpha_o + 2*dAlpha])
        
        FM_dAlpha = np.zeros((4,6))
        
        for i in range(len(dAlpha_array)):
            
            FX, FY, FZ, Mx, My, Mz = self.aeroModel.aero_CG_offset_results(dAlpha_array[i], beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o, tau_o,
                                                       self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                       M = self.M, use_Anderson = True, enforce_stall=self.stall)
            
            FM_dAlpha[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])


        dFX_dAlpha = self.force_derivs(FM_dAlpha[0,0],FM_dAlpha[1,0],FM_dAlpha[2,0],FM_dAlpha[3,0], dAlpha)
        dFY_dAlpha = self.force_derivs(FM_dAlpha[0,1],FM_dAlpha[1,1],FM_dAlpha[2,1],FM_dAlpha[3,1], dAlpha)
        dFZ_dAlpha = self.force_derivs(FM_dAlpha[0,2],FM_dAlpha[1,2],FM_dAlpha[2,2],FM_dAlpha[3,2], dAlpha)
            
        dMX_dAlpha = self.force_derivs(FM_dAlpha[0,3],FM_dAlpha[1,3],FM_dAlpha[2,3],FM_dAlpha[3,3], dAlpha)
        dMY_dAlpha = self.force_derivs(FM_dAlpha[0,4],FM_dAlpha[1,4],FM_dAlpha[2,4],FM_dAlpha[3,4], dAlpha)
        dMZ_dAlpha = self.force_derivs(FM_dAlpha[0,5],FM_dAlpha[1,5],FM_dAlpha[2,5],FM_dAlpha[3,5], dAlpha)
        
        '''dBeta Data'''
                
        dBeta_array = np.array([beta_o - 2*dBeta, beta_o - dBeta, beta_o + dBeta, beta_o + 2*dBeta])

        FM_dBeta = np.zeros((4,6))
        
        for i in range(len(dBeta_array)):
            
            # FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, dBeta_array[i], pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V)
            
            FX, FY, FZ, Mx, My, Mz = self.aeroModel.aero_CG_offset_results(alpha_o, dBeta_array[i], pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o, tau_o,
                                                       self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                       M = self.M, use_Anderson = True, enforce_stall=self.stall)


            FM_dBeta[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])

        dFX_dBeta = self.force_derivs(FM_dBeta[0,0],FM_dBeta[1,0],FM_dBeta[2,0],FM_dBeta[3,0], dBeta)
        dFY_dBeta = self.force_derivs(FM_dBeta[0,1],FM_dBeta[1,1],FM_dBeta[2,1],FM_dBeta[3,1], dBeta)
        dFZ_dBeta = self.force_derivs(FM_dBeta[0,2],FM_dBeta[1,2],FM_dBeta[2,2],FM_dBeta[3,2], dBeta)
            
        dMX_dBeta = self.force_derivs(FM_dBeta[0,3],FM_dBeta[1,3],FM_dBeta[2,3],FM_dBeta[3,3], dBeta)
        dMY_dBeta = self.force_derivs(FM_dBeta[0,4],FM_dBeta[1,4],FM_dBeta[2,4],FM_dBeta[3,4], dBeta)
        dMZ_dBeta = self.force_derivs(FM_dBeta[0,5],FM_dBeta[1,5],FM_dBeta[2,5],FM_dBeta[3,5], dBeta)
        
        
        '''Thrust Derivative'''
        
        # dT_dV = tau_o*self.dthrust_dV(self.V, self.rho, self.rho_0)
        if self.simple_thrust == True:
            dT_dV = self.thrust_derivatives_simple(self.V)
        else:
            dT_dV = self.thrust_derivatives(tau_o,self.V,self.H)
        
        
        print('dT_dV: ', dT_dV)
        
        # self.eq_FM is dimensional so it needs to be nondimensionalized and then
        # multiplied by the values
        dFX_dV = (2*FX0/self.V) + dT_dV
        # dFX_dV = 2*FX0/self.V
        dFY_dV = 2*FY0/self.V
        dFZ_dV = 2*FZ0/self.V
        
        '''CHECK THIS'''
        dMX_dV = 2*Mx0/self.V
        dMY_dV = 2*My0/self.V
        dMZ_dV = 2*Mz0/self.V
        
        '''Could condense these calculations into arrays****'''
        
        '''derivative of body-fixed forces with respect to u'''
        self.Fxb_u = (u_o/self.V)*dFX_dV - (w_o/(u_o*u_o + w_o*w_o))*dFX_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFX_dBeta
        self.Fyb_u = (u_o/self.V)*dFY_dV - (w_o/(u_o*u_o + w_o*w_o))*dFY_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFY_dBeta
        self.Fzb_u = (u_o/self.V)*dFZ_dV - (w_o/(u_o*u_o + w_o*w_o))*dFZ_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to u'''
        self.Mxb_u = (u_o/self.V)*dMX_dV - (w_o/(u_o*u_o + w_o*w_o))*dMX_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMX_dBeta
        self.Myb_u = (u_o/self.V)*dMY_dV - (w_o/(u_o*u_o + w_o*w_o))*dMY_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMY_dBeta
        self.Mzb_u = (u_o/self.V)*dMZ_dV - (w_o/(u_o*u_o + w_o*w_o))*dMZ_dAlpha - ((u_o*v_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMZ_dBeta
        
        '''derivative of body-fixed forces with respect to v'''
        self.Fxb_v = (v_o/self.V)*dFX_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFX_dBeta
        self.Fyb_v = (v_o/self.V)*dFY_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFY_dBeta
        self.Fzb_v = (v_o/self.V)*dFZ_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to v'''
        self.Mxb_v = (v_o/self.V)*dMX_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMX_dBeta
        self.Myb_v = (v_o/self.V)*dMY_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMY_dBeta
        self.Mzb_v = (v_o/self.V)*dMZ_dV + (np.sqrt(u_o*u_o + w_o*w_o)/(self.V*self.V))*dMZ_dBeta
        
        '''derivative of body-fixed forces with respect to w'''
        self.Fxb_w = (w_o/self.V)*dFX_dV + (u_o/(u_o*u_o + w_o*w_o))*dFX_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFX_dBeta
        self.Fyb_w = (w_o/self.V)*dFY_dV + (u_o/(u_o*u_o + w_o*w_o))*dFY_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFY_dBeta
        self.Fzb_w = (w_o/self.V)*dFZ_dV + (u_o/(u_o*u_o + w_o*w_o))*dFZ_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dFZ_dBeta
        
        '''derivative of body-fixed moments with respect to w'''
        self.Mxb_w = (w_o/self.V)*dMX_dV + (u_o/(u_o*u_o + w_o*w_o))*dMX_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMX_dBeta
        self.Myb_w = (w_o/self.V)*dMY_dV + (u_o/(u_o*u_o + w_o*w_o))*dMY_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMY_dBeta
        self.Mzb_w = (w_o/self.V)*dMZ_dV + (u_o/(u_o*u_o + w_o*w_o))*dMZ_dAlpha - ((v_o*w_o)/(self.V*self.V*(u_o*u_o + w_o*w_o)))*dMZ_dBeta
        
        
        '''delta rotation rate Data'''
        dpbar = dp*self.bw/(2.*self.V)
        dqbar = dq*self.cw/(2.*self.V)
        drbar = dr*self.bw/(2.*self.V)
        
        dPbar_array = np.array([pbar_o - 2*dpbar, pbar_o - dpbar, pbar_o + dpbar, pbar_o + 2*dpbar])
        dQbar_array = np.array([qbar_o - 2*dqbar, qbar_o - dqbar, qbar_o + dqbar, qbar_o + 2*dqbar])
        dRbar_array = np.array([rbar_o - 2*drbar, rbar_o - drbar, rbar_o + drbar, rbar_o + 2*drbar])
        
        FM_dPbar = np.zeros((4,6))
        FM_dQbar = np.zeros((4,6))
        FM_dRbar = np.zeros((4,6))
        
        for i in range(len(dPbar_array)):
            
            # FXp, FYp, FZp, Mxp, Myp, Mzp = self.run_FM_body(np.array([alpha_o, beta_o, dPbar_array[i], qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            # FM_dPbar[i,:] = np.array([FXp, FYp, FZp, Mxp, Myp, Mzp])
            
            # FXq, FYq, FZq, Mxq, Myq, Mzq = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, dQbar_array[i], rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            # FM_dQbar[i,:] = np.array([FXq, FYq, FZq, Mxq, Myq, Mzq])
            
            # FXr, FYr, FZr, Mxr, Myr, Mzr = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, dRbar_array[i], da_o, de_o, dr_o]), tau_o, self.V) 
            # FM_dRbar[i,:] = np.array([FXr, FYr, FZr, Mxr, Myr, Mzr])
            
            FXp, FYp, FZp, Mxp, Myp, Mzp = self.aeroModel.aero_CG_offset_results(alpha_o, beta_o, dPbar_array[i], qbar_o, rbar_o, da_o, de_o, dr_o, tau_o,
                                                       self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                       M = self.M, use_Anderson = True, enforce_stall=self.stall)
            FM_dPbar[i,:] = np.array([FXp, FYp, FZp, Mxp, Myp, Mzp])
            
            FXq, FYq, FZq, Mxq, Myq, Mzq = self.aeroModel.aero_CG_offset_results(alpha_o, beta_o, pbar_o, dQbar_array[i], rbar_o, da_o, de_o, dr_o, tau_o,
                                                       self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                       M = self.M, use_Anderson = True, enforce_stall=self.stall)
            FM_dQbar[i,:] = np.array([FXq, FYq, FZq, Mxq, Myq, Mzq])
            
            FXr, FYr, FZr, Mxr, Myr, Mzr = self.aeroModel.aero_CG_offset_results(alpha_o, beta_o, pbar_o, qbar_o, dRbar_array[i], da_o, de_o, dr_o, tau_o,
                                                       self.V, self.H, self.rho_0, self.rho, self.cg_shift, compressible = self.compressible,
                                                       M = self.M, use_Anderson = True, enforce_stall=self.stall)
            FM_dRbar[i,:] = np.array([FXr, FYr, FZr, Mxr, Myr, Mzr])
        
        self.Fxb_p = self.force_derivs(FM_dPbar[0,0],FM_dPbar[1,0],FM_dPbar[2,0],FM_dPbar[3,0], dp)
        self.Fyb_p = self.force_derivs(FM_dPbar[0,1],FM_dPbar[1,1],FM_dPbar[2,1],FM_dPbar[3,1], dp)
        self.Fzb_p = self.force_derivs(FM_dPbar[0,2],FM_dPbar[1,2],FM_dPbar[2,2],FM_dPbar[3,2], dp)
            
        self.Mxb_p = self.force_derivs(FM_dPbar[0,3],FM_dPbar[1,3],FM_dPbar[2,3],FM_dPbar[3,3], dp)
        self.Myb_p = self.force_derivs(FM_dPbar[0,4],FM_dPbar[1,4],FM_dPbar[2,4],FM_dPbar[3,4], dp)
        self.Mzb_p = self.force_derivs(FM_dPbar[0,5],FM_dPbar[1,5],FM_dPbar[2,5],FM_dPbar[3,5], dp)
        
        self.Fxb_q = self.force_derivs(FM_dQbar[0,0],FM_dQbar[1,0],FM_dQbar[2,0],FM_dQbar[3,0], dq)
        self.Fyb_q = self.force_derivs(FM_dQbar[0,1],FM_dQbar[1,1],FM_dQbar[2,1],FM_dQbar[3,1], dq)
        self.Fzb_q = self.force_derivs(FM_dQbar[0,2],FM_dQbar[1,2],FM_dQbar[2,2],FM_dQbar[3,2], dq)
            
        self.Mxb_q = self.force_derivs(FM_dQbar[0,3],FM_dQbar[1,3],FM_dQbar[2,3],FM_dQbar[3,3], dq)
        self.Myb_q = self.force_derivs(FM_dQbar[0,4],FM_dQbar[1,4],FM_dQbar[2,4],FM_dQbar[3,4], dq)
        self.Mzb_q = self.force_derivs(FM_dQbar[0,5],FM_dQbar[1,5],FM_dQbar[2,5],FM_dQbar[3,5], dq)
        
        self.Fxb_r = self.force_derivs(FM_dRbar[0,0],FM_dRbar[1,0],FM_dRbar[2,0],FM_dRbar[3,0], dr)
        self.Fyb_r = self.force_derivs(FM_dRbar[0,1],FM_dRbar[1,1],FM_dRbar[2,1],FM_dRbar[3,1], dr)
        self.Fzb_r = self.force_derivs(FM_dRbar[0,2],FM_dRbar[1,2],FM_dRbar[2,2],FM_dRbar[3,2], dr)
            
        self.Mxb_r = self.force_derivs(FM_dRbar[0,3],FM_dRbar[1,3],FM_dRbar[2,3],FM_dRbar[3,3], dr)
        self.Myb_r = self.force_derivs(FM_dRbar[0,4],FM_dRbar[1,4],FM_dRbar[2,4],FM_dRbar[3,4], dr)
        self.Mzb_r = self.force_derivs(FM_dRbar[0,5],FM_dRbar[1,5],FM_dRbar[2,5],FM_dRbar[3,5], dr)
        
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,u derivatvies:',self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,v derivatvies:',self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,w derivatvies:',self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,p derivatvies:',self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,q derivatvies:',self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        # print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,r derivatvies:',self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        # print('\n')
        

        self.acceleration_derivatives()
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

    def print_derivs(self):
        
        print('Fxb,u, Fyb,u, Fzb,u, Mxb,u, Myb,u, Mzb,u')
        print('Fxb,v, Fyb,v, Fzb,v, Mxb,v, Myb,v, Mzb,v')
        print('Fxb,w, Fyb,w, Fzb,w, Mxb,w, Myb,w, Mzb,w')
        print('Fxb,p, Fyb,p, Fzb,p, Mxb,p, Myb,p, Mzb,p')
        print('Fxb,q, Fyb,q, Fzb,q, Mxb,q, Myb,q, Mzb,q')
        print('Fxb,r, Fyb,r, Fzb,r, Mxb,r, Myb,r, Mzb,r')
        dstr = '{:>24.16f}{:>24.16f}{:>24.16f}{:>24.16f}{:>24.16f}{:>24.16f}'
        print(dstr.format(self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        print(dstr.format(self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        print(dstr.format(self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        print(dstr.format(self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        print(dstr.format(self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        print(dstr.format(self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        print('\n')

    def set_deriv_solution(self):
        self.derivs_sol.Fxb_u = self.Fxb_u
        self.derivs_sol.Fxb_v = self.Fxb_v
        self.derivs_sol.Fxb_w = self.Fxb_w
        self.derivs_sol.Fxb_p = self.Fxb_p
        self.derivs_sol.Fxb_q = self.Fxb_q
        self.derivs_sol.Fxb_r = self.Fxb_r
        
        
        self.derivs_sol.Fyb_u = self.Fyb_u
        self.derivs_sol.Fyb_v = self.Fyb_v
        self.derivs_sol.Fyb_w = self.Fyb_w
        self.derivs_sol.Fyb_p = self.Fyb_p
        self.derivs_sol.Fyb_q = self.Fyb_q
        self.derivs_sol.Fyb_r = self.Fyb_r
        
        self.derivs_sol.Fzb_u = self.Fzb_u
        self.derivs_sol.Fzb_v = self.Fzb_v
        self.derivs_sol.Fzb_w = self.Fzb_w
        self.derivs_sol.Fzb_p = self.Fzb_p
        self.derivs_sol.Fzb_q = self.Fzb_q
        self.derivs_sol.Fzb_r = self.Fzb_r
                
        self.derivs_sol.Mxb_u = self.Mxb_u
        self.derivs_sol.Mxb_v = self.Mxb_v
        self.derivs_sol.Mxb_w = self.Mxb_w
        self.derivs_sol.Mxb_p = self.Mxb_p
        self.derivs_sol.Mxb_q = self.Mxb_q
        self.derivs_sol.Mxb_r = self.Mxb_r
        
        self.derivs_sol.Myb_u = self.Myb_u
        self.derivs_sol.Myb_v = self.Myb_v
        self.derivs_sol.Myb_w = self.Myb_w
        self.derivs_sol.Myb_p = self.Myb_p
        self.derivs_sol.Myb_q = self.Myb_q
        self.derivs_sol.Myb_r = self.Myb_r
        
        self.derivs_sol.Mzb_u = self.Mzb_u
        self.derivs_sol.Mzb_v = self.Mzb_v
        self.derivs_sol.Mzb_w = self.Mzb_w
        self.derivs_sol.Mzb_p = self.Mzb_p
        self.derivs_sol.Mzb_q = self.Mzb_q
        self.derivs_sol.Mzb_r = self.Mzb_r
        
        self.derivs_sol.Fzb_wdot = self.Fzb_wdot
        self.derivs_sol.Myb_wdot = self.Myb_wdot
        self.derivs_sol.Fxb_udot = self.Fxb_udot
        self.derivs_sol.Fxb_vdot = self.Fxb_vdot
        self.derivs_sol.Fxb_wdot = self.Fxb_wdot
        self.derivs_sol.Fyb_udot = self.Fyb_udot
        self.derivs_sol.Fyb_vdot = self.Fyb_vdot
        self.derivs_sol.Fyb_wdot = self.Fyb_wdot
        self.derivs_sol.Fzb_vdot = self.Fzb_vdot
        
        self.derivs_sol.Fzb_udot = self.Fzb_udot
        self.derivs_sol.Myb_udot = self.Myb_udot


class test_derivs:
    '''A class for parsing the data for example 9.8.1 in Phillips MOF'''
    def __init__(self):
        
        self.derivs_sol = derivs()
    
    def get_JSON_inputs(self, filename):
        
        json_data = json.loads(open(filename).read()) # reads json input fil
        
        self.g = json_data["operating"]["g[ft/s^2]"]
        
        # AIRCRAFT INPUTS
        self.W = json_data["operating"]["weight[lbf]"]
        self.Sw = json_data["aircraft"]["wing_area[ft^2]"] #ft^2
        self.bw = json_data["aircraft"]["wing_span[ft]"] #ft
        self.cw = self.Sw/self.bw
        
        self.Ixxb = json_data["reference"]["Ixx[slugs*ft^2]"]
        self.Iyyb = json_data["reference"]["Iyy[slugs*ft^2]"]
        self.Izzb = json_data["reference"]["Izz[slugs*ft^2]"]
        self.Ixyb = json_data["reference"]["Ixy[slugs*ft^2]"]
        self.Ixzb = json_data["reference"]["Ixz[slugs*ft^2]"]
        self.Iyzb = json_data["reference"]["Iyz[slugs*ft^2]"]
        
        self.hxb = json_data["reference"]["hx[slug*ft^2/s]"]
        self.hyb = json_data["reference"]["hy[slug*ft^2/s]"]
        self.hzb = json_data["reference"]["hz[slug*ft^2/s]"]

        
        self.V = json_data["operating"]["airspeed[ft/s]"]
        u_o = self.V
        v_o = 0.0
        w_o = 0.0
        
        # t_o = 0.0
        
        theta_o = np.deg2rad(json_data["operating"]["elevation_angle[deg]"])
        phi_o = np.deg2rad(json_data["operating"]["bank_angle[deg]"])
        
        ST_o = np.sin(theta_o)
        CT_o = np.cos(theta_o)
        # TT_o = np.tan(theta_o)
        
        SP_o = np.sin(phi_o)
        CP_o = np.cos(phi_o)
        
        # Omega = (self.g*SP_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o) #radians/s
        # print('Omega [deg/s]:', np.rad2deg(Omega))
        
        # SO_o = np.sin(Omega*t_o)
        # CO_o = np.cos(Omega*t_o)
        
        p_o = (-self.g*SP_o*ST_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        q_o = (self.g*SP_o*SP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        r_o = (self.g*SP_o*CP_o*CT_o*CT_o)/(w_o*ST_o + u_o*CP_o*CT_o)
        
        self.eq_euler = np.array([phi_o,theta_o,0.0])
        self.eq_vel = np.array([u_o, v_o, w_o])
        self.eq_rot = np.array([p_o, q_o, r_o])
        
        self.rho = json_data["operating"]["density[slugs/ft^3]"]
        
        redim_coeff = 0.5*self.rho*self.V*self.V*self.Sw
        
        CXo = json_data["reference"]["CXo"]
        CZo = json_data["reference"]["CZo"]
        Cmo = json_data["reference"]["Cmo"]
        
        self.Fxb_u = (1/self.V)*redim_coeff*(2*CXo + json_data["reference"]["CX,mu"])
        self.Fyb_u = (1/self.V)*redim_coeff*json_data["reference"]["CY,mu"]
        self.Fzb_u = (1/self.V)*redim_coeff*(2*CZo + json_data["reference"]["CZ,mu"])
        self.Mxb_u = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,mu"]
        self.Myb_u = (1/self.V)*self.cw*redim_coeff*(2*Cmo + json_data["reference"]["Cm,mu"])
        self.Mzb_u = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,mu"]
        
        self.Fxb_v = (1/self.V)*redim_coeff*json_data["reference"]["CX,beta"]
        self.Fyb_v = (1/self.V)*redim_coeff*json_data["reference"]["CY,beta"]
        self.Fzb_v = (1/self.V)*redim_coeff*json_data["reference"]["CZ,beta"]
        self.Mxb_v = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,beta"]
        self.Myb_v = (1/self.V)*self.cw*redim_coeff*json_data["reference"]["Cm,beta"]
        self.Mzb_v = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,beta"]
        
        self.Fxb_w = (1/self.V)*redim_coeff*json_data["reference"]["CX,alpha"]
        self.Fyb_w = (1/self.V)*redim_coeff*json_data["reference"]["CY,alpha"]
        self.Fzb_w = (1/self.V)*redim_coeff*json_data["reference"]["CZ,alpha"]
        self.Mxb_w = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cl,alpha"]
        self.Myb_w = (1/self.V)*self.cw*redim_coeff*json_data["reference"]["Cm,alpha"]
        self.Mzb_w = (1/self.V)*self.bw*redim_coeff*json_data["reference"]["Cn,alpha"]
        
        self.Fxb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CX,alphahat"]
        self.Fzb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CZ,alphahat"]
        self.Myb_wdot = (self.cw/(2*self.V*self.V))*redim_coeff*self.cw*json_data["reference"]["Cm,alphahat"]
        
        self.Fxb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CX,muhat"]
        self.Fzb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*json_data["reference"]["CZ,muhat"]
        self.Myb_udot = (self.cw/(2*self.V*self.V))*redim_coeff*self.cw*json_data["reference"]["Cm,muhat"]

        
        self.Fxb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CX,pbar"]
        self.Fyb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CY,pbar"]
        self.Fzb_p = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,pbar"]
        self.Mxb_p = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,pbar"]
        self.Myb_p = (self.bw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,pbar"]
        self.Mzb_p = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,pbar"]
        
        self.Fxb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CX,qbar"]
        self.Fyb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CY,qbar"]
        self.Fzb_q = (self.cw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,qbar"]
        self.Mxb_q = (self.cw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,qbar"]
        self.Myb_q = (self.cw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,qbar"]
        self.Mzb_q = (self.cw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,qbar"]
        
        self.Fxb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CX,rbar"]
        self.Fyb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CY,rbar"]
        self.Fzb_r = (self.bw/(2*self.V))*redim_coeff*json_data["reference"]["CZ,rbar"]
        self.Mxb_r = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cl,rbar"]
        self.Myb_r = (self.bw/(2*self.V))*self.cw*redim_coeff*json_data["reference"]["Cm,rbar"]
        self.Mzb_r = (self.bw/(2*self.V))*self.bw*redim_coeff*json_data["reference"]["Cn,rbar"]

    def solve_derivs(self):
        
        self.set_deriv_solution()
        
        return self.derivs_sol
    
    def set_deriv_solution(self):
        self.derivs_sol.Fxb_u = self.Fxb_u
        self.derivs_sol.Fxb_v = self.Fxb_v
        self.derivs_sol.Fxb_w = self.Fxb_w
        self.derivs_sol.Fxb_p = self.Fxb_p
        self.derivs_sol.Fxb_q = self.Fxb_q
        self.derivs_sol.Fxb_r = self.Fxb_r
        
        
        self.derivs_sol.Fyb_u = self.Fyb_u
        self.derivs_sol.Fyb_v = self.Fyb_v
        self.derivs_sol.Fyb_w = self.Fyb_w
        self.derivs_sol.Fyb_p = self.Fyb_p
        self.derivs_sol.Fyb_q = self.Fyb_q
        self.derivs_sol.Fyb_r = self.Fyb_r
        
        self.derivs_sol.Fzb_u = self.Fzb_u
        self.derivs_sol.Fzb_v = self.Fzb_v
        self.derivs_sol.Fzb_w = self.Fzb_w
        self.derivs_sol.Fzb_p = self.Fzb_p
        self.derivs_sol.Fzb_q = self.Fzb_q
        self.derivs_sol.Fzb_r = self.Fzb_r
                
        self.derivs_sol.Mxb_u = self.Mxb_u
        self.derivs_sol.Mxb_v = self.Mxb_v
        self.derivs_sol.Mxb_w = self.Mxb_w
        self.derivs_sol.Mxb_p = self.Mxb_p
        self.derivs_sol.Mxb_q = self.Mxb_q
        self.derivs_sol.Mxb_r = self.Mxb_r
        
        self.derivs_sol.Myb_u = self.Myb_u
        self.derivs_sol.Myb_v = self.Myb_v
        self.derivs_sol.Myb_w = self.Myb_w
        self.derivs_sol.Myb_p = self.Myb_p
        self.derivs_sol.Myb_q = self.Myb_q
        self.derivs_sol.Myb_r = self.Myb_r
        
        self.derivs_sol.Mzb_u = self.Mzb_u
        self.derivs_sol.Mzb_v = self.Mzb_v
        self.derivs_sol.Mzb_w = self.Mzb_w
        self.derivs_sol.Mzb_p = self.Mzb_p
        self.derivs_sol.Mzb_q = self.Mzb_q
        self.derivs_sol.Mzb_r = self.Mzb_r
        
        self.derivs_sol.Fzb_wdot = self.Fzb_wdot
        self.derivs_sol.Myb_wdot = self.Myb_wdot
        self.derivs_sol.Fxb_udot = self.Fxb_udot
        # self.derivs_sol.Fxb_vdot = self.Fxb_vdot
        self.derivs_sol.Fxb_wdot = self.Fxb_wdot
        # self.derivs_sol.Fyb_udot = self.Fyb_udot
        # self.derivs_sol.Fyb_vdot = self.Fyb_vdot
        # self.derivs_sol.Fyb_wdot = self.Fyb_wdot
        # self.derivs_sol.Fzb_vdot = self.Fzb_vdot
        
        self.derivs_sol.Fzb_udot = self.Fzb_udot
        self.derivs_sol.Myb_udot = self.Myb_udot