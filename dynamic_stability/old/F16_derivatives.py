import numpy as np
import json
import sys


aero_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/aerodynamics_model/'
trim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/trim/'
sim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/flight_simulation/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, trim_directory)
sys.path.insert(1, sim_directory)

from F16_aero import F16Aero

from aircraft_properties import AircraftProperties
from trim_functions import solve_trim

class F16_derivs:

    def __init__(self, aeroModel = F16Aero):
        self.aeroModel = aeroModel

    def solve_derivatives(self, dAlpha, dBeta, dp, dq, dr):
        
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
        
        FX0, FY0, FZ0, Mx0, My0, Mz0 = self.run_FM_body(alpha_o, beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o, 0.0, 
                                   self.V, self.H, self.rho_0, self.rho, cg_shift=self.cg_shift, compressible=self.compressible,
                                   M=113.0, use_Anderson=True, enforce_stall=self.stall)
        
        FX0, FY0, FZ0, Mx0, My0, Mz0 = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), 0.0, self.V)
                
        '''dAlpha Data'''
        
        dAlpha_array = np.array([alpha_o - 2*dAlpha, alpha_o - dAlpha, alpha_o + dAlpha, alpha_o + 2*dAlpha])
        
        FM_dAlpha = np.zeros((4,6))
        
        for i in range(len(dAlpha_array)):

            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([dAlpha_array[i], beta_o, pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            
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
            
            FX, FY, FZ, Mx, My, Mz = self.run_FM_body(np.array([alpha_o, dBeta_array[i], pbar_o, qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            
            FM_dBeta[i,:] = np.array([FX, FY, FZ, Mx, My, Mz])
        
        dFX_dBeta = self.force_derivs(FM_dBeta[0,0],FM_dBeta[1,0],FM_dBeta[2,0],FM_dBeta[3,0], dBeta)
        dFY_dBeta = self.force_derivs(FM_dBeta[0,1],FM_dBeta[1,1],FM_dBeta[2,1],FM_dBeta[3,1], dBeta)
        dFZ_dBeta = self.force_derivs(FM_dBeta[0,2],FM_dBeta[1,2],FM_dBeta[2,2],FM_dBeta[3,2], dBeta)
            
        dMX_dBeta = self.force_derivs(FM_dBeta[0,3],FM_dBeta[1,3],FM_dBeta[2,3],FM_dBeta[3,3], dBeta)
        dMY_dBeta = self.force_derivs(FM_dBeta[0,4],FM_dBeta[1,4],FM_dBeta[2,4],FM_dBeta[3,4], dBeta)
        dMZ_dBeta = self.force_derivs(FM_dBeta[0,5],FM_dBeta[1,5],FM_dBeta[2,5],FM_dBeta[3,5], dBeta)
        
        
        '''Thrust Derivative'''
        
        dT_dV = tau_o*self.dthrust_dV(self.V, self.rho, self.rho_0)
        
        # self.eq_FM is dimensional so it needs to be nondimensionalized and then
        # multiplied by the values
        dFX_dV = 2*FX0/self.V + dT_dV
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
            
            FXp, FYp, FZp, Mxp, Myp, Mzp = self.run_FM_body(np.array([alpha_o, beta_o, dPbar_array[i], qbar_o, rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dPbar[i,:] = np.array([FXp, FYp, FZp, Mxp, Myp, Mzp])
            
            FXq, FYq, FZq, Mxq, Myq, Mzq = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, dQbar_array[i], rbar_o, da_o, de_o, dr_o]), tau_o, self.V) 
            FM_dQbar[i,:] = np.array([FXq, FYq, FZq, Mxq, Myq, Mzq])
            
            FXr, FYr, FZr, Mxr, Myr, Mzr = self.run_FM_body(np.array([alpha_o, beta_o, pbar_o, qbar_o, dRbar_array[i], da_o, de_o, dr_o]), tau_o, self.V) 
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
        
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,u derivatvies:',self.Fxb_u,self.Fyb_u,self.Fzb_u,self.Mxb_u,self.Myb_u,self.Mzb_u))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,v derivatvies:',self.Fxb_v,self.Fyb_v,self.Fzb_v,self.Mxb_v,self.Myb_v,self.Mzb_v))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,w derivatvies:',self.Fxb_w,self.Fyb_w,self.Fzb_w,self.Mxb_w,self.Myb_w,self.Mzb_w))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,p derivatvies:',self.Fxb_p,self.Fyb_p,self.Fzb_p,self.Mxb_p,self.Myb_p,self.Mzb_p))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,q derivatvies:',self.Fxb_q,self.Fyb_q,self.Fzb_q,self.Mxb_q,self.Myb_q,self.Mzb_q))
        print('{:<16}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}{:>20.12f}'.format('FM,r derivatvies:',self.Fxb_r,self.Fyb_r,self.Fzb_r,self.Mxb_r,self.Myb_r,self.Mzb_r))
        print('\n')
        
        self.Fzb_wdot = -((self.rho*self.Sw*self.Sh*self.lwt)/(np.pi*self.bw*self.bw))*self.CLw_a*self.CLh_a
        self.Myb_wdot = -self.xbh*self.Fzb_wdot
        self.Fxb_udot = 0.0
        self.Fxb_vdot = 0.0
        self.Fxb_wdot = 0.0
        self.Fyb_udot = 0.0
        self.Fyb_vdot = 0.0
        self.Fyb_wdot = 0.0
        self.Fzb_vdot = 0.0
        
        self.Fzb_udot = 0.0
        self.Myb_udot = 0.0
        
        print('dFXdalpha: ', dFX_dAlpha)
        print('dFXdbeta: ', dFX_dBeta)
        print('dTdV: ', dT_dV)
        print('\n')