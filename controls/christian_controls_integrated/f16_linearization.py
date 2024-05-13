#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:49:57 2022

@author: christian
"""

import sys
aero_directory = '../aerodynamics_model/'
mass_directory = '../mass_properties/'
trim_directory = '../trim/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, mass_directory)
sys.path.insert(1, trim_directory)

import numpy as np
from f16_aero import F16Aero
import aero_trim as trim
from hunsaker_atm import stdatm_english
from control import ctrb, lqr, place
import matplotlib.pyplot as plt
from os.path import exists

class Lin_Results:
    def __init__(self, N, M):
        self.A = np.zeros((N, N))
        self.B = np.zeros((N, M))
        self.C = np.zeros((N, N))
        self.K = np.zeros((M, N))
        self.eigs = np.zeros(N)
        self.aircraft = "F16"

class LinearizationBaseline:
    def __init__(self, props, aero_dir=aero_directory, N=8, M=4):
        self.N = N
        self.M = M
        self.x_hat = np.zeros(N + M)
        self.u_hat = np.zeros(M)
        self.alpha_hat = 0.
        self.beta_hat = 0.
        self.V_hat = 0.
        self.props = props
        self.rho = props.rho
        self.rho_0 = props.rho_0
        self.S_w = props.S_w
        self.b_w = props.b_w
        self.c_w = props.c_w
        self.W = props.W
        self.g = props.g
        self.aero_dir = aero_dir
        self.tc_tau = 0.05
        self.tc_da = 0.05
        self.tc_de = 0.05
        self.tc_dr = 0.05
        self.rate_da = 80.*np.pi/180.
        self.rate_de = 60*np.pi/180.
        self.rate_dr = 120*np.pi/180.

    def set_linearization_point(self, x_hat, u_hat, alpha_hat, beta_hat, FM_hat,
                                cg_shift):
        self.x_hat = x_hat
        self.u_hat = u_hat
        self.alpha_hat = alpha_hat
        self.beta_hat = beta_hat
        self.V_hat = np.sqrt(np.sum(np.square(self.x_hat[:3])))
        [self.CD_hat, self.CS_hat, self.CL_hat,
         self.Cl_hat, self.Cm_hat, self.Cn_hat] = FM_hat
        self.I_inv = self._I_inv()
        self._W_matrix()
        self.dVinv_dz = self._dVinv_dz()
        self.dV_dz = self._dV_dz()
        self.da_dz = self._dalpha_dz()
        self.db_dz = self._dbeta_dz()
        self.Dx = cg_shift[0]
        self.Dy = cg_shift[1]
        self.Dz = cg_shift[2]
        self.dp_dz = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
        self.dq_dz = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
        self.dr_dz = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
        self.dde_du = np.array([0., 0., 1., 0.])
        self.dda_du = np.array([0., 1., 0., 0.])
        self.dtau_du = np.array([1., 0., 0., 0.])
        self.ddr_du = np.array([0., 0., 0., 1.])
        self.dde2_du = 2.*self.u_hat[2]*self.dde_du
        aero = F16Aero(self.aero_dir,thrust_dir=self.aero_dir)
        self.CL_0 = aero.CL0
        self.CL_a = aero.CLa
        self.CL_q = aero.CLq
        self.CL_de = aero.CLde
        self.CL1_hat = self.CL_0 + self.CL_a*self.alpha_hat
        self.CS_b = aero.CSb
        self.CS_Lp = aero.CSLp
        self.CS_p = aero.CSp
        self.CS_r = aero.CSr
        self.CS_da = aero.CSda
        self.CS_dr = aero.CSdr
        self.CD_L = aero.CDL
        self.CD_L2 = aero.CDL2
        self.CD_S2 = aero.CDS2
        self.CS1_hat = self.CS_b*self.beta_hat
        self.CD_Sp = aero.CDSp
        self.CD_L2q = aero.CDL2q
        self.CD_Lq = aero.CDLq
        self.CD_q = aero.CDq
        self.CD_Sr = aero.CDSr
        self.CD_Sda = aero.CDSda
        self.CD_Lde = aero.CDLde
        self.CD_de = aero.CDde
        self.CD_Sdr = aero.CDSdr
        self.CD_de2 = aero.CDde2
        self.Cl_b = aero.Clb
        self.Cl_p = aero.Clp
        self.Cl_Lr = aero.ClLr
        self.Cl_r = aero.Clr
        self.Cl_da = aero.Clda
        self.Cl_dr = aero.Cldr
        self.Cm_a = aero.Cma
        self.Cm_q = aero.Cmq
        self.Cm_de = aero.Cmde
        self.Cn_b = aero.Cnb
        self.Cn_Lp = aero.CnLp
        self.Cn_p = aero.Cnp
        self.Cn_r = aero.Cnr
        self.Cn_Lda = aero.CnLda
        self.Cn_da = aero.Cnda
        self.Cn_dr = aero.Cndr

    def _det_I(self):
        props = self.props
        C1 = props.Ixx*(props.Iyy*props.Izz - props.Iyz*props.Izy)
        C2 = props.Ixy*(props.Iyx*props.Izz + props.Iyz*props.Izx)
        C3 = props.Ixz*(props.Iyx*props.Izy + props.Iyy*props.Izx)
        return C1 - C2 - C3

    def _I_inv(self):
        props = self.props
        det_I = self._det_I()
        I_inv = np.zeros((3, 3))
        I_inv[0, 0] = props.Iyy*props.Izz - props.Iyz*props.Izy
        I_inv[0, 1] = props.Ixy*props.Izz + props.Ixz*props.Izy
        I_inv[0, 2] = props.Ixy*props.Iyz + props.Ixz*props.Iyy
        I_inv[1, 0] = props.Iyx*props.Izz + props.Iyz*props.Izx
        I_inv[1, 1] = props.Ixx*props.Izz - props.Ixz*props.Izx
        I_inv[1, 2] = props.Ixx*props.Iyz + props.Ixy*props.Ixz
        I_inv[2, 0] = props.Iyz*props.Ixy + props.Iyy*props.Izx
        I_inv[2, 1] = props.Ixx*props.Izy + props.Ixy*props.Izx
        I_inv[2, 2] = props.Ixx*props.Iyy - props.Ixy*props.Iyx
        I_inv = I_inv/det_I
        return I_inv
    
    def _H_matrix(self):
        props = self.props
        H_mat = np.zeros((3,3))
        H_mat[0,1] = -props.hz
        H_mat[0,2] =  props.hy
        H_mat[1,0] =  props.hz
        H_mat[1,2] = -props.hx
        H_mat[2,0] = -props.hy
        H_mat[2,1] =  props.hx
        return H_mat

    def _dz1_dz(self):
        dz1_dz = np.zeros(self.N)
        dFxdz = self._dFx_dz()
        dz1_dz = self.props.g/self.props.W*dFxdz
        dz1_dz[1] += self.x_hat[5]
        dz1_dz[2] -= self.x_hat[4]
        dz1_dz[4] -= self.x_hat[2]
        dz1_dz[5] += self.x_hat[1]
        dz1_dz[7] -= self.props.g*np.cos(self.x_hat[7])
        return dz1_dz

    def _dz2_dz(self):
        dz2_dz = np.zeros(self.N)
        dFydz = self._dFy_dz()
        dz2_dz = self.props.g/self.props.W*dFydz
        dz2_dz[0] -= self.x_hat[5]
        dz2_dz[2] += self.x_hat[3]
        dz2_dz[3] += self.x_hat[2]
        dz2_dz[5] -= self.x_hat[0]
        dz2_dz[6] += self.props.g*np.cos(self.x_hat[6])*np.cos(self.x_hat[7])
        dz2_dz[7] -= self.props.g*np.sin(self.x_hat[6])*np.sin(self.x_hat[7])
        return dz2_dz

    def _dz3_dz(self):
        dz3_dz = np.zeros(self.N)
        dFzdz = self._dFz_dz()
        dz3_dz = self.props.g/self.props.W*dFzdz
        dz3_dz[0] += self.x_hat[4]
        dz3_dz[1] -= self.x_hat[3]
        dz3_dz[3] -= self.x_hat[1]
        dz3_dz[4] += self.x_hat[0]
        dz3_dz[6] -= self.props.g*np.sin(self.x_hat[6])*np.cos(self.x_hat[7])
        dz3_dz[7] -= self.props.g*np.cos(self.x_hat[6])*np.sin(self.x_hat[7])
        return dz3_dz

    def _dz4_dz(self):
        dz4_dz = np.zeros(self.N)
        dMxdz = self._dMx_dz()
        dMydz = self._dMy_dz()
        dMzdz = self._dMz_dz()
        dM = np.array([dMxdz, dMydz, dMzdz])
        R = np.zeros((3, self.N + self.M))
        Hmat = np.block([np.zeros((3,3)),self._H_matrix(),np.zeros((3,2))])
        R = dM + Hmat + self.W_mat
        dz4_dz = np.matmul(self.I_inv, R)[0, :]
        return dz4_dz

    def _dz5_dz(self):
        dz5_dz = np.zeros(self.N)
        dMxdz = self._dMx_dz()
        dMydz = self._dMy_dz()
        dMzdz = self._dMz_dz()
        dM = np.array([dMxdz, dMydz, dMzdz])
        R = np.zeros((3, self.N + self.M))
        Hmat = np.block([np.zeros((3,3)),self._H_matrix(),np.zeros((3,2))])
        R = dM + Hmat + self.W_mat
        dz5_dz = np.matmul(self.I_inv, R)[1, :]
        return dz5_dz

    def _dz6_dz(self):
        dz6_dz = np.zeros(self.N)
        dMxdz = self._dMx_dz()
        dMydz = self._dMy_dz()
        dMzdz = self._dMz_dz()
        dM = np.array([dMxdz, dMydz, dMzdz])
        R = np.zeros((3, self.N + self.M))
        Hmat = np.block([np.zeros((3,3)),self._H_matrix(),np.zeros((3,2))])
        R = dM + Hmat + self.W_mat
        dz6_dz = np.matmul(self.I_inv, R)[2, :]
        return dz6_dz

    def _dz7_dz(self):
        dz7_dz = np.zeros(self.N)
        s_7 = np.sin(self.x_hat[6])
        c_7 = np.cos(self.x_hat[6])
        s_8 = np.sin(self.x_hat[7])
        c_8 = np.cos(self.x_hat[7])
        t_8 = s_8/c_8
        dz7_dz[3] = 1.
        dz7_dz[4] = s_7*t_8
        dz7_dz[5] = c_7*t_8
        dz7_dz[6] = t_8*(c_7*self.x_hat[4] - s_7*self.x_hat[5])
        dz7_dz[7] = s_7/(c_8**2)*self.x_hat[4] + c_7/(c_8**2)*self.x_hat[5]
        return dz7_dz

    def _dz8_dz(self):
        dz8_dz = np.zeros(self.N)
        s_7 = np.sin(self.x_hat[6])
        c_7 = np.cos(self.x_hat[6])
        dz8_dz[4] = c_7
        dz8_dz[5] = -s_7
        dz8_dz[6] = -s_7*self.x_hat[4] - c_7*self.x_hat[5]
        return dz8_dz

    def _dFx_dz(self):
        dFxdz = np.zeros(self.N)
        dCXdz = self._dCX_dz()
        dVdz = self.dV_dz
        dTxdz = self._dTX_dz()
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CX = -(self.CD_hat*c_a*c_b + self.CS_hat*c_a*s_b -
               self.CL_hat*s_a)
        dFxdz = (0.5*self.rho*self.V_hat**2*self.S_w*dCXdz +
                 self.rho*self.V_hat*self.S_w*CX*dVdz + dTxdz)
        return dFxdz

    def _dFy_dz(self):
        dFydz = np.zeros(self.N)
        dCYdz = self._dCY_dz()
        dVdz = self.dV_dz
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CY = self.CS_hat*c_b - self.CD_hat*s_b
        dFydz = (0.5*self.rho*self.V_hat**2*self.S_w*dCYdz +
                 self.rho*self.V_hat*self.S_w*CY*dVdz)
        return dFydz

    def _dFz_dz(self):
        dFzdz = np.zeros(self.N)
        dCZdz = self._dCZ_dz()
        dVdz = self.dV_dz
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CZ = -(self.CD_hat*s_a*c_b + self.CS_hat*s_a*s_b +
               self.CL_hat*c_a)
        dFzdz = (0.5*self.rho*self.V_hat**2*self.S_w*dCZdz +
                 self.rho*self.V_hat*self.S_w*CZ*dVdz)
        return dFzdz

    def _dMx_dz(self):
        dMxdz = np.zeros(self.N)
        dCldz = self._dCl_dz()
        dFzdz = self._dFz_dz()
        dFydz = self._dFy_dz()
        dVdz = self.dV_dz
        dy = self.Dy
        dz = self.Dz
        dMxdz = (0.5*self.rho*self.V_hat**2*self.S_w*self.b_w*dCldz +
                 self.rho*self.V_hat*self.S_w*self.b_w*self.Cl_hat*dVdz -
                 dFzdz*dy +
                 dFydz*dz)
        return dMxdz

    def _dMy_dz(self):
        dMydz = np.zeros(self.N)
        dCmdz = self._dCm_dz()
        dFzdz = self._dFz_dz()
        dFxdz = self._dFx_dz()
        dVdz = self.dV_dz
        dx = self.Dx
        dz = self.Dz
        dMydz = (0.5*self.rho*self.V_hat**2*self.S_w*self.c_w*dCmdz +
                 self.rho*self.V_hat*self.S_w*self.c_w*self.Cm_hat*dVdz -
                 dFxdz*dz +
                 dFzdz*dx)
        return dMydz

    def _dMz_dz(self):
        dMzdz = np.zeros(self.N)
        dCndz = self._dCn_dz()
        dFxdz = self._dFx_dz()
        dFydz = self._dFy_dz()
        dVdz = self.dV_dz
        dx = self.Dx
        dy = self.Dy
        dMzdz = (0.5*self.rho*self.V_hat**2*self.S_w*self.b_w*dCndz +
                 self.rho*self.V_hat*self.S_w*self.b_w*self.Cn_hat*dVdz -
                 dFydz*dx +
                 dFxdz*dy)
        return dMzdz

    def _dV_dz(self):
        dVdz = np.zeros(self.N)
        dVdz[0] = self.x_hat[0]/self.V_hat
        dVdz[1] = self.x_hat[1]/self.V_hat
        dVdz[2] = self.x_hat[2]/self.V_hat
        return dVdz

    def _dVinv_dz(self):
        dVinvdz = np.zeros(self.N)
        dVinvdz[0] = -self.x_hat[0]/self.V_hat**3
        dVinvdz[1] = -self.x_hat[1]/self.V_hat**3
        dVinvdz[2] = -self.x_hat[2]/self.V_hat**3
        return dVinvdz

    def _dTX_dz(self):
        dVdz = self._dV_dz()
        H = self.props.H
        a_mil = self.props.a_mil(H)
        T1_mil = self.props.T1_mil(H)
        T2_mil = self.props.T2_mil(H)
        rho_ratio = (self.rho/self.rho_0)
        dTmil_dz = rho_ratio**a_mil*(T1_mil*dVdz + 2.*T2_mil*self.V_hat*dVdz)
        if self.u_hat[0] < 0.77:
            a_idle = self.props.a_idle(H)
            T1_idle = self.props.T1_idle(H)
            T2_idle = self.props.T2_idle(H)
            dTidle_dz = rho_ratio**a_idle*(T1_idle*dVdz + 2.*T2_idle*self.V_hat*dVdz)
            P1 = 64.94*self.u_hat[0]/50.
            dTX_dz = P1*(dTmil_dz - dTidle_dz) + dTidle_dz
        else:
            a_max = self.props.a_max(H)
            T1_max = self.props.T1_max(H)
            T2_max = self.props.T2_max(H)
            dTmax_dz = rho_ratio**a_max*(T1_max*dVdz + 2.*T2_max*self.V_hat*dVdz)
            P1 = (217.38*self.u_hat[0] - 117.38 - 50.)/50.
            dTX_dz = P1*(dTmax_dz - dTmil_dz) + dTmil_dz
        return dTX_dz


    def _dCX_dz(self):
        dCXdz = np.zeros(self.N)
        dCDdz = self._dCD_dz()
        dCSdz = self._dCS_dz()
        dCLdz = self._dCL_dz()
        dadz = self.da_dz
        dbdz = self.db_dz
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CD = self.CD_hat
        CS = self.CS_hat
        CL = self.CL_hat
        dCXdz = (-dCDdz*c_a*c_b + CD*s_a*c_b*dadz + CD*c_a*s_b*dbdz -
                 dCSdz*c_a*s_b + CS*s_a*s_b*dadz - CS*c_a*c_b*dbdz +
                 dCLdz*s_a + CL*c_a*dadz)
        return dCXdz

    def _dCY_dz(self):
        dCYdz = np.zeros(self.N)
        dCDdz = self._dCD_dz()
        dCSdz = self._dCS_dz()
        dbdz = self.db_dz
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CD = self.CD_hat
        CS = self.CS_hat
        dCYdz = dCSdz*c_b - CS*s_b*dbdz - dCDdz*s_b - CD*c_b*dbdz
        return dCYdz

    def _dCZ_dz(self):
        dCZdz = np.zeros(self.N)
        dCDdz = self._dCD_dz()
        dCSdz = self._dCS_dz()
        dCLdz = self._dCL_dz()
        dadz = self.da_dz
        dbdz = self.db_dz
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CD = self.CD_hat
        CS = self.CS_hat
        CL = self.CL_hat
        dCZdz = (-dCDdz*s_a*c_b - CD*c_a*c_b*dadz + CD*s_a*s_b*dbdz -
                 dCSdz*s_a*s_b - CS*c_a*s_b*dadz - CS*s_a*c_b*dbdz -
                 dCLdz*c_a + CL*s_a*dadz)
        return dCZdz

    def _dalpha_dz(self):
        dadz = np.zeros(self.N)
        C1 = self.x_hat[0]**2 + self.x_hat[2]**2
        dadz[0] = -self.x_hat[2]/C1
        dadz[2] = self.x_hat[0]/C1
        return dadz

    def _dbeta_dz(self):
        dbdz = np.zeros(self.N)
        C1 = np.sqrt(self.x_hat[0]**2 + self.x_hat[2]**2)
        C2 = (self.V_hat**2)*C1
        dbdz[0] = -self.x_hat[1]*self.x_hat[0]/C2
        dbdz[1] = C1/(self.V_hat**2)
        dbdz[2] = -self.x_hat[1]*self.x_hat[2]/C2
        return dbdz

    def _dCL_dz(self):
        dCLdz = np.zeros(self.N)
        dCL1dz = self._dCL1_dz()
        dqbardz = self._dqbar_dz()
        dCLdz = dCL1dz + self.CL_q*dqbardz
        return dCLdz

    def _dCL1_dz(self):
        dCL1dz = self.CL_a*self.da_dz
        return dCL1dz

    def _dqbar_dz(self):
        dqdz = self.dq_dz
        dVidz = self.dVinv_dz
        dqbardz = dqdz*self.c_w/(2.*self.V_hat) + dVidz*self.c_w*self.x_hat[4]/2.
        return dqbardz

    def _dCS_dz(self):
        dCSdz = np.zeros(self.N)
        dCS1dz = self._dCS1_dz()
        dCL1dz = self._dCL1_dz()
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        dpbardz = self._dpbar_dz()
        drbardz = self._drbar_dz()
        dCSdz = (dCS1dz +
                 self.CS_Lp*dCL1dz*xb_4 +
                 (self.CS_Lp*self.CL1_hat + self.CS_p)*dpbardz +
                 self.CS_r*drbardz)
        return dCSdz

    def _dCS1_dz(self):
        dCS1dz = self.CS_b*self.db_dz
        return dCS1dz

    def _dpbar_dz(self):
        dpdz = self.dp_dz
        dVidz = self.dVinv_dz
        dpbardz = dpdz*self.b_w/(2.*self.V_hat) + dVidz*self.b_w*self.x_hat[3]/2.
        return dpbardz

    def _drbar_dz(self):
        drdz = self.dr_dz
        dVidz = self.dVinv_dz
        drbardz = drdz*self.b_w/(2.*self.V_hat) + dVidz*self.b_w*self.x_hat[5]/2.
        return drbardz

    def _dCD_dz(self):
        dCDdz = np.zeros(self.N)
        dCL1dz = self._dCL1_dz()
        dCS1dz = self._dCS1_dz()
        dCL12dz = 2.*self.CL1_hat*dCL1dz
        dCS12dz = 2.*self.CS1_hat*dCS1dz
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dpbardz = self._dpbar_dz()
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        CL1 = self.CL1_hat
        CS1 = self.CS1_hat
        dCDdz = (self.CD_L*dCL1dz + self.CD_L2*dCL12dz + self.CD_S2*dCS12dz +
                 self.CD_Sp*dCS1dz*xb_4 + self.CD_Sp*CS1*dpbardz +
                 (self.CD_L2q*dCL12dz + self.CD_Lq*dCL1dz)*xb_5 +
                 (self.CD_L2q*CL1**2 + self.CD_Lq*CL1 + self.CD_q)*dqbardz +
                 self.CD_Sr*dCS1dz*xb_6 + self.CD_Sr*CS1*drbardz +
                 self.CD_Sda*dCS1dz*self.u_hat[1] +
                 self.CD_Lde*dCL1dz*self.u_hat[2] +
                 self.CD_Sdr*dCS1dz*self.u_hat[3])
        return dCDdz

    def _dCl_dz(self):
        dCldz = np.zeros(self.N)
        dbdz = self.db_dz
        dpbardz = self._dpbar_dz()
        drbardz = self._drbar_dz()
        dCL1dz = self._dCL1_dz()
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        CL1 = self.CL1_hat
        dCldz = (self.Cl_b*dbdz + self.Cl_p*dpbardz + self.Cl_Lr*dCL1dz*xb_6 +
                 (self.Cl_Lr*CL1 + self.Cl_r)*drbardz)
        return dCldz

    def _dCm_dz(self):
        dCmdz = np.zeros(self.N)
        dadz = self.da_dz
        dqbardz = self._dqbar_dz()
        dCmdz = self.Cm_a*dadz + self.Cm_q*dqbardz
        return dCmdz

    def _dCn_dz(self):
        dCndz = np.zeros(self.N)
        dbdz = self.db_dz
        dpbardz = self._dpbar_dz()
        drbardz = self._drbar_dz()
        dCL1dz = self._dCL1_dz()
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        CL1 = self.CL1_hat
        dCndz = (self.Cn_b*dbdz + self.Cn_Lp*dCL1dz*xb_4 +
                 (self.Cn_Lp*CL1 + self.Cn_p)*dpbardz + self.Cn_r*drbardz +
                 self.Cn_Lda*dCL1dz*self.u_hat[1])
        return dCndz

    def _dz1_du(self):
        dFxdu = self._dFx_du()
        dz1du = self.g/self.W*dFxdu
        return dz1du

    def _dz2_du(self):
        dFydu = self._dFy_du()
        dz2du = self.g/self.W*dFydu
        return dz2du

    def _dz3_du(self):
        dFzdu = self._dFz_du()
        dz3du = self.g/self.W*dFzdu
        return dz3du

    def _dz4_du(self):
        dMxdu = self._dMx_du()
        dMydu = self._dMy_du()
        dMzdu = self._dMz_du()
        dM = np.array([dMxdu, dMydu, dMzdu])
        dz4du = np.matmul(self.I_inv, dM)[0, :]
        return dz4du

    def _dz5_du(self):
        dMxdu = self._dMx_du()
        dMydu = self._dMy_du()
        dMzdu = self._dMz_du()
        dM = np.array([dMxdu, dMydu, dMzdu])
        dz5du = np.matmul(self.I_inv, dM)[1, :]
        return dz5du

    def _dz6_du(self):
        dMxdu = self._dMx_du()
        dMydu = self._dMy_du()
        dMzdu = self._dMz_du()
        dM = np.array([dMxdu, dMydu, dMzdu])
        dz6du = np.matmul(self.I_inv, dM)[2, :]
        return dz6du

    def _dCL_du(self):
        dCLdu = self.CL_de*self.dde_du
        return dCLdu

    def _dCS_du(self):
        dCSdu = self.CS_da*self.dda_du + self.CS_dr*self.ddr_du
        return dCSdu

    def _dCD_du(self):
        dCDdu = (self.CD_Sda*self.CS1_hat*self.dda_du +
                 (self.CD_Lde*self.CL1_hat + self.CD_de)*self.dde_du +
                 self.CD_de2*self.dde2_du +
                 self.CD_Sdr*self.CS1_hat*self.ddr_du)
        return dCDdu

    def _dCl_du(self):
        dCldu = self.Cl_da*self.dda_du + self.Cl_dr*self.ddr_du
        return dCldu

    def _dCm_du(self):
        dCmdu = self.Cm_de*self.dde_du
        return dCmdu

    def _dCn_du(self):
        dCndu = ((self.Cn_Lda*self.CL1_hat + self.Cn_da)*self.dda_du +
                 self.Cn_dr*self.ddr_du)
        return dCndu

    def _dTx_du(self):
        a_mil = self.props.a_mil(self.props.H)
        T0_mil = self.props.T0_mil(self.props.H)
        T1_mil = self.props.T1_mil(self.props.H)
        T2_mil = self.props.T2_mil(self.props.H)
        V = self.props.V
        T_mil = (self.rho/self.rho_0)**a_mil*(T0_mil + T1_mil*V + T2_mil*V**2)
        if self.u_hat[0] < 0.77:
            a_idle = self.props.a_idle(self.props.H)
            T0_idle = self.props.T0_idle(self.props.H)
            T1_idle = self.props.T1_idle(self.props.H)
            T2_idle = self.props.T2_idle(self.props.H)
            T_idle = (self.rho/self.rho_0)**a_idle*(T0_idle + T1_idle*V +
                                                    T2_idle*V**2)
            dTxdu = 64.94/50.*(T_mil - T_idle)
        else:
            a_max = self.props.a_max(self.props.H)
            T0_max = self.props.T0_max(self.props.H)
            T1_max = self.props.T1_max(self.props.H)
            T2_max = self.props.T2_max(self.props.H)
            T_max = (self.rho/self.rho_0)**a_max*(T0_max + T1_max*V + T2_max*V**2)
            dTxdu = 217.38/50.*(T_max - T_mil)
        return np.array([dTxdu,0.0,0.0,0.0])

    def _dCX_du(self):
        dCDdu = self._dCD_du()
        dCSdu = self._dCS_du()
        dCLdu = self._dCL_du()
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        dCXdu = -(dCDdu*c_a*c_b + dCSdu*c_a*s_b - dCLdu*s_a)
        return dCXdu

    def _dCY_du(self):
        dCDdu = self._dCD_du()
        dCSdu = self._dCS_du()
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        dCYdu = dCSdu*c_b - dCDdu*s_b
        return dCYdu

    def _dCZ_du(self):
        dCDdu = self._dCD_du()
        dCSdu = self._dCS_du()
        dCLdu = self._dCL_du()
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        dCZdu = -(dCDdu*s_a*c_b + dCSdu*s_a*s_b + dCLdu*c_a)
        return dCZdu

    def _dFx_du(self):
        dCXdu = self._dCX_du()
        dTxdu = self._dTx_du()
        dFxdu = 0.5*self.rho*self.V_hat**2*self.S_w*dCXdu + dTxdu
        return dFxdu

    def _dFy_du(self):
        dCYdu = self._dCY_du()
        dFydu = 0.5*self.rho*self.V_hat**2*self.S_w*dCYdu
        return dFydu

    def _dFz_du(self):
        dCZdu = self._dCZ_du()
        dFzdu = 0.5*self.rho*self.V_hat**2*self.S_w*dCZdu
        return dFzdu

    def _dMx_du(self):
        dCldu = self._dCl_du()
        dFzdu = self._dFz_du()
        dFydu = self._dFy_du()
        dMxdu = (0.5*self.rho*self.V_hat**2*self.S_w*self.b_w*dCldu -
                 dFzdu*self.Dy +
                 dFydu*self.Dz)
        return dMxdu

    def _dMy_du(self):
        dCmdu = self._dCm_du()
        dFzdu = self._dFz_du()
        dFxdu = self._dFx_du()
        dMydu = (0.5*self.rho*self.V_hat**2*self.S_w*self.c_w*dCmdu -
                 dFxdu*self.Dz +
                 dFzdu*self.Dx)
        return dMydu

    def _dMz_du(self):
        dCndu = self._dCn_du()
        dFydu = self._dFy_du()
        dFxdu = self._dFx_du()
        dMzdu = (0.5*self.rho*self.V_hat**2*self.S_w*self.b_w*dCndu -
                 dFydu*self.Dx +
                 dFxdu*self.Dy)
        return dMzdu

    def _W_matrix(self):
        self.W_mat = np.zeros((3, self.N))
        self.W_mat[:, 3] = np.array([self.props.Ixz*self.x_hat[4] -
                                     self.props.Ixy*self.x_hat[5],
                                     (self.props.Izz - self.props.Ixx)*self.x_hat[5] -
                                     2.*self.props.Ixz*self.x_hat[3] -
                                     self.props.Iyz*self.x_hat[4],
                                     (self.props.Ixx - self.props.Iyy)*self.x_hat[4] +
                                     2.*self.props.Ixy*self.x_hat[3] +
                                     self.props.Iyz*self.x_hat[5]])
        self.W_mat[:, 4] = np.array([(self.props.Iyy - self.props.Izz)*self.x_hat[5] +
                                     2.*self.props.Iyz*self.x_hat[4] +
                                     self.props.Ixz*self.x_hat[3],
                                     self.props.Ixy*self.x_hat[5] -
                                     self.props.Iyz*self.x_hat[3],
                                     (self.props.Ixx - self.props.Iyy)*self.x_hat[3] -
                                     2.*self.props.Ixy*self.x_hat[4] - 
                                     self.props.Ixz*self.x_hat[5]])
        self.W_mat[:, 5] = np.array([(self.props.Iyy - self.props.Izz)*self.x_hat[4] -
                                     2.*self.props.Iyz*self.x_hat[5] -
                                     self.props.Ixy*self.x_hat[3],
                                     (self.props.Izz - self.props.Ixx)*self.x_hat[3] +
                                     2.*self.props.Ixz*self.x_hat[5] +
                                     self.props.Ixy*self.x_hat[4],
                                     self.props.Iyz*self.x_hat[3] - 
                                     self.props.Ixz*self.x_hat[4]])


    def create_A_matrix(self):
        A = np.zeros((self.N, self.N))
        A[0, :] = self._dz1_dz()
        A[1, :] = self._dz2_dz()
        A[2, :] = self._dz3_dz()
        A[3, :] = self._dz4_dz()
        A[4, :] = self._dz5_dz()
        A[5, :] = self._dz6_dz()
        A[6, :] = self._dz7_dz()
        A[7, :] = self._dz8_dz()
        return A

    def create_B_matrix(self):
        B = np.zeros((self.N, self.M))
        B[0, :] = self._dz1_du()
        B[1, :] = self._dz2_du()
        B[2, :] = self._dz3_du()
        B[3, :] = self._dz4_du()
        B[4, :] = self._dz5_du()
        B[5, :] = self._dz6_du()
        return B

    def create_C_matrix(self):
        C = np.eye(self.N)
        return C

    def create_E_matrix(self):
        E = np.zeros((self.N, 3))
        E[0, 0] = 1.
        E[1, 1] = 1.
        E[2, 2] = 1.
        return E

def create_feedback_control(trim_solution, V, H, Gamma, cg_shift,
                            p=-np.arange(1., 9.), lqr_flag=True,
                            Q=np.eye(8), R=np.eye(4),
                            N=np.zeros((8, 4))):
    # aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    x_hat = trim_solution.states
    alpha_hat = trim_solution.x[1]
    beta_hat = trim_solution.x[2]
    u_hat = trim_solution.inputs
    FM_hat = trim_solution.FM
    props = trim.AircraftProperties(V, H, Gamma)#, aero_dir)
    # system =
    linearization = LinearizationBaseline(props)#, aero_dir)
    linearization.set_linearization_point(x_hat, u_hat, alpha_hat, beta_hat, FM_hat,
                                          cg_shift)
    A = linearization.create_A_matrix()
    B = linearization.create_B_matrix()
    C = linearization.create_C_matrix()
    D = np.zeros((linearization.N, linearization.M))
    E = linearization.create_E_matrix()
    G = ctrb(A, B)
    print(np.linalg.matrix_rank(G))
    if lqr_flag:
        K, S, E = lqr(A, B, Q, R, N)
    else:
        K = place(A, B, p)
    eig_check, v_check = np.linalg.eig(A - np.matmul(B, K))
    assert all(np.real(eig_check) < 0.)
    lin_res = Lin_Results(linearization.N, linearization.M)
    lin_res.A = A
    lin_res.B = B
    lin_res.C = C
    lin_res.D = D
    lin_res.K = K
    lin_res.E = E
    lin_res.eigs = eig_check
    return lin_res


def rep2D(array, name = "ans", predecimals = 5, decimals = 4,print_format="f"):

    printname = "{} = ".format(name)
    lenname = len(printname)
    for i in range(array.shape[0]):
        if i == 0:
            print(printname,end="")
        else:
            print(" "*lenname,end="")
        
        if print_format == "e":
            num = 4
        else:
            num = 0
        
        for j in range(array.shape[1]):
            print("{:> {}.{}{}}".format(array[i,j],decimals+predecimals+num,\
                decimals,print_format),end="")
            if j != array.shape[1]-1:
                print(",",end="")
        print()
    print()


if __name__ == "__main__":
    np.set_printoptions(precision=16)
    H = 15000.
    a = stdatm_english(H)[-1]
    M = 0.6
    V = M*a
    V = 634.4133153512274
    b_w = 30.
    c_w = 11.32
    gamma = np.deg2rad(0.)
    phi = np.deg2rad(20.)
    Gamma = 0.1
    cg_shift = [1., 1., 1.]
    aero_dir = aero_directory#'/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    trim_solution = trim.trim(V, H, gamma, phi, Gamma, fixed_point=False,cg_shift=cg_shift,
                              aero_dir=aero_dir,compressible=True, stall=True)
    print()
    x_hat = trim_solution.states
    alpha_hat = trim_solution.x[1]
    beta_hat = trim_solution.x[2]
    # print(alpha_hat,beta_hat)
    alpha_hat = np.arctan2(x_hat[2],x_hat[0])
    V = ( x_hat[0]**2. + x_hat[1]**2. + x_hat[2]**2. )**0.5
    beta_hat = np.arcsin(x_hat[1]/V)
    # print(alpha_hat,beta_hat)
    u_hat = trim_solution.inputs
    # rep2D(x_hat[:,np.newaxis],decimals=20)
    # rng = [1,2,3,0]
    # rep2D(u_hat[rng,np.newaxis],decimals=20)
    FM_hat = trim_solution.FM
    props = trim.AircraftProperties(V, H, Gamma, aero_dir)
    linearization = LinearizationBaseline(props, aero_dir)
    linearization.set_linearization_point(x_hat, u_hat, alpha_hat, beta_hat,
                                          FM_hat, cg_shift)
    A = linearization.create_A_matrix()
    B = linearization.create_B_matrix()
    C = linearization.create_C_matrix()

    # print("g =", linearization.g)
    rep2D(A[:,:],"  A  ",decimals=16) # [0:1][:,0:3]
    rng = [1,2,3,0] # [1,2,3,0] # 
    rep2D(B[:,rng],"  B  ",decimals=16) # [0:8][:,0:4]