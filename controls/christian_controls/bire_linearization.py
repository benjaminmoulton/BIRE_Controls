#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:49:57 2022

@author: christian
"""

import numpy as np
from bire_aero import BIREAero
import aero_trim as trim
from stdatmos import stdatm_english
from control import ctrb, lqr
import matplotlib.pyplot as plt
import json
from os.path import exists
import pickle

class Lin_Results:
    def __init__(self, N, M):
        self.A = np.zeros((N, N))
        self.B = np.zeros((N, M))
        self.C = np.zeros((N, N))
        self.K = np.zeros((M, N))
        self.eigs = np.zeros(N)
        self.aircraft = "BIRE"

class LinearizationBIRE:
    def __init__(self, props, aero_dir='./', N=8, M=4):
        self.N = N
        self.M = M
        self.x_hat = np.zeros(N)
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
        I_model = json.load(open('./bire_inertia_model.json'))
        Ixx = I_model["Ixx"]
        Iyy = I_model["Iyy"]
        Izz = I_model["Izz"]
        Ixz = I_model["Ixz"]
        Ixy = I_model["Ixy"]
        Iyz = I_model["Iyz"]
        self.I_xx = lambda dB : Ixx["A"]*np.sin(Ixx["w"]*dB + Ixx["phi"]) + Ixx["z"]
        self.I_yy = lambda dB : Iyy["A"]*np.sin(Iyy["w"]*dB + Iyy["phi"]) + Iyy["z"]
        self.I_zz = lambda dB : Izz["A"]*np.sin(Izz["w"]*dB + Izz["phi"]) + Izz["z"]
        self.I_yz = lambda dB : Iyz["A"]*np.abs(np.sin(Iyz["w"]*dB + Iyz["phi"])) + Iyz["z"]
        self.I_xy = lambda dB : Ixy["A"]*np.sin(Ixy["w"]*dB + Ixy["phi"]) + Ixy["z"]
        self.I_xz = lambda dB : Ixz["A"]*np.sin(Ixz["w"]*dB + Ixz["phi"]) + Ixz["z"]
        self.dI_xx = lambda dB : np.array([0., 0., 0.,
                                           Ixx["A"]*Ixx["w"]*np.cos(Ixx["w"]*dB +
                                                                    Ixx["phi"])])
        self.dI_yy = lambda dB : np.array([0., 0., 0.,
                                           Iyy["A"]*Iyy["w"]*np.cos(Iyy["w"]*dB +
                                                                    Iyy["phi"])])
        self.dI_zz = lambda dB : np.array([0., 0., 0.,
                                           Izz["A"]*Izz["w"]*np.cos(Izz["w"]*dB +
                                                                    Izz["phi"])])
        self.dI_xy = lambda dB : np.array([0., 0., 0.,
                                           Ixy["A"]*Ixy["w"]*np.cos(Ixy["w"]*dB +
                                                                    Ixy["phi"])])
        self.dI_xz = lambda dB : np.array([0., 0., 0.,
                                           Ixz["A"]*Ixz["w"]*np.cos(Ixz["w"]*dB +
                                                                    Ixz["phi"])])
        self.tc_tau = 0.05
        self.tc_da = 0.05
        self.tc_de = 0.05
        self.tc_dr = 0.05
        self.rate_da = 80.*np.pi/180.
        self.rate_de = 60*np.pi/180.
        self.rate_dB = 120*np.pi/180.

    def dI_yz(self, dB):
        I_model = json.load(open('./bire_inertia_model.json'))
        Iyz = I_model["Iyz"]
        if dB == 0.:
            dI_yz = np.array([0., 0., 0., 0.])
        elif abs(dB) == np.pi:
            dI_yz = np.array([0., 0., 0., 0.])
        else:
            dI_yz = np.array([0., 0., 0.,
                              Iyz["A"]*Iyz["w"]*np.sin(2.*Iyz["w"]*dB +
                                                       Iyz["phi"])/
                                                       np.abs(np.sin(Iyz["w"]*dB))])
        return dI_yz


    def set_linearization_point(self, x_hat, u_hat, alpha_hat, beta_hat, FM_hat,
                                cg_shift):
        self.x_hat = x_hat
        self.u_hat = u_hat
        self.dB_hat = self.u_hat[3]
        self.alpha_hat = alpha_hat
        self.beta_hat = beta_hat
        self.V_hat = np.sqrt(np.sum(np.square(self.x_hat[:3])))
        [self.CD_hat, self.CS_hat, self.CL_hat,
         self.Cl_hat, self.Cm_hat, self.Cn_hat] = FM_hat
        self.I_inv = self._I_inv(self.dB_hat)
        self._W_matrix()
        self.dVinv_dz = self._dVinv_dz()
        self.dV_dz = self._dV_dz()
        self.da_dz = self._dalpha_dz()
        self.db_dz = self._dbeta_dz()
        self.Dx = cg_shift[0]
        self.Dy = cg_shift[1]
        self.Dz = cg_shift[2]
        dim_const = 0.5*self.rho*self.V_hat**2*self.S_w
        CZ = -(self.CD_hat*np.sin(self.alpha_hat)*np.cos(self.beta_hat) +
               self.CS_hat*np.sin(self.alpha_hat)*np.sin(self.beta_hat) +
               self.CL_hat*np.cos(self.alpha_hat))
        CY = self.CS_hat*np.cos(self.beta_hat) - self.CD_hat*np.sin(self.beta_hat)
        CX = -(self.CD_hat*np.cos(self.alpha_hat)*np.cos(self.beta_hat) +
               self.CS_hat*np.cos(self.alpha_hat)*np.sin(self.beta_hat) -
               self.CL_hat*np.sin(self.alpha_hat))
        Tx = trim.thrust(self.u_hat[0], self.V_hat, self.props)
        FX = dim_const*CX + self.u_hat[0]*Tx
        FY = dim_const*CY
        FZ = dim_const*CZ
        self.Mx_hat = dim_const*self.b_w*self.Cl_hat - FZ*self.Dy + FY*self.Dz
        self.My_hat = dim_const*self.c_w*self.Cm_hat - FZ*self.Dx + FX*self.Dz
        self.Mz_hat = dim_const*self.b_w*self.Cn_hat - FY*self.Dx + FX*self.Dy
        self.dp_dz = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
        self.dq_dz = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
        self.dr_dz = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
        self.dde_du = np.array([0., 0., 1., 0.])
        self.dda_du = np.array([0., 1., 0., 0.])
        self.dtau_du = np.array([1., 0., 0., 0.])
        self.ddB_du = np.array([0., 0., 0., 1.])
        self.dde2_du = 2.*self.u_hat[2]*self.dde_du
        aero = BIREAero(self.aero_dir)
        aero.evaluate_coeffs(self.dB_hat)
        aero.evaluate_derivatives(self.dB_hat)

        self.CL_0 = aero.CL0
        self.CL_a = aero.CLa
        self.CL_b = aero.CLb
        self.CL_p = aero.CLp
        self.CL_q = aero.CLq
        self.CL_r = aero.CLr
        self.CL_da = aero.CLda
        self.CL_de = aero.CLde
        self.CL1_hat = self.CL_0 + self.CL_a*self.alpha_hat

        self.CS_0 = aero.CS0
        self.CS_a = aero.CSa
        self.CS_b = aero.CSb
        self.CS1_hat = self.CS_0 + self.CS_b*self.beta_hat
        self.CS_Lp = aero.CSLp
        self.CS_p = aero.CSp
        self.CS_q = aero.CSq
        self.CS_r = aero.CSr
        self.CS_da = aero.CSda
        self.CS_de = aero.CSde

        self.CD_0 = aero.CD0
        self.CD_L = aero.CDL
        self.CD_L2 = aero.CDL2
        self.CD_S = aero.CDS
        self.CD_S2 = aero.CDS2
        self.CD_Sp = aero.CDSp
        self.CD_p = aero.CDp
        self.CD_L2q = aero.CDL2q
        self.CD_Lq = aero.CDLq
        self.CD_q = aero.CDq
        self.CD_Sr = aero.CDSr
        self.CD_r = aero.CDr
        self.CD_Sda = aero.CDSda
        self.CD_da = aero.CDda
        self.CD_Lde = aero.CDLde
        self.CD_de = aero.CDde
        self.CD_de2 = aero.CDde2

        self.Cl_0 = aero.Cl0
        self.Cl_a = aero.Cla
        self.Cl_b = aero.Clb
        self.Cl_p = aero.Clp
        self.Cl_q = aero.Clq
        self.Cl_Lr = aero.ClLr
        self.Cl_r = aero.Clr
        self.Cl_da = aero.Clda
        self.Cl_de = aero.Clde

        self.Cm_0 = aero.Cm0
        self.Cm_a = aero.Cma
        self.Cm_b = aero.Cmb
        self.Cm_p = aero.Cmp
        self.Cm_q = aero.Cmq
        self.Cm_r = aero.Cmr
        self.Cm_da = aero.Cmda
        self.Cm_de = aero.Cmde

        self.Cn_0 = aero.Cn0
        self.Cn_a = aero.Cna
        self.Cn_b = aero.Cnb
        self.Cn_Lp = aero.CnLp
        self.Cn_p = aero.Cnp
        self.Cn_q = aero.Cnq
        self.Cn_r = aero.Cnr
        self.Cn_Lda = aero.CnLda
        self.Cn_da = aero.Cnda
        self.Cn_de = aero.Cnde

        self.dCL_0 = aero.dCL0*self.ddB_du
        self.dCL_a = aero.dCLa*self.ddB_du
        self.dCL_b = aero.dCLb*self.ddB_du
        self.dCL_p = aero.dCLp*self.ddB_du
        self.dCL_q = aero.dCLq*self.ddB_du
        self.dCL_r = aero.dCLr*self.ddB_du
        self.dCL_da = aero.dCLda*self.ddB_du
        self.dCL_de = aero.dCLde*self.ddB_du
        self.dCL1_hat = self.dCL_0 + self.dCL_a*self.alpha_hat

        self.dCS_0 = aero.dCS0*self.ddB_du
        self.dCS_a = aero.dCSa*self.ddB_du
        self.dCS_b = aero.dCSb*self.ddB_du
        self.dCS1_hat = self.dCS_0 + self.dCS_b*self.beta_hat
        self.dCS_Lp = aero.dCSLp*self.ddB_du
        self.dCS_p = aero.dCSp*self.ddB_du
        self.dCS_q = aero.dCSq*self.ddB_du
        self.dCS_r = aero.dCSr*self.ddB_du
        self.dCS_da = aero.dCSda*self.ddB_du
        self.dCS_de = aero.dCSde*self.ddB_du

        self.dCD_0 = aero.dCD0*self.ddB_du
        self.dCD_L = aero.dCDL*self.ddB_du
        self.dCD_L2 = aero.dCDL2*self.ddB_du
        self.dCD_S = aero.dCDS*self.ddB_du
        self.dCD_S2 = aero.dCDS2*self.ddB_du
        self.dCD_Sp = aero.dCDSp*self.ddB_du
        self.dCD_p = aero.dCDp*self.ddB_du
        self.dCD_L2q = aero.dCDL2q*self.ddB_du
        self.dCD_Lq = aero.dCDLq*self.ddB_du
        self.dCD_q = aero.dCDq*self.ddB_du
        self.dCD_Sr = aero.dCDSr*self.ddB_du
        self.dCD_r = aero.dCDr*self.ddB_du
        self.dCD_Sda = aero.dCDSda*self.ddB_du
        self.dCD_da = aero.dCDda*self.ddB_du
        self.dCD_Lde = aero.dCDLde*self.ddB_du
        self.dCD_de = aero.dCDde*self.ddB_du
        self.dCD_de2 = aero.dCDde2*self.ddB_du

        self.dCl_0 = aero.dCl0*self.ddB_du
        self.dCl_a = aero.dCla*self.ddB_du
        self.dCl_b = aero.dClb*self.ddB_du
        self.dCl_p = aero.dClp*self.ddB_du
        self.dCl_q = aero.dClq*self.ddB_du
        self.dCl_Lr = aero.dClLr*self.ddB_du
        self.dCl_r = aero.dClr*self.ddB_du
        self.dCl_da = aero.dClda*self.ddB_du
        self.dCl_de = aero.dClde*self.ddB_du

        self.dCm_0 = aero.dCm0*self.ddB_du
        self.dCm_a = aero.dCma*self.ddB_du
        self.dCm_b = aero.dCmb*self.ddB_du
        self.dCm_p = aero.dCmp*self.ddB_du
        self.dCm_q = aero.dCmq*self.ddB_du
        self.dCm_r = aero.dCmr*self.ddB_du
        self.dCm_da = aero.dCmda*self.ddB_du
        self.dCm_de = aero.dCmde*self.ddB_du

        self.dCn_0 = aero.dCn0*self.ddB_du
        self.dCn_a = aero.dCna*self.ddB_du
        self.dCn_b = aero.dCnb*self.ddB_du
        self.dCn_Lp = aero.dCnLp*self.ddB_du
        self.dCn_p = aero.dCnp*self.ddB_du
        self.dCn_q = aero.dCnq*self.ddB_du
        self.dCn_r = aero.dCnr*self.ddB_du
        self.dCn_Lda = aero.dCnLda*self.ddB_du
        self.dCn_da = aero.dCnda*self.ddB_du
        self.dCn_de = aero.dCnde*self.ddB_du

    def _det_I(self, dB):
        Ixx = self.I_xx(dB)
        Iyy = self.I_yy(dB)
        Izz = self.I_zz(dB)
        Iyz = self.I_yz(dB)
        Ixz = self.I_xz(dB)
        Ixy = self.I_xy(dB)
        Izy = Iyz
        Izx = Ixz
        Iyx = Ixy
        C1 = Ixx*(Iyy*Izz - Iyz*Izy)
        C2 = Ixy*(Iyx*Izz + Iyz*Izx)
        C3 = Ixz*(Iyx*Izy + Iyy*Izx)
        return C1 - C2 - C3

    def _ddetI_du(self):
        Ixx = self.I_xx(self.dB_hat)
        Iyy = self.I_yy(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        dIxx = self.dI_xx(self.dB_hat)
        dIyy = self.dI_yy(self.dB_hat)
        dIzz = self.dI_zz(self.dB_hat)
        dIyz = self.dI_yz(self.dB_hat)
        dIxz = self.dI_xz(self.dB_hat)
        dIxy = self.dI_xy(self.dB_hat)
        ddetIdu = (dIxx*(Iyy*Izz - Iyz**2) +
                   Ixx*(dIyy*Izz + Iyy*dIzz - 2.*Iyz*dIyz) -
                   dIxy*(Ixy*Izz + Iyz*Ixz) -
                   Ixy*(dIxy*Izz + Ixy*dIzz + dIyz*Ixz + Iyz*dIxz) -
                   dIxz*(Ixy*Iyz + Iyy*Ixz) -
                   Ixz*(dIxy*Iyz + Ixy*dIyz + dIyy*Ixz + Iyy*dIxz))
        return ddetIdu

    def _Istar(self):
        Ixx = self.I_xx(self.dB_hat)
        Iyy = self.I_yy(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        Istar = np.zeros((3, 3))
        Istar[0, 0] = Iyy*Izz - Iyz**2
        Istar[0, 1] = Ixy*Izz + Ixz*Iyz
        Istar[0, 2] = Ixy*Iyz + Ixz*Iyy
        Istar[1, 0] = Istar[0, 1]
        Istar[1, 1] = Ixx*Izz - Ixz**2
        Istar[1, 2] = Ixx*Iyz + Ixy*Ixz
        Istar[2, 0] = Istar[0, 2]
        Istar[2, 1] = Istar[1, 2]
        Istar[2, 2] = Ixx*Iyy - Ixy**2
        return Istar

    def _dIstar_du(self):
        Ixx = self.I_xx(self.dB_hat)
        Iyy = self.I_yy(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        dIxx = self.dI_xx(self.dB_hat)
        dIyy = self.dI_yy(self.dB_hat)
        dIzz = self.dI_zz(self.dB_hat)
        dIyz = self.dI_yz(self.dB_hat)
        dIxz = self.dI_xz(self.dB_hat)
        dIxy = self.dI_xy(self.dB_hat)
        dIstardu = np.zeros((3, 3, self.M))
        dIstardu[0, 0, :] = dIyy*Izz + Iyy*dIzz - 2.*Iyz*dIyz
        dIstardu[0, 1, :] = dIxy*Izz + Ixy*dIzz + dIxz*Iyz + Ixz*dIyz
        dIstardu[0, 2, :] = dIxy*Iyz + Ixy*dIyz + dIxz*Iyy + Ixz*dIyy
        dIstardu[1, 0, :] = dIxy*Izz + Ixy*dIzz + dIyz*Ixz + Iyz*dIxz
        dIstardu[1, 1, :] = dIxx*Izz + Ixx*dIzz - 2.*Ixz*dIxz
        dIstardu[1, 2, :] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
        dIstardu[2, 0, :] = dIxy*Iyz + Ixy*dIyz + dIyy*Ixz + Iyy*dIxz
        dIstardu[2, 1, :] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
        dIstardu[2, 2, :] = dIxx*Iyy + Ixx*dIyy - 2.*Ixy*dIxy
        return dIstardu

    def _I_inv(self, dB):
        Ixx = self.I_xx(dB)
        Iyy = self.I_yy(dB)
        Izz = self.I_zz(dB)
        Iyz = self.I_yz(dB)
        Ixz = self.I_xz(dB)
        Ixy = self.I_xy(dB)
        Izy = Iyz
        Izx = Ixz
        Iyx = Ixy
        det_I = self._det_I(dB)
        I_inv = np.zeros((3, 3))
        I_inv[0, 0] = Iyy*Izz - Iyz*Izy
        I_inv[0, 1] = Ixy*Izz + Ixz*Izy
        I_inv[0, 2] = Ixy*Iyz + Ixz*Iyy
        I_inv[1, 0] = Iyx*Izz + Iyz*Izx
        I_inv[1, 1] = Ixx*Izz - Ixz*Izx
        I_inv[1, 2] = Ixx*Iyz + Ixz*Iyz
        I_inv[2, 0] = Iyz*Izy + Iyy*Izx
        I_inv[2, 1] = Ixx*Izy + Ixy*Izx
        I_inv[2, 2] = Ixx*Iyy - Ixy*Iyx
        I_inv = I_inv/det_I
        return I_inv

    def _dIinv_du(self):
        dIinvdu = np.zeros((3, 3, self.M))
        dIstardu = self._dIstar_du()
        ddetI = self._ddetI_du()
        detI = self._det_I(self.dB_hat)
        Istar = self._Istar()
        dIinvdu = (detI*dIstardu - Istar[:, :, None]*ddetI[None, :])/(detI**2)
        return dIinvdu

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
        R = dM + self.W_mat
        dz4_dz = np.matmul(self.I_inv, R)[0, :]
        return dz4_dz

    def _dz5_dz(self):
        dz5_dz = np.zeros(self.N)
        dMxdz = self._dMx_dz()
        dMydz = self._dMy_dz()
        dMzdz = self._dMz_dz()
        dM = np.array([dMxdz, dMydz, dMzdz])
        R = np.zeros((3, self.N + self.M))
        R = dM + self.W_mat
        dz5_dz = np.matmul(self.I_inv, R)[1, :]
        return dz5_dz

    def _dz6_dz(self):
        dz6_dz = np.zeros(self.N)
        dMxdz = self._dMx_dz()
        dMydz = self._dMy_dz()
        dMzdz = self._dMz_dz()
        dM = np.array([dMxdz, dMydz, dMzdz])
        R = np.zeros((3, self.N + self.M))
        R = dM + self.W_mat
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
        dTXdz = self._dTX_dz()
        c_a = np.cos(self.alpha_hat)
        s_a = np.sin(self.alpha_hat)
        c_b = np.cos(self.beta_hat)
        s_b = np.sin(self.beta_hat)
        CX = -(self.CD_hat*c_a*c_b + self.CS_hat*c_a*s_b -
               self.CL_hat*s_a)
        dFxdz = (0.5*self.rho*self.V_hat**2*self.S_w*dCXdz +
                 self.rho*self.V_hat*self.S_w*CX*dVdz + dTXdz)
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
                 dFzdz*dx +
                 dFxdz*dz)
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
        dpbardz = self._dpbar_dz()
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        dCLdz = (dCL1dz + self.CL_b*self.db_dz +
                 self.CL_p*dpbardz + self.CL_q*dqbardz + self.CL_r*drbardz)
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
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        dCSdz = (self.CS_a*self.da_dz + dCS1dz +
                 self.CS_Lp*dCL1dz*xb_4 +
                 (self.CS_Lp*self.CL1_hat + self.CS_p)*dpbardz +
                 self.CS_q*dqbardz +
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
        dCDdz = (self.CD_L*dCL1dz + self.CD_L2*dCL12dz +
                 self.CD_S*dCS1dz + self.CD_S2*dCS12dz +
                 self.CD_Sp*dCS1dz*xb_4 +
                 (self.CD_Sp*CS1 + self.CD_p)*dpbardz +
                 (self.CD_L2q*dCL12dz + self.CD_Lq*dCL1dz)*xb_5 +
                 (self.CD_L2q*CL1**2 + self.CD_Lq*CL1 + self.CD_q)*dqbardz +
                 self.CD_Sr*dCS1dz*xb_6 +
                 (self.CD_Sr*CS1 + self.CD_r)*drbardz +
                 self.CD_Sda*dCS1dz*self.u_hat[1] +
                 self.CD_Lde*dCL1dz*self.u_hat[2])
        return dCDdz

    def _dCl_dz(self):
        dCldz = np.zeros(self.N)
        dadz = self.da_dz
        dbdz = self.db_dz
        dpbardz = self._dpbar_dz()
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        dCL1dz = self._dCL1_dz()
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        CL1 = self.CL1_hat
        dCldz = (self.Cl_a*dadz + self.Cl_b*dbdz + self.Cl_p*dpbardz +
                 self.Cl_q*dqbardz + self.Cl_Lr*dCL1dz*xb_6 +
                 (self.Cl_Lr*CL1 + self.Cl_r)*drbardz)
        return dCldz

    def _dCm_dz(self):
        dCmdz = np.zeros
        dadz = self.da_dz
        dbdz = self.db_dz
        dpbardz = self._dpbar_dz()
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        dCmdz = (self.Cm_a*dadz + self.Cm_b*dbdz + self.Cm_p*dpbardz +
                 self.Cm_q*dqbardz + self.Cm_r*drbardz)
        return dCmdz

    def _dCn_dz(self):
        dCndz = np.zeros(self.N)
        dadz = self.da_dz
        dbdz = self.db_dz
        dpbardz = self._dpbar_dz()
        dqbardz = self._dqbar_dz()
        drbardz = self._drbar_dz()
        dCL1dz = self._dCL1_dz()
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        CL1 = self.CL1_hat
        dCndz = (self.Cn_a*dadz + self.Cn_b*dbdz + self.Cn_Lp*dCL1dz*xb_4 +
                 (self.Cn_Lp*CL1 + self.Cn_p)*dpbardz + self.Cn_q*dqbardz +
                 self.Cn_r*drbardz + self.Cn_Lda*dCL1dz*self.u_hat[1])
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

    def _M1(self):
        Iyy = self.I_yy(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        M1 = (self.Mx_hat +
              (Iyy - Izz)*self.x_hat[4]*self.x_hat[5] +
              Iyz*(self.x_hat[4]**2 - self.x_hat[5]**2) +
              Ixz*self.x_hat[3]*self.x_hat[4] -
              Ixy*self.x_hat[3]*self.x_hat[5])
        return M1

    def _dM1_du(self):
        dIyy = self.dI_yy(self.dB_hat)
        dIzz = self.dI_zz(self.dB_hat)
        dIyz = self.dI_yz(self.dB_hat)
        dIxz = self.dI_xz(self.dB_hat)
        dIxy = self.dI_xy(self.dB_hat)
        dMxdu = self._dMx_du()
        dM1du = (dMxdu +
              (dIyy - dIzz)*self.x_hat[4]*self.x_hat[5] +
              dIyz*(self.x_hat[4]**2 - self.x_hat[5]**2) +
              dIxz*self.x_hat[3]*self.x_hat[4] -
              dIxy*self.x_hat[3]*self.x_hat[5])
        return dM1du

    def _M2(self):
        Ixx = self.I_xx(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        M2 = (self.My_hat +
              (Izz - Ixx)*self.x_hat[3]*self.x_hat[5] +
              Ixz*(self.x_hat[5]**2 - self.x_hat[3]**2) +
              Ixy*self.x_hat[4]*self.x_hat[5] -
              Iyz*self.x_hat[3]*self.x_hat[4])
        return M2

    def _dM2_du(self):
        dIxx = self.dI_xx(self.dB_hat)
        dIzz = self.dI_zz(self.dB_hat)
        dIyz = self.dI_yz(self.dB_hat)
        dIxz = self.dI_xz(self.dB_hat)
        dIxy = self.dI_xy(self.dB_hat)
        dMydu = self._dMy_du()
        dM2du = (dMydu +
              (dIzz - dIxx)*self.x_hat[3]*self.x_hat[5] +
              dIxz*(self.x_hat[5]**2 - self.x_hat[3]**2) +
              dIxy*self.x_hat[4]*self.x_hat[5] -
              dIyz*self.x_hat[3]*self.x_hat[4])
        return dM2du

    def _M3(self):
        Ixx = self.I_xx(self.dB_hat)
        Iyy = self.I_yy(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        M3 = (self.Mz_hat +
              (Ixx - Iyy)*self.x_hat[3]*self.x_hat[4] +
              Ixy*(self.x_hat[3]**2 - self.x_hat[4]**2) +
              Iyz*self.x_hat[3]*self.x_hat[5] -
              Ixz*self.x_hat[4]*self.x_hat[5])
        return M3

    def _dM3_du(self):
        dIxx = self.dI_xx(self.dB_hat)
        dIyy = self.dI_yy(self.dB_hat)
        dIyz = self.dI_yz(self.dB_hat)
        dIxz = self.dI_xz(self.dB_hat)
        dIxy = self.dI_xy(self.dB_hat)
        dMzdu = self._dMz_du()
        dM3du = (dMzdu +
                 (dIxx - dIyy)*self.x_hat[3]*self.x_hat[4] +
                 dIxy*(self.x_hat[3]**2 - self.x_hat[4]**2) +
                 dIyz*self.x_hat[3]*self.x_hat[5] -
                 dIxz*self.x_hat[4]*self.x_hat[5])
        return dM3du

    def _dz4_du(self):
        dIinv = self._dIinv_du()
        M1 = self._M1()
        M2 = self._M2()
        M3 = self._M3()
        dM1du = self._dM1_du()
        dM2du = self._dM2_du()
        dM3du = self._dM3_du()
        M = np.array([M1, M2, M3])
        dM = np.array([dM1du, dM2du, dM3du])
        dz4du = np.zeros(self.M)
        for i in range(self.M):
            dz4du[i] = np.matmul(dIinv[:, :, i], M)[0]
        dz4du = dz4du + np.matmul(self.I_inv, dM)[0, :]
        return dz4du

    def _dz5_du(self):
        dIinv = self._dIinv_du()
        M1 = self._M1()
        M2 = self._M2()
        M3 = self._M3()
        dM1du = self._dM1_du()
        dM2du = self._dM2_du()
        dM3du = self._dM3_du()
        M = np.array([M1, M2, M3])
        dM = np.array([dM1du, dM2du, dM3du])
        dz5du = np.zeros(self.M)
        for i in range(self.M):
            dz5du[i] = np.matmul(dIinv[:, :, i], M)[1]
        dz5du = dz5du + np.matmul(self.I_inv, dM)[1, :]
        return dz5du

    def _dz6_du(self):
        dIinv = self._dIinv_du()
        M1 = self._M1()
        M2 = self._M2()
        M3 = self._M3()
        dM1du = self._dM1_du()
        dM2du = self._dM2_du()
        dM3du = self._dM3_du()
        M = np.array([M1, M2, M3])
        dM = np.array([dM1du, dM2du, dM3du])
        dz6du = np.zeros(self.M)
        for i in range(self.M):
            dz6du[i] = np.matmul(dIinv[:, :, i], M)[2]
        dz6du = dz6du + np.matmul(self.I_inv, dM)[2, :]
        return dz6du

    def _dCL_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCLdu = (self.dCL1_hat +
                 self.dCL_b*self.beta_hat +
                 self.dCL_p*xb_4 +
                 self.dCL_q*xb_5 +
                 self.dCL_r*xb_6 +
                 self.dCL_da*self.u_hat[1] +
                 self.CL_da*self.dda_du +
                 self.dCL_de*self.u_hat[2] +
                 self.CL_de*self.dde_du)
        return dCLdu

    def _dCS_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCSdu = (self.dCS1_hat +
                 self.dCS_a*self.alpha_hat +
                 (self.dCS_Lp*self.CL1_hat +
                  self.CS_Lp*self.dCL1_hat +
                  self.dCS_p)*xb_4 +
                 self.dCS_q*xb_5 +
                 self.dCS_r*xb_6 +
                 self.dCS_da*self.u_hat[1] +
                 self.CS_da*self.dda_du +
                 self.dCS_de*self.u_hat[2] +
                 self.CS_de*self.dde_du)
        return dCSdu

    def _dCD_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCL12du = 2.*self.CL1_hat*self.dCL1_hat
        dCS12du = 2.*self.CS1_hat*self.dCS1_hat
        dCDdu = (self.dCD_0 +
                 self.dCD_L*self.CL1_hat +
                 self.CD_L*self.dCL1_hat +
                 self.dCD_L2*self.CL1_hat**2 +
                 self.CD_L2*dCL12du +
                 self.dCD_S*self.CS1_hat +
                 self.CD_S*self.dCS1_hat +
                 self.dCD_S2*self.CS1_hat**2 +
                 self.CD_S2*dCS12du +
                 (self.dCD_Sp*self.CS1_hat +
                  self.CD_Sp*self.dCS1_hat +
                  self.dCD_p)*xb_4 +
                 (self.dCD_L2q*self.CL1_hat**2 +
                  self.CD_L2q*dCL12du +
                  self.dCD_Lq*self.CL1_hat +
                  self.CD_Lq*self.dCL1_hat +
                  self.dCD_q)*xb_5 +
                 (self.dCD_Sr*self.CS1_hat +
                  self.CD_Sr*self.dCS1_hat +
                  self.dCD_r)*xb_6 +
                 (self.dCD_Sda*self.CS1_hat +
                  self.CD_Sda*self.dCS1_hat +
                  self.dCD_da)*self.u_hat[1] +
                 (self.CD_Sda*self.CS1_hat + self.CD_da)*self.dda_du +
                 (self.dCD_Lde*self.CL1_hat +
                  self.CD_Lde*self.dCL1_hat +
                  self.dCD_de)*self.u_hat[2] +
                 (self.CD_Lde*self.CL1_hat + self.CD_de)*self.dde_du +
                 self.dCD_de2*self.u_hat[2]**2 +
                 self.CD_de2*self.dde2_du)
        return dCDdu

    def _dCl_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCldu = (self.dCl_0 +
                 self.dCl_a*self.alpha_hat +
                 self.dCl_b*self.beta_hat +
                 self.dCl_p*xb_4 +
                 self.dCl_q*xb_5 +
                 (self.dCl_Lr*self.CL1_hat +
                  self.Cl_Lr*self.dCL1_hat +
                  self.dCl_r)*xb_6 +
                 self.dCl_da*self.u_hat[1] +
                 self.Cl_da*self.dda_du +
                 self.dCl_de*self.u_hat[2] +
                 self.Cl_de*self.dde_du)
        return dCldu

    def _dCm_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCmdu = (self.dCm_0 +
                 self.dCm_a*self.alpha_hat +
                 self.dCm_b*self.beta_hat +
                 self.dCm_p*xb_4 +
                 self.dCm_q*xb_5 +
                 self.dCm_r*xb_6 +
                 self.dCm_da*self.u_hat[1] +
                 self.Cm_da*self.dda_du +
                 self.dCm_de*self.u_hat[2] +
                 self.Cm_de*self.dde_du)
        return dCmdu

    def _dCn_du(self):
        xb_4 = self.b_w*self.x_hat[3]/(2.*self.V_hat)
        xb_5 = self.c_w*self.x_hat[4]/(2.*self.V_hat)
        xb_6 = self.b_w*self.x_hat[5]/(2.*self.V_hat)
        dCndu = (self.dCn_0 +
                 self.dCn_a*self.alpha_hat +
                 self.dCn_b*self.beta_hat +
                 (self.dCn_Lp*self.CL1_hat +
                  self.Cn_Lp*self.dCL1_hat +
                  self.dCn_p)*xb_4 +
                 self.dCn_q*xb_5 +
                 self.dCn_r*xb_6 +
                 (self.dCn_Lda*self.CL1_hat +
                  self.Cn_Lda*self.dCL1_hat +
                  self.dCn_da)*self.u_hat[1] +
                 (self.Cn_Lda*self.CL1_hat +
                  self.Cn_da)*self.dda_du +
                 self.dCn_de*self.u_hat[2] +
                 self.Cn_de*self.dde_du)
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
        return dTxdu

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
                 dFzdu*self.Dx +
                 dFxdu*self.Dz)
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
        Ixx = self.I_xx(self.dB_hat)
        Ixy = self.I_xy(self.dB_hat)
        Ixz = self.I_xz(self.dB_hat)
        Iyy = self.I_yy(self.dB_hat)
        Izz = self.I_zz(self.dB_hat)
        Iyz = self.I_yz(self.dB_hat)
        self.W_mat[:, 3] = np.array([Ixz*self.x_hat[4] - Ixy*self.x_hat[5],
                                     (Izz - Ixx)*self.x_hat[5] -
                                     2.*Ixz*self.x_hat[3] - Iyz*self.x_hat[4],
                                     (Ixx - Iyy)*self.x_hat[4] +
                                     2.*Ixy*self.x_hat[3] + Iyz*self.x_hat[5]])
        self.W_mat[:, 4] = np.array([(Iyy - Izz)*self.x_hat[5] -
                                     2.*Iyz*self.x_hat[4] + Ixz*self.x_hat[3],
                                     Ixy*self.x_hat[5] - Iyz*self.x_hat[3],
                                     (Ixx - Iyy)*self.x_hat[3] -
                                     2.*Ixy*self.x_hat[4] - Ixz*self.x_hat[5]])
        self.W_mat[:, 5] = np.array([(Iyy - Izz)*self.x_hat[4] +
                                     2.*Iyz*self.x_hat[5] - Ixy*self.x_hat[3],
                                     (Izz - Ixx)*self.x_hat[3] +
                                     2.*Ixz*self.x_hat[5] + Ixy*self.x_hat[4],
                                     Iyz*self.x_hat[3] - Ixz*self.x_hat[4]])


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

def create_feedback_control(trim_solution, V, H, Gamma, cg_shift, Q, R):
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    x_hat = trim_solution.states
    alpha_hat = trim_solution.x[1]
    beta_hat = trim_solution.x[2]
    u_hat = trim_solution.inputs
    FM_hat = trim_solution.FM
    props = trim.AircraftProperties(V, H, Gamma, aero_dir, bire=True)
    linearization = LinearizationBIRE(props, aero_dir)
    linearization.set_linearization_point(x_hat, u_hat, alpha_hat, beta_hat, FM_hat,
                                          cg_shift)
    A = linearization.create_A_matrix()
    B = linearization.create_B_matrix()
    C = linearization.create_C_matrix()
    G = ctrb(A, B)
    print(np.rad2deg(x_hat[-1]), np.linalg.matrix_rank(G))
    K, S, E = lqr(A, B, Q, R)
    eig_check, v_check = np.linalg.eig(A - np.matmul(B, K))
    try:
        assert all(np.real(eig_check) < 0.)
    except  AssertionError:
        print("Not able to stabilize.")

    results = Lin_Results(linearization.N, linearization.M)
    results.A = A
    results.B = B
    results.C = C
    results.K = K
    results.eigs = eig_check
    return results

if __name__ == "__main__":
    plt.close('all')
    H = 15000.
    a = stdatm_english(H)[-1]
    M = 0.6
    V = M*a
    b_w = 30.
    c_w = 11.32
    gamma = np.deg2rad(0.)
    phi = np.deg2rad(0.)
    Gamma = 0.1
    cg_shift = [0., 0., 0.]
    aero_dir = '/home/christian/Python Projects/AFRL BIRE/Static Analysis/main/'
    trim_solution = trim.trim(V, H, gamma, phi, Gamma, fixed_point=False,
                              aero_dir=aero_dir, bire=True)
    x_hat = trim_solution.states
    alpha_hat = trim_solution.x[1]
    beta_hat = trim_solution.x[2]
    u_hat = trim_solution.inputs
    FM_hat = trim_solution.FM
    props = trim.AircraftProperties(V, H, Gamma, aero_dir)
    linearization = LinearizationBIRE(props, aero_dir)
    linearization.set_linearization_point(x_hat, u_hat, alpha_hat, beta_hat, FM_hat,
                                          cg_shift)
    A = linearization.create_A_matrix()
    B = linearization.create_B_matrix()
    C = linearization.create_C_matrix()