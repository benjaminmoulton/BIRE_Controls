import numpy as np
import json
from matplotlib import pyplot as plt
from scipy.integrate import ode, odeint
from controller_simulation import Aircraft,monte_carlo_perturbations,run_single_simulation
from numpy import cos,sin, arctan2 as atan2,arcsin as asin

def simple_model_euler(self,t,x,
    is_controlled=True,given_control=False,u="o",
    force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u)

    # read in states
    V = x[0]
    b = x[1]
    a = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    xf = x[6]
    yf = x[7]
    zf = x[8]
    ph = x[9]
    th = x[10]
    ps = x[11]
    da = u[0]
    de = u[1]
    dB = u[2]
    tau = u[3]

    # # disturbance model
    # ## INTSTATE
    # V = x[0]
    # Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    # Vg = [Du,Dv,Dw]
    # Wg = [Dp,Dq,Dr]

    ## INTSTATE
    # Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    # a = np.atan2(Vw,Vu)
    # V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    # b = np.asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-self.x_trim[8]) # zf) ## ## ## ## ## ## ## 
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (p)*self.bw/2./V
    qbar = (q)*self.cw/2./V
    rbar = (r)*self.bw/2./V

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(0.0) # u[3]) # ## ## ## ##
    # Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    # calculate Q's
    Q = 0.5*rho*V**2.*self.Sw
    Qxx = Q*self.bw/Ixx
    Qyy = Q*self.cw/Iyy
    Qzz = Q*self.bw/Izz
    bam = bire.aero_model

    # aerodynamic terms
    CL1Az = (bam.CL_0_A + bam.CL_0_z) + bam.CL_a_z*a
    CS1n = (bam.CS_0_A*bam.CS_0_w)*dB + (bam.CS_b_A + bam.CS_b_z)*b
    ###
    CLqAz = bam.CL_q_A + bam.CL_q_z
    CSbAz = bam.CS_b_A + bam.CS_b_z
    CSrAz = bam.CS_r_A + bam.CS_r_z
    CDLqAz = bam.CD_Lq_A + bam.CD_Lq_z
    CLdeAz = bam.CL_de_A + bam.CL_de_z
    CDS2Az = bam.CD_S2_A + bam.CD_S2_z
    #
    CldaAz = bam.Cl_da_A + bam.Cl_da_z
    CmaAz = bam.Cm_a_A + bam.Cm_a_z
    CmqAz = bam.Cm_q_A + bam.Cm_q_z
    CmdeAz = bam.Cm_de_A + bam.Cm_de_z
    CnbAz = bam.Cn_b_A + bam.Cn_b_z
    CnrAz = bam.Cn_r_A + bam.Cn_r_z
    ###
    CLbAw = bam.CL_b_A*bam.CL_b_w
    CLrAw = bam.CL_r_A*bam.CL_r_w
    CS0Aw = bam.CS_0_A*bam.CS_0_w
    CSaAw = bam.CS_a_A*bam.CS_a_w
    CSqAw = bam.CS_q_A*bam.CS_q_w
    CSdeAw = bam.CS_de_A*bam.CS_de_w
    CDSAw = bam.CD_S_A*bam.CD_S_w
    #
    CmbAw = bam.Cm_b_A*bam.Cm_b_w
    CmrAw = bam.Cm_r_A*bam.Cm_r_w
    CnaAw = bam.Cn_a_A*bam.Cn_a_w
    CnqAw = bam.Cn_q_A*bam.Cn_q_w
    CndeAw = bam.Cn_de_A*bam.Cn_de_w
    
    # # # Dynamics
    dx = x * 0.
    #######################################################################
    # reconvert back to Vx,Vy,Vz
    xnl = x*1.
    V = x[0]*1.; a = x[2]*1.; b = x[1]*1.
    xnl[0] = V*np.cos(a)*np.cos(b)
    xnl[1] = V*np.sin(b)
    xnl[2] = V*np.sin(a)*np.cos(b)
    dx_xyz = self._nonlinear_euler_dynamics(t,xnl,is_controlled,given_control,u)
    dxnl = dx_xyz*1.
    dxnl[0] = 1./V*(xnl[0]*dx_xyz[0] + xnl[1]*dx_xyz[1] + xnl[2]*dx_xyz[2])
    dxnl[1] = (dx_xyz[1]*V - xnl[1]*dxnl[0])/(V*(xnl[0]**2. + xnl[2]**2.)**0.5)
    dxnl[2] = (xnl[0]*dx_xyz[2] - dx_xyz[0]*xnl[2])/(xnl[0]**2. + xnl[2]**2.)
    #######################################################################
    #
    # V
    trm_1 = (CL1Az + CLqAz*qbar)*a
    trm_2_1 = bam.CD_0_z + bam.CD_L_z*CL1Az + bam.CD_L2_z*CL1Az**2.
    trm_2_2 = (CSbAz**2.*b**2. + CDLqAz*CL1Az + bam.CD_q_z)*qbar
    trm_2 = trm_2_1 + trm_2_2
    trm_3 = -(bam.CD_Lde_A*CL1Az*de + bam.CD_de2_z*de**2.) + CLdeAz*a*de
    trm_4_1 = (bam.CL_b_A*bam.CL_b_w*b + bam.CL_r_A*bam.CL_r_w*rbar)*a*dB
    trm_4_2 = (CDSAw*CSbAz + CS0Aw*CSbAz)*b*dB
    trm_4_3 = (CDSAw*CS0Aw + CDS2Az*CS0Aw**2.)*dB**2.
    trm_4 = trm_4_1 - (trm_4_2 + trm_4_3)
    trm_5_1 = bam.get_thrust(tau,-self.x_trim[8],V) # -zf,V)) # ## ## ## ## ## ## 
    trm_5 = 2.*(trm_5_1*g/W - g*th + r*b*V - q*a*V)
    dx[0] = 2.*Q*g/W*(trm_1 - trm_2 + trm_3 + trm_4) + trm_5
    # dx[0] = dxnl[0]*1.
    #
    # b
    prt_1 = CSbAz*b + bam.CS_Lp_z*CL1Az*pbar + CSrAz*rbar
    prt_2 = (CS0Aw + CSaAw*a + CSqAw*qbar)*dB + CSdeAw*de*dB
    prt_3 = g*ph + p*a*V - r*V
    dx[1] = Q/V*g/W*(prt_1 + prt_2) + prt_3
    # dx[1] = dxnl[1]*1.
    #
    # a
    cmp_1 = -(CL1Az + CLqAz*qbar) + g
    cmp_2 = -CLdeAz*de - (CLbAw*b + CLrAw*rbar)*dB
    cmp_3 = g + q*V - p*b*V
    dx[2] = Q/V*g/W*(cmp_1 + cmp_2) + cmp_3
    # dx[2] = dxnl[2]*1.
    #
    # p
    bit_1 = bam.Cl_b_z*b + bam.Cl_p_z*pbar + (bam.Cl_Lr_z*CL1Az + bam.Cl_r_z)*rbar
    bit_2 = CldaAz*da + bam.Cl_de_z*de
    bit_3 = (Iyy - Izz)*q*r
    dx[3] = Qxx*(bit_1 + bit_2) + 1./Ixx*bit_3
    # dx[3] = dxnl[3]*1.
    #
    # q
    byt_1 = CmaAz*a + CmqAz*qbar
    byt_2 = CmdeAz*de + (CmbAw*b + CmrAw*rbar)*dB
    byt_3 = -hx*r + (Izz - Ixx)*p*r
    dx[4] = Qyy*(byt_1 + byt_2) + 1./Iyy*byt_3
    # dx[4] = dxnl[4]*1.
    #
    # r
    byt_1 = CnbAz*b + CnrAz*rbar
    byt_2 = (CnaAw*a + CnqAw*qbar)*dB + CndeAw*de*dB
    byt_3 = hx*q + (Ixx - Iyy)*p*q
    dx[5] = Qzz*(byt_1 + byt_2) + 1./Izz*byt_3
    # dx[5] = dxnl[5]*1.
    #
    # xf
    dx[6] = V*(1. + th*a)
    # dx[6] = dxnl[6]*1.
    #
    # yf
    dx[7] = V*(ps - ph*a)
    # dx[7] = dxnl[7]*1.
    #
    # zf
    dx[8] = V*(a - th)
    # dx[8] = dxnl[8]*1.
    #
    # ph
    dx[9] = p + th*r
    # dx[9] = dxnl[9]*1.
    #
    # th
    dx[10] = q - ph*r
    # dx[10] = dxnl[10]*1.
    # 
    # ps
    dx[11] = q*ph + r
    # dx[11] = dxnl[11]*1.

    return dx


def true_nonlinear_no_comp(self,t,x,
    is_controlled=True,given_control=False,u="o",
    use_stall=True,force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) + BAM.CL_0_z + BAM.CL_0_d
    CLa = BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p) + BAM.CL_a_z + BAM.CL_a_d
    CLb = BAM.CL_b_A*sin(BAM.CL_b_w*dB + BAM.CL_b_p) + BAM.CL_b_z + BAM.CL_b_d
    CLp = BAM.CL_p_A*sin(BAM.CL_p_w*dB + BAM.CL_p_p) + BAM.CL_p_z + BAM.CL_p_d
    CLq = BAM.CL_q_A*sin(BAM.CL_q_w*dB + BAM.CL_q_p) + BAM.CL_q_z + BAM.CL_q_d
    CLr = BAM.CL_r_A*sin(BAM.CL_r_w*dB + BAM.CL_r_p) + BAM.CL_r_z + BAM.CL_r_d
    CLda = BAM.CL_da_A*sin(BAM.CL_da_w*dB + BAM.CL_da_p) + BAM.CL_da_z + BAM.CL_da_d
    CLde = BAM.CL_de_A*sin(BAM.CL_de_w*dB + BAM.CL_de_p) + BAM.CL_de_z + BAM.CL_de_d
    #
    CS0 = BAM.CS_0_A*sin(BAM.CS_0_w*dB + BAM.CS_0_p) + BAM.CS_0_z + BAM.CS_0_d
    CSa = BAM.CS_a_A*sin(BAM.CS_a_w*dB + BAM.CS_a_p) + BAM.CS_a_z + BAM.CS_a_d
    CSb = BAM.CS_b_A*sin(BAM.CS_b_w*dB + BAM.CS_b_p) + BAM.CS_b_z + BAM.CS_b_d
    CSp = BAM.CS_p_A*sin(BAM.CS_p_w*dB + BAM.CS_p_p) + BAM.CS_p_z + BAM.CS_p_d
    CSLp = BAM.CS_Lp_A*sin(BAM.CS_Lp_w*dB + BAM.CS_Lp_p) + BAM.CS_Lp_z + BAM.CS_Lp_d
    CSq = BAM.CS_q_A*sin(BAM.CS_q_w*dB + BAM.CS_q_p) + BAM.CS_q_z + BAM.CS_q_d
    CSr = BAM.CS_r_A*sin(BAM.CS_r_w*dB + BAM.CS_r_p) + BAM.CS_r_z + BAM.CS_r_d
    CSda = BAM.CS_da_A*sin(BAM.CS_da_w*dB + BAM.CS_da_p) + BAM.CS_da_z + BAM.CS_da_d
    CSde = BAM.CS_de_A*sin(BAM.CS_de_w*dB + BAM.CS_de_p) + BAM.CS_de_z + BAM.CS_de_d
    #
    CD0 = BAM.CD_0_A*sin(BAM.CD_0_w*dB + BAM.CD_0_p) + BAM.CD_0_z + BAM.CD_0_d
    CDL = BAM.CD_L_A*sin(BAM.CD_L_w*dB + BAM.CD_L_p) + BAM.CD_L_z + BAM.CD_L_d
    CDL2 = BAM.CD_L2_A*sin(BAM.CD_L2_w*dB + BAM.CD_L2_p) + BAM.CD_L2_z + BAM.CD_L2_d
    CDS = BAM.CD_S_A*sin(BAM.CD_S_w*dB + BAM.CD_S_p) + BAM.CD_S_z + BAM.CD_S_d
    CDS2 = BAM.CD_S2_A*sin(BAM.CD_S2_w*dB + BAM.CD_S2_p) + BAM.CD_S2_z + BAM.CD_S2_d
    CDp = BAM.CD_p_A*sin(BAM.CD_p_w*dB + BAM.CD_p_p) + BAM.CD_p_z + BAM.CD_p_d
    CDSp = BAM.CD_Sp_A*sin(BAM.CD_Sp_w*dB + BAM.CD_Sp_p) + BAM.CD_Sp_z + BAM.CD_Sp_d
    CDq = BAM.CD_q_A*sin(BAM.CD_q_w*dB + BAM.CD_q_p) + BAM.CD_q_z + BAM.CD_q_d
    CDLq = BAM.CD_Lq_A*sin(BAM.CD_Lq_w*dB + BAM.CD_Lq_p) + BAM.CD_Lq_z + BAM.CD_Lq_d
    CDL2q = BAM.CD_L2q_A*sin(BAM.CD_L2q_w*dB + BAM.CD_L2q_p) + BAM.CD_L2q_z + BAM.CD_L2q_d
    CDr = BAM.CD_r_A*sin(BAM.CD_r_w*dB + BAM.CD_r_p) + BAM.CD_r_z + BAM.CD_r_d
    CDSr = BAM.CD_Sr_A*sin(BAM.CD_Sr_w*dB + BAM.CD_Sr_p) + BAM.CD_Sr_z + BAM.CD_Sr_d
    CDda = BAM.CD_da_A*sin(BAM.CD_da_w*dB + BAM.CD_da_p) + BAM.CD_da_z + BAM.CD_da_d
    CDSda = BAM.CD_Sda_A*sin(BAM.CD_Sda_w*dB + BAM.CD_Sda_p) + BAM.CD_Sda_z + BAM.CD_Sda_d
    CDde = BAM.CD_de_A*sin(BAM.CD_de_w*dB + BAM.CD_de_p) + BAM.CD_de_z + BAM.CD_de_d
    CDLde = BAM.CD_Lde_A*sin(BAM.CD_Lde_w*dB + BAM.CD_Lde_p) + BAM.CD_Lde_z + BAM.CD_Lde_d
    CDde2 = BAM.CD_de2_A*sin(BAM.CD_de2_w*dB + BAM.CD_de2_p) + BAM.CD_de2_z + BAM.CD_de2_d
    #
    Cl0 = BAM.Cl_0_A*sin(BAM.Cl_0_w*dB + BAM.Cl_0_p) + BAM.Cl_0_z + BAM.Cl_0_d
    Cla = BAM.Cl_a_A*sin(BAM.Cl_a_w*dB + BAM.Cl_a_p) + BAM.Cl_a_z + BAM.Cl_a_d
    Clb = BAM.Cl_b_A*sin(BAM.Cl_b_w*dB + BAM.Cl_b_p) + BAM.Cl_b_z + BAM.Cl_b_d
    Clp = BAM.Cl_p_A*sin(BAM.Cl_p_w*dB + BAM.Cl_p_p) + BAM.Cl_p_z + BAM.Cl_p_d
    Clq = BAM.Cl_q_A*sin(BAM.Cl_q_w*dB + BAM.Cl_q_p) + BAM.Cl_q_z + BAM.Cl_q_d
    Clr = BAM.Cl_r_A*sin(BAM.Cl_r_w*dB + BAM.Cl_r_p) + BAM.Cl_r_z + BAM.Cl_r_d
    ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p) + BAM.Cl_Lr_z + BAM.Cl_Lr_d
    Clda = BAM.Cl_da_A*sin(BAM.Cl_da_w*dB + BAM.Cl_da_p) + BAM.Cl_da_z + BAM.Cl_da_d
    Clde = BAM.Cl_de_A*sin(BAM.Cl_de_w*dB + BAM.Cl_de_p) + BAM.Cl_de_z + BAM.Cl_de_d
    #
    Cm0 = BAM.Cm_0_A*sin(BAM.Cm_0_w*dB + BAM.Cm_0_p) + BAM.Cm_0_z + BAM.Cm_0_d
    Cma = BAM.Cm_a_A*sin(BAM.Cm_a_w*dB + BAM.Cm_a_p) + BAM.Cm_a_z + BAM.Cm_a_d
    Cmb = BAM.Cm_b_A*sin(BAM.Cm_b_w*dB + BAM.Cm_b_p) + BAM.Cm_b_z + BAM.Cm_b_d
    Cmp = BAM.Cm_p_A*sin(BAM.Cm_p_w*dB + BAM.Cm_p_p) + BAM.Cm_p_z + BAM.Cm_p_d
    Cmq = BAM.Cm_q_A*sin(BAM.Cm_q_w*dB + BAM.Cm_q_p) + BAM.Cm_q_z + BAM.Cm_q_d
    Cmr = BAM.Cm_r_A*sin(BAM.Cm_r_w*dB + BAM.Cm_r_p) + BAM.Cm_r_z + BAM.Cm_r_d
    Cmda = BAM.Cm_da_A*sin(BAM.Cm_da_w*dB + BAM.Cm_da_p) + BAM.Cm_da_z + BAM.Cm_da_d
    Cmde = BAM.Cm_de_A*sin(BAM.Cm_de_w*dB + BAM.Cm_de_p) + BAM.Cm_de_z + BAM.Cm_de_d
    #
    Cn0 = BAM.Cn_0_A*sin(BAM.Cn_0_w*dB + BAM.Cn_0_p) + BAM.Cn_0_z + BAM.Cn_0_d
    Cna = BAM.Cn_a_A*sin(BAM.Cn_a_w*dB + BAM.Cn_a_p) + BAM.Cn_a_z + BAM.Cn_a_d
    Cnb = BAM.Cn_b_A*sin(BAM.Cn_b_w*dB + BAM.Cn_b_p) + BAM.Cn_b_z + BAM.Cn_b_d
    Cnp = BAM.Cn_p_A*sin(BAM.Cn_p_w*dB + BAM.Cn_p_p) + BAM.Cn_p_z + BAM.Cn_p_d
    CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p) + BAM.Cn_Lp_z + BAM.Cn_Lp_d
    Cnq = BAM.Cn_q_A*sin(BAM.Cn_q_w*dB + BAM.Cn_q_p) + BAM.Cn_q_z + BAM.Cn_q_d
    Cnr = BAM.Cn_r_A*sin(BAM.Cn_r_w*dB + BAM.Cn_r_p) + BAM.Cn_r_z + BAM.Cn_r_d
    Cnda = BAM.Cn_da_A*sin(BAM.Cn_da_w*dB + BAM.Cn_da_p) + BAM.Cn_da_z + BAM.Cn_da_d
    CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p) + BAM.Cn_Lda_z + BAM.Cn_Lda_d
    Cnde = BAM.Cn_de_A*sin(BAM.Cn_de_w*dB + BAM.Cn_de_p) + BAM.Cn_de_z + BAM.Cn_de_d
    #
    A_CL1 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) \
        + (BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p))*a
    z_CL1 = BAM.CL_0_z + BAM.CL_0_d + (BAM.CL_a_z + BAM.CL_a_d)*a
    A_ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p)
    z_ClLr = BAM.Cl_Lr_z + BAM.Cl_Lr_d
    A_CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p)
    z_CnLp = BAM.Cn_Lp_z + BAM.Cn_Lp_d
    A_CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p)
    z_CnLda = BAM.Cn_Lda_z + BAM.Cn_Lda_d

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr + A_ClLr*A_CL1 + z_ClLr*A_CL1 + A_ClLr*z_CL1 + z_ClLr*z_CL1)*rbar \
        + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta \
        + (Cnp + A_CnLp*A_CL1 + z_CnLp*A_CL1 + A_CnLp*z_CL1 + z_CnLp*z_CL1)*pbar \
        + Cnq*qbar + Cnr*rbar \
        + (Cnda + A_CnLda*A_CL1 + z_CnLda*A_CL1 + A_CnLda*z_CL1 + z_CnLda*z_CL1)*da \
        + Cnde*de)
    # return Cn

    # stall
    if use_stall:
        S_M = 7.0
        S_ab = np.deg2rad(45.0)
        # determine flat plate forces and moment
        CLplate = 2.*np.sign(a)*np.sin(a)**2.*np.cos(a)
        CDplate = 2.*np.sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = np.exp(-S_M*(a-S_ab))
        expMplu = np.exp(S_M*(a+S_ab))
        sig = (1. + expMmin + expMplu)/(1. + expMmin)/(1. + expMplu)

        # add stall effects
        CL = (1. - sig)*CL + sig*CLplate
        CD = (1. - sig)*CD + sig*CDplate
        Cm = (1. - sig)*Cm + sig*Cmplate
    
    

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx


def nonlinear_noC_psdolin(self,t,x,
    is_controlled=True,given_control=False,u="o",
    use_stall=True,force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) + BAM.CL_0_z + BAM.CL_0_d
    CLa = BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p) + BAM.CL_a_z + BAM.CL_a_d
    CLb = BAM.CL_b_A*sin(BAM.CL_b_w*dB + BAM.CL_b_p) + BAM.CL_b_z + BAM.CL_b_d
    CLp = BAM.CL_p_A*sin(BAM.CL_p_w*dB + BAM.CL_p_p) + BAM.CL_p_z + BAM.CL_p_d
    CLq = BAM.CL_q_A*sin(BAM.CL_q_w*dB + BAM.CL_q_p) + BAM.CL_q_z + BAM.CL_q_d
    CLr = BAM.CL_r_A*sin(BAM.CL_r_w*dB + BAM.CL_r_p) + BAM.CL_r_z + BAM.CL_r_d
    CLda = BAM.CL_da_A*sin(BAM.CL_da_w*dB + BAM.CL_da_p) + BAM.CL_da_z + BAM.CL_da_d
    CLde = BAM.CL_de_A*sin(BAM.CL_de_w*dB + BAM.CL_de_p) + BAM.CL_de_z + BAM.CL_de_d
    #
    CS0 = BAM.CS_0_A*sin(BAM.CS_0_w*dB + BAM.CS_0_p) + BAM.CS_0_z + BAM.CS_0_d
    CSa = BAM.CS_a_A*sin(BAM.CS_a_w*dB + BAM.CS_a_p) + BAM.CS_a_z + BAM.CS_a_d
    CSb = BAM.CS_b_A*sin(BAM.CS_b_w*dB + BAM.CS_b_p) + BAM.CS_b_z + BAM.CS_b_d
    CSp = BAM.CS_p_A*sin(BAM.CS_p_w*dB + BAM.CS_p_p) + BAM.CS_p_z + BAM.CS_p_d
    CSLp = BAM.CS_Lp_A*sin(BAM.CS_Lp_w*dB + BAM.CS_Lp_p) + BAM.CS_Lp_z + BAM.CS_Lp_d
    CSq = BAM.CS_q_A*sin(BAM.CS_q_w*dB + BAM.CS_q_p) + BAM.CS_q_z + BAM.CS_q_d
    CSr = BAM.CS_r_A*sin(BAM.CS_r_w*dB + BAM.CS_r_p) + BAM.CS_r_z + BAM.CS_r_d
    CSda = BAM.CS_da_A*sin(BAM.CS_da_w*dB + BAM.CS_da_p) + BAM.CS_da_z + BAM.CS_da_d
    CSde = BAM.CS_de_A*sin(BAM.CS_de_w*dB + BAM.CS_de_p) + BAM.CS_de_z + BAM.CS_de_d
    #
    CD0 = BAM.CD_0_A*sin(BAM.CD_0_w*dB + BAM.CD_0_p) + BAM.CD_0_z + BAM.CD_0_d
    CDL = BAM.CD_L_A*sin(BAM.CD_L_w*dB + BAM.CD_L_p) + BAM.CD_L_z + BAM.CD_L_d
    CDL2 = BAM.CD_L2_A*sin(BAM.CD_L2_w*dB + BAM.CD_L2_p) + BAM.CD_L2_z + BAM.CD_L2_d
    CDS = BAM.CD_S_A*sin(BAM.CD_S_w*dB + BAM.CD_S_p) + BAM.CD_S_z + BAM.CD_S_d
    CDS2 = BAM.CD_S2_A*sin(BAM.CD_S2_w*dB + BAM.CD_S2_p) + BAM.CD_S2_z + BAM.CD_S2_d
    CDp = BAM.CD_p_A*sin(BAM.CD_p_w*dB + BAM.CD_p_p) + BAM.CD_p_z + BAM.CD_p_d
    CDSp = BAM.CD_Sp_A*sin(BAM.CD_Sp_w*dB + BAM.CD_Sp_p) + BAM.CD_Sp_z + BAM.CD_Sp_d
    CDq = BAM.CD_q_A*sin(BAM.CD_q_w*dB + BAM.CD_q_p) + BAM.CD_q_z + BAM.CD_q_d
    CDLq = BAM.CD_Lq_A*sin(BAM.CD_Lq_w*dB + BAM.CD_Lq_p) + BAM.CD_Lq_z + BAM.CD_Lq_d
    CDL2q = BAM.CD_L2q_A*sin(BAM.CD_L2q_w*dB + BAM.CD_L2q_p) + BAM.CD_L2q_z + BAM.CD_L2q_d
    CDr = BAM.CD_r_A*sin(BAM.CD_r_w*dB + BAM.CD_r_p) + BAM.CD_r_z + BAM.CD_r_d
    CDSr = BAM.CD_Sr_A*sin(BAM.CD_Sr_w*dB + BAM.CD_Sr_p) + BAM.CD_Sr_z + BAM.CD_Sr_d
    CDda = BAM.CD_da_A*sin(BAM.CD_da_w*dB + BAM.CD_da_p) + BAM.CD_da_z + BAM.CD_da_d
    CDSda = BAM.CD_Sda_A*sin(BAM.CD_Sda_w*dB + BAM.CD_Sda_p) + BAM.CD_Sda_z + BAM.CD_Sda_d
    CDde = BAM.CD_de_A*sin(BAM.CD_de_w*dB + BAM.CD_de_p) + BAM.CD_de_z + BAM.CD_de_d
    CDLde = BAM.CD_Lde_A*sin(BAM.CD_Lde_w*dB + BAM.CD_Lde_p) + BAM.CD_Lde_z + BAM.CD_Lde_d
    CDde2 = BAM.CD_de2_A*sin(BAM.CD_de2_w*dB + BAM.CD_de2_p) + BAM.CD_de2_z + BAM.CD_de2_d
    #
    Cl0 = BAM.Cl_0_A*sin(BAM.Cl_0_w*dB + BAM.Cl_0_p) + BAM.Cl_0_z + BAM.Cl_0_d
    Cla = BAM.Cl_a_A*sin(BAM.Cl_a_w*dB + BAM.Cl_a_p) + BAM.Cl_a_z + BAM.Cl_a_d
    Clb = BAM.Cl_b_A*sin(BAM.Cl_b_w*dB + BAM.Cl_b_p) + BAM.Cl_b_z + BAM.Cl_b_d
    Clp = BAM.Cl_p_A*sin(BAM.Cl_p_w*dB + BAM.Cl_p_p) + BAM.Cl_p_z + BAM.Cl_p_d
    Clq = BAM.Cl_q_A*sin(BAM.Cl_q_w*dB + BAM.Cl_q_p) + BAM.Cl_q_z + BAM.Cl_q_d
    Clr = BAM.Cl_r_A*sin(BAM.Cl_r_w*dB + BAM.Cl_r_p) + BAM.Cl_r_z + BAM.Cl_r_d
    ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p) + BAM.Cl_Lr_z + BAM.Cl_Lr_d
    Clda = BAM.Cl_da_A*sin(BAM.Cl_da_w*dB + BAM.Cl_da_p) + BAM.Cl_da_z + BAM.Cl_da_d
    Clde = BAM.Cl_de_A*sin(BAM.Cl_de_w*dB + BAM.Cl_de_p) + BAM.Cl_de_z + BAM.Cl_de_d
    #
    Cm0 = BAM.Cm_0_A*sin(BAM.Cm_0_w*dB + BAM.Cm_0_p) + BAM.Cm_0_z + BAM.Cm_0_d
    Cma = BAM.Cm_a_A*sin(BAM.Cm_a_w*dB + BAM.Cm_a_p) + BAM.Cm_a_z + BAM.Cm_a_d
    Cmb = BAM.Cm_b_A*sin(BAM.Cm_b_w*dB + BAM.Cm_b_p) + BAM.Cm_b_z + BAM.Cm_b_d
    Cmp = BAM.Cm_p_A*sin(BAM.Cm_p_w*dB + BAM.Cm_p_p) + BAM.Cm_p_z + BAM.Cm_p_d
    Cmq = BAM.Cm_q_A*sin(BAM.Cm_q_w*dB + BAM.Cm_q_p) + BAM.Cm_q_z + BAM.Cm_q_d
    Cmr = BAM.Cm_r_A*sin(BAM.Cm_r_w*dB + BAM.Cm_r_p) + BAM.Cm_r_z + BAM.Cm_r_d
    Cmda = BAM.Cm_da_A*sin(BAM.Cm_da_w*dB + BAM.Cm_da_p) + BAM.Cm_da_z + BAM.Cm_da_d
    Cmde = BAM.Cm_de_A*sin(BAM.Cm_de_w*dB + BAM.Cm_de_p) + BAM.Cm_de_z + BAM.Cm_de_d
    #
    Cn0 = BAM.Cn_0_A*sin(BAM.Cn_0_w*dB + BAM.Cn_0_p) + BAM.Cn_0_z + BAM.Cn_0_d
    Cna = BAM.Cn_a_A*sin(BAM.Cn_a_w*dB + BAM.Cn_a_p) + BAM.Cn_a_z + BAM.Cn_a_d
    Cnb = BAM.Cn_b_A*sin(BAM.Cn_b_w*dB + BAM.Cn_b_p) + BAM.Cn_b_z + BAM.Cn_b_d
    Cnp = BAM.Cn_p_A*sin(BAM.Cn_p_w*dB + BAM.Cn_p_p) + BAM.Cn_p_z + BAM.Cn_p_d
    CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p) + BAM.Cn_Lp_z + BAM.Cn_Lp_d
    Cnq = BAM.Cn_q_A*sin(BAM.Cn_q_w*dB + BAM.Cn_q_p) + BAM.Cn_q_z + BAM.Cn_q_d
    Cnr = BAM.Cn_r_A*sin(BAM.Cn_r_w*dB + BAM.Cn_r_p) + BAM.Cn_r_z + BAM.Cn_r_d
    Cnda = BAM.Cn_da_A*sin(BAM.Cn_da_w*dB + BAM.Cn_da_p) + BAM.Cn_da_z + BAM.Cn_da_d
    CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p) + BAM.Cn_Lda_z + BAM.Cn_Lda_d
    Cnde = BAM.Cn_de_A*sin(BAM.Cn_de_w*dB + BAM.Cn_de_p) + BAM.Cn_de_z + BAM.Cn_de_d
    #
    A_CL1 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) \
        + (BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p))*a
    z_CL1 = BAM.CL_0_z + BAM.CL_0_d + (BAM.CL_a_z + BAM.CL_a_d)*a
    A_ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p)
    z_ClLr = BAM.Cl_Lr_z + BAM.Cl_Lr_d
    A_CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p)
    z_CnLp = BAM.Cn_Lp_z + BAM.Cn_Lp_d
    A_CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p)
    z_CnLda = BAM.Cn_Lda_z + BAM.Cn_Lda_d

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr)*rbar \
        + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta \
        + (Cnp)*pbar \
        + Cnq*qbar + Cnr*rbar \
        + (Cnda)*da \
        + Cnde*de)
    # return Cn

    # stall
    if use_stall:
        S_M = 7.0
        S_ab = np.deg2rad(45.0)
        # determine flat plate forces and moment
        CLplate = 2.*np.sign(a)*np.sin(a)**2.*np.cos(a)
        CDplate = 2.*np.sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = np.exp(-S_M*(a-S_ab))
        expMplu = np.exp(S_M*(a+S_ab))
        sig = (1. + expMmin + expMplu)/(1. + expMmin)/(1. + expMplu)

        # add stall effects
        CL = (1. - sig)*CL + sig*CLplate
        CD = (1. - sig)*CD + sig*CDplate
        Cm = (1. - sig)*Cm + sig*Cmplate
    
    

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx


def nonlinear_noC_noSp(self,t,x,
    is_controlled=True,given_control=False,u="o",
    use_stall=True,force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) + BAM.CL_0_z + BAM.CL_0_d
    CLa = BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p) + BAM.CL_a_z + BAM.CL_a_d
    CLb = BAM.CL_b_A*sin(BAM.CL_b_w*dB + BAM.CL_b_p) + BAM.CL_b_z + BAM.CL_b_d
    CLp = BAM.CL_p_A*sin(BAM.CL_p_w*dB + BAM.CL_p_p) + BAM.CL_p_z + BAM.CL_p_d
    CLq = BAM.CL_q_A*sin(BAM.CL_q_w*dB + BAM.CL_q_p) + BAM.CL_q_z + BAM.CL_q_d
    CLr = BAM.CL_r_A*sin(BAM.CL_r_w*dB + BAM.CL_r_p) + BAM.CL_r_z + BAM.CL_r_d
    CLda = BAM.CL_da_A*sin(BAM.CL_da_w*dB + BAM.CL_da_p) + BAM.CL_da_z + BAM.CL_da_d
    CLde = BAM.CL_de_A*sin(BAM.CL_de_w*dB + BAM.CL_de_p) + BAM.CL_de_z + BAM.CL_de_d
    #
    CS0 = BAM.CS_0_A*sin(BAM.CS_0_w*dB + BAM.CS_0_p) + BAM.CS_0_z + BAM.CS_0_d
    CSa = BAM.CS_a_A*sin(BAM.CS_a_w*dB + BAM.CS_a_p) + BAM.CS_a_z + BAM.CS_a_d
    CSb = BAM.CS_b_A*sin(BAM.CS_b_w*dB + BAM.CS_b_p) + BAM.CS_b_z + BAM.CS_b_d
    CSp = BAM.CS_p_A*sin(BAM.CS_p_w*dB + BAM.CS_p_p) + BAM.CS_p_z + BAM.CS_p_d
    CSLp = BAM.CS_Lp_A*sin(BAM.CS_Lp_w*dB + BAM.CS_Lp_p) + BAM.CS_Lp_z + BAM.CS_Lp_d
    CSq = BAM.CS_q_A*sin(BAM.CS_q_w*dB + BAM.CS_q_p) + BAM.CS_q_z + BAM.CS_q_d
    CSr = BAM.CS_r_A*sin(BAM.CS_r_w*dB + BAM.CS_r_p) + BAM.CS_r_z + BAM.CS_r_d
    CSda = BAM.CS_da_A*sin(BAM.CS_da_w*dB + BAM.CS_da_p) + BAM.CS_da_z + BAM.CS_da_d
    CSde = BAM.CS_de_A*sin(BAM.CS_de_w*dB + BAM.CS_de_p) + BAM.CS_de_z + BAM.CS_de_d
    #
    CD0 = BAM.CD_0_A*sin(BAM.CD_0_w*dB + BAM.CD_0_p) + BAM.CD_0_z + BAM.CD_0_d
    CDL = BAM.CD_L_A*sin(BAM.CD_L_w*dB + BAM.CD_L_p) + BAM.CD_L_z + BAM.CD_L_d
    CDL2 = BAM.CD_L2_A*sin(BAM.CD_L2_w*dB + BAM.CD_L2_p) + BAM.CD_L2_z + BAM.CD_L2_d
    CDS = BAM.CD_S_A*sin(BAM.CD_S_w*dB + BAM.CD_S_p) + BAM.CD_S_z + BAM.CD_S_d
    CDS2 = BAM.CD_S2_A*sin(BAM.CD_S2_w*dB + BAM.CD_S2_p) + BAM.CD_S2_z + BAM.CD_S2_d
    CDp = BAM.CD_p_A*sin(BAM.CD_p_w*dB + BAM.CD_p_p) + BAM.CD_p_z + BAM.CD_p_d
    CDSp = BAM.CD_Sp_A*sin(BAM.CD_Sp_w*dB + BAM.CD_Sp_p) + BAM.CD_Sp_z + BAM.CD_Sp_d
    CDq = BAM.CD_q_A*sin(BAM.CD_q_w*dB + BAM.CD_q_p) + BAM.CD_q_z + BAM.CD_q_d
    CDLq = BAM.CD_Lq_A*sin(BAM.CD_Lq_w*dB + BAM.CD_Lq_p) + BAM.CD_Lq_z + BAM.CD_Lq_d
    CDL2q = BAM.CD_L2q_A*sin(BAM.CD_L2q_w*dB + BAM.CD_L2q_p) + BAM.CD_L2q_z + BAM.CD_L2q_d
    CDr = BAM.CD_r_A*sin(BAM.CD_r_w*dB + BAM.CD_r_p) + BAM.CD_r_z + BAM.CD_r_d
    CDSr = BAM.CD_Sr_A*sin(BAM.CD_Sr_w*dB + BAM.CD_Sr_p) + BAM.CD_Sr_z + BAM.CD_Sr_d
    CDda = BAM.CD_da_A*sin(BAM.CD_da_w*dB + BAM.CD_da_p) + BAM.CD_da_z + BAM.CD_da_d
    CDSda = BAM.CD_Sda_A*sin(BAM.CD_Sda_w*dB + BAM.CD_Sda_p) + BAM.CD_Sda_z + BAM.CD_Sda_d
    CDde = BAM.CD_de_A*sin(BAM.CD_de_w*dB + BAM.CD_de_p) + BAM.CD_de_z + BAM.CD_de_d
    CDLde = BAM.CD_Lde_A*sin(BAM.CD_Lde_w*dB + BAM.CD_Lde_p) + BAM.CD_Lde_z + BAM.CD_Lde_d
    CDde2 = BAM.CD_de2_A*sin(BAM.CD_de2_w*dB + BAM.CD_de2_p) + BAM.CD_de2_z + BAM.CD_de2_d
    #
    Cl0 = BAM.Cl_0_A*sin(BAM.Cl_0_w*dB + BAM.Cl_0_p) + BAM.Cl_0_z + BAM.Cl_0_d
    Cla = BAM.Cl_a_A*sin(BAM.Cl_a_w*dB + BAM.Cl_a_p) + BAM.Cl_a_z + BAM.Cl_a_d
    Clb = BAM.Cl_b_A*sin(BAM.Cl_b_w*dB + BAM.Cl_b_p) + BAM.Cl_b_z + BAM.Cl_b_d
    Clp = BAM.Cl_p_A*sin(BAM.Cl_p_w*dB + BAM.Cl_p_p) + BAM.Cl_p_z + BAM.Cl_p_d
    Clq = BAM.Cl_q_A*sin(BAM.Cl_q_w*dB + BAM.Cl_q_p) + BAM.Cl_q_z + BAM.Cl_q_d
    Clr = BAM.Cl_r_A*sin(BAM.Cl_r_w*dB + BAM.Cl_r_p) + BAM.Cl_r_z + BAM.Cl_r_d
    ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p) + BAM.Cl_Lr_z + BAM.Cl_Lr_d
    Clda = BAM.Cl_da_A*sin(BAM.Cl_da_w*dB + BAM.Cl_da_p) + BAM.Cl_da_z + BAM.Cl_da_d
    Clde = 0.0        *sin(BAM.Cl_de_w*dB + BAM.Cl_de_p) + BAM.Cl_de_z + BAM.Cl_de_d
    #
    Cm0 = BAM.Cm_0_A*sin(BAM.Cm_0_w*dB + BAM.Cm_0_p) + BAM.Cm_0_z + BAM.Cm_0_d
    Cma = BAM.Cm_a_A*sin(BAM.Cm_a_w*dB + BAM.Cm_a_p) + BAM.Cm_a_z + BAM.Cm_a_d
    Cmb = BAM.Cm_b_A*sin(BAM.Cm_b_w*dB + BAM.Cm_b_p) + BAM.Cm_b_z + BAM.Cm_b_d
    Cmp = BAM.Cm_p_A*sin(BAM.Cm_p_w*dB + BAM.Cm_p_p) + BAM.Cm_p_z + BAM.Cm_p_d
    Cmq = BAM.Cm_q_A*sin(BAM.Cm_q_w*dB + BAM.Cm_q_p) + BAM.Cm_q_z + BAM.Cm_q_d
    Cmr = BAM.Cm_r_A*sin(BAM.Cm_r_w*dB + BAM.Cm_r_p) + BAM.Cm_r_z + BAM.Cm_r_d
    Cmda = 0.0        *sin(BAM.Cm_da_w*dB + BAM.Cm_da_p) + 0.0         + 0.0        
    Cmde = BAM.Cm_de_A*sin(BAM.Cm_de_w*dB + BAM.Cm_de_p) + BAM.Cm_de_z + BAM.Cm_de_d
    #
    Cn0 = BAM.Cn_0_A*sin(BAM.Cn_0_w*dB + BAM.Cn_0_p) + BAM.Cn_0_z + BAM.Cn_0_d
    Cna = BAM.Cn_a_A*sin(BAM.Cn_a_w*dB + BAM.Cn_a_p) + BAM.Cn_a_z + BAM.Cn_a_d
    Cnb = BAM.Cn_b_A*sin(BAM.Cn_b_w*dB + BAM.Cn_b_p) + BAM.Cn_b_z + BAM.Cn_b_d
    Cnp = BAM.Cn_p_A*sin(BAM.Cn_p_w*dB + BAM.Cn_p_p) + BAM.Cn_p_z + BAM.Cn_p_d
    CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p) + BAM.Cn_Lp_z + BAM.Cn_Lp_d
    Cnq = BAM.Cn_q_A*sin(BAM.Cn_q_w*dB + BAM.Cn_q_p) + BAM.Cn_q_z + BAM.Cn_q_d
    Cnr = BAM.Cn_r_A*sin(BAM.Cn_r_w*dB + BAM.Cn_r_p) + BAM.Cn_r_z + BAM.Cn_r_d
    Cnda = BAM.Cn_da_A*sin(BAM.Cn_da_w*dB + BAM.Cn_da_p) + BAM.Cn_da_z + BAM.Cn_da_d
    CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p) + BAM.Cn_Lda_z + BAM.Cn_Lda_d
    Cnde = BAM.Cn_de_A*sin(BAM.Cn_de_w*dB + BAM.Cn_de_p) + BAM.Cn_de_z + BAM.Cn_de_d
    #
    A_CL1 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) \
        + (BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p))*a
    z_CL1 = BAM.CL_0_z + BAM.CL_0_d + (BAM.CL_a_z + BAM.CL_a_d)*a
    A_ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p)
    z_ClLr = BAM.Cl_Lr_z + BAM.Cl_Lr_d
    A_CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p)
    z_CnLp = BAM.Cn_Lp_z + BAM.Cn_Lp_d
    A_CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p)
    z_CnLda = BAM.Cn_Lda_z + BAM.Cn_Lda_d

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr + A_ClLr*A_CL1 + z_ClLr*A_CL1 + A_ClLr*z_CL1 + z_ClLr*z_CL1)*rbar \
        + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta \
        + (Cnp + A_CnLp*A_CL1 + z_CnLp*A_CL1 + A_CnLp*z_CL1 + z_CnLp*z_CL1)*pbar \
        + Cnq*qbar + Cnr*rbar \
        + (Cnda + A_CnLda*A_CL1 + z_CnLda*A_CL1 + A_CnLda*z_CL1 + z_CnLda*z_CL1)*da \
        + Cnde*de)
    # return Cn

    # stall
    if use_stall:
        S_M = 7.0
        S_ab = np.deg2rad(45.0)
        # determine flat plate forces and moment
        CLplate = 2.*np.sign(a)*np.sin(a)**2.*np.cos(a)
        CDplate = 2.*np.sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = np.exp(-S_M*(a-S_ab))
        expMplu = np.exp(S_M*(a+S_ab))
        sig = (1. + expMmin + expMplu)/(1. + expMmin)/(1. + expMplu)

        # add stall effects
        CL = (1. - sig)*CL + sig*CLplate
        CD = (1. - sig)*CD + sig*CDplate
        Cm = (1. - sig)*Cm + sig*Cmplate
    
    

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx


def nonlinear_noC_noSpCl(self,t,x,
    is_controlled=True,given_control=False,u="o",
    use_stall=True,force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) + BAM.CL_0_z + BAM.CL_0_d
    CLa = BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p) + BAM.CL_a_z + BAM.CL_a_d
    CLb = BAM.CL_b_A*sin(BAM.CL_b_w*dB + BAM.CL_b_p) + BAM.CL_b_z + BAM.CL_b_d
    CLp = BAM.CL_p_A*sin(BAM.CL_p_w*dB + BAM.CL_p_p) + BAM.CL_p_z + BAM.CL_p_d
    CLq = BAM.CL_q_A*sin(BAM.CL_q_w*dB + BAM.CL_q_p) + BAM.CL_q_z + BAM.CL_q_d
    CLr = BAM.CL_r_A*sin(BAM.CL_r_w*dB + BAM.CL_r_p) + BAM.CL_r_z + BAM.CL_r_d
    CLda = BAM.CL_da_A*sin(BAM.CL_da_w*dB + BAM.CL_da_p) + BAM.CL_da_z + BAM.CL_da_d
    CLde = BAM.CL_de_A*sin(BAM.CL_de_w*dB + BAM.CL_de_p) + BAM.CL_de_z + BAM.CL_de_d
    #
    CS0 = BAM.CS_0_A*sin(BAM.CS_0_w*dB + BAM.CS_0_p) + BAM.CS_0_z + BAM.CS_0_d
    CSa = BAM.CS_a_A*sin(BAM.CS_a_w*dB + BAM.CS_a_p) + BAM.CS_a_z + BAM.CS_a_d
    CSb = BAM.CS_b_A*sin(BAM.CS_b_w*dB + BAM.CS_b_p) + BAM.CS_b_z + BAM.CS_b_d
    CSp = BAM.CS_p_A*sin(BAM.CS_p_w*dB + BAM.CS_p_p) + BAM.CS_p_z + BAM.CS_p_d
    CSLp = BAM.CS_Lp_A*sin(BAM.CS_Lp_w*dB + BAM.CS_Lp_p) + BAM.CS_Lp_z + BAM.CS_Lp_d
    CSq = BAM.CS_q_A*sin(BAM.CS_q_w*dB + BAM.CS_q_p) + BAM.CS_q_z + BAM.CS_q_d
    CSr = BAM.CS_r_A*sin(BAM.CS_r_w*dB + BAM.CS_r_p) + BAM.CS_r_z + BAM.CS_r_d
    CSda = BAM.CS_da_A*sin(BAM.CS_da_w*dB + BAM.CS_da_p) + BAM.CS_da_z + BAM.CS_da_d
    CSde = BAM.CS_de_A*sin(BAM.CS_de_w*dB + BAM.CS_de_p) + BAM.CS_de_z + BAM.CS_de_d
    #
    CD0 = BAM.CD_0_A*sin(BAM.CD_0_w*dB + BAM.CD_0_p) + BAM.CD_0_z + BAM.CD_0_d
    CDL = BAM.CD_L_A*sin(BAM.CD_L_w*dB + BAM.CD_L_p) + BAM.CD_L_z + BAM.CD_L_d
    CDL2 = BAM.CD_L2_A*sin(BAM.CD_L2_w*dB + BAM.CD_L2_p) + BAM.CD_L2_z + BAM.CD_L2_d
    CDS = BAM.CD_S_A*sin(BAM.CD_S_w*dB + BAM.CD_S_p) + BAM.CD_S_z + BAM.CD_S_d
    CDS2 = BAM.CD_S2_A*sin(BAM.CD_S2_w*dB + BAM.CD_S2_p) + BAM.CD_S2_z + BAM.CD_S2_d
    CDp = BAM.CD_p_A*sin(BAM.CD_p_w*dB + BAM.CD_p_p) + BAM.CD_p_z + BAM.CD_p_d
    CDSp = BAM.CD_Sp_A*sin(BAM.CD_Sp_w*dB + BAM.CD_Sp_p) + BAM.CD_Sp_z + BAM.CD_Sp_d
    CDq = BAM.CD_q_A*sin(BAM.CD_q_w*dB + BAM.CD_q_p) + BAM.CD_q_z + BAM.CD_q_d
    CDLq = BAM.CD_Lq_A*sin(BAM.CD_Lq_w*dB + BAM.CD_Lq_p) + BAM.CD_Lq_z + BAM.CD_Lq_d
    CDL2q = BAM.CD_L2q_A*sin(BAM.CD_L2q_w*dB + BAM.CD_L2q_p) + BAM.CD_L2q_z + BAM.CD_L2q_d
    CDr = BAM.CD_r_A*sin(BAM.CD_r_w*dB + BAM.CD_r_p) + BAM.CD_r_z + BAM.CD_r_d
    CDSr = BAM.CD_Sr_A*sin(BAM.CD_Sr_w*dB + BAM.CD_Sr_p) + BAM.CD_Sr_z + BAM.CD_Sr_d
    CDda = BAM.CD_da_A*sin(BAM.CD_da_w*dB + BAM.CD_da_p) + BAM.CD_da_z + BAM.CD_da_d
    CDSda = BAM.CD_Sda_A*sin(BAM.CD_Sda_w*dB + BAM.CD_Sda_p) + BAM.CD_Sda_z + BAM.CD_Sda_d
    CDde = BAM.CD_de_A*sin(BAM.CD_de_w*dB + BAM.CD_de_p) + BAM.CD_de_z + BAM.CD_de_d
    CDLde = BAM.CD_Lde_A*sin(BAM.CD_Lde_w*dB + BAM.CD_Lde_p) + BAM.CD_Lde_z + BAM.CD_Lde_d
    CDde2 = BAM.CD_de2_A*sin(BAM.CD_de2_w*dB + BAM.CD_de2_p) + BAM.CD_de2_z + BAM.CD_de2_d
    #
    Cl0 = 0.00      *sin(BAM.Cl_0_w*dB + BAM.Cl_0_p) + BAM.Cl_0_z + BAM.Cl_0_d
    Cla = 0.00      *sin(BAM.Cl_a_w*dB + BAM.Cl_a_p) + BAM.Cl_a_z + BAM.Cl_a_d
    Clb = BAM.Cl_b_A*sin(BAM.Cl_b_w*dB + BAM.Cl_b_p) + BAM.Cl_b_z + BAM.Cl_b_d
    Clp = 0.00      *sin(BAM.Cl_p_w*dB + BAM.Cl_p_p) + BAM.Cl_p_z + BAM.Cl_p_d
    Clq = BAM.Cl_q_A*sin(BAM.Cl_q_w*dB + BAM.Cl_q_p) + BAM.Cl_q_z + BAM.Cl_q_d
    Clr = BAM.Cl_r_A*sin(BAM.Cl_r_w*dB + BAM.Cl_r_p) + BAM.Cl_r_z + BAM.Cl_r_d
    ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p) + BAM.Cl_Lr_z + BAM.Cl_Lr_d
    Clda = BAM.Cl_da_A*sin(BAM.Cl_da_w*dB + BAM.Cl_da_p) + BAM.Cl_da_z + BAM.Cl_da_d
    Clde = 0.0        *sin(BAM.Cl_de_w*dB + BAM.Cl_de_p) + BAM.Cl_de_z + BAM.Cl_de_d
    #
    Cm0 = BAM.Cm_0_A*sin(BAM.Cm_0_w*dB + BAM.Cm_0_p) + BAM.Cm_0_z + BAM.Cm_0_d
    Cma = BAM.Cm_a_A*sin(BAM.Cm_a_w*dB + BAM.Cm_a_p) + BAM.Cm_a_z + BAM.Cm_a_d
    Cmb = BAM.Cm_b_A*sin(BAM.Cm_b_w*dB + BAM.Cm_b_p) + BAM.Cm_b_z + BAM.Cm_b_d
    Cmp = BAM.Cm_p_A*sin(BAM.Cm_p_w*dB + BAM.Cm_p_p) + BAM.Cm_p_z + BAM.Cm_p_d
    Cmq = BAM.Cm_q_A*sin(BAM.Cm_q_w*dB + BAM.Cm_q_p) + BAM.Cm_q_z + BAM.Cm_q_d
    Cmr = BAM.Cm_r_A*sin(BAM.Cm_r_w*dB + BAM.Cm_r_p) + BAM.Cm_r_z + BAM.Cm_r_d
    Cmda = 0.0        *sin(BAM.Cm_da_w*dB + BAM.Cm_da_p) + 0.0         + 0.0        
    Cmde = BAM.Cm_de_A*sin(BAM.Cm_de_w*dB + BAM.Cm_de_p) + BAM.Cm_de_z + BAM.Cm_de_d
    #
    Cn0 = BAM.Cn_0_A*sin(BAM.Cn_0_w*dB + BAM.Cn_0_p) + BAM.Cn_0_z + BAM.Cn_0_d
    Cna = BAM.Cn_a_A*sin(BAM.Cn_a_w*dB + BAM.Cn_a_p) + BAM.Cn_a_z + BAM.Cn_a_d
    Cnb = BAM.Cn_b_A*sin(BAM.Cn_b_w*dB + BAM.Cn_b_p) + BAM.Cn_b_z + BAM.Cn_b_d
    Cnp = BAM.Cn_p_A*sin(BAM.Cn_p_w*dB + BAM.Cn_p_p) + BAM.Cn_p_z + BAM.Cn_p_d
    CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p) + BAM.Cn_Lp_z + BAM.Cn_Lp_d
    Cnq = BAM.Cn_q_A*sin(BAM.Cn_q_w*dB + BAM.Cn_q_p) + BAM.Cn_q_z + BAM.Cn_q_d
    Cnr = BAM.Cn_r_A*sin(BAM.Cn_r_w*dB + BAM.Cn_r_p) + BAM.Cn_r_z + BAM.Cn_r_d
    Cnda = BAM.Cn_da_A*sin(BAM.Cn_da_w*dB + BAM.Cn_da_p) + BAM.Cn_da_z + BAM.Cn_da_d
    CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p) + BAM.Cn_Lda_z + BAM.Cn_Lda_d
    Cnde = BAM.Cn_de_A*sin(BAM.Cn_de_w*dB + BAM.Cn_de_p) + BAM.Cn_de_z + BAM.Cn_de_d
    #
    A_CL1 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) \
        + (BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p))*a
    z_CL1 = BAM.CL_0_z + BAM.CL_0_d + (BAM.CL_a_z + BAM.CL_a_d)*a
    A_ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p)
    z_ClLr = BAM.Cl_Lr_z + BAM.Cl_Lr_d
    A_CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p)
    z_CnLp = BAM.Cn_Lp_z + BAM.Cn_Lp_d
    A_CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p)
    z_CnLda = BAM.Cn_Lda_z + BAM.Cn_Lda_d

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr + A_ClLr*A_CL1 + z_ClLr*A_CL1 + A_ClLr*z_CL1 + z_ClLr*z_CL1)*rbar \
        + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta \
        + (Cnp + 0.00       + 0.00       + 0.00       + z_CnLp*z_CL1)*pbar \
        + Cnq*qbar + Cnr*rbar \
        + (Cnda + 0.00       + 0.00       + 0.00       + z_CnLda*z_CL1)*da \
        + Cnde*de)
    # return Cn

    # stall
    if use_stall:
        S_M = 7.0
        S_ab = np.deg2rad(45.0)
        # determine flat plate forces and moment
        CLplate = 2.*np.sign(a)*np.sin(a)**2.*np.cos(a)
        CDplate = 2.*np.sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = np.exp(-S_M*(a-S_ab))
        expMplu = np.exp(S_M*(a+S_ab))
        sig = (1. + expMmin + expMplu)/(1. + expMmin)/(1. + expMplu)

        # add stall effects
        CL = (1. - sig)*CL + sig*CLplate
        CD = (1. - sig)*CD + sig*CDplate
        Cm = (1. - sig)*Cm + sig*Cmplate
    
    

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx


def nonlinear_noC_noSpClDi(self,t,x,
    is_controlled=True,given_control=False,u="o",
    use_stall=True,force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) + BAM.CL_0_z + BAM.CL_0_d
    CLa = BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p) + BAM.CL_a_z + BAM.CL_a_d
    CLb = BAM.CL_b_A*sin(BAM.CL_b_w*dB + BAM.CL_b_p) + BAM.CL_b_z + BAM.CL_b_d
    CLp = BAM.CL_p_A*sin(BAM.CL_p_w*dB + BAM.CL_p_p) + BAM.CL_p_z + BAM.CL_p_d
    CLq = BAM.CL_q_A*sin(BAM.CL_q_w*dB + BAM.CL_q_p) + BAM.CL_q_z + BAM.CL_q_d
    CLr = BAM.CL_r_A*sin(BAM.CL_r_w*dB + BAM.CL_r_p) + BAM.CL_r_z + BAM.CL_r_d
    CLda = BAM.CL_da_A*sin(BAM.CL_da_w*dB + BAM.CL_da_p) + BAM.CL_da_z + BAM.CL_da_d
    CLde = BAM.CL_de_A*sin(BAM.CL_de_w*dB + BAM.CL_de_p) + BAM.CL_de_z + BAM.CL_de_d
    #
    CS0 = BAM.CS_0_A*sin(BAM.CS_0_w*dB + BAM.CS_0_p) + BAM.CS_0_z + BAM.CS_0_d
    CSa = BAM.CS_a_A*sin(BAM.CS_a_w*dB + BAM.CS_a_p) + BAM.CS_a_z + BAM.CS_a_d
    CSb = BAM.CS_b_A*sin(BAM.CS_b_w*dB + BAM.CS_b_p) + BAM.CS_b_z + BAM.CS_b_d
    CSp = BAM.CS_p_A*sin(BAM.CS_p_w*dB + BAM.CS_p_p) + BAM.CS_p_z + BAM.CS_p_d
    CSLp = BAM.CS_Lp_A*sin(BAM.CS_Lp_w*dB + BAM.CS_Lp_p) + BAM.CS_Lp_z + BAM.CS_Lp_d
    CSq = BAM.CS_q_A*sin(BAM.CS_q_w*dB + BAM.CS_q_p) + BAM.CS_q_z + BAM.CS_q_d
    CSr = BAM.CS_r_A*sin(BAM.CS_r_w*dB + BAM.CS_r_p) + BAM.CS_r_z + BAM.CS_r_d
    CSda = BAM.CS_da_A*sin(BAM.CS_da_w*dB + BAM.CS_da_p) + BAM.CS_da_z + BAM.CS_da_d
    CSde = BAM.CS_de_A*sin(BAM.CS_de_w*dB + BAM.CS_de_p) + BAM.CS_de_z + BAM.CS_de_d
    #
    CD0 = BAM.CD_0_A*sin(BAM.CD_0_w*dB + BAM.CD_0_p) + BAM.CD_0_z + BAM.CD_0_d
    CDL = BAM.CD_L_A*sin(BAM.CD_L_w*dB + BAM.CD_L_p) + BAM.CD_L_z + BAM.CD_L_d
    CDL2 = BAM.CD_L2_A*sin(BAM.CD_L2_w*dB + BAM.CD_L2_p) + BAM.CD_L2_z + BAM.CD_L2_d
    CDS = BAM.CD_S_A*sin(BAM.CD_S_w*dB + BAM.CD_S_p) + BAM.CD_S_z + BAM.CD_S_d
    CDS2 = BAM.CD_S2_A*sin(BAM.CD_S2_w*dB + BAM.CD_S2_p) + BAM.CD_S2_z + BAM.CD_S2_d
    CDp = BAM.CD_p_A*sin(BAM.CD_p_w*dB + BAM.CD_p_p) + BAM.CD_p_z + BAM.CD_p_d
    CDSp = BAM.CD_Sp_A*sin(BAM.CD_Sp_w*dB + BAM.CD_Sp_p) + BAM.CD_Sp_z + BAM.CD_Sp_d
    CDq = BAM.CD_q_A*sin(BAM.CD_q_w*dB + BAM.CD_q_p) + BAM.CD_q_z + BAM.CD_q_d
    CDLq = BAM.CD_Lq_A*sin(BAM.CD_Lq_w*dB + BAM.CD_Lq_p) + BAM.CD_Lq_z + BAM.CD_Lq_d
    CDL2q = BAM.CD_L2q_A*sin(BAM.CD_L2q_w*dB + BAM.CD_L2q_p) + BAM.CD_L2q_z + BAM.CD_L2q_d
    CDr = BAM.CD_r_A*sin(BAM.CD_r_w*dB + BAM.CD_r_p) + BAM.CD_r_z + BAM.CD_r_d
    CDSr = BAM.CD_Sr_A*sin(BAM.CD_Sr_w*dB + BAM.CD_Sr_p) + BAM.CD_Sr_z + BAM.CD_Sr_d
    CDda = BAM.CD_da_A*sin(BAM.CD_da_w*dB + BAM.CD_da_p) + BAM.CD_da_z + BAM.CD_da_d
    CDSda = BAM.CD_Sda_A*sin(BAM.CD_Sda_w*dB + BAM.CD_Sda_p) + BAM.CD_Sda_z + BAM.CD_Sda_d
    CDde = BAM.CD_de_A*sin(BAM.CD_de_w*dB + BAM.CD_de_p) + BAM.CD_de_z + BAM.CD_de_d
    CDLde = BAM.CD_Lde_A*sin(BAM.CD_Lde_w*dB + BAM.CD_Lde_p) + BAM.CD_Lde_z + BAM.CD_Lde_d
    CDde2 = BAM.CD_de2_A*sin(BAM.CD_de2_w*dB + BAM.CD_de2_p) + BAM.CD_de2_z + BAM.CD_de2_d
    #
    Cl0 = 0.00      *sin(BAM.Cl_0_w*dB + BAM.Cl_0_p) + BAM.Cl_0_z + BAM.Cl_0_d
    Cla = 0.00      *sin(BAM.Cl_a_w*dB + BAM.Cl_a_p) + BAM.Cl_a_z + BAM.Cl_a_d
    Clb = 0.000     *sin(BAM.Cl_b_w*dB + BAM.Cl_b_p) + BAM.Cl_b_z + BAM.Cl_b_d
    Clp = 0.00      *sin(BAM.Cl_p_w*dB + BAM.Cl_p_p) + BAM.Cl_p_z + BAM.Cl_p_d
    Clq = BAM.Cl_q_A*sin(BAM.Cl_q_w*dB + BAM.Cl_q_p) + BAM.Cl_q_z + BAM.Cl_q_d
    Clr = BAM.Cl_r_A*sin(BAM.Cl_r_w*dB + BAM.Cl_r_p) + BAM.Cl_r_z + BAM.Cl_r_d
    ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p) + BAM.Cl_Lr_z + BAM.Cl_Lr_d
    Clda = BAM.Cl_da_A*sin(BAM.Cl_da_w*dB + BAM.Cl_da_p) + BAM.Cl_da_z + BAM.Cl_da_d
    Clde = 0.0        *sin(BAM.Cl_de_w*dB + BAM.Cl_de_p) + BAM.Cl_de_z + BAM.Cl_de_d
    #
    Cm0 = 0.000     *sin(BAM.Cm_0_w*dB + BAM.Cm_0_p) + BAM.Cm_0_z + BAM.Cm_0_d
    Cma = 0.000     *sin(BAM.Cm_a_w*dB + BAM.Cm_a_p) + BAM.Cm_a_z + BAM.Cm_a_d
    Cmb = BAM.Cm_b_A*sin(BAM.Cm_b_w*dB + BAM.Cm_b_p) + BAM.Cm_b_z + BAM.Cm_b_d
    Cmp = BAM.Cm_p_A*sin(BAM.Cm_p_w*dB + BAM.Cm_p_p) + BAM.Cm_p_z + BAM.Cm_p_d
    Cmq = BAM.Cm_q_A*sin(BAM.Cm_q_w*dB + BAM.Cm_q_p) + BAM.Cm_q_z + BAM.Cm_q_d
    Cmr = BAM.Cm_r_A*sin(BAM.Cm_r_w*dB + BAM.Cm_r_p) + BAM.Cm_r_z + BAM.Cm_r_d
    Cmda = 0.0        *sin(BAM.Cm_da_w*dB + BAM.Cm_da_p) + 0.0         + 0.0        
    Cmde = BAM.Cm_de_A*sin(BAM.Cm_de_w*dB + BAM.Cm_de_p) + BAM.Cm_de_z + BAM.Cm_de_d
    #
    Cn0 = 0.000     *sin(BAM.Cn_0_w*dB + BAM.Cn_0_p) + BAM.Cn_0_z + BAM.Cn_0_d
    Cna = 0.000     *sin(BAM.Cn_a_w*dB + BAM.Cn_a_p) + BAM.Cn_a_z + BAM.Cn_a_d
    Cnb = BAM.Cn_b_A*sin(BAM.Cn_b_w*dB + BAM.Cn_b_p) + BAM.Cn_b_z + BAM.Cn_b_d
    Cnp = BAM.Cn_p_A*sin(BAM.Cn_p_w*dB + BAM.Cn_p_p) + BAM.Cn_p_z + BAM.Cn_p_d
    CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p) + BAM.Cn_Lp_z + BAM.Cn_Lp_d
    Cnq = BAM.Cn_q_A*sin(BAM.Cn_q_w*dB + BAM.Cn_q_p) + BAM.Cn_q_z + BAM.Cn_q_d
    Cnr = BAM.Cn_r_A*sin(BAM.Cn_r_w*dB + BAM.Cn_r_p) + BAM.Cn_r_z + BAM.Cn_r_d
    Cnda = BAM.Cn_da_A*sin(BAM.Cn_da_w*dB + BAM.Cn_da_p) + BAM.Cn_da_z + BAM.Cn_da_d
    CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p) + BAM.Cn_Lda_z + BAM.Cn_Lda_d
    Cnde = BAM.Cn_de_A*sin(BAM.Cn_de_w*dB + BAM.Cn_de_p) + BAM.Cn_de_z + BAM.Cn_de_d
    #
    A_CL1 = BAM.CL_0_A*sin(BAM.CL_0_w*dB + BAM.CL_0_p) \
        + (BAM.CL_a_A*sin(BAM.CL_a_w*dB + BAM.CL_a_p))*a
    z_CL1 = BAM.CL_0_z + BAM.CL_0_d + (BAM.CL_a_z + BAM.CL_a_d)*a
    A_ClLr = BAM.Cl_Lr_A*sin(BAM.Cl_Lr_w*dB + BAM.Cl_Lr_p)
    z_ClLr = BAM.Cl_Lr_z + BAM.Cl_Lr_d
    A_CnLp = BAM.Cn_Lp_A*sin(BAM.Cn_Lp_w*dB + BAM.Cn_Lp_p)
    z_CnLp = BAM.Cn_Lp_z + BAM.Cn_Lp_d
    A_CnLda = BAM.Cn_Lda_A*sin(BAM.Cn_Lda_w*dB + BAM.Cn_Lda_p)
    z_CnLda = BAM.Cn_Lda_z + BAM.Cn_Lda_d

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr + A_ClLr*A_CL1 + 0.000       + A_ClLr*z_CL1 + z_ClLr*z_CL1)*rbar \
        + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta \
        + (Cnp + 0.00       + 0.00       + 0.00       + z_CnLp*z_CL1)*pbar \
        + Cnq*qbar + Cnr*rbar \
        + (Cnda + 0.00       + 0.00       + 0.00       + z_CnLda*z_CL1)*da \
        + Cnde*de)
    # return Cn

    # stall
    if use_stall:
        S_M = 7.0
        S_ab = np.deg2rad(45.0)
        # determine flat plate forces and moment
        CLplate = 2.*np.sign(a)*np.sin(a)**2.*np.cos(a)
        CDplate = 2.*np.sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = np.exp(-S_M*(a-S_ab))
        expMplu = np.exp(S_M*(a+S_ab))
        sig = (1. + expMmin + expMplu)/(1. + expMmin)/(1. + expMplu)

        # add stall effects
        CL = (1. - sig)*CL + sig*CLplate
        CD = (1. - sig)*CD + sig*CDplate
        Cm = (1. - sig)*Cm + sig*Cmplate
    
    

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Im1 = self.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx


def simple_model_euler2_lyap(self,t,x,
    is_controlled=True,given_control=False,u="o",
    force_control_to_inputs=False):

    # get control
    u,inputs = self._get_control(t,x,is_controlled,given_control,u,
        force_control_to_inputs = force_control_to_inputs)

    # disturbance model
    ## INTSTATE
    V = (x[0]**2. + x[1]**2. + x[2]**2.)**0.5
    Du,Dv,Dw,Dp,Dq,Dr = self.get_disturbance(t,V)
    Vg = [Du,Dv,Dw]
    Wg = [Dp,Dq,Dr]

    # get aero forces
    ###########################################################################
    # Fx,Fy,Fz,Mx,My,Mz,g = self._aerodynamics(x,inputs,Vg=Vg,Wg=Wg)
    # def _aerodynamics(self,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0],
    #     is_trim=False,is_VAB_format=False):
    is_trim=False
    is_VAB_format=False
    # aero conditions
    ## INTSTATE
    if is_VAB_format:
        Vu = x[0]*cos(x[1])*cos(x[2]) + Vg[0]
        Vv = x[0]*sin(x[2])           + Vg[1]
        Vw = x[0]*sin(x[1])*cos(x[2]) + Vg[2]
    else:
        Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = asin(Vv/V)
    _,g,_,_,rho,sos = self.stdatm(-x[8])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*self.bw/2./V
    qbar = (x[4]+Wg[1])*self.cw/2./V
    rbar = (x[5]+Wg[2])*self.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    ###########################################################################
    # # use aircraft model
    # aero_results = self.aero_model.aero_results(*[
    #     a,b,pbar,qbar,rbar,ail,ele,rud,
    #     self.is_compressible,M,self.use_anderson,self.has_stall
    # ])
    # # add in errors
    # [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + self.FM_errors[i]) \
    #     for i in range(len(aero_results))]
    
    # BAM
    BAM = self.aero_model
    dB = rud
    de = ele
    da = ail
    alpha = a
    beta = b

    # coefficients
    #
    CL0 = BAM._CL0(dB)
    CLa = BAM._CL_alpha(dB)
    CLb = BAM._CL_beta(dB)
    CLp = BAM._CL_pbar(dB)
    CLq = BAM._CL_qbar(dB)
    CLr = BAM._CL_rbar(dB)
    CLda = BAM._CL_da(dB)
    CLde = BAM._CL_de(dB)
    #
    CS0 = BAM._CS0(dB)
    CSa = BAM._CS_alpha(dB)
    CSb = BAM._CS_beta(dB)
    CSp = BAM._CS_pbar(dB)
    CSLp = BAM._CS_Lpbar(dB)
    CSq = BAM._CS_qbar(dB)
    CSr = BAM._CS_rbar(dB)
    CSda = BAM._CS_da(dB)
    CSde = BAM._CS_de(dB)
    #
    CD0 = BAM._CD0(dB)
    CDL = BAM._CD_L(dB)
    CDL2 = BAM._CD_L2(dB)
    CDS = BAM._CD_S(dB)
    CDS2 = BAM._CD_S2(dB)
    CDp = BAM._CD_pbar(dB)
    CDSp = BAM._CD_Spbar(dB)
    CDq = BAM._CD_qbar(dB)
    CDLq = BAM._CD_Lqbar(dB)
    CDL2q = BAM._CD_L2qbar(dB)
    CDr = BAM._CD_rbar(dB)
    CDSr = BAM._CD_Srbar(dB)
    CDda = BAM._CD_da(dB)
    CDSda = BAM._CD_Sda(dB)
    CDde = BAM._CD_de(dB)
    CDLde = BAM._CD_Lde(dB)
    CDde2 = BAM._CD_de2(dB)
    #
    Cl0 = BAM._Cl0(dB)
    Cla = BAM._Cl_alpha(dB)
    Clb = BAM._Cl_beta(dB)
    Clp = BAM._Cl_pbar(dB)
    Clq = BAM._Cl_qbar(dB)
    Clr = BAM._Cl_rbar(dB)
    ClLr = BAM._Cl_Lrbar(dB)
    Clda = BAM._Cl_da(dB)
    Clde = BAM._Cl_de(dB)
    #
    Cm0 = BAM._Cm0(dB)
    Cma = BAM._Cm_alpha(dB)
    Cmb = BAM._Cm_beta(dB)
    Cmp = BAM._Cm_pbar(dB)
    Cmq = BAM._Cm_qbar(dB)
    Cmr = BAM._Cm_rbar(dB)
    Cmda = BAM._Cm_da(dB)
    Cmde = BAM._Cm_de(dB)
    #
    Cn0 = BAM._Cn0(dB)
    Cna = BAM._Cn_alpha(dB)
    Cnb = BAM._Cn_beta(dB)
    Cnp = BAM._Cn_pbar(dB)
    CnLp = BAM._Cn_Lpbar(dB)
    Cnq = BAM._Cn_qbar(dB)
    Cnr = BAM._Cn_rbar(dB)
    Cnda = BAM._Cn_da(dB)
    CnLda = BAM._Cn_Lda(dB)
    Cnde = BAM._Cn_de(dB)
    # ASSUME ClLr  = CnLp  = CnLda = Cmda = Cmp = Cnp = Clr = 0
    ClLr  = CnLp  = CnLda = Cmda = Cmp = Cnp = Clr = 0.0

    # def _CL(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    CL1 = CL0 + CLa*alpha
    CL = (CL1 + CLb*beta + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de)
    # return CL

    # def _CS(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS = (CS0 + CSa*alpha + CSb*beta + \
        (CSp + CSLp*CL1)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de)
    # return CS

    # def _CD(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    CS1 = CS0 + CSb*beta
    CD = (CD0 + CDL*CL1 + CDL2*CL1**2 + CDS*CS1 + CDS2*CS1**2 + \
        (CDp + CDSp*CS1)*pbar +(CDq + CDLq*CL1 + CDL2q*CL1**2)*qbar + \
        (CDr + CDSr*CS1)*rbar + (CDda + CDSda*CS1)*da + 
        (CDde + CDLde*CL1)*de + CDde2*de**2)
    # return CD

    # def _Cl(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cl = (Cl0 + Cla*alpha + Clb*beta + Clp*pbar + Clq*qbar + \
        (Clr + ClLr*CL1)*rbar + Clda*da + Clde*de)
    # return Cl

    # def _Cm(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    Cm = (Cm0 + Cma*alpha + Cmb*beta + Cmp*pbar + \
        Cmq*qbar + Cmr*rbar + Cmda*da + Cmde*de)
    # return Cm

    # def _Cn(BAM, alpha, beta, pbar, qbar, rbar, da, de, dB):
    # CL1 = BAM._CL0(dB) + BAM._CL_alpha(dB)*alpha
    Cn = (Cn0 + Cna*alpha + Cnb*beta + (Cnp + CnLp*CL1)*pbar + \
        Cnq*qbar + Cnr*rbar + (Cnda + CnLda*CL1)*da + Cnde*de)
    # return Cn

    ###########################################################################

    # thrust forces
    ## INTSTATE
    T = self._get_thrust_model(thr,thr,-x[8],V,M,is_trim)
    FP = T  * self.T_dir
    MP = [
        FP[2] * self.T_loc[1] - FP[1] * self.T_loc[2],
        FP[0] * self.T_loc[2] - FP[2] * self.T_loc[0],
        FP[1] * self.T_loc[0] - FP[0] * self.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * self.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * self.bw
    My = MP[1] + Cm * dynF * self.cw
    Mz = MP[2] + Cn * dynF * self.bw
    # # SAL ay
    # self._SAL_ay = Fy/self.inertia_model.W

    # add in CG effects
    cg = self.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    # return Fx,Fy,Fz,Mx,My,Mz,g

    ###########################################################################

    # read in mass properties
    W = self.inertia_model.W
    # ASSUME Ixx, Iyy, Izz fixed at dB; Ixy = Ixz = Iyz = 0
    # Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(u[3])
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = self.inertia_model.inertia_results(0.0)
    Ixy = Ixz = Iyz = 0.0
    # Im1 = self.inertia_model.inverse_tensor(u[3])
    ## ASSUME
    Im1 = np.array([
        [1./Ixx,   0.0,   0.0],
        [   0.0,1./Iyy,   0.0],
        [   0.0,   0.0,1./Izz]
    ])
    hx,hy,hz = self.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[9],x[10],x[11] # self._euler_angles(x) # 
    cp = cos(ph); sp = sin(ph)
    ct = cos(th); st = sin(th)
    cs = cos(ps); ss = sin(ps)

    # u,v,w
    ## INTSTATE
    dx[0] = g/W*Fx - g*st    + r*Vv - q*Vw
    dx[1] = g/W*Fy + g*sp*ct + p*Vw - r*Vu
    dx[2] = g/W*Fz + g*cp*ct + q*Vu - p*Vv

    # rhs for p,q,r
    pq = p*q; pr = p*r; qr = q*r
    p2, q2, r2 = p**2., q**2., r**2.
    rhs0 = r*hy - q*hz + Mx + (Iyy-Izz)*qr + Iyz*(q2-r2) + Ixz*pq - Ixy*pr
    rhs1 = p*hz - r*hx + My + (Izz-Ixx)*pr + Ixz*(r2-p2) + Ixy*qr - Iyz*pq
    rhs2 = q*hx - p*hy + Mz + (Ixx-Iyy)*pq + Ixy*(p2-q2) + Iyz*pr - Ixz*qr
    # p,q,r
    ## INTSTATE
    dx[3] = Im1[0][0]*rhs0 + Im1[0][1]*rhs1 + Im1[0][2]*rhs2
    dx[4] = Im1[1][0]*rhs0 + Im1[1][1]*rhs1 + Im1[1][2]*rhs2
    dx[5] = Im1[2][0]*rhs0 + Im1[2][1]*rhs1 + Im1[2][2]*rhs2
    
    # x,y,z
    mat = [
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ]
    ## INTSTATE
    dx[6] = mat[0][0]*Vu + mat[0][1]*Vv + mat[0][2]*Vw
    dx[7] = mat[1][0]*Vu + mat[1][1]*Vv + mat[1][2]*Vw
    dx[8] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 9] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[10] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r
    dx[11] = mat[2][0]*p + mat[2][1]*q + mat[2][2]*r

    # actuator dynamics
    if self.order == 1:
        dx[12:16] = self._actuation_dynamics(x,u)
    elif self.order == 2:
        dx[12:20] = self._actuation_dynamics(x,u)
    
    # integral states
    r = self._get_reference(t)[self.xPi]
    dx[self.xIi] = r - x[self.xPi]*1.

    return dx



if __name__ == "__main__":

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 2./15.,
        "transparent" : False,
        "format" : "pdf"
    }

    # bire aero err dict
    bire_errs = { }
    # bire inertia
    bire_iner = { }
    # base make f16 aero err dict
    base_errs = { }
    # base inertia
    base_iner = { }
    # bire FM
    bire_FM_errs = [0.25]*6
    # base FM
    base_FM_errs = [0.25]*6

    flight_conditions = {
        "T1" : { "m" : 0.2 , "h" :  1000., "V" : 222., "Re" : 15641000. },
        "T2" : { "m" : 0.19, "h" : 15000., "V" : 201., "Re" :  9919000. },
        "C1" : { "m" : 0.8 , "h" :  1000., "V" : 890., "Re" : 62563000. },
        "C2" : { "m" : 0.6 , "h" : 15000., "V" : 634., "Re" : 31324000. },
        "C3" : { "m" : 0.8 , "h" : 30000., "V" : 796., "Re" : 25828000. }
    }
    f1 = "C2"
    f2 = "C3"
    state_threshold = [
        10., 15., 15.,
        20., 10., 10., # 
        1., 1., 50., # 
        25., 10., 1., # 
        5., 5., 5., 0.05
    ]

    run_base = {
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 15., # 120., # 
        # "time_step" : 0.01,
        "initial_mach" : flight_conditions[f1]["m"]*1.,
        "initial_altitude" : flight_conditions[f1]["h"]*1.,
        "trim_bank" : 30.0, # 75.5224878, # 78.463041, # 80.4059318, # 60.0, # 
        "trim_climb" : 0.0,
        "start_climbing" : False,
        "end_gs_climbing" : False,
        "final_mach" : flight_conditions[f1]["m"]*1., # f2]["m"]*1., # 
        "final_altitude" : flight_conditions[f1]["h"]*1., # f2]["h"]*1., # 
        "t_gain_schedule" : 0.1, # 90., # 
        "gain_steps" : 30,
        "cut_mine" : True,
        "save_data" : True,
        "statistical" : True,
        "has_turbulence" : False, # True, # 
        "turbulence_setting" : "light", # "moderate", # "severe", # 
        "has_model_error" : False, # True, # 
        "aero_model_errors" : base_errs,
        "inertia_model_errors" : base_iner,
        "FM_errors" : base_FM_errs,
        "state_threshold" : state_threshold, # 64.0, # 
        "rerandomize_turbulence" : True,
        "mrrr" : [6,7,11], # 0,1,2,8,9,10,
        # "mrrc" : [2,3], # [3], # [2], # 
        "get_aero_FM" : True,
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "name_end" : "_" + f1 + "_BK_3"#4_wSd" # _1e1pqr" #+ "_" + name
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire = {**run_base}
    run_bire["aero_model_errors"] = bire_errs
    run_bire["inertia_model_errors"] = bire_iner
    run_bire["FM_errors"] = bire_FM_errs

    # run bire simulation
    bire_dict["simulation"]["use_quaternions"] = False
    # bire_dict["simulation"]["include_compressibility"] = False
    # bire_dict["simulation"]["include_stall"] = False
    bire_dict["initial"]["mach"] = 0.6
    bire_dict["initial"]["altitude[ft]"] = 15000.
    # bire_dict["initial"].pop("mach")
    # bire_dict["initial"]["airspeed[ft/s]"] = 100.
    # bire_dict["initial"]["altitude[ft]"] = 0.
    # print(bire_dict)
    bire = Aircraft(bire_dict)
    tf = 15.
    num = 1500
    ts = np.linspace(0.,tf,num=num)

    u_fun = lambda t : np.array([
        bire.u_trim[0] + np.sin(t)*bire.max_da/2.,
        bire.u_trim[1] + np.sin(t)*bire.max_de/2.,
        bire.u_trim[2] + np.sin(t)*bire.max_dr/1.5,
        bire.u_trim[3] + np.sin(t)*bire.max_tau/50.
    ])
    # print(np.rad2deg(bire.max_da/2.), np.rad2deg(bire.max_de/2.), 
    #     np.rad2deg(bire.max_dr/1.5), bire.max_tau/50.)
    # quit()
    dynamics = lambda t,x : bire._dynamics(t,x,True,True,u_fun(t))
    xs_nlin = odeint(dynamics,bire.x_trim,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # linear dynamics
    bire._build_controller(report=False,
                save_matrices=False,mrrr=None,mrrc=None,
                include_stall_derivatives=False,
                run_freq=False)
    bire._get_dynamics = bire._linear_euler_dynamics
    xs_lin = odeint(dynamics,bire.x_trim,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # kinda_nonlinear dynamics
    use_stall = True
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : true_nonlinear_no_comp(bire,t,x,True,True,u_fun(t),use_stall)
    xs_knln = odeint(dynamics,x0,ts,tfirst=True,
                    ).T
    use_stall = False
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : true_nonlinear_no_comp(bire,t,x,True,True,u_fun(t),use_stall)
    xs_knlS = odeint(dynamics,x0,ts,tfirst=True,
                    ).T
    
    # simple dynamics
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : nonlinear_noC_psdolin(bire,t,x,True,True,u_fun(t),use_stall)
    xs_psln = odeint(dynamics,x0,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # simple dynamics
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : nonlinear_noC_noSp(bire,t,x,True,True,u_fun(t),use_stall)
    xs_nS__ = odeint(dynamics,x0,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # simple dynamics
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : nonlinear_noC_noSpCl(bire,t,x,True,True,u_fun(t),use_stall)
    xs_nSC_ = odeint(dynamics,x0,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # simple dynamics
    x0 = bire.x_trim*1.0
    dynamics = lambda t,x : nonlinear_noC_noSpClDi(bire,t,x,True,True,u_fun(t),use_stall)
    xs_nSCD = odeint(dynamics,x0,ts,tfirst=True,
                    # atol=1e-10,rtol=1e-10
                    ).T
    
    # # simple dynamics
    # x0 = bire.x_trim*1.0
    # # Vu = x0[0]*1.; Vv = x0[1]*1.; Vw = x0[2]*1.
    # # x0[2] = np.arctan2(Vw,Vu)
    # # x0[0] = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    # # x0[1] = np.arcsin(Vv/x0[0])
    # dynamics = lambda t,x : simple_model_euler(bire,t,x,True,True,u_fun(t))
    # dynamics = lambda t,x : simple_model_euler2_lyap(bire,t,x,True,True,u_fun(t))
    # # dynamics = lambda t,x : true_nonlinear_no_comp(bire,t,x,True,True,u_fun(t))
    # xs_simp = odeint(dynamics,x0,ts,tfirst=True,
    #                 # atol=1e-10,rtol=1e-10
    #                 ).T
    # # # reconvert back to Vx,Vy,Vz
    # # Vs = xs_simp[0]*1.; afs = xs_simp[2]*1.; bs = xs_simp[1]*1.
    # # xs_simp[0] = Vs*np.cos(afs)*np.cos(bs)
    # # xs_simp[1] = Vs*np.sin(bs)
    # # xs_simp[2] = Vs*np.sin(afs)*np.cos(bs)
    # # xs_simp = xs_nlin*1.

    
    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 6.0
    plt.rcParams["axes.labelsize"] = 6.0
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.dpi'] = 300.0

    ylabels = [
        ["Vxb","Vyb","Vzb"],
        ["p","q","r"],
        ["xf","yf","zf"],
        ["phi","theta","psi"]
    ]
    
    figs = []
    axs = []
    for i in range(4):
        fig,ax = plt.subplots(3,1,figsize=(3.25,3.5),sharex=True,constrained_layout=True)
        figs.append(fig)
        axs.append(ax)
    for i in range(4):
        for j in range(3):
            axs[i][j].plot(ts,xs_nlin[j+i*3],"k-",marker="o",markevery=100,ms=2.0,label="NL")
            axs[i][j].plot(ts,xs_lin [j+i*3],"r",label="L")
            axs[i][j].plot(ts,xs_knln[j+i*3],"m",label="NL no C")
            axs[i][j].plot(ts,xs_knlS[j+i*3],"m--",label="NL no C S")
            axs[i][j].plot(ts,xs_psln[j+i*3],"b",label="NL no C S psdolin")
            # axs[i][j].plot(ts,xs_nS__[j+i*3],"y",label=r"NL no C S $\spadesuit$")
            # axs[i][j].plot(ts,xs_nSC_[j+i*3],"b",label=r"NL no C S $\spadesuit$ $\clubsuit$")
            # axs[i][j].plot(ts,xs_nSCD[j+i*3],"c",label=r"NL no C S $\spadesuit$ $\clubsuit$ $\diamondsuit$")
            # axs[i][j].plot(ts,xs_simp[j+i*3],"b",label="smp nonlin")
            axs[i][j].set_xlim(ts[0],ts[-1])
            axs[i][j].set_yscale("symlog")
            axs[i][j].set_ylabel(ylabels[i][j])
            if j == 0:
                axs[i][j].legend()
            elif j == 2:
                axs[i][j].set_xlabel("time, [s]")
    # save figs
    save_figs = True # False # 
    save_folder = "da_de_CcTCc_determinant/simpler_system_plots/"
    save_names = ["pslin_vels","pslin_rats","pslin_posn","pslin_ornt"]
    file_type = "png" # "pdf" # 
    dpi = 300.0
    if save_figs:
        for i in range(4):
            figs[i].savefig(save_folder+save_names[i]+"."+file_type,
                format=file_type,dpi=dpi)
    
    if False: # True # 
        plt.show()
    else:
        plt.close("all")

    

