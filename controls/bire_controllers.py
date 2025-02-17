import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.axes3d as ax3
from matplotlib.animation import FuncAnimation
from numpy import sign, matmul as mm
from datetime import datetime
import control as co
from scipy.linalg import block_diag
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d, interpn
from scipy.optimize import curve_fit,minimize,minimize_scalar,newton
from scipy.io import savemat, loadmat
from scipy.signal import tf2zpk as scipy_tf2zpk
# from math import pi, sin, cos, tan, exp, asin, acos, atan, atan2
from numpy import pi, sin, cos, tan, exp, arcsin as asin, arccos as acos, arctan as atan, arctan2 as atan2
from std_atm import stdatm_english
from quat import quat_mult, euler_2_quat, quat_2_euler, quat_norm, body_2_fixed, fixed_2_body, eulerdot_2_quatdot, quatdot_2_eulerdot
from linearization import linearization as lin,Anderson_correction_der_coeff,Anderson_correction_der_M

from controller_simulation import Aircraft,run_single_simulation, \
    monte_carlo_perturbations, report_latex, report_eigprops, rep2D,BIREAero


class NonlinearDynamicInversionAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True

        self.use_transformed_controls = False # True # 
        self.include_stall_ders_in_LM = True # False # 
        self.include_alt_ders_in_LM = True # False # 
        self.LDI_on_det_sign_flip = False # True # 
        self.first_LQDI_step = True
        
        self.u_til_next_update = self.u_trim*1.0

        I = np.eye(3)
        Z = np.zeros((3,3))
        A = np.block([[Z,I,Z],[Z,Z,I],[Z,Z,Z]])
        B = np.block([[Z],[Z],[I]])
        q_z1 = 0.1; q_z2 = 1000.0; q_z3 = 10.0
        qqr1 = qpr1 = -q_z1/5.; qqr2 = qpr2 = -q_z2/5.; qqr3 = qpr3 = -q_z3/5.
        q1 = np.array([[q_z1,0.0,qpr1],[0.0,q_z1,qqr1],[qpr1,qqr1,q_z1]])
        q2 = np.array([[q_z2,0.0,qpr2],[0.0,q_z2,qqr2],[qpr2,qqr2,q_z2]])
        q3 = np.array([[q_z3,0.0,qpr3],[0.0,q_z3,qqr3],[qpr3,qqr3,q_z3]])
        Q = block_diag(q1,q2,q3)
        r_d1 = 0.1; r_d2 = 0.1; r_d3 = 1.0
        r_aB = 0.01; r_eB = 0.05
        R = np.array([[r_d1,0.0,r_aB],[0.0,r_d2,r_eB],[r_aB,r_eB,r_d3]])
        K,_,K_eigs = co.lqr(A,B,Q,R)
        self.K_FB_2 = K
        # print(K)
        # rep2D(K,"K",decimals=15,np_array=True)
        report_latex(Q,"Q")
        report_latex(R,"R")
        report_latex(K,"K_{lqr}")
        report_latex(K_eigs,r"\lambda_{cl \, lqr}")

        self.Ndets = []
        self.Ndet = 1.0
        self.mindet = 1.0e100


        self.vI = np.zeros((3,))

        # zt = 0.7
        # wn = 10.0
        # pv = 1.0
        # k1 = pv*wn**2. # inte
        # k2 = wn**2. + 2.*wn*zt*pv# e
        # k3 = 2.*wn*zt + pv # edot
        # K1 = np.diag([k1]*3)#; K1[0,2] = k1**2.
        # K2 = np.diag([k2]*3)#; K2[0,2] = k2**2. # self.Lin_Model.KI
        # K3 = np.diag([k3]*3)#; K3[0,2] = k3**2. # self.Lin_Model.K
        # K = np.block([K1,K2,K3])
        # K_eigs,_ = np.linalg.eig(A - np.matmul(B,K))
        # report_latex(K,"K_{3ord}")
        # report_latex(K_eigs,r"\lambda_{cl \, 3ord}")
        # # quit()
    
    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # feedback linearization
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                V_xb    = x_euler[ 0] #  self.x_trim_euler[ 0] # 
                V_yb    = x_euler[ 1] #  self.x_trim_euler[ 1] # 
                V_zb    = x_euler[ 2] #  self.x_trim_euler[ 2] # 
                p       = x_euler[ 3] #  self.x_trim_euler[ 3] # 
                q       = x_euler[ 4] #  self.x_trim_euler[ 4] # 
                r       = x_euler[ 5] #  self.x_trim_euler[ 5] # 
                z_f     = x_euler[ 8] #  self.x_trim_euler[ 8] # 
                da      = x_euler[12] #  self.x_trim_euler[12] # 
                de      = x_euler[13] #  self.x_trim_euler[13] # 
                dB      = x_euler[14] #  self.x_trim_euler[14] # 
                tau     = x_euler[15] #  self.x_trim_euler[15] # 
                # da,de,dB,tau = self.u_til_next_update*1.0
                epI     = x_euler[self.xIi_eul[1]]
                eqI     = x_euler[self.xIi_eul[2]]
                erI     = x_euler[self.xIi_eul[3]]
                # Derived Quantities
                V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
                V_xb_ss = self.x_trim[0]
                V_yb_ss = self.x_trim[1]
                V_zb_ss = self.x_trim[2]
                V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
                aero = 0
                if aero == 0:
                    a   = np.arctan2(V_zb,V_xb)
                    b   = asin(V_yb/V_tot)
                    V = V_tot
                    V_xb_in = V_xb*1.; V_yb_in = V_yb*1.; V_zb_in = V_zb*1.
                elif aero == 1:
                    a   = 0.0
                    b   = 0.0
                    V = V_tot
                    V_xb_in = V_tot*1.; V_yb_in = 0.0; V_zb_in = 0.0
                elif aero == 2:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V     = V_tot
                    V_xb_in = V*np.cos(a)*np.cos(b)
                    V_yb_in = V          *np.sin(b)
                    V_zb_in = V*np.sin(a)*np.cos(b)
                elif aero == 3:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V = V_xb
                    V_xb_in = V_xb*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                elif aero == 4:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V = V_ss
                    V_xb_in = V_xb_ss*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                #
                # pull in matrices from linearization code
                x_in = x_euler*1.0
                x_in[0:3] = [V_xb_in,V_yb_in,V_zb_in]
                u_in = np.array([da,de,dB,tau])
                self.Lin_Model.report = False
                self.Lin_Model.include_stall_ders = self.include_stall_ders_in_LM
                self.Lin_Model.include_alt_ders = self.include_alt_ders_in_LM
                A,B = self.Lin_Model.build_jacobians(x_in, u_in,self.cgshift)
                # # #
                if self.use_transformed_controls:
                    # copy B
                    Bo = B*1.0
                    # trim values
                    dm = de*cos(dB)
                    dn = de*sin(dB)
                    # transform
                    dedm = dm/(dm**2.+dn**2.)**0.5
                    dedn = dn/(dm**2.+dn**2.)**0.5
                    dBdm = -dn/dm**2./(1. + dn**2./dm**2.)
                    dBdn =  1./dm    /(1. + dn**2./dm**2.)
                    # apply
                    B[:,1] = Bo[:,1]*dedm + Bo[:,2]*dBdm
                    B[:,2] = Bo[:,1]*dedn + Bo[:,2]*dBdn
                # # #
                if self.constant_density:
                    _,g,_,_,rho,sos = self.stdatm(self.H0)
                else:
                    _,g,_,_,rho,sos = self.stdatm(-z_f)
                pbar = p*self.bw/2./V
                qbar = q*self.cw/2./V
                rbar = r*self.bw/2./V
                params = a, b, pbar, qbar, rbar, da, de, dB
                # pull out parts of state
                # preliminaries
                Sw = self.Sw
                bw = self.bw
                cw = self.cw
                h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
                hmat = np.array([
                    [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
                Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                    self.inertia_model.inertia_results(dB)
                Iinv  = self.inertia_model.inverse_tensor(dB)
                Qdyn = 0.5*rho*V**2.*Sw
                G = Qdyn*np.diag([bw,cw,bw])
                Imult = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                Sigma = np.matmul(hmat,[p,q,r]) + Imult
                #
                # values for later use
                Ca = cos(a); Sa = sin(a)
                Cb = cos(b); Sb = sin(b)
                #
                M = V / sos
                #
                # get forces and moments at the specified condition
                [CL, CS, CD, Cl, Cm, Cn] = \
                    self.aero_model.aero_results(*params,M=M,**{
                        "compressible" : self.is_compressible,
                        "use_Anderson" : self.use_anderson,
                        "enforce_stall" : self.has_stall
                })
                #
                # # thrust state derivatives
                dfdw = A[3:6,3:6]
                #
                dfdzf = A[3:6,8:9]
                #
                dfdy = A[3:6,0:3]
                #
                # evaluate at condition for Mx, My, Mz
                T = self.aero_model.Prop.get_thrust(tau,-z_f,V)
                #
                Fx = Qdyn*(  CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
                Fy = Qdyn*(          CS   *Cb - CD   *Sb)
                Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
                #
                dfdd = B[3:6,0:3]
                #
                S = np.diag([self.s_da,self.s_de,self.s_dr])
                N = dfddS = np.matmul(dfdd,S)
                # print(t,np.linalg.det(B[3:6,0:3])) # ,np.linalg.det(N))
                Nadj = np.array([
                    [ (N[1,1]*N[2,2]-N[1,2]*N[2,1]),
                        -(N[0,1]*N[2,2]-N[0,2]*N[2,1]),
                         (N[0,1]*N[1,2]-N[0,2]*N[1,1])],
                    [-(N[1,0]*N[2,2]-N[1,2]*N[2,0]),
                         (N[0,0]*N[2,2]-N[0,2]*N[2,0]),
                        -(N[0,0]*N[1,2]-N[0,2]*N[1,0])],
                    [ (N[1,0]*N[2,1]-N[1,1]*N[2,0]),
                        -(N[0,0]*N[2,1]-N[0,1]*N[2,0]),
                         (N[0,0]*N[1,1]-N[0,1]*N[1,0])]
                ])
                Ndet = N[0,0]*(N[1,1]*N[2,2] - N[1,2]*N[2,1]) \
                    -  N[0,1]*(N[1,0]*N[2,2] - N[1,2]*N[2,0]) \
                    +  N[0,2]*(N[1,0]*N[2,1] - N[1,1]*N[2,0])
                dfddSinv = Nadj/Ndet
                # print()
                # print("x =",x)
                # print("t = {:>+8.3f}, Ndet = {:>+12.3e}".format(t,Ndet))
                #
                ph,th,ps = x_euler[9],x_euler[10],x_euler[11]
                cp = cos(ph); sp = sin(ph)
                ct = cos(th); st = sin(th)
                # cs = cos(ps); ss = sin(ps)
                # u,v,w
                ## INTSTATE
                W = self.inertia_model.W
                dy = np.array([
                    g/W*Fx - g*st    + r*V_yb - q*V_zb,
                    g/W*Fy + g*sp*ct + p*V_zb - r*V_xb,
                    g/W*Fz + g*cp*ct + q*V_xb - p*V_yb
                ])
                dzf = np.matmul([[-st, sp*ct, cp*ct]],[V_xb,V_yb,V_zb])
                # vectors
                Mxyz = np.matmul(G,[ Cl, Cm, Cn ]) + np.array([
                    Fy * self.cgshift[2] - Fz * self.cgshift[1],
                    Fz * self.cgshift[0] - Fx * self.cgshift[2],
                    Fx * self.cgshift[1] - Fy * self.cgshift[0]])
                z3 = np.matmul(Iinv,Mxyz + Sigma) # omega dot
                z2 = x_euler[self.Lin_Model.Cslice] - ref
                z1 = np.array([epI,eqI,erI])
                delta = x_euler[12:15]*1.
                # # #
                if self.use_transformed_controls:
                    delta[1] = dm; delta[2] = dn
                # # #
                #
                K = self.K_FB_2
                z = np.concatenate((z1,z2,z3))
                v_cl = - np.matmul(K,z)
                self.v_cl = v_cl*1.0
                self.z3_cl = z3*1.0
                self.v_Fx = Fx
                self.v_Fy = Fy
                self.v_Fz = Fz
                self.v_Mx = Mxyz[0]
                self.v_My = Mxyz[1]
                self.v_Mz = Mxyz[2]
                self.v_CL = CL
                self.v_CS = CS
                self.v_CD = CD
                self.v_Cl = Cl
                self.v_Cm = Cm
                self.v_Cn = Cn
                self.v_params = params
                rest = - np.matmul(dfdw,z3) + np.matmul(dfddS,delta) \
                    + v_cl - np.matmul(dfdy,dy) - np.matmul(dfdzf,dzf)
                v = np.matmul(dfddSinv,rest)
                # # # #
                if self.use_transformed_controls:
                    dm = v[1]; dn = v[2]
                    dB = atan2(dn,dm)
                    if dB < -np.pi/2.:
                        # print("-np.pi/2.")
                        e2s = e1s = abs(dB) // np.pi
                        mult = +1.0
                    elif dB > +np.pi/2.:
                        # print("+np.pi/2.")
                        e2s = e1s = abs(dB) // np.pi
                        mult = -1.0
                    else: # if True:#
                        e2s = -1
                        e1s = 1
                        mult = +1.0
                    dB += mult*(e2s + 1)*np.pi
                    de = (-1.0)**(e1s + 1)*(dm**2. + dn**2.)**0.5 # 
                    v[1] = de; v[2] = dB
                # # # # #
                # #
                # if v[2] > np.pi:
                #     v[2] -= 2.0*np.pi
                # elif v[2] < -np.pi:
                #     v[2] += 2.0*np.pi
                # #
                # ##  ##  ##  ##  ##  ##  ##  ##
                # de = v[1]
                # dB = v[2]
                # if np.isnan(dB) or abs(dB) > 1.0e5:
                #     dB = self.u_trim[2]
                #     de = self.u_trim[1]
                # while dB >  np.pi/2.0:
                #     dB -= np.pi
                #     de *= -1.0
                # while dB < -np.pi/2.0:
                #     dB += np.pi
                #     de *= -1.0
                # #
                # v[1] = de
                # v[2] = dB
                # ##  ##  ##  ##  ##  ##  ##  ##

                # dynamic inversion!!!
                if self.LDI_on_det_sign_flip and self.first_LQDI_step:
                    # build system, solve LQR problem
                    A_tr = self.Lin_Model.A_min
                    B_tr = self.Lin_Model.B_min
                    Z = np.zeros((3,3))
                    I = np.eye(3)
                    A = np.block([[Z,I],[Z,A_tr]])
                    B = np.block([[Z],[B_tr]])
                    # Q = np.diag([1.0e+0,1.0e+0,1.0e+0]+[1.0e+0,1.0e+0,1.0e+0])
                    Q = np.diag([2.0e+1,2.0e+2,2.0e+2]+[2.0e+3,2.0e+4,2.0e+4])
                    Q[0,2] = Q[2,0] = 1.0e+1
                    Q[1,2] = Q[2,1] = 1.0e+2
                    # Q[3,5] = Q[5,3] = 1.0e+2
                    # Q[4,5] = Q[5,4] = 1.0e+3
                    N = np.array([
                        [ 0.0e+0, 0.0e+0, 2.0e+0],
                        [ 0.0e+0, 0.0e+0, 2.0e+0],
                        [ 0.0e+0, 0.0e+0, 1.0e+0],
                        [ 0.0e+0, 0.0e+0, 2.0e+1],
                        [ 0.0e+0, 0.0e+0, 2.0e+1],
                        [ 0.0e+0, 0.0e+0, 1.0e+1]
                    ])
                    R = np.diag([1.0e+0,1.0e+0,1.0e+0])
                    K,_,K_eigs = co.lqr(A,B,Q,R,N)
                    self.KI_DI,self.KP_DI = K[:,0:3],K[:,3:6]
                    # print(K)
                    print(self.KI_DI)
                    print(self.KP_DI)
                    print(K_eigs)
                    self.first_LQDI_step = False
                    self.Ndet = Ndet*1.0
                #
                use_linear = self.Ndet*Ndet <= 0.0
                self.Ndet = Ndet*1.0
                if abs(self.Ndet) < self.mindet:
                    self.mindet = abs(self.Ndet)
                ## # ## # ## # ## # ## # ## # ## # ## # ## #
                if self.LDI_on_det_sign_flip and use_linear:
                    w  = np.array([  p,  q,  r])
                    eI = np.array([epI,eqI,erI])
                    x_trim = self.x_trim
                    dref = ref - x_trim[3:6]
                    e = w - ref
                    A = self.Lin_Model.A_min
                    Binv = self.Lin_Model.Binv_min
                    
                    v = - np.matmul(self.Lin_Model.K,e) \
                        - np.matmul(self.Lin_Model.KI,eI)
                    
                    delta = np.matmul(Binv,
                                      - np.matmul(A,e) - np.matmul(A,dref) + v)
                    vcom = delta + self.u_trim[0:3]
                    ## # ## # ## # ## # ## # ## # ## # ## # ## #
                else:
                    vcom = v*1.0
                # # # 
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((vcom,[tcom]))
                # print("u =",u)


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs

    def _add_to_delta_x0(self,delta_x0):
        # delta_beta_deg = 0.6
        # vx_trim = self.x_trim[0]; vy_trim = self.x_trim[1]; vz_trim = self.x_trim[2]
        # V_trim = (vx_trim**2.0 + vy_trim**2.0 + vz_trim**2.0)**0.5
        # a_trim = atan2(vz_trim,vx_trim)
        # b_trim = asin(vy_trim/V_trim)
        # Vnew = V_trim
        # anew = a_trim
        # bnew = b_trim + np.deg2rad(delta_beta_deg)
        # delta_x0[0] = - vx_trim + Vnew*cos(anew)*cos(bnew)
        # delta_x0[1] = - vy_trim + Vnew*sin(bnew)
        # delta_x0[2] = - vz_trim + Vnew*sin(anew)*cos(bnew)
        return delta_x0

    def _empty_call_after_get_control(self):
        self.Ndets.append(self.Ndet*1.0)
        return

    def returns_zero(self,tarr,xarr,uarr,subdict,xticks,perc_zoom,predir,
        format,savedict,save_plot):
        # plot Ndets over time
        print("Min det(N) =",self.mindet)
        Ndets = np.array([self.Ndets[0]] + self.Ndets)
        #
        # # Det plot
        Ndet_fig, Ndet_axs = plt.subplots(1,1,**subdict)
        # axis labels, legends
        altcol = "0.5"
        Ndet_fig.supxlabel(r"Time, s")
        Ndet_fig.supylabel(r"det($N$)")
        # xticks
        Ndet_axs.set_xticks(ticks=xticks)
        # grid, axis labels, legends
        Ndet_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
        #
        Ndet_axs.plot(tarr,Ndets,c="k",ls="-" )
        #
        Ndet_axs.set_yscale("symlog")
        #
        Ndet_axs.set_xlim((0.,perc_zoom*self.tf))
        # Ndet_axs.set_ylim((1.0e-3,))
        if save_plot:
            Ndet_fig.savefig(predir+"determinant."+format,**savedict)
        plt.close(Ndet_fig)
        #
        return 0


class DynamicInversionAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.LQDI = False # True # 
        self.LQR_CL = True # False # 
        self.first_LQDI_step = True # False # 
        #
        if not self.LQDI:
            self.first_LQDI_step = False

        if self.LQR_CL:
            I = np.eye(3)
            Z = np.zeros((3,3))
            A = np.block([[Z,I],[Z,Z]])
            B = np.block([[Z],[I]])
            Q = np.diag([1.0e+1,1.0e+3,2.0e+2] + [1.0e+3,1.0e+5,2.0e+4])
            # Q[0,2] = Q[2,0] = 1.0e+0
            Q[1,2] = Q[2,1] = -1.0e+2
            # Q[3,5] = Q[5,3] = 1.0e+0
            # Q[4,5] = Q[5,4] = 1.0e+0
            R = np.diag([1.0e+0,1.0e+0] + [1.0e+3])
            K,_,K_eigs = co.lqr(A,B,Q,R)
            self.KI_DI,self.KP_DI = K[:,0:3],K[:,3:6]
            # print(K)
            print(self.KI_DI)
            print(self.KP_DI)
            print(K_eigs)
            # quit()
    
    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # dynamic inversion!!!
                if self.first_LQDI_step:
                    # build system, solve LQR problem
                    A_tr = self.Lin_Model.A_min
                    B_tr = self.Lin_Model.B_min
                    Z = np.zeros((3,3))
                    I = np.eye(3)
                    A = np.block([[Z,I],[Z,A_tr]])
                    B = np.block([[Z],[B_tr]])
                    # Q = np.diag([1.0e+0,1.0e+0,1.0e+0]+[1.0e+0,1.0e+0,1.0e+0])
                    Q = np.diag([2.0e+1,2.0e+2,2.0e+2]+[2.0e+3,2.0e+4,2.0e+4])
                    Q[0,2] = Q[2,0] = 1.0e+1
                    Q[1,2] = Q[2,1] = 1.0e+2
                    # Q[3,5] = Q[5,3] = 1.0e+2
                    # Q[4,5] = Q[5,4] = 1.0e+3
                    N = np.array([
                        [ 0.0e+0, 0.0e+0, 2.0e+0],
                        [ 0.0e+0, 0.0e+0, 2.0e+0],
                        [ 0.0e+0, 0.0e+0, 1.0e+0],
                        [ 0.0e+0, 0.0e+0, 2.0e+1],
                        [ 0.0e+0, 0.0e+0, 2.0e+1],
                        [ 0.0e+0, 0.0e+0, 1.0e+1]
                    ])
                    R = np.diag([1.0e+0,1.0e+0,1.0e+0])
                    K,_,K_eigs = co.lqr(A,B,Q,R,N)
                    self.KI_DI,self.KP_DI = K[:,0:3],K[:,3:6]
                    # print(K)
                    print(self.KI_DI)
                    print(self.KP_DI)
                    print(K_eigs)
                    self.first_LQDI_step = False
                    # quit()

                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                epI     = x_euler[self.xIi_eul[1]]
                eqI     = x_euler[self.xIi_eul[2]]
                erI     = x_euler[self.xIi_eul[3]]
                w  = np.array([  p,  q,  r])
                eI = np.array([epI,eqI,erI])
                x_trim = self.x_trim
                dref = ref - x_trim[3:6]
                e = w - ref
                A = self.Lin_Model.A_min
                Binv = self.Lin_Model.Binv_min
                # A = np.block([[A,np.zeros((3,3))],[np.zeros((3,3)),np.eye(3)]])
                # B = np.block([[self.Lin_Model.B_min],[np.zeros((3,3))]])
                # Q = np.diag([1.]*3 + [100.]*3)
                # R = np.diag([1.]*2 + [100.])
                # K,_,_ = co.lqr(A,B,Q,R)
                # print(K)
                # quit()
                
                if not self.LQR_CL and not self.LQDI:
                    v = - np.matmul(self.Lin_Model.K,e) \
                        - np.matmul(self.Lin_Model.KI,eI)
                else:
                    v = - np.matmul(self.KP_DI,e) \
                        - np.matmul(self.KI_DI,eI)
                
                if self.LQDI:
                    delta = np.matmul(self.Lin_Model.nBiA_min,dref) + v
                else:
                    delta = np.matmul(Binv, - np.matmul(A,e) - np.matmul(A,dref) + v)
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((delta + self.u_trim[0:3],[tcom]))


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class ControlAllocationMomentAssignmentAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.bool_plot_limit_inputs = True # False # 
        self.pseudo_inverse_method = True # False # 
        self.do_line_search = False # True # # if false, use prev calc
        self.ls_dB_lim = 45.0 # 30.0 # 
        self.ls_num = 21 # 11 # 
        self.opt_tol = 1.0e-12
        self.opt_max_iter = 50 # 1000 # 10 # 100 # 
        self.report_error_threshold = 1.0e10 # 1.0e-2 # 
        #
        self.scalar_options = ["Golden","Brent"]
        self.scipy_options = ["SLSQP","Nelder-Mead","trust-exact","BFGS"]
        line_search_options = ["Newton"] + self.scalar_options + self.scipy_options
        self.line_method = line_search_options[0] # "None" # [1] # [0] # 
        self.line_method = "Newton_Root"
        # self.add_tail_lag_eq = False # True # 
        # bire aero model for derivs
        self.dBAM = BIREAero(**self.aero_dict)
        self.dBAM.deriv = True
        self.ddBAM = BIREAero(**self.aero_dict)
        self.ddBAM._make_double_derivative_model()
        self.u_til_next_update = self.u_trim*1.0
        
        # use LQR to design v
        I = np.eye(3)
        Z = np.zeros((3,3))
        A = np.block([[Z,I],[Z,Z]])
        B = np.block([[Z],[I]])
        C = np.eye(6)
        # ## # # vv as
        Q = np.diag([1.0e+2,5.0e+2,5.0e+2] + [2.0e+0,5.0e+1,5.0e+1])
        Q[0,2] = Q[2,0] = 1.0e+2
        Q[1,2] = Q[2,1] = 3.5e+2
        Q[3,5] = Q[5,3] = 1.0e+0 # potential increase, was 1.0e+0
        Q[4,5] = Q[5,4] = 5.0e+0
        # I don't think I want correlation here
        Q[0,1] = Q[1,0] = 0.0e+00
        Q[3,4] = Q[4,3] = 0.0e+00
        # seem to have no effect
        Q[0,3] = Q[3,0] = 0.0e+00
        Q[1,4] = Q[4,1] = 0.0e+00
        Q[2,5] = Q[5,2] = 0.0e+00
        # e and int(e)
        Q[0,4] = Q[4,0] = 0.0e+0
        Q[0,5] = Q[5,0] = 0.0e+0 #
        Q[1,3] = Q[3,1] = 0.0e+0
        Q[1,5] = Q[5,1] = 0.0e+0 #
        # Q[2,3] = Q[3,2] = 1.0e+2 # # remove this, from as, now ar
        Q[2,4] = Q[4,2] = 0.0e+0 #
        #
        R = np.diag([1.0e+0,1.0e+0,2.5e+2])
        R[0,1] = R[1,0] = 0.0
        R[0,2] = R[2,0] = 0.0
        R[1,2] = R[2,1] = 5.0e-2
        # # # # # # # # # # vvvv OLD QR from optimization setup
        # Q = np.diag([8.4e+2] + [4.2e+3]*2 + [4.0e+0] + [4.0e+1]*2)
        # Q[0,2] = Q[2,0] = 1.0e2 # 5.0e2
        # Q[1,2] = Q[2,1] = 5.0e2
        # Q[3,5] = Q[5,3] = 1.0e+0 # 1.0e+1
        # Q[4,5] = Q[5,4] = 1.0e+1
        # R = np.diag([1.0e+0,1.0e+0,2.0e+3])
        # # # # # # # # # # ^^^^ OLD QR from optimization setup
        # # similar response to SMD system
        # Q = np.diag([4.2e+3]*3 + [4.0e+1]*3)
        # R = np.diag([1.0e+0,1.0e+0,1.0e+0])
        K,_,K_eigs = co.lqr(A,B,Q,R)
        self.KI_DI,self.KP_DI = K[:,0:3],K[:,3:6]
        # #
        K = np.block([self.KI_DI,self.KP_DI])
        # print(K)
        print("Keigs =",K_eigs)
        rep2D(self.KI_DI,"KI",decimals=3)# print("KI =",self.KI_DI)
        rep2D(self.KP_DI,"KP",decimals=3)# print("KP =",self.KP_DI)

        # search functions
        def delta_E_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # determine da, de
            Cs = np.array([Cls,Cms,Cns])
            Cc = np.array([[Clda,Clde],[Cmda,Cmde],[Cnda,Cnde]])
            Qdyn = 0.5*rho*V**2.*self.Sw
            G = Qdyn*np.diag([self.bw,self.cw,self.bw])
            GCs = mm(G,Cs)
            GCc = mm(G,Cc)
            dai,dei = mm(np.linalg.pinv(GCc),Md - GCs)
            M = GCs + mm(GCc,[dai,dei])
            Error = np.linalg.norm(M-Md)
            return dai,dei,Error
        
        def delta_E_fun_sum(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # determine da, de
            Cs = np.array([Cls,Cms,Cns])
            Cc = np.array([[Clda,Clde],[Cmda,Cmde],[Cnda,Cnde]])
            Qdyn = 0.5*rho*V**2.*self.Sw
            G = Qdyn*np.diag([self.bw,self.cw,self.bw])
            GCs = mm(G,Cs)
            GCc = mm(G,Cc)
            dai,dei = mm(np.linalg.pinv(GCc),Md - GCs)
            M = GCs + mm(GCc,[dai,dei])
            Error = np.sum(M-Md)
            return dai,dei,Error

        def delta_E_fun_sq(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            da,de,E = delta_E_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
            return da,de,E**2.0
        
        def delta_E_dE_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            # previously
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # determine da, de
            Cs = np.array([Cls,Cms,Cns])
            Cc = np.array([[Clda,Clde],[Cmda,Cmde],[Cnda,Cnde]])
            Qdyn = 0.5*rho*V**2.*self.Sw
            G = Qdyn*np.diag([self.bw,self.cw,self.bw])
            GCs = mm(G,Cs)
            GCc = mm(G,Cc)
            GCcp = np.linalg.pinv(GCc)
            dai,dei = mm(GCcp,Md - GCs)
            M = GCs + mm(GCc,[dai,dei])
            Error = np.linalg.norm(M-Md)
            # derivatives
            DAM = self.dBAM
            dCL1 = DAM._CL0(dBj) + DAM._CL_alpha(dBj)*a
            dCls = (DAM._Cl0(dBj) + DAM._Cl_alpha(dBj)*a +
                DAM._Cl_beta(dBj)*b + DAM._Cl_pbar(dBj)*pbar +
                DAM._Cl_qbar(dBj)*qbar +
                (DAM._Cl_rbar(dBj) + DAM._Cl_Lrbar(dBj)*CL1 + 
                BAM._Cl_Lrbar(dBj)*dCL1)*rbar)
            dClda = DAM._Cl_da(dBj)
            dClde = DAM._Cl_de(dBj)
            dCms = (DAM._Cm0(dBj) + DAM._Cm_alpha(dBj)*a +
                DAM._Cm_beta(dBj)*b + DAM._Cm_pbar(dBj)*pbar +
                DAM._Cm_qbar(dBj)*qbar + DAM._Cm_rbar(dBj)*rbar)
            dCmda = DAM._Cm_da(dBj)
            dCmde = DAM._Cm_de(dBj)
            dCns = (DAM._Cn0(dBj) + DAM._Cn_alpha(dBj)*a +
                DAM._Cn_beta(dBj)*b +
                (DAM._Cn_pbar(dBj) + DAM._Cn_Lpbar(dBj)*CL1 + 
                BAM._Cn_Lpbar(dBj)*dCL1)*pbar +
                DAM._Cn_qbar(dBj)*qbar + DAM._Cn_rbar(dBj)*rbar)
            dCnda = DAM._Cn_da(dBj) + DAM._Cn_Lda(dBj)*CL1 + \
                BAM._Cn_Lda(dBj)*dCL1
            dCnde = DAM._Cn_de(dBj)
            # determine da, de
            dCs = np.array([dCls,dCms,dCns])
            dCc = np.array([[dClda,dClde],[dCmda,dCmde],[dCnda,dCnde]])
            dGCs = mm(G,dCs)
            dGCc = mm(G,dCc)
            dA = dGCc; A = GCc; B = GCcp
            # The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate. Author(s): G. H. Golub and V. Pereyra. Source: SIAM Journal on Numerical Analysis, Vol. 10, No. 2 (Apr., 1973), pp. 413-432
            dGCcp = -mm(mm(B,dA),B) \
                + mm(mm(mm(B,B.T),dA.T),\
                (np.eye(3) - mm(A,B))) \
                + mm(mm(mm((np.eye(2) - mm(B,A)),\
                dA.T),B.T),B)
            Ddai,Ddei = mm(dGCcp,Md - GCs) + mm(GCcp,-dGCs)
            dM = dGCs + mm(dGCc,[dai,dei]) + mm(GCc,[Ddai,Ddei])
            dE = mm(dM.T,(M-Md))/Error
            return dai,dei,Error,dE
        
        def delta_E_dE_fun_sq(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            da,de,E,dE = delta_E_dE_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
            return da,de,E**2.0,2.0*dE*E
        
        def delta_E_dE_wE_fun_sq(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            # previously
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # determine da, de
            Cs = np.array([Cls,Cms,Cns])
            Cc = np.array([[Clda,Clde],[Cmda,Cmde],[Cnda,Cnde]])
            Qdyn = 0.5*rho*V**2.*self.Sw
            G = Qdyn*np.diag([self.bw,self.cw,self.bw])
            GCs = mm(G,Cs)
            GCc = mm(G,Cc)
            GCcp = np.linalg.pinv(GCc)
            dai,dei = mm(GCcp,Md - GCs)
            M = GCs + mm(GCc,[dai,dei])
            Error = np.linalg.norm(M-Md)**2.0
            # derivatives
            DAM = self.dBAM
            dCL1 = DAM._CL0(dBj) + DAM._CL_alpha(dBj)*a
            dCls = (DAM._Cl0(dBj) + DAM._Cl_alpha(dBj)*a +
                DAM._Cl_beta(dBj)*b + DAM._Cl_pbar(dBj)*pbar +
                DAM._Cl_qbar(dBj)*qbar +
                (DAM._Cl_rbar(dBj) + DAM._Cl_Lrbar(dBj)*CL1 + 
                BAM._Cl_Lrbar(dBj)*dCL1)*rbar)
            dClda = DAM._Cl_da(dBj)
            dClde = DAM._Cl_de(dBj)
            dCms = (DAM._Cm0(dBj) + DAM._Cm_alpha(dBj)*a +
                DAM._Cm_beta(dBj)*b + DAM._Cm_pbar(dBj)*pbar +
                DAM._Cm_qbar(dBj)*qbar + DAM._Cm_rbar(dBj)*rbar)
            dCmda = DAM._Cm_da(dBj)
            dCmde = DAM._Cm_de(dBj)
            dCns = (DAM._Cn0(dBj) + DAM._Cn_alpha(dBj)*a +
                DAM._Cn_beta(dBj)*b +
                (DAM._Cn_pbar(dBj) + DAM._Cn_Lpbar(dBj)*CL1 + 
                BAM._Cn_Lpbar(dBj)*dCL1)*pbar +
                DAM._Cn_qbar(dBj)*qbar + DAM._Cn_rbar(dBj)*rbar)
            dCnda = DAM._Cn_da(dBj) + DAM._Cn_Lda(dBj)*CL1 + \
                BAM._Cn_Lda(dBj)*dCL1
            dCnde = DAM._Cn_de(dBj)
            # determine da, de
            dCs = np.array([dCls,dCms,dCns])
            dCc = np.array([[dClda,dClde],[dCmda,dCmde],[dCnda,dCnde]])
            dGCs = mm(G,dCs)
            dGCc = mm(G,dCc)
            dA = dGCc; A = GCc; B = GCcp
            # The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate. Author(s): G. H. Golub and V. Pereyra. Source: SIAM Journal on Numerical Analysis, Vol. 10, No. 2 (Apr., 1973), pp. 413-432
            dGCcp = -mm(mm(B,dA),B) \
                + mm(mm(mm(B,B.T),dA.T),\
                (np.eye(3) - mm(A,B))) \
                + mm(mm(mm((np.eye(2) - mm(B,A)),\
                dA.T),B.T),B)
            Ddai,Ddei = mm(dGCcp,Md - GCs) + mm(GCcp,-dGCs)
            dM = dGCs + mm(dGCc,[dai,dei]) + mm(GCc,[Ddai,Ddei])
            dE = 2.0*mm(dM.T,(M-Md))
            # double derivatives
            WAM = self.ddBAM
            wCL1 = WAM._CL0(dBj) + WAM._CL_alpha(dBj)*a
            wCls = (WAM._Cl0(dBj) + WAM._Cl_alpha(dBj)*a +
                WAM._Cl_beta(dBj)*b + WAM._Cl_pbar(dBj)*pbar +
                WAM._Cl_qbar(dBj)*qbar +
                (WAM._Cl_rbar(dBj) + 
                WAM._Cl_Lrbar(dBj)*CL1 + 2.0*DAM._Cl_Lrbar(dBj)*dCL1 + 
                BAM._Cl_Lrbar(dBj)*wCL1)*rbar)
            wClda = WAM._Cl_da(dBj)
            wClde = WAM._Cl_de(dBj)
            wCms = (WAM._Cm0(dBj) + WAM._Cm_alpha(dBj)*a +
                WAM._Cm_beta(dBj)*b + WAM._Cm_pbar(dBj)*pbar +
                WAM._Cm_qbar(dBj)*qbar + WAM._Cm_rbar(dBj)*rbar)
            wCmda = WAM._Cm_da(dBj)
            wCmde = WAM._Cm_de(dBj)
            wCns = (WAM._Cn0(dBj) + WAM._Cn_alpha(dBj)*a +
                WAM._Cn_beta(dBj)*b +
                (WAM._Cn_pbar(dBj) + 
                WAM._Cn_Lpbar(dBj)*CL1 + 2.0*DAM._Cn_Lpbar(dBj)*dCL1 + 
                BAM._Cn_Lpbar(dBj)*wCL1)*pbar +
                WAM._Cn_qbar(dBj)*qbar + WAM._Cn_rbar(dBj)*rbar)
            wCnda = WAM._Cn_da(dBj) + \
                WAM._Cn_Lda(dBj)*CL1 + 2.0*DAM._Cn_Lda(dBj)*dCL1 + \
                BAM._Cn_Lda(dBj)*wCL1
            wCnde = WAM._Cn_de(dBj)
            # determine da, de
            wCs = np.array([wCls,wCms,wCns])
            wCc = np.array([[wClda,wClde],[wCmda,wCmde],[wCnda,wCnde]])
            wGCs = mm(G,wCs)
            wGCc = mm(G,wCc)
            wA = wGCc; dA = dGCc; A = GCc; dB = dGCcp; B = GCcp
            wGCcp = -mm(mm(dB,dA),B) - mm(mm(B,wA),B) - mm(mm(B,dA),dB) \
                + mm( mm(mm(dB,B.T),dA.T) + mm(mm(B,dB.T),dA.T) \
                + mm(mm(B,B.T),wA.T) ,np.eye(3) - mm(A,B)) \
                + mm( mm(mm(B,B.T),dA.T) , - mm(dA,B) - mm(A,dB) ) \
                + mm( - mm(dB,A) - mm(B,dA) , mm(mm(dA.T,B.T),B) ) \
                + mm( np.eye(2) - mm(B,A) , mm(mm(wA.T,B.T),B) \
                + mm(mm(dA.T,dB.T),B) + mm(mm(dA.T,B.T),dB) )
            Wdai,Wdei = mm(wGCcp,Md - GCs) + 2.0*mm(dGCcp,-dGCs) \
                + mm(GCcp,-wGCs)
            wM = wGCs + mm(wGCc,[dai,dei]) + 2.0*mm(dGCc,[Ddai,Ddei]) \
                + mm(GCc,[Wdai,Wdei])
            wE = 2.0*mm(wM.T,(M-Md)) + 2.0*mm(dM.T,dM)

            return dai,dei,Error,dE,wE
        
        def sine_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            # determine desired moment coefficients
            CMd_lref = Md/0.5/rho/V**2.0/self.Sw
            Cld = CMd_lref[0]#/self.bw
            Cmd = CMd_lref[1]#/self.cw
            Cnd = CMd_lref[2]#/self.bw

            # aero trig
            ca = cos(a); sa = sin(a)
            cb = cos(b); sb = sin(b)

            # determine moment constants
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            CS1 = BAM._CS0(dBj) + BAM._CS_beta(dBj)*b
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            # Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            # Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)

            # force constants
            CLs = (CL1 + BAM._CL_beta(dBj)*b + BAM._CL_pbar(dBj)*pbar +
                BAM._CL_qbar(dBj)*qbar + BAM._CL_rbar(dBj)*rbar)
            # CLda = BAM._CL_da(dBj)
            CLde = BAM._CL_de(dBj)
            CSs = (CS1 + BAM._CS_alpha(dBj)*a + 
                (BAM._CS_pbar(dBj) + BAM._CS_Lpbar(dBj)*CL1)*pbar +
                BAM._CS_qbar(dBj)*qbar + BAM._CS_rbar(dBj)*rbar)
            CSda = BAM._CS_da(dBj)
            CSde = BAM._CS_de(dBj)
            CDs = (BAM._CD0(dBj) + BAM._CD_L(dBj)*CL1 + BAM._CD_L2(dBj)*CL1**2 +
                BAM._CD_S(dBj)*CS1 + BAM._CD_S2(dBj)*CS1**2 +
                (BAM._CD_pbar(dBj) + BAM._CD_Spbar(dBj)*CS1)*pbar +
                (BAM._CD_qbar(dBj) + BAM._CD_Lqbar(dBj)*CL1 + 
                BAM._CD_L2qbar(dBj)*CL1**2)*qbar +
                (BAM._CD_rbar(dBj) + BAM._CD_Srbar(dBj)*CS1)*rbar)
            CDda = (BAM._CD_da(dBj) + BAM._CD_Sda(dBj)*CS1)
            CDde = (BAM._CD_de(dBj) + BAM._CD_Lde(dBj)*CL1) # dropped squared part

            # add to corresponding moments
            Cls  = self.bw*Cls
            Clda = self.bw*Clda
            Cms  = self.cw*Cms +self.cgshift[0]*(-ca*CLs -sa*sb*CSs -sa*cb*CDs )
            Cmde = self.cw*Cmde+self.cgshift[0]*(-ca*CLde-sa*sb*CSde-sa*cb*CDde)
            Cns  = self.cw*Cns -self.cgshift[0]*( cb*CSs -sb*CDs )
            Cnde = self.cw*Cnde-self.cgshift[0]*( cb*CSde-sb*CDde)
            Cnda = self.cw*Cnda-self.cgshift[0]*( cb*CSda-sb*CDda)

            # differences
            dl = Cld - Cls
            dm = Cmd - Cms
            dn = Cnd - Cns

            # da, de calc
            dai = dl/Clda
            dei = dm/Cmde

            # determine equation that should be zero
            zero  =  Cnde* dm* Clda +  Cnda* dl* Cmde -  dn* Cmde* Clda
            # print("Z = {:> 13.10f}, da = {:> 7.3f}, de = {:> 7.3f}, dB = {:> 7.3f}".format(
            #     zero,dai*180.0/np.pi,dei*180.0/np.pi,dBj*180.0/np.pi
            # ))

            return dai,dei,zero
        
        def sine_dsine_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            # determine desired moment coefficients
            CMd_lref = Md/0.5/rho/V**2.0/self.Sw
            Cld = CMd_lref[0]#/self.bw
            Cmd = CMd_lref[1]#/self.cw
            Cnd = CMd_lref[2]#/self.bw

            # aero trig
            ca = cos(a); sa = sin(a)
            cb = cos(b); sb = sin(b)

            # determine moment constants
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            CS1 = BAM._CS0(dBj) + BAM._CS_beta(dBj)*b
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            # Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            # Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # derivatives
            DAM = self.dBAM
            dCL1 = DAM._CL0(dBj) + DAM._CL_alpha(dBj)*a
            dCS1 = DAM._CS0(dBj) + DAM._CS_beta(dBj)*b
            dCls = (DAM._Cl0(dBj) + DAM._Cl_alpha(dBj)*a +
                DAM._Cl_beta(dBj)*b + DAM._Cl_pbar(dBj)*pbar +
                DAM._Cl_qbar(dBj)*qbar +
                (DAM._Cl_rbar(dBj) + DAM._Cl_Lrbar(dBj)*CL1 + 
                BAM._Cl_Lrbar(dBj)*dCL1)*rbar)
            dClda = DAM._Cl_da(dBj)
            # dClde = DAM._Cl_de(dBj)
            dCms = (DAM._Cm0(dBj) + DAM._Cm_alpha(dBj)*a +
                DAM._Cm_beta(dBj)*b + DAM._Cm_pbar(dBj)*pbar +
                DAM._Cm_qbar(dBj)*qbar + DAM._Cm_rbar(dBj)*rbar)
            # dCmda = DAM._Cm_da(dBj)
            dCmde = DAM._Cm_de(dBj)
            dCns = (DAM._Cn0(dBj) + DAM._Cn_alpha(dBj)*a +
                DAM._Cn_beta(dBj)*b +
                (DAM._Cn_pbar(dBj) + DAM._Cn_Lpbar(dBj)*CL1 + 
                BAM._Cn_Lpbar(dBj)*dCL1)*pbar +
                DAM._Cn_qbar(dBj)*qbar + DAM._Cn_rbar(dBj)*rbar)
            dCnda = DAM._Cn_da(dBj) + DAM._Cn_Lda(dBj)*CL1 + \
                BAM._Cn_Lda(dBj)*dCL1
            dCnde = DAM._Cn_de(dBj)

            # force constants
            CLs = (CL1 + BAM._CL_beta(dBj)*b + BAM._CL_pbar(dBj)*pbar +
                BAM._CL_qbar(dBj)*qbar + BAM._CL_rbar(dBj)*rbar)
            # CLda = BAM._CL_da(dBj)
            CLde = BAM._CL_de(dBj)
            CSs = (CS1 + BAM._CS_alpha(dBj)*a + 
                (BAM._CS_pbar(dBj) + BAM._CS_Lpbar(dBj)*CL1)*pbar +
                BAM._CS_qbar(dBj)*qbar + BAM._CS_rbar(dBj)*rbar)
            CSda = BAM._CS_da(dBj)
            CSde = BAM._CS_de(dBj)
            CDs = (BAM._CD0(dBj) + BAM._CD_L(dBj)*CL1 + BAM._CD_L2(dBj)*CL1**2 +
                BAM._CD_S(dBj)*CS1 + BAM._CD_S2(dBj)*CS1**2 +
                (BAM._CD_pbar(dBj) + BAM._CD_Spbar(dBj)*CS1)*pbar +
                (BAM._CD_qbar(dBj) + BAM._CD_Lqbar(dBj)*CL1 + 
                BAM._CD_L2qbar(dBj)*CL1**2)*qbar +
                (BAM._CD_rbar(dBj) + BAM._CD_Srbar(dBj)*CS1)*rbar)
            CDda = (BAM._CD_da(dBj) + BAM._CD_Sda(dBj)*CS1)
            CDde = (BAM._CD_de(dBj) + BAM._CD_Lde(dBj)*CL1) # dropped squared part
            # derivatives
            dCLs = (dCL1 + DAM._CL_beta(dBj)*b + DAM._CL_pbar(dBj)*pbar +
                DAM._CL_qbar(dBj)*qbar + DAM._CL_rbar(dBj)*rbar)
            # dCLda = DAM._CL_da(dBj)
            dCLde = DAM._CL_de(dBj)
            dCSs = (dCS1 + DAM._CS_alpha(dBj)*a + 
                (DAM._CS_pbar(dBj) + DAM._CS_Lpbar(dBj)*CL1 + 
                    BAM._CS_Lpbar(dBj)*dCL1)*pbar +
                DAM._CS_qbar(dBj)*qbar + DAM._CS_rbar(dBj)*rbar)
            dCSda = DAM._CS_da(dBj)
            dCSde = DAM._CS_de(dBj)
            dCDs = (DAM._CD0(dBj) + DAM._CD_L(dBj)*CL1 + BAM._CD_L(dBj)*dCL1 + 
                DAM._CD_L2(dBj)*CL1**2 + 2.0*BAM._CD_L2(dBj)*CL1*dCL1 +
                DAM._CD_S(dBj)*CS1 + BAM._CD_S(dBj)*dCS1 + 
                DAM._CD_S2(dBj)*CS1**2 + 2.0*BAM._CD_S2(dBj)*CS1*dCS1 +
                (DAM._CD_pbar(dBj) + DAM._CD_Spbar(dBj)*CS1 + 
                    BAM._CD_Spbar(dBj)*dCS1)*pbar +
                (DAM._CD_qbar(dBj) + DAM._CD_Lqbar(dBj)*CL1 + 
                    BAM._CD_Lqbar(dBj)*dCL1 + DAM._CD_L2qbar(dBj)*CL1**2 + 
                    2.0*BAM._CD_L2qbar(dBj)*CL1*dCL1)*qbar +
                (DAM._CD_rbar(dBj) + DAM._CD_Srbar(dBj)*CS1 + 
                    BAM._CD_Srbar(dBj)*dCS1)*rbar)
            dCDda = (DAM._CD_da(dBj) + DAM._CD_Sda(dBj)*CS1 + 
                BAM._CD_Sda(dBj)*dCS1)
            dCDde = (DAM._CD_de(dBj) + DAM._CD_Lde(dBj)*CL1 + 
                BAM._CD_Lde(dBj)*dCL1) # dropped squared part
            
            # add to corresponding moments
            Cls  = self.bw*Cls
            Clda = self.bw*Clda
            Cms  = self.cw*Cms +self.cgshift[0]*(-ca*CLs -sa*sb*CSs -sa*cb*CDs )
            Cmde = self.cw*Cmde+self.cgshift[0]*(-ca*CLde-sa*sb*CSde-sa*cb*CDde)
            Cns  = self.cw*Cns -self.cgshift[0]*( cb*CSs -sb*CDs )
            Cnde = self.cw*Cnde-self.cgshift[0]*( cb*CSde-sb*CDde)
            Cnda = self.cw*Cnda-self.cgshift[0]*( cb*CSda-sb*CDda)
            # derivatives
            dCls  = self.bw*dCls
            dClda = self.bw*dClda
            dCms  = self.cw*dCms +self.cgshift[0]*(-ca*dCLs -sa*sb*dCSs -sa*cb*dCDs )
            dCmde = self.cw*dCmde+self.cgshift[0]*(-ca*dCLde-sa*sb*dCSde-sa*cb*dCDde)
            dCns  = self.cw*dCns -self.cgshift[0]*( cb*dCSs -sb*dCDs )
            dCnde = self.cw*dCnde-self.cgshift[0]*( cb*dCSde-sb*dCDde)
            dCnda = self.cw*dCnda-self.cgshift[0]*( cb*dCSda-sb*dCDda)

            # differences
            dl = Cld - Cls
            dm = Cmd - Cms
            dn = Cnd - Cns
            # derivatives
            ddl = - dCls
            ddm = - dCms
            ddn = - dCns

            # da, de calc
            dai = dl/Clda
            dei = dm/Cmde

            # determine equation that should be zero
            zero  =  Cnde* dm* Clda +  Cnda* dl* Cmde -  dn* Cmde* Clda
            # derivative
            dzero = \
                + dCnde* dm* Clda +  Cnde*ddm* Clda +  Cnde* dm*dClda \
                + dCnda* dl* Cmde +  Cnda*ddl* Cmde +  Cnda* dl*dCmde \
                - ddn* Cmde* Clda -  dn*dCmde* Clda -  dn* Cmde*dClda

            return dai,dei,zero,dzero
        
        def sine_dsine_wsine_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md):
            # determine desired moment coefficients
            CMd_lref = Md/0.5/rho/V**2.0/self.Sw
            Cld = CMd_lref[0]#/self.bw
            Cmd = CMd_lref[1]#/self.cw
            Cnd = CMd_lref[2]#/self.bw

            # aero trig
            ca = cos(a); sa = sin(a)
            cb = cos(b); sb = sin(b)

            # determine moment constants
            BAM = self.aero_model
            CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
            CS1 = BAM._CS0(dBj) + BAM._CS_beta(dBj)*b
            Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                BAM._Cl_qbar(dBj)*qbar +
                (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
            Clda = BAM._Cl_da(dBj)
            # Clde = BAM._Cl_de(dBj)
            Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
            # Cmda = BAM._Cm_da(dBj)
            Cmde = BAM._Cm_de(dBj)
            Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                BAM._Cn_beta(dBj)*b +
                (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
            Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
            Cnde = BAM._Cn_de(dBj)
            # derivatives
            DAM = self.dBAM
            dCL1 = DAM._CL0(dBj) + DAM._CL_alpha(dBj)*a
            dCS1 = DAM._CS0(dBj) + DAM._CS_beta(dBj)*b
            dCls = (DAM._Cl0(dBj) + DAM._Cl_alpha(dBj)*a +
                DAM._Cl_beta(dBj)*b + DAM._Cl_pbar(dBj)*pbar +
                DAM._Cl_qbar(dBj)*qbar +
                (DAM._Cl_rbar(dBj) + DAM._Cl_Lrbar(dBj)*CL1 + 
                BAM._Cl_Lrbar(dBj)*dCL1)*rbar)
            dClda = DAM._Cl_da(dBj)
            # dClde = DAM._Cl_de(dBj)
            dCms = (DAM._Cm0(dBj) + DAM._Cm_alpha(dBj)*a +
                DAM._Cm_beta(dBj)*b + DAM._Cm_pbar(dBj)*pbar +
                DAM._Cm_qbar(dBj)*qbar + DAM._Cm_rbar(dBj)*rbar)
            # dCmda = DAM._Cm_da(dBj)
            dCmde = DAM._Cm_de(dBj)
            dCns = (DAM._Cn0(dBj) + DAM._Cn_alpha(dBj)*a +
                DAM._Cn_beta(dBj)*b +
                (DAM._Cn_pbar(dBj) + DAM._Cn_Lpbar(dBj)*CL1 + 
                BAM._Cn_Lpbar(dBj)*dCL1)*pbar +
                DAM._Cn_qbar(dBj)*qbar + DAM._Cn_rbar(dBj)*rbar)
            dCnda = DAM._Cn_da(dBj) + DAM._Cn_Lda(dBj)*CL1 + \
                BAM._Cn_Lda(dBj)*dCL1
            dCnde = DAM._Cn_de(dBj)
            # double derivatives
            WAM = self.ddBAM
            wCL1 = WAM._CL0(dBj) + WAM._CL_alpha(dBj)*a
            wCS1 = WAM._CS0(dBj) + WAM._CS_beta(dBj)*b
            wCls = (WAM._Cl0(dBj) + WAM._Cl_alpha(dBj)*a +
                WAM._Cl_beta(dBj)*b + WAM._Cl_pbar(dBj)*pbar +
                WAM._Cl_qbar(dBj)*qbar +
                (WAM._Cl_rbar(dBj) + 
                WAM._Cl_Lrbar(dBj)*CL1 + 2.0*DAM._Cl_Lrbar(dBj)*dCL1 + 
                BAM._Cl_Lrbar(dBj)*wCL1)*rbar)
            wClda = WAM._Cl_da(dBj)
            wClde = WAM._Cl_de(dBj)
            wCms = (WAM._Cm0(dBj) + WAM._Cm_alpha(dBj)*a +
                WAM._Cm_beta(dBj)*b + WAM._Cm_pbar(dBj)*pbar +
                WAM._Cm_qbar(dBj)*qbar + WAM._Cm_rbar(dBj)*rbar)
            wCmda = WAM._Cm_da(dBj)
            wCmde = WAM._Cm_de(dBj)
            wCns = (WAM._Cn0(dBj) + WAM._Cn_alpha(dBj)*a +
                WAM._Cn_beta(dBj)*b +
                (WAM._Cn_pbar(dBj) + 
                WAM._Cn_Lpbar(dBj)*CL1 + 2.0*DAM._Cn_Lpbar(dBj)*dCL1 + 
                BAM._Cn_Lpbar(dBj)*wCL1)*pbar +
                WAM._Cn_qbar(dBj)*qbar + WAM._Cn_rbar(dBj)*rbar)
            wCnda = WAM._Cn_da(dBj) + \
                WAM._Cn_Lda(dBj)*CL1 + 2.0*DAM._Cn_Lda(dBj)*dCL1 + \
                BAM._Cn_Lda(dBj)*wCL1
            wCnde = WAM._Cn_de(dBj)

            # force constants
            CLs = (CL1 + BAM._CL_beta(dBj)*b + BAM._CL_pbar(dBj)*pbar +
                BAM._CL_qbar(dBj)*qbar + BAM._CL_rbar(dBj)*rbar)
            # CLda = BAM._CL_da(dBj)
            CLde = BAM._CL_de(dBj)
            CSs = (CS1 + BAM._CS_alpha(dBj)*a + 
                (BAM._CS_pbar(dBj) + BAM._CS_Lpbar(dBj)*CL1)*pbar +
                BAM._CS_qbar(dBj)*qbar + BAM._CS_rbar(dBj)*rbar)
            CSda = BAM._CS_da(dBj)
            CSde = BAM._CS_de(dBj)
            CDs = (BAM._CD0(dBj) + BAM._CD_L(dBj)*CL1 + BAM._CD_L2(dBj)*CL1**2 +
                BAM._CD_S(dBj)*CS1 + BAM._CD_S2(dBj)*CS1**2 +
                (BAM._CD_pbar(dBj) + BAM._CD_Spbar(dBj)*CS1)*pbar +
                (BAM._CD_qbar(dBj) + BAM._CD_Lqbar(dBj)*CL1 + 
                BAM._CD_L2qbar(dBj)*CL1**2)*qbar +
                (BAM._CD_rbar(dBj) + BAM._CD_Srbar(dBj)*CS1)*rbar)
            CDda = (BAM._CD_da(dBj) + BAM._CD_Sda(dBj)*CS1)
            CDde = (BAM._CD_de(dBj) + BAM._CD_Lde(dBj)*CL1) # dropped squared part
            # derivatives
            dCLs = (dCL1 + DAM._CL_beta(dBj)*b + DAM._CL_pbar(dBj)*pbar +
                DAM._CL_qbar(dBj)*qbar + DAM._CL_rbar(dBj)*rbar)
            # dCLda = DAM._CL_da(dBj)
            dCLde = DAM._CL_de(dBj)
            dCSs = (dCS1 + DAM._CS_alpha(dBj)*a + 
                (DAM._CS_pbar(dBj) + DAM._CS_Lpbar(dBj)*CL1 + 
                    BAM._CS_Lpbar(dBj)*dCL1)*pbar +
                DAM._CS_qbar(dBj)*qbar + DAM._CS_rbar(dBj)*rbar)
            dCSda = DAM._CS_da(dBj)
            dCSde = DAM._CS_de(dBj)
            dCDs = (DAM._CD0(dBj) + DAM._CD_L(dBj)*CL1 + BAM._CD_L(dBj)*dCL1 + 
                DAM._CD_L2(dBj)*CL1**2 + 2.0*BAM._CD_L2(dBj)*CL1*dCL1 +
                DAM._CD_S(dBj)*CS1 + BAM._CD_S(dBj)*dCS1 + 
                DAM._CD_S2(dBj)*CS1**2 + 2.0*BAM._CD_S2(dBj)*CS1*dCS1 +
                (DAM._CD_pbar(dBj) + DAM._CD_Spbar(dBj)*CS1 + 
                    BAM._CD_Spbar(dBj)*dCS1)*pbar +
                (DAM._CD_qbar(dBj) + DAM._CD_Lqbar(dBj)*CL1 + 
                    BAM._CD_Lqbar(dBj)*dCL1 + DAM._CD_L2qbar(dBj)*CL1**2 + 
                    2.0*BAM._CD_L2qbar(dBj)*CL1*dCL1)*qbar +
                (DAM._CD_rbar(dBj) + DAM._CD_Srbar(dBj)*CS1 + 
                    BAM._CD_Srbar(dBj)*dCS1)*rbar)
            dCDda = (DAM._CD_da(dBj) + DAM._CD_Sda(dBj)*CS1 + 
                BAM._CD_Sda(dBj)*dCS1)
            dCDde = (DAM._CD_de(dBj) + DAM._CD_Lde(dBj)*CL1 + 
                BAM._CD_Lde(dBj)*dCL1) # dropped squared part
            # double derivatives
            wCLs = (wCL1 + WAM._CL_beta(dBj)*b + WAM._CL_pbar(dBj)*pbar +
                WAM._CL_qbar(dBj)*qbar + WAM._CL_rbar(dBj)*rbar)
            # wCLda = WAM._CL_da(dBj)
            wCLde = WAM._CL_de(dBj)
            wCSs = (wCS1 + WAM._CS_alpha(dBj)*a + 
                (WAM._CS_pbar(dBj) + WAM._CS_Lpbar(dBj)*CL1 + 
                    2.0*DAM._CS_Lpbar(dBj)*dCL1 + 
                    BAM._CS_Lpbar(dBj)*wCL1)*pbar +
                WAM._CS_qbar(dBj)*qbar + WAM._CS_rbar(dBj)*rbar)
            wCSda = WAM._CS_da(dBj)
            wCSde = WAM._CS_de(dBj)
            wCDs = (WAM._CD0(dBj) + WAM._CD_L(dBj)*CL1 + 
                2.0*DAM._CD_L(dBj)*dCL1 + BAM._CD_L(dBj)*wCL1 + 
                WAM._CD_L2(dBj)*CL1**2 + 4.0*DAM._CD_L2(dBj)*CL1*dCL1 + 
                4.0*BAM._CD_L2(dBj)*dCL1 + 2.0*BAM._CD_L2(dBj)*CL1*wCL1 +
                WAM._CD_S(dBj)*CS1 + 
                2.0*BAM._CD_S(dBj)*dCS1 + BAM._CD_S(dBj)*wCS1 + 
                WAM._CD_S2(dBj)*CS1**2 + 2.0*DAM._CD_S2(dBj)*CS1*dCS1 + 
                2.0*WAM._CD_S2(dBj)*CS1*dCS1 + 4.0*BAM._CD_S2(dBj)*dCS1 + 
                2.0*BAM._CD_S2(dBj)*CS1*wCS1 + 
                (WAM._CD_pbar(dBj) + WAM._CD_Spbar(dBj)*CS1 + 
                    2.0*DAM._CD_Spbar(dBj)*dCS1 + BAM._CD_Spbar(dBj)*wCS1)*pbar +
                (WAM._CD_qbar(dBj) + 
                    WAM._CD_Lqbar(dBj)*CL1 + DAM._CD_Lqbar(dBj)*dCL1 + 
                    DAM._CD_Lqbar(dBj)*dCL1 + BAM._CD_Lqbar(dBj)*wCL1 + 
                    WAM._CD_L2qbar(dBj)*CL1**2 + 2.0*DAM._CD_L2qbar(dBj)*CL1*dCL1 + 
                    2.0*WAM._CD_L2qbar(dBj)*CL1*dCL1 + 
                    4.0*BAM._CD_L2qbar(dBj)*dCL1 + 
                    2.0*BAM._CD_L2qbar(dBj)*CL1*wCL1)*qbar +
                (WAM._CD_rbar(dBj) + WAM._CD_Srbar(dBj)*CS1 + 
                    2.0*DAM._CD_Srbar(dBj)*dCS1 + 
                    BAM._CD_Srbar(dBj)*wCS1)*rbar)
            wCDda = (WAM._CD_da(dBj) + WAM._CD_Sda(dBj)*CS1 + 
                        2.0*DAM._CD_Sda(dBj)*dCS1 + 
                        BAM._CD_Sda(dBj)*wCS1)
            wCDde = (WAM._CD_de(dBj) + WAM._CD_Lde(dBj)*CL1 + 
                        2.0*DAM._CD_Lde(dBj)*dCL1 + 
                        BAM._CD_Lde(dBj)*wCL1) # dropped squared part
            
            # add to corresponding moments
            Cls  = self.bw*Cls
            Clda = self.bw*Clda
            Cms  = self.cw*Cms +self.cgshift[0]*(-ca*CLs -sa*sb*CSs -sa*cb*CDs )
            Cmde = self.cw*Cmde+self.cgshift[0]*(-ca*CLde-sa*sb*CSde-sa*cb*CDde)
            Cns  = self.cw*Cns -self.cgshift[0]*( cb*CSs -sb*CDs )
            Cnde = self.cw*Cnde-self.cgshift[0]*( cb*CSde-sb*CDde)
            Cnda = self.cw*Cnda-self.cgshift[0]*( cb*CSda-sb*CDda)
            # derivatives
            dCls  = self.bw*dCls
            dClda = self.bw*dClda
            dCms  = self.cw*dCms +self.cgshift[0]*(-ca*dCLs -sa*sb*dCSs -sa*cb*dCDs )
            dCmde = self.cw*dCmde+self.cgshift[0]*(-ca*dCLde-sa*sb*dCSde-sa*cb*dCDde)
            dCns  = self.cw*dCns -self.cgshift[0]*( cb*dCSs -sb*dCDs )
            dCnde = self.cw*dCnde-self.cgshift[0]*( cb*dCSde-sb*dCDde)
            dCnda = self.cw*dCnda-self.cgshift[0]*( cb*dCSda-sb*dCDda)
            # double derivatives
            wCls  = self.bw*wCls
            wClda = self.bw*wClda
            wCms  = self.cw*wCms +self.cgshift[0]*(-ca*wCLs -sa*sb*wCSs -sa*cb*wCDs )
            wCmde = self.cw*wCmde+self.cgshift[0]*(-ca*wCLde-sa*sb*wCSde-sa*cb*wCDde)
            wCns  = self.cw*wCns -self.cgshift[0]*( cb*wCSs -sb*wCDs )
            wCnde = self.cw*wCnde-self.cgshift[0]*( cb*wCSde-sb*wCDde)
            wCnda = self.cw*wCnda-self.cgshift[0]*( cb*wCSda-sb*wCDda)

            # differences
            dl = Cld - Cls
            dm = Cmd - Cms
            dn = Cnd - Cns
            # derivatives
            ddl = - dCls
            ddm = - dCms
            ddn = - dCns
            # double derivatives
            wdl = - wCls
            wdm = - wCms
            wdn = - wCns

            # da, de calc
            dai = dl/Clda
            dei = dm/Cmde

            # determine equation that should be zero
            zero  =  Cnde* dm* Clda +  Cnda* dl* Cmde -  dn* Cmde* Clda
            # derivative
            dzero = \
                + dCnde* dm* Clda +  Cnde*ddm* Clda +  Cnde* dm*dClda \
                + dCnda* dl* Cmde +  Cnda*ddl* Cmde +  Cnda* dl*dCmde \
                - ddn* Cmde* Clda -  dn*dCmde* Clda -  dn* Cmde*dClda
            # double derivative
            wzero = \
                + wCnde* dm* Clda + dCnde*ddm* Clda + dCnde* dm*dClda \
                + dCnde*ddm* Clda +  Cnde*wdm* Clda +  Cnde*ddm*dClda \
                + dCnde* dm*dClda +  Cnde*ddm*dClda +  Cnde* dm*wClda \
                \
                + wCnda* dl* Cmde + dCnda*ddl* Cmde + dCnda* dl*dCmde \
                + dCnda*ddl* Cmde +  Cnda*wdl* Cmde +  Cnda*ddl*dCmde \
                + dCnda* dl*dCmde +  Cnda*ddl*dCmde +  Cnda* dl*wCmde \
                \
                - wdn* Cmde* Clda - ddn*dCmde* Clda - ddn* Cmde*dClda \
                - ddn*dCmde* Clda -  dn*wCmde* Clda -  dn*dCmde*dClda \
                - ddn* Cmde*dClda -  dn*dCmde*dClda -  dn* Cmde*wClda

            return dai,dei,zero,dzero,wzero
      
        ###
        self.delta_E_fun = delta_E_fun
        self.delta_E_fun_sum = delta_E_fun_sum
        self.delta_E_dE_fun = delta_E_dE_fun
        self.delta_E_fun_sq = delta_E_fun_sq
        self.delta_E_dE_fun_sq = delta_E_dE_fun_sq
        self.delta_E_dE_wE_fun_sq = delta_E_dE_wE_fun_sq
        self.            sine_fun =             sine_fun
        self.      sine_dsine_fun =       sine_dsine_fun
        self.sine_dsine_wsine_fun = sine_dsine_wsine_fun
        #
        self.prev_E = 0.0
        ###
        self.time_check = 100.0 # 1.0 # 
        self.dt_check = 0.000001 # 0.1 # 0.05 # 0.01 # 
        self._err_plot_pause_time = 0.25 # 0.1 # 0.000001 # 5.0 # 1.0 # 
        self._end_plot_time = 1.5 # 
        self.have_saved = False
        self.log_scale = False # True # 
        self.symlog_scale = False # True # 
        self.first_plot = True # False # 
        self.feval = 0

        self.plot_alternate_solns = False # True # 
        self.poss_ts     = []
        self.poss_dadegs = []
        self.poss_dedegs = []
        self.poss_dBdegs = []
        self.poss_Es     = []
        self.poss_check_num = 10000 # 20000 # 5000 # 
        self.save_poss_every = 5 # 10 # 2 # 100 # 
        self._Evals_threshold = 5.0e-7 # 1.0e-7 # 
        self.poss_dB_lim = 90.0 # 180.0 # 
        self.save_poss_counter = 0
        self._final_on_rk4 = False

    def __del__(self):
        # # report gain matrix
        # rep2D(self.KI_DI,"KI",decimals=3)
        # rep2D(self.KP_DI,"KP",decimals=3)
        pass

    def returns_zero(self,tarr,xarr,uarr,errs_axs,ctrl_axs,
        subdict,xticks,perc_zoom,
        predir,format,savedict,save_plot):
        # calculate Error
        MErr = []
        MErrnew = []
        Mds = []
        dBdiff = []
        fevals = []
        nits = []
        devals = []
        for k in range(tarr.shape[0]):
            x_at_t = xarr[:,k]
            u_at_t = uarr[:,k]
            t = tarr[k]
            #
            ref = self._get_reference(t)[self.Lin_Model.Cslice]
            V_xb    = x_at_t[ 0]
            V_yb    = x_at_t[ 1]
            V_zb    = x_at_t[ 2]
            p       = np.deg2rad(x_at_t[ 3])
            q       = np.deg2rad(x_at_t[ 4])
            r       = np.deg2rad(x_at_t[ 5])
            z_f     = x_at_t[ 8]
            dB      = np.deg2rad(x_at_t[14])
            dB_comm = np.deg2rad(u_at_t[ 2]) # u_at_t[ 2] # 
            epI     = np.deg2rad(x_at_t[self.xIi_eul[1]])
            eqI     = np.deg2rad(x_at_t[self.xIi_eul[2]])
            erI     = np.deg2rad(x_at_t[self.xIi_eul[3]])
            # Derived Quantities
            V = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
            a   = np.arctan2(V_zb,V_xb)
            b   = asin(V_yb/V)
            if self.constant_density:
                _,g,_,_,rho,sos = self.stdatm(self.H0)
            else:
                _,g,_,_,rho,sos = self.stdatm(-z_f)
            M = V/sos
            pbar = p*self.bw/2./V
            qbar = q*self.cw/2./V
            rbar = r*self.bw/2./V
            # pull out parts of state
            # preliminaries
            Sw = self.Sw
            bw = self.bw
            cw = self.cw
            h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
            hmat = np.array([
                [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
            Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                self.inertia_model.inertia_results(dB)
            I     = self.inertia_model.inertia_tensor(dB)
            Iinv  = self.inertia_model.inverse_tensor(dB)
            Qdyn = 0.5*rho*V**2.*Sw
            G = Qdyn*np.diag([bw,cw,bw])
            Om = np.array([
                (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
            # determine desired moment
            w  = np.array([  p,  q,  r])
            eI = np.array([epI,eqI,erI])
            e = w - ref
            # LM = self.Lin_Model
            # v = - np.matmul(LM.K,e) - np.matmul(LM.KI,eI)
            v = - np.matmul(self.KP_DI,e) - np.matmul(self.KI_DI,eI)
            Md = np.matmul(I,(v - np.matmul(Iinv,np.matmul(hmat,w) + Om)))
            # correct moment
            Md = np.matmul(G,self.aero_model.uncorrect_M(
                np.matmul(1./Qdyn*np.diag([1./bw,1./cw,1./bw]),Md),a,
                self.is_compressible,M,self.use_anderson,self.has_stall))
            Mds.append(Md)
            # Error
            # dB_commanded = dB_comm
            x = x_at_t*1.0
            iconv = self.xicnv
            x[iconv] = np.deg2rad(x[iconv])
            x = self.euler2quat_state(x)
            ucomm,incomm = self._get_control(t,x,True,False,"o",False)
            fevals.append(self.feval)
            nits.append(self.nit)
            devals.append(self.deval)
            dB_commanded = ucomm[2]
            dBdiff.append(np.rad2deg(dB_commanded - dB_comm))
            # print(t,np.rad2deg(dB_commanded),np.rad2deg(dB_comm),np.rad2deg(dB_commanded-dB_comm))
            MErr.append(self.delta_E_fun_sq(
                rho,V,dB_comm,a,b,pbar,qbar,rbar,Md)[2])
            MErrnew.append(self.delta_E_fun_sq(
                rho,V,dB_commanded,a,b,pbar,qbar,rbar,Md)[2])
            # MErr.append(Md)
        Mds = np.array(Mds).T
        dBdiff = np.array(dBdiff)
        dBcomm = uarr[2]
        dBnew = dBdiff + dBcomm
        # print(dBcomm)
        # print(dBnew)
        #
        # # Error plots
        ErMg_fig, ErMg_axs = plt.subplots(1,1,**subdict)
        ErMg_ax2 = ErMg_axs.twinx()
        ErMn_fig, ErMn_axs = plt.subplots(1,1,**subdict)
        ErMn_ax2 = ErMn_axs.twinx()
        fevl_fig, fevl_axs = plt.subplots(1,1,**subdict)
        fevl_ax2 = fevl_axs.twinx()
        # axis labels, legends
        altcol = "0.5"
        ErMg_fig.supxlabel(r"Time, s")
        ErMg_fig.supylabel(r"Moment Error, lbf$^2$-ft$^2$")
        ErMg_ax2.set_ylabel(r"Desired Moment, lbf-ft",c=altcol)
        ErMn_fig.supxlabel(r"Time, s")
        ErMn_fig.supylabel(r"Moment Error, lbf$^2$-ft$^2$")
        ErMn_ax2.set_ylabel(r"$\Delta \delta_B$ difference, deg",c=altcol)
        fevl_fig.supxlabel(r"Time, s")
        fevl_fig.supylabel(r"Evaluations")
        fevl_ax2.set_ylabel(r"Iterations",c=altcol)
        # xticks
        ErMg_axs.set_xticks(ticks=xticks)
        ErMn_axs.set_xticks(ticks=xticks)
        fevl_axs.set_xticks(ticks=xticks)
        # ErMg_ax2.set_yticks(ticks=ErMg_ax2.get_yticks(),color=altcol)
        # ErMn_ax2.set_yticks(ticks=ErMn_ax2.get_yticks(),color=altcol)
        # grid, axis labels, legends
        ErMg_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
        ErMn_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
        fevl_axs.grid(which="major",lw=0.6,ls="-",c="0.75")
        #
        ErMg_ax2.plot(tarr,Mds[0],c=altcol,ls="-" )
        ErMg_ax2.plot(tarr,Mds[1],c=altcol,ls="--")
        ErMg_ax2.plot(tarr,Mds[2],c=altcol,ls="-.")
        ErMg_axs.plot(tarr,MErr,c="k")
        ErMn_ax2.plot(tarr,dBdiff,c=altcol)
        ErMn_axs.plot(tarr,MErrnew,c="k")
        fevl_axs.plot(tarr,fevals,ls="-" ,c="k",label="fun",zorder=2)
        fevl_axs.plot(tarr,devals,ls="--",c="k",label="jac",zorder=3)
        fevl_ax2.plot(tarr,nits,c=altcol,zorder=1)
        legend = fevl_axs.legend()
        legend.set_zorder(4)
        #
        if self.plot_alternate_solns:
            for i in range(len(self.poss_ts)):
                for j in range(len(self.poss_dadegs[i])):
                    ctrl_axs[0].plot(self.poss_ts[i],self.poss_dadegs[i][j],"b.",ms=0.5)
                    ctrl_axs[1].plot(self.poss_ts[i],self.poss_dedegs[i][j],"r.",ms=0.5)
                    ctrl_axs[2].plot(self.poss_ts[i],self.poss_dBdegs[i][j],"g.",ms=0.5)
                    ErMg_axs   .plot(self.poss_ts[i],self.poss_Es    [i][j],"b.",ms=0.5)
        # limit control plots
        if not(self.bool_limit_inputs):
            min_da      = np.rad2deg( self.min_da)
            max_da      = np.rad2deg( self.max_da)
            min_de_opt  = np.rad2deg( self.min_de)
            max_de_opt  = np.rad2deg( self.max_de)
            min_dr      = np.rad2deg( self.min_dr)
            max_dr      = np.rad2deg( self.max_dr)
            ctrl_axs[0].set_ylim((min_da-5.,max_da+5.))
            ctrl_axs[1].set_ylim((min_de_opt-5.,max_de_opt+5.))
            ctrl_axs[2].set_ylim((min_dr-5.,max_dr+5.))
        
        if self.tracking:
            # TESTING MACA
            errs_axs.set_ylim((-150.0,150.0))
            # TESTING MACA

        ErMg_axs.set_yscale("log")
        ErMg_axs.set_xlim((0.,perc_zoom*self.tf))
        ErMg_ax2.set_xlim((0.,perc_zoom*self.tf))
        ErMn_axs.set_xlim((0.,perc_zoom*self.tf))
        ErMn_ax2.set_xlim((0.,perc_zoom*self.tf))
        fevl_axs.set_xlim((0.,perc_zoom*self.tf))
        fevl_ax2.set_xlim((0.,perc_zoom*self.tf))
        if save_plot:
            ErMg_fig.savefig(predir+"moment_error."+format,**savedict)
            ErMn_fig.savefig(predir+"moment_error_new."+format,**savedict)
            fevl_fig.savefig(predir+"function_evaluations."+format,**savedict)
        plt.close(ErMg_fig)
        plt.close(ErMn_fig)
        plt.close(fevl_fig)
        #
        return 0
    
    def _empty_call_after_rk4(self,t):
        # save values
        if self.plot_alternate_solns:
            self.poss_ts    .append(t)
            self.poss_dadegs.append(self.poss_dadeg)
            self.poss_dedegs.append(self.poss_dedeg)
            self.poss_dBdegs.append(self.poss_dBdeg)
            self.poss_Es    .append(self.poss_E    )
        return
    
    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # feedback linearization
                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                V_xb    = x_euler[ 0] #  self.x_trim_euler[ 0] # 
                V_yb    = x_euler[ 1] #  self.x_trim_euler[ 1] # 
                V_zb    = x_euler[ 2] #  self.x_trim_euler[ 2] # 
                p       = x_euler[ 3] #  self.x_trim_euler[ 3] # 
                q       = x_euler[ 4] #  self.x_trim_euler[ 4] # 
                r       = x_euler[ 5] #  self.x_trim_euler[ 5] # 
                z_f     = x_euler[ 8] #  self.x_trim_euler[ 8] # 
                # da      = x_euler[12] #  self.x_trim_euler[12] # 
                # de      = x_euler[13] #  self.x_trim_euler[13] # 
                if self.order > 0:
                    dB  = x_euler[14] #  self.x_trim_euler[14] # 
                else:
                    dB  = self.u_til_next_update[2] # 
                # tau     = x_euler[15] #  self.x_trim_euler[15] # 
                epI     = x_euler[self.xIi_eul[1]]
                eqI     = x_euler[self.xIi_eul[2]]
                erI     = x_euler[self.xIi_eul[3]]
                # Derived Quantities
                V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
                V_xb_ss = self.x_trim[0]
                V_yb_ss = self.x_trim[1]
                V_zb_ss = self.x_trim[2]
                V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
                aero = 0
                if aero == 0:
                    a   = np.arctan2(V_zb,V_xb)
                    b   = asin(V_yb/V_tot)
                    V = V_tot
                    V_xb_in = V_xb*1.; V_yb_in = V_yb*1.; V_zb_in = V_zb*1.
                elif aero == 1:
                    a   = 0.0
                    b   = 0.0
                    V = V_tot
                    V_xb_in = V_tot*1.; V_yb_in = 0.0; V_zb_in = 0.0
                elif aero == 2:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V     = V_tot
                    V_xb_in = V*np.cos(a)*np.cos(b)
                    V_yb_in = V          *np.sin(b)
                    V_zb_in = V*np.sin(a)*np.cos(b)
                elif aero == 3:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V = V_xb
                    V_xb_in = V_xb*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                elif aero == 4:
                    a = np.arctan2(V_zb_ss,V_xb_ss)
                    b = asin(V_yb_ss/V_ss)
                    V = V_ss
                    V_xb_in = V_xb_ss*1.; V_yb_in = V_yb_ss*1.; V_zb_in = V_zb_ss*1.
                #
                if self.constant_density:
                    _,g,_,_,rho,sos = self.stdatm(self.H0)
                else:
                    _,g,_,_,rho,sos = self.stdatm(-z_f)
                M = V/sos
                pbar = p*self.bw/2./V
                qbar = q*self.cw/2./V
                rbar = r*self.bw/2./V
                # pull out parts of state
                # preliminaries
                Sw = self.Sw
                bw = self.bw
                cw = self.cw
                h_xb,h_yb,h_zb = self.inertia_model.angular_momentum_results()
                hmat = np.array([
                    [0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
                Ixx,Iyy,Izz,Ixy,Ixz,Iyz = \
                    self.inertia_model.inertia_results(dB)
                I     = self.inertia_model.inertia_tensor(dB)
                Iinv  = self.inertia_model.inverse_tensor(dB)
                Qdyn = 0.5*rho*V**2.*Sw
                G = Qdyn*np.diag([bw,cw,bw])
                Om = np.array([
                    (Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
                # determine desired moment
                w  = np.array([  p,  q,  r])
                eI = np.array([epI,eqI,erI])
                e = w - ref
                # LM = self.Lin_Model
                # v = - np.matmul(LM.K,e) - np.matmul(LM.KI,eI)
                v = - np.matmul(self.KP_DI,e) - np.matmul(self.KI_DI,eI)
                Md = np.matmul(I,(v - np.matmul(Iinv,np.matmul(hmat,w) + Om)))
                CMd_c = np.matmul(1./Qdyn*np.diag([1./bw,1./cw,1./bw]),Md)
                # correct moment
                Md = np.matmul(G,self.aero_model.uncorrect_M(
                    np.matmul(1./Qdyn*np.diag([1./bw,1./cw,1./bw]),Md),a,
                    self.is_compressible,M,self.use_anderson,self.has_stall))
                CMd_u = np.matmul(1./Qdyn*np.diag([1./bw,1./cw,1./bw]),Md)
                
                # ######################################
                # # # checking second derivative. works!
                # E = lambda dBj : self.delta_E_dE_wE_fun_sq(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                # dBtest = np.deg2rad(np.linspace(-90.0,90.0,1000))
                # dME = np.zeros((len(dBtest),))
                # dCS = np.zeros((len(dBtest),))
                # pows = np.zeros((len(dBtest),))
                # h = np.deg2rad(1.0e-6)
                # for i in range(len(dBtest)):
                #     dME[i] = E(dBtest[i])[4] # hessian
                #     dBtesti = complex(dBtest[i],h)
                #     dCS[i] = np.imag(E(dBtesti)[3])/h # hessian
                # pd = (dME - dCS)/dME
                # print(pd)
                # print(np.linalg.norm(pd)**2.)
                # quit()
                # ######################################
                # ######################################
                # # # checking second derivative. works!
                # E = lambda dBj : self.sine_dsine_wsine_fun(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                # dBtest = np.deg2rad(np.linspace(-90.0,90.0,1000))
                # dME = np.zeros((len(dBtest),))
                # dCS = np.zeros((len(dBtest),))
                # pows = np.zeros((len(dBtest),))
                # h = np.deg2rad(1.0e-6)
                # # # hessian test
                # # fi = 3; di = 4
                # # jacobian test
                # fi = 2; di = 3
                # for i in range(len(dBtest)):
                #     dME[i] = E(dBtest[i])[di] # hessian
                #     dBtesti = complex(dBtest[i],h)
                #     dCS[i] = np.imag(E(dBtesti)[fi])/h # hessian
                # pd = (dME - dCS)/dME
                # print(pd)
                # print(np.linalg.norm(pd))#**2.)
                # quit()
                # ######################################

                if self.pseudo_inverse_method:
                    # if self.add_tail_lag_eq:
                    #     Md = np.concatenate((Md,[0.0]))
                    # run through a cycle of dB's and determine which minimizes 
                    #   the problem.
                    dB_lim = self.ls_dB_lim
                    num = self.ls_num
                    if self.line_method == "None" or self.do_line_search:
                        dBs = np.deg2rad(np.linspace(-dB_lim,dB_lim,num=num))
                        dBs += self.u_trim[2]
                        err = 1e10; da_d = self.u_trim[0]
                        de_d = self.u_trim[1]; dB_d = self.u_trim[2]; i_d = 0
                        for i,dBi in enumerate(dBs):
                            # if self.add_tail_lag_eq:
                            #     dBj = dB
                            # else:
                            dBj = dBi
                            # BAM = self.aero_model
                            # CL1 = BAM._CL0(dBj) + BAM._CL_alpha(dBj)*a
                            # Cls = (BAM._Cl0(dBj) + BAM._Cl_alpha(dBj)*a +
                            #     BAM._Cl_beta(dBj)*b + BAM._Cl_pbar(dBj)*pbar +
                            #     BAM._Cl_qbar(dBj)*qbar +
                            #     (BAM._Cl_rbar(dBj) + BAM._Cl_Lrbar(dBj)*CL1)*rbar)
                            # Clda = BAM._Cl_da(dBj)
                            # Clde = BAM._Cl_de(dBj)
                            # Cms = (BAM._Cm0(dBj) + BAM._Cm_alpha(dBj)*a +
                            #     BAM._Cm_beta(dBj)*b + BAM._Cm_pbar(dBj)*pbar +
                            #     BAM._Cm_qbar(dBj)*qbar + BAM._Cm_rbar(dBj)*rbar)
                            # Cmda = BAM._Cm_da(dBj)
                            # Cmde = BAM._Cm_de(dBj)
                            # Cns = (BAM._Cn0(dBj) + BAM._Cn_alpha(dBj)*a +
                            #     BAM._Cn_beta(dBj)*b +
                            #     (BAM._Cn_pbar(dBj) + BAM._Cn_Lpbar(dBj)*CL1)*pbar +
                            #     BAM._Cn_qbar(dBj)*qbar + BAM._Cn_rbar(dBj)*rbar)
                            # Cnda = BAM._Cn_da(dBj) + BAM._Cn_Lda(dBj)*CL1
                            # Cnde = BAM._Cn_de(dBj)
                            # # determine da, de
                            # Cs = np.array([Cls,Cms,Cns])
                            # Cc = np.array([[Clda,Clde],[Cmda,Cmde],[Cnda,Cnde]])
                            # GCs = np.matmul(G,Cs)
                            # GCc = np.matmul(G,Cc)
                            # # if self.add_tail_lag_eq:
                            # #     GCs = np.concatenate((GCs,[-self.s_dr*dBj]))
                            # #     GCc = np.block([[GCc,np.zeros((3,1))],[np.zeros((1,2)),np.array([self.s_dr])]])
                            # #     dai,dei,dBc = np.matmul(np.linalg.pinv(GCc),Md - GCs)
                            # #     M = GCs + np.matmul(GCc,[dai,dei,dBc])
                            # # else:
                            # dai,dei = np.matmul(np.linalg.pinv(GCc),Md - GCs)
                            # M = GCs + np.matmul(GCc,[dai,dei])
                            # new_err = np.linalg.norm(M-Md)
                            #
                            # dai,dei = self.delta(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                            # new_err = self.Err(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                            # #
                            dai,dei,new_err = \
                                self.delta_E_fun_sq(rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                            # print(new_err)
                            if new_err < err:
                                err = new_err*1.
                                da_d,de_d,dB_d = dai*1.,dei*1.,dBj*1.
                                i_d = i*1
                        i_d += 1
                        dB_lim = dB_lim + abs(np.rad2deg(dBs[1] - dBs[0]))
                        dBSs = np.deg2rad(np.linspace(-dB_lim,dB_lim,num=num+2))
                        dBSs += self.u_trim[2]
                        bracket = (dBSs[i_d-1],dBSs[i_d],dBSs[i_d+1])
                    else:
                        da_d,de_d,dB_d = self.u_til_next_update[0:3]
                        # dB_d = (dB_d + dB)/2.0
                        # dB_d = np.deg2rad(-30.0)
                        step = np.deg2rad(self.ls_dB_lim)*2/(self.ls_num-1)
                        # step = self.max_drdot*self.dt
                        bracket = (dB_d - step, dB_d, dB_d + step)
                        # if t >= 2.0:
                        #     dB_d = 0.0
                        # # if abs(dB_d) > np.pi/2.0:
                        # #     dB_d = 0.0
                        dBbrack = np.deg2rad(90.0)
                        bracket = (-dBbrack, dB_d, dBbrack)

                    # if self._final_on_rk4 and self.plot_alternate_solns:
                    #     if self.save_poss_counter % self.save_poss_every == 0:
                    #         # determine all "zero" controls
                    #         da_de_Eall = lambda dBj : self.delta_E_fun_sq(\
                    #             rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                    #         dBvals_deg = np.linspace(-90.0,90.0,self.poss_check_num)
                    #         dBvals = np.deg2rad(dBvals_deg)
                    #         Evals = [da_de_Eall(dBvals[i]) for i in range(len(dBvals))]
                    #         Evals = np.array(Evals)
                    #         Evals_threshold = 5e2
                    #         poss_inds = np.argwhere(Evals[:,2] < Evals_threshold)[:,0]
                    #         poss_Evals = Evals[poss_inds]
                    #         self.poss_dadeg = np.rad2deg(poss_Evals[:,0])
                    #         self.poss_dedeg = np.rad2deg(poss_Evals[:,1])
                    #         self.poss_dBdeg = dBvals_deg[poss_inds]
                    #         self.poss_E     = poss_Evals[:,2]
                    #         print(t,self.poss_E)
                    #         self.save_poss_counter += 1
                    #     else:
                    #         self.save_poss_counter += 1


                    if self._final_on_rk4 and self.plot_alternate_solns:
                        if self.save_poss_counter % self.save_poss_every == 0:
                            # determine all "zero" controls
                            da_de_Eall = lambda dBj : self.sine_fun(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)
                            dBvals_deg = np.linspace(-self.poss_dB_lim,self.poss_dB_lim,self.poss_check_num)
                            dBvals = np.deg2rad(dBvals_deg)
                            Evals = [da_de_Eall(dBvals[i]) for i in range(len(dBvals))]
                            Evals = np.array(Evals)
                            Evals_threshold = self._Evals_threshold
                            poss_inds = np.argwhere(np.abs(Evals[:,2]) < Evals_threshold)[:,0]
                            poss_Evals = Evals[poss_inds]
                            self.poss_dadeg = np.rad2deg(poss_Evals[:,0])
                            self.poss_dedeg = np.rad2deg(poss_Evals[:,1])
                            self.poss_dBdeg = dBvals_deg[poss_inds]
                            self.poss_E     = poss_Evals[:,2]
                            # print(t,self.poss_E)
                            self.save_poss_counter += 1
                        else:
                            self.save_poss_counter += 1

                    if self.line_method == "Newton_Root":
                        zero  = lambda dBj : self.sine_fun(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                        dzero = lambda dBj : self.sine_dsine_fun(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[3]
                        wzero = lambda dBj : self.sine_dsine_wsine_fun(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[4]
                        # guess = ref[0]/np.pi*self.max_dr
                        # if abs(dB_d) > np.pi: dB_d = dB*1.0
                        # if zero(dB)*self.prev_E < 0.0:
                        #     dB_d = dB
                        # self.prev_E = zero(dB)
                        dB_d,res = newton(zero,dB_d, # dB, # 0.0, # 
                            fprime=dzero,
                            fprime2=wzero,
                            maxiter=self.opt_max_iter, # 1000, # 
                            tol=self.opt_tol,
                            disp=False, # True, # 
                            full_output=True)
                        # print()
                        # dB_d_2,res_2 = newton(zero,dB,
                        #     fprime=dzero,
                        #     fprime2=wzero,
                        #     maxiter=self.opt_max_iter, # 1000, # 
                        #     tol=self.opt_tol,
                        #     disp=False, # True, # 
                        #     full_output=True)
                        # # pick which one
                        # if abs(dB_d_1 - dB) < abs(dB_d_2 - dB):
                        #     dB_d = dB_d_1
                        #     res = res_1
                        # else:
                        #     dB_d = dB_d_2
                        #     res = res_2
                        # # # # # rebound
                        if dB_d != 0.0: # and abs(abs(dB) - self.max_dr) < 1.0*np.pi/180.0:
                            dBsgn = np.sign(dB_d)
                            dB_d = dBsgn*(abs(dB_d) % np.pi) # 2.0*
                            # if abs(zero(dB_d)) >= self.opt_tol:
                            #     dB_d,res = newton(zero,dB_d, # dB, # 0.0, # 
                            #         fprime=dzero,
                            #         fprime2=wzero,
                            #         maxiter=self.opt_max_iter, # 1000, # 
                            #         tol=self.opt_tol,
                            #         disp=False, # True, # 
                            #         full_output=True)
                            #     # print()
                            # if   dB_d >  np.pi/2.0:
                            #     dB_d -= np.pi
                            # elif dB_d < -np.pi/2.0:
                            #     dB_d += np.pi
                        # if dB_d > np.pi:
                        #     dBsgn = np.sign(dB_d)
                        #     dB_d = dBsgn*(abs(dB_d) % np.pi)
                        #
                        #
                        dB_rep = dB_d*1.0
                        # if abs(dB_d-dB) > 4.0*self.max_drdot*self.dt:
                        #     dB_d = dB + np.sign(dB_d-dB)*4.0*self.max_drdot*self.dt

                        # # # # # # # #
                        res.fun = zero(dB_d)
                        res.nit = res.iterations
                        self.nit = res.iterations
                        self.feval = res.function_calls
                        self.deval = res.function_calls
                    # other
                    elif self.line_method == "Newton":
                        E = lambda dBj : self.delta_E_fun_sq(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                        dE = lambda dBj : self.delta_E_dE_fun_sq(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[3]
                        # # # # # # # #
                        try:
                            dB_1,res_1 = newton(E,dB_d, # res.x, # 
                                dE,
                                maxiter=self.opt_max_iter,tol=self.opt_tol,
                                disp=False,
                                full_output=True)
                            if dB_1 >= np.pi/2.0:
                                dB_1 -= np.pi # dB_1 = np.pi/2.0 # 
                            elif dB_1 <= -np.pi/2.0:
                                dB_1 += np.pi # dB_1 = -np.pi/2.0 # 
                            res_1.fun = self.delta_E_fun_sq(\
                                rho,V,dB_1,a,b,pbar,qbar,rbar,Md)[2]
                            fail_1 = False
                        except:
                            fail_1 = True
                        try:
                            dB_2,res_2 = newton(E,0.0, # res.x, # 
                                dE,
                                maxiter=self.opt_max_iter,tol=self.opt_tol,
                                disp=False,
                                full_output=True)
                            # if dB_2 >= np.pi/2.0:
                            #     dB_2 -= np.pi # dB_2 = np.pi/2.0 # 
                            # elif dB_2 <= -np.pi/2.0:
                            #     dB_2 += np.pi # dB_2 = -np.pi/2.0 # 
                            res_2.fun = self.delta_E_fun_sq(\
                                rho,V,dB_2,a,b,pbar,qbar,rbar,Md)[2]
                            fail_2 = False
                        except:
                            fail_2 = True
                        #
                        if   not(fail_1) and not(fail_2):
                            if res_1.fun > res_2.fun:
                                dB_d = dB_2
                                res = res_2
                            else:
                                dB_d = dB_1
                                res = res_1
                        elif not(fail_1) and    fail_2 :
                            dB_d = dB_1
                            res = res_1
                        elif     fail_1 and not(fail_2):
                            dB_d = dB_2
                            res = res_2
                        else:
                            raise RuntimeError("Newton Solver Failed!!!!")
                        # # # # # # # #
                        res.fun = self.delta_E_fun_sq(\
                            rho,V,dB_d,a,b,pbar,qbar,rbar,Md)[2]
                        res.nit = res.iterations
                        self.nit = res.iterations
                        self.feval = res.function_calls
                        self.deval = res.function_calls
                    # remaining search functions
                    elif self.line_method in self.scipy_options:
                        odict = dict(tol=1.0e-12)
                        if self.line_method == "SLSQP":
                            E = lambda dBj : self.delta_E_dE_fun_sq(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2:4]
                            odict["jac"] = True
                            odict["bounds"] = [(-np.pi/2.,np.pi/2.)]
                        elif self.line_method == "BFGS":
                            E = lambda dBj : self.delta_E_dE_fun_sq(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2:4]
                            odict["jac"] = True
                        elif self.line_method == "trust-exact":
                            E = lambda dBj : self.delta_E_dE_fun_sq(
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2:4]
                            wE = lambda dBj : self.delta_E_dE_wE_fun_sq(
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[4]
                            odict["jac"],odict["hess"] = True,wE
                            odict["options"] = {
                                "initial_trust_radius" : 1.0e-6,
                                "max_trust_radius" : 1.0e-2,
                            }
                        else: # "Nelder-Mead"
                            E = lambda dBj : self.delta_E_fun(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                            odict["jac"] = False
                            odict["bounds"] = [(-np.pi/2.,np.pi/2.)]
                        odict["options"] = odict.get("options",{})
                        odict["options"]["maxiter"] = self.opt_max_iter
                        res = minimize(E,dB_d,
                            method=self.line_method,
                            tol=self.opt_tol,
                            **odict)
                        # print(t,res)
                        dB_d = res.x[0]
                        self.nit = res.nit
                        self.feval = res.nfev
                        if self.line_method != "Nelder-Mead":
                            self.deval = res.njev
                        else:
                            self.deval = 0
                        if self.line_method == "trust-exact":
                            self.heval = res.nhev
                    elif self.line_method in self.scalar_options:
                        E = lambda dBj : self.delta_E_fun(\
                            rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                        # # search
                        res = minimize_scalar(E,bracket,
                            method=self.line_method,
                            options={"maxiter": self.opt_max_iter}, # 5}, # 20}, # 
                            tol=self.opt_tol,)
                        dB_d = res.x
                        self.nit = res.nit
                        self.feval = res.nfev
                        self.deval = 0.0
                    

                    if self.line_method != "None":
                        E_d = res.fun
                        i_d = res.nit
                        # the below is technically not true, but since E is 
                        # not returned, it doesn't matter (some need _sq)
                        if self.line_method == "Newton_Root":
                            da_d,de_d = self.sine_fun(rho,V,dB_d,a,b,pbar,qbar,rbar,Md)[0:2]
                            if abs(E_d) > 1/self.report_error_threshold:
                                print("t = {:>10.3f}, i = {:>6d}, Z = {:> 12.3e}".format(t,i_d,E_d))
                        else:
                            da_d,de_d = self.delta_E_fun(rho,V,dB_d,a,b,pbar,qbar,rbar,Md)[0:2]
                            if E_d > self.report_error_threshold:
                                print("t = {:>10.3f}, i = {:>6d}, E = {:> 10.3f}".format(t,i_d,E_d))
                    else:
                        self.nit = 0.0
                        self.feval = 0.0
                        self.deval = 0.0
                    

                    # print(t) # ,self.integrator) # 
                    if t >= self.time_check and t <= self._end_plot_time and not(self.have_saved):
                        if self.first_plot:
                            plt.xlabel("Tail rotation, deg")
                            plt.ylabel("f(dB)")# $E$")#  = ||M - M_d||$") # ^2$") # 
                            plt.plot([-360,360],[0.0,0.0],"-",c="k",lw=0.5)
                            if self.log_scale:
                                plt.yscale("log")
                            if self.symlog_scale:
                                plt.yscale("symlog")
                        if self.line_method == "Newton_Root":
                            E = lambda dBj : self.sine_fun(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                        else:
                            E = lambda dBj : self.delta_E_fun_sq(\
                                rho,V,dBj,a,b,pbar,qbar,rbar,Md)[2]
                        #
                        dBvals_deg = np.linspace(-360.0,360.0,20000) # -90.0,90.0,10000) # 
                        dBvals = np.deg2rad(dBvals_deg)
                        Evals = [E(dBvals[i]) for i in range(len(dBvals))]
                        dBcol = plt.plot(dBvals_deg,Evals)[0].get_color()
                        plt.plot(np.rad2deg(dB),E(dB),"o",c="k",ms=2.0,mfc=dBcol)
                        plt.plot(np.rad2deg(dB_d),E(dB_d),"o",c=dBcol,ms=2.0,mfc="w")
                        i_neg = int(abs((-np.pi*2.0-dB_d)/np.pi))
                        for ineg in range(i_neg):
                            dB_in = dB_d - (ineg+1)*np.pi
                            plt.plot(np.rad2deg(dB_in),0.0,"o",c=dBcol,ms=2.0)#,mfc=dBcol)
                        i_pos = int(abs(( np.pi*2.0-dB_d)/np.pi))
                        for ipos in range(i_pos):
                            dB_ip = dB_d + (ipos+1)*np.pi
                            plt.plot(np.rad2deg(dB_ip),0.0,"o",c=dBcol,ms=2.0)#,mfc=dBcol)
                        if self.line_method in self.scalar_options:
                            plt.plot(np.rad2deg(bracket[0]),E(bracket[0]),"x",c=dBcol,ms=3.0)
                            plt.plot(np.rad2deg(bracket[2]),E(bracket[2]),"x",c=dBcol,ms=3.0)
                        plt.title("t = {:> 7.3f}".format(t))
                        # plt.yscale("log")
                        plt.show(block=False)
                        if not(self.have_saved) and \
                            self._end_plot_time - self.dt_check <= t:
                            print("end of times!!!")
                            now = datetime.now()
                            ct = now.strftime("%Y-%m-%d_%H-%M-%S")
                            plt.savefig("/home/ben/Desktop/plotfig_"+ct+".png")
                            plt.close()
                            self.have_saved = True
                        else:
                            plt.pause(self._err_plot_pause_time)
                        self.time_check += self.dt_check
                    # quit()

                else:
                    # define aerodynamics
                    BAM = self.aero_model
                    CL1 = lambda dBi : BAM._CL0(dBi) + BAM._CL_alpha(dBi)*a
                    Cls = lambda dBi : (BAM._Cl0(dBi) + BAM._Cl_alpha(dBi)*a +
                        BAM._Cl_beta(dBi)*b + BAM._Cl_pbar(dBi)*pbar +
                        BAM._Cl_qbar(dBi)*qbar +
                        (BAM._Cl_rbar(dBi) + BAM._Cl_Lrbar(dBi)*CL1(dBi))*rbar)
                    Clda = lambda dBi : BAM._Cl_da(dBi)
                    Clde = lambda dBi : BAM._Cl_de(dBi)
                    Cms = lambda dBi : (BAM._Cm0(dBi) + BAM._Cm_alpha(dBi)*a +
                        BAM._Cm_beta(dBi)*b + BAM._Cm_pbar(dBi)*pbar +
                        BAM._Cm_qbar(dBi)*qbar + BAM._Cm_rbar(dBi)*rbar)
                    Cmda = lambda dBi : BAM._Cm_da(dBi)
                    Cmde = lambda dBi : BAM._Cm_de(dBi)
                    Cns = lambda dBi : (BAM._Cn0(dBi) + BAM._Cn_alpha(dBi)*a +
                        BAM._Cn_beta(dBi)*b +
                        (BAM._Cn_pbar(dBi) + BAM._Cn_Lpbar(dBi)*CL1(dBi))*pbar +
                        BAM._Cn_qbar(dBi)*qbar + BAM._Cn_rbar(dBi)*rbar)
                    Cnda = lambda dBi : BAM._Cn_da(dBi) \
                        + BAM._Cn_Lda(dBi)*CL1(dBi)
                    Cnde = lambda dBi : BAM._Cn_de(dBi)
                    #
                    # determine da, de
                    Cs = lambda dBi : np.array([Cls(dBi),Cms(dBi),Cns(dBi)])
                    Cc = lambda dBi : np.array([
                        [Clda(dBi),Clde(dBi)],
                        [Cmda(dBi),Cmde(dBi)],
                        [Cnda(dBi),Cnde(dBi)]])
                    GCs = lambda dBi : np.matmul(G,Cs(dBi))
                    GCc = lambda dBi : np.matmul(G,Cc(dBi))
                    M = lambda dai,dei,dBi : GCs(dBi) \
                        + np.matmul(GCc(dBi),np.array([dai,dei]))
                    E = lambda u : np.linalg.norm(M(u[0],u[1],u[2])-Md)
                    res = minimize(E,self.u_trim[0:3])
                    da_d,de_d,dB_d = res.x
                    # print(np.rad2deg(da_d),np.rad2deg(de_d),np.rad2deg(dB_d))
                    # quit()
                delta = np.array([da_d,de_d,dB_d])
                # print("{:> 6.3f}::{:> 8.3f} deg, {:> 8.3f} deg, {:> 8.3f} deg"\
                #     .format(t,np.rad2deg(delta[0]),np.rad2deg(delta[1]),\
                #     np.rad2deg(delta[2])))
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((delta,[tcom]))

                # report # # CHECKING CAMA
                ca = cos(a); sa = sin(a)
                cb = cos(b); sb = sin(b)
                #
                CM__u = self.aero_model.aero_results(*[
                    a,b,pbar,qbar,rbar,da_d,de_d,dB_rep, # dB_d, # 
                    False,M,False,False,True
                ])#[3:]
                CM__u[4] = CM__u[4] \
                    + self.cgshift[0]*\
                    (-ca*CM__u[0] -sa*sb*CM__u[1] -sa*cb*CM__u[2] )/self.cw
                CM__u[5] = CM__u[5] \
                    + self.cgshift[0]*(cb*CM__u[1] -sb*CM__u[2] )/self.bw
                CM__u = CM__u[3:]
                CM__c = self.aero_model.aero_results(*[
                    a,b,pbar,qbar,rbar,da_d,de_d,dB_rep, # dB_d, # 
                    self.is_compressible,M,self.use_anderson,self.has_stall,True
                ])#[3:]
                CM__c[4] = CM__c[4] \
                    + self.cgshift[0]*\
                    (-ca*CM__c[0] -sa*sb*CM__c[1] -sa*cb*CM__c[2] )/self.cw
                CM__c[5] = CM__c[5] \
                    + self.cgshift[0]*(cb*CM__c[1] -sb*CM__c[2] )/self.bw
                CM__c = CM__c[3:]
                pd_u = (CMd_u - CM__u)/CMd_u*100.0
                pd_c = (CMd_c - CM__c)/CMd_c*100.0
                threshold = 1.0e-06
                if np.linalg.norm(pd_u) > threshold or np.linalg.norm(pd_c) > threshold:
                    print("CMd   corr =",CMd_c)
                    print("CMd uncorr =",CMd_u)
                    print("CM  uncorr =",CM__u)
                    print("CM    corr =",CM__c)
                    print("unc % diff =",pd_u)
                    print("cor % diff =",pd_c)
                    print()
                    # quit()


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class ITPIAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # gains
        # damping and natural frequency in each axis
        # # # old
        # z_p = 3.0
        # z_q = 2.0
        # z_r = 0.8
        # wn_p = 8.0
        # wn_q = 8.0
        # wn_r = 8.0
        # # # # new
        # z_p = 3.0
        # z_q = 2.0
        # z_r = 0.8
        # wn_p = 8.0
        # wn_q = 8.0
        # wn_r = 6.0
        # # # # new
        # z_p = 3.0
        # z_q = 3.0
        # z_r = 0.8
        # wn_p = 8.0
        # wn_q = 8.0
        # wn_r = 6.0
        # # # new
        z_p = 3.0
        z_q = 4.0
        z_r = 0.8
        wn_p = 8.0
        wn_q = 8.0
        wn_r = 6.0
        #
        self.z = np.diag([ z_p, z_q, z_r])
        self.w = np.diag([wn_p,wn_q,wn_r])

        # invoke init of parent
        self.first_step = True # False # 
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.update_AB = False # True # 
        # below true to get A and B, then put in below


    def _report_trim_other(self,u):
        # calculate cartesian controls
        dm_trim = u[1]*cos(u[2])
        dn_trim = u[1]*sin(u[2])

        # report cartesian controls
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"alt.-pitch[deg,rad]\"",dm_trim*self.rtod,dm_trim))
        print("    {:<23s} : {:> 23.16f} : {:> 23.16f}".format(\
            "\"alt.-yaw[deg,rad]\"",dn_trim*self.rtod,dn_trim))
        return
  
    def _overwrite_initial_x_u(self,x,u):
        # # Build Controller!!!!
        if self.first_step:
            # solve for trim in SLF and build linear system
            phi_trim = self.phi_trim*1.0
            self.phi_trim = 0.0
            u_trim,x_trim = self.run_trim(verbose=False,no_report=True)
            x_trim_euler = np.delete(x_trim,9)
            x_trim_euler[9:12] = self._euler_angles(x_trim)
            x_trim_euler[12:] = x_trim[13:]*1.
            self.x_trim = x_trim; self.u_trim = u_trim
            # print(self.x_trim)
            self.x_trim2 = x_trim; self.u_trim2 = u_trim
            self.x_trim2_euler = x_trim_euler*1.0
            _,self.Lin_Model = self._build_controller(x_tr = x_trim_euler,u_tr = u_trim,
                report=False,save_matrices=False,
                mrrr=[0,1,2,6,7,8,9,10,11],mrrc=[3],
                include_stall_derivatives=False,run_freq=False)
            #
            # transform system
            A = self.Lin_Model.A_min
            Bo = self.Lin_Model.B_min
            Avxvyvz = self.Lin_Model.A[3:6,0:3]
            # An,Bn = self.Lin_Model.build_jacobians(self.x_trim,
            #     self.u_trim,[1.0,0.0,0.0],
            #     numerical = True,
            #     numerical_dynamics = self._nonlinear_euler_dynamics)
            # Avxvyvz = An[3:6,0:3]
            # A = An[3:6,3:6]
            # Bo = Bn[3:6,0:3]
            # print(repr(An))
            # print(repr(Bn))
            V = (self.x_trim2[0]**2. + self.x_trim2[1]**2. + self.x_trim2[2]**2.)**0.5
            a = atan2(self.x_trim2[2],self.x_trim2[0])
            b = asin(self.x_trim2[1]/V)
            self.Abeta = Abeta = np.matmul(Avxvyvz,np.array([
                -V*cos(a)*sin(b), V*cos(b), -V*sin(a)*sin(b)
            ]))
            # trim values
            de_trim = self.u_trim2[1]*1.0
            dB_trim = self.u_trim2[2]*1.0
            dm_trim = de_trim*cos(dB_trim)
            dn_trim = de_trim*sin(dB_trim)
            # print(dm_trim,dn_trim)
            # # transform
            dedm =  abs(dm_trim)/(dn_trim**2. + dm_trim**2.)**0.5
            dedn =  np.sign(dm_trim)*dn_trim/(dn_trim**2. + dm_trim**2.)**0.5
            # dedm =  dm_trim/(dn_trim**2. + dm_trim**2.)**0.5
            # dedn =  dn_trim/(dn_trim**2. + dm_trim**2.)**0.5
            dBdm = -dn_trim/(dn_trim**2. + dm_trim**2.)
            dBdn =  dm_trim/(dn_trim**2. + dm_trim**2.)
            # apply
            self.T = T = np.array([
                [1.0, 0.0, 0.0],
                [0.0,dedm,dedn],
                [0.0,dBdm,dBdn],
            ])
            B = mm(Bo,T)
            #
            Go = co.ctrb(A,Bo); rGo = np.linalg.matrix_rank(Go)
            G  = co.ctrb(A,B) ; rG  = np.linalg.matrix_rank(G )

            self.A = A
            self.B = B
            self.Binv = Binv = np.linalg.solve(B,np.eye(3))
            self.kP = mm(Binv,mm(2.0*self.z,self.w) + A)
            self.kI = mm(Binv,mm(self.w,self.w))

            # prepare for integrator states
            da_trim = self.u_trim2[0]
            dm_trim = self.u_trim2[1]*cos(self.u_trim2[2])
            dn_trim = self.u_trim2[1]*sin(self.u_trim2[2])
            self.altu_trim = np.array([da_trim,dm_trim,dn_trim])

        # add in integrator states
        ref = x[3:6] - self.x_trim2[self.xPi[1:]]
        da,de,dB = u[0:3]
        dm = de*np.cos(dB)
        dn = de*np.sin(dB)
        delta = [da,dm,dn]
        uff = - mm(self.Binv,mm(self.A,ref)) #- mm(self.Binv,self.Abeta*(b-b_trim)) #  *0.0 # 
        eI = np.linalg.solve(self.kI, uff + self.altu_trim - delta)
        x[self.xIi[1:]] = eI

        if self.first_step:
            # return to previous trim solution, linear system
            self.phi_trim = phi_trim
            u_trim,x_trim = self.run_trim(verbose=False,no_report=True)
            x_trim_euler = np.delete(x_trim,9)
            x_trim_euler[9:12] = self._euler_angles(x_trim)
            x_trim_euler[12:] = x_trim[13:]*1.
            self.x_trim = x_trim; self.u_trim = u_trim
            _,self.Lin_Model = self._build_controller(x_tr = x_trim_euler,u_tr = u_trim,
                report=False,save_matrices=False,
                mrrr=self.Lin_Model.mrrr,mrrc=self.Lin_Model.mrrc,
                include_stall_derivatives=False,run_freq=False)

            # report
            print("Bo =",repr(Bo))
            print()
            print("Bo^-1 =",repr(np.linalg.solve(Bo,np.eye(3))))
            print()
            # print("np.diag(Bo) =",repr(np.diag(Bo)))
            print("cond(Bo) =",repr(np.linalg.cond(Bo)))
            print()
            # print("cond(Bo**) dB = 100*db =",repr(np.linalg.cond(mm(Bo,[
            #     [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,100.0]]))))
            # print("Bo**^-1 =",repr(np.linalg.solve(mm(Bo,[
            #     [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,100.0]]),np.eye(3))))
            # print()
            print("rank(Go) =",rGo)
            print()
            print()
            print("Abeta =",repr(Abeta))
            print()
            print("A =",repr(A))
            print()
            # print("np.diag(A) =",repr(np.diag(A)))
            print("T =",repr(T))
            print()
            print("B =",repr(B))
            print()
            print("B^-1 =",repr(Binv))
            print()
            # print("np.diag(B) =",repr(np.diag(B)))
            print("cond(B) =",repr(np.linalg.cond(B)))
            print()
            print("rank(G) =",rG)
            print()
            print("eI0 =",eI)
            print()
            self.first_step=False
        return x,u

    def __del__(self):
        # report gain matrix
        rep2D(self.kI,"Ki",decimals=3)
        rep2D(self.kP,"Kp",decimals=3)
        pass

    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                ref = ref - self.x_trim2_euler[self.xPi_eul[1:]]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # # # transform system
                # if self.first_step:
                #     self.first_step = False
                #     # quit()

                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                Vx      = x_euler[0]
                Vy      = x_euler[1]
                Vz      = x_euler[2]
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                epI     = x_euler[self.xIi_eul[1]]
                eqI     = x_euler[self.xIi_eul[2]]
                erI     = x_euler[self.xIi_eul[3]]
                w  = np.array([  p,  q,  r]) - self.x_trim2_euler[self.xPi_eul[1:]]
                eI = np.array([epI,eqI,erI]) - self.x_trim2_euler[self.xIi_eul[1:]]
                e = w - ref
                #
                V = (Vx**2. + Vy**2. + Vz**2.)**0.5
                a = atan2(Vz,Vx)
                b = asin(Vy/V)
                #
                V_trim = (self.x_trim2[0]**2. + self.x_trim2[1]**2. + self.x_trim2[2]**2.)**0.5
                a_trim = atan2(self.x_trim2[2],self.x_trim2[0])
                b_trim = asin(self.x_trim2[1]/V_trim)
                # # # # # # # # # 
                if self.update_AB:
                    self.Lin_Model.report = False
                    A,B = self.Lin_Model.build_jacobians(x_euler, x_euler[12:16])
                    Avxvyvz = A[3:6,0:3]
                    Abeta = np.matmul(Avxvyvz,np.array([ # self.Abeta =  # 
                        -V*cos(a)*sin(b), V*cos(b), -V*sin(a)*sin(b)
                    ]))
                    rows = [3,4,5]
                    cols = [0,1,2]
                    A  = (A[rows,:])[:,rows]
                    Bo = (B[rows,:])[:,cols]
                    # transform
                    da_state = x_euler[12]
                    de_state = x_euler[13]
                    dB_state = x_euler[14]
                    dm_state = de_state*cos(dB_state)
                    dn_state = de_state*sin(dB_state)
                    # self.altu_trim = np.array([da_state,dm_state,dn_state])
                    # transform
                    dedm = 2.*dm_state/(dm_state**2.+dn_state**2.)**0.5
                    dedn = 2.*dn_state/(dm_state**2.+dn_state**2.)**0.5
                    dBdm = -dn_state/(dn_state**2. + dm_state**2.)
                    dBdn =  dm_state/(dn_state**2. + dm_state**2.)
                    # apply
                    B = Bo*1.0
                    B[:,1] = Bo[:,1]*dedm + Bo[:,2]*dBdm
                    B[:,2] = Bo[:,1]*dedn + Bo[:,2]*dBdn
                    # A # self.A = 
                    # B # self.B = 
                    Binv = np.linalg.solve(B,np.eye(3)) # self.Binv = 
                    # fix initial integrator states so we start at the trim state
                    self.kP = mm(mm(2.0*self.z,self.w) + A,Binv)
                    self.kI = mm(mm(self.w,self.w),Binv)
                # # # # # # # # # 
                uff = - mm(self.Binv,mm(self.A,ref)) #- mm(self.Binv,self.Abeta*(b-b_trim)) #  *0.0 # 
                delta = - mm(self.kP,e) - mm(self.kI,eI) + uff + self.altu_trim
                da,dm,dn = delta
                # dm = -(dm - self.altu_trim[1]) + self.altu_trim[1]
                # 
                de = (dm**2. + dn**2.)**0.5
                # dB = atan2(dn,dm)
                dB = atan(dn/dm)
                # print(t,np.rad2deg(de),np.rad2deg(dB))
                if dm < 0.0:
                    de *= -1.0
                #     dB += -np.pi*np.sign(dn)
                
                # if   dB < -np.pi/2.: de,dB = -de,dB+np.pi
                # elif dB > +np.pi/2.: de,dB = -de,dB-np.pi
                # print("   ",t,np.rad2deg(de),np.rad2deg(dB))
                # print(t,np.rad2deg(de - self.u_trim[1]),np.rad2deg(dB - self.u_trim[2]))

                v = np.array([da,de,dB])
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((v,[tcom]))# #


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class LinearQuadraticTrackingAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        #
        self.first_LQT_step = True
        # self.second_LQT_step = True
        self.use_transform = False # True # 

    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # build controller
                if self.first_LQT_step: # or (self.second_LQT_step and t >= 2.0):
                    # if self.first_LQT_step:
                    #     r0 = np.deg2rad(np.array([[1.0],[0.01],[0.01]])) # /3.0**0.5
                    # elif self.second_LQT_step and t >= 2.0:
                    r0 = np.deg2rad(np.ones((3,1))) # np.array([[1.0],[0.0],[0.0]])) # /3.0**0.5
                    # build system, solve LQR problem
                    A_tr = self.Lin_Model.A_min
                    B_tr = self.Lin_Model.B_min
                    if self.use_transform:
                        # trim values
                        de_trim = self.u_trim[1]*1.0
                        dB_trim = self.u_trim[2]*1.0
                        dm_trim = de_trim*cos(dB_trim)
                        dn_trim = de_trim*sin(dB_trim)
                        # transform
                        dedm = 2.*dm_trim/(dm_trim**2. + dn_trim**2.)**0.5
                        dedn = 2.*dn_trim/(dm_trim**2. + dn_trim**2.)**0.5
                        dBdm =  - dn_trim/(dm_trim**2. + dn_trim**2.)
                        dBdn =    dm_trim/(dm_trim**2. + dn_trim**2.)
                        T = np.array([[1.,0.,0.],[0.,dedm,dedn],[0.,dBdm,dBdn]])
                        # apply
                        B_tr = np.matmul(B_tr,T)
                    Z = np.zeros((3,3))
                    I = np.eye(3)
                    A = np.block([[A_tr,Z],[I,Z]]) # np.block([[Z,I],[Z,A_tr]])
                    B = np.block([[B_tr],[Z]])
                    H = np.block([[I,Z]])
                    Q = mm(H.T,H)
                    if self.use_transform:
                        Q[0:3,0:3] = np.diag([1.0e+0]*3)
                        Q[3:6,3:6] = np.diag([5.0e+0]*3)
                        R = np.diag([1.0e-2,1.0e-2,1.0e-2])
                        GK = np.zeros((3,6))
                        # GK = np.array([
                        #     [0.0e+0,2.0e+0,0.0e+0,0.0e+0,2.0e+1,0.0e+0],
                        #     [1.0e+1,0.0e+0,0.0e+0,1.0e+2,0.0e+0,0.0e+0],
                        #     [1.0e+1,2.0e+0,0.0e+0,1.0e+2,2.0e+1,0.0e+0]
                        # ])
                    else:
                        Q[0:3,0:3] = np.diag([5.0e-1,1.0e+0,1.0e+0])
                        Q[3:6,3:6] = np.diag([5.0e+0,1.0e+1,5.0e+0])
                        R = np.diag([1.e-2]*2 + [10.0]) # 1.0e+1*I # 
                        # R = 1.0e+0*I
                        GK = np.array([
                            [0.0e+0,2.0e+0,0.0e+0,0.0e+0,2.0e+1,0.0e+0],
                            [1.0e+1,0.0e+0,0.0e+0,1.0e+2,0.0e+0,0.0e+0],
                            [1.0e+1,2.0e+0,0.0e+0,1.0e+2,2.0e+1,0.0e+0]
                        ])
                    # A,Q obsv
                    if np.linalg.matrix_rank(co.obsv(A,Q**0.5)) < 6:
                        raise ValueError("A,sqrt(Q) must be observable!!!")
                    C = np.block([[I,Z],[Z,I]]) # np.eye(6) # 
                    # initialize K
                    Kshape = (3,6)
                    Kflat = (18,)
                    k0,_,_ = co.lqr(A,B,Q,R)
                    print(k0)
                    # k0 = np.zeros(Kshape)
                    # k0 = np.ones(Kshape)
                    # k0 = np.block([[I,I]])
                    K0 = k0.reshape(Kflat)
                    G = np.block([[ Z],[-I]])
                    F = np.block([[-I],[ Z]])
                    V = Z*0.
                    def minJ(K,A,B,C,Q,R,F,G,H,V,r0):
                        K = K.reshape(Kshape)
                        CKRKC = mm(C.T,mm(K.T,mm(R,mm(K,C))))
                        CKRKC = (CKRKC.T + CKRKC)/2.
                        QCKRKC = Q + CKRKC
                        Ac = A - mm(B,mm(K,C))
                        Bc = G - mm(B,mm(K,F))
                        P = co.lyap(Ac.T,QCKRKC)
                        # print(Ac)
                        Acinv = np.linalg.solve(Ac,np.eye(Ac.shape[0]))
                        X = mm(Acinv,mm(Bc,mm(r0,mm(r0.T,mm(Bc.T,Acinv.T)))))
                        ebar = mm(1. + mm(H,mm(Acinv,Bc)),r0)
                        J = 0.5*np.trace(mm(P,X)) \
                            + 0.5*mm(ebar.T,mm(V,ebar))[0,0]
                        J = abs(J)
                        # print(J)
                        return J
                    def minJGK(K,A,B,C,Q,R,GK,F,G,H,V,r0):
                        K = K.reshape(Kshape)
                        CKRKC = mm(C.T,mm(K.T,mm(R,mm(K,C))))
                        CKRKC = (CKRKC.T + CKRKC)/2.
                        QCKRKC = Q + CKRKC
                        Ac = A - mm(B,mm(K,C))
                        Bc = G - mm(B,mm(K,F))
                        P = co.lyap(Ac.T,QCKRKC)
                        # print(Ac)
                        Acinv = np.linalg.solve(Ac,np.eye(Ac.shape[0]))
                        X = mm(Acinv,mm(Bc,mm(r0,mm(r0.T,mm(Bc.T,Acinv.T)))))
                        ebar = mm(1. + mm(H,mm(Acinv,Bc)),r0)
                        J = 0.5*np.trace(mm(P,X)) \
                            + 0.5*mm(ebar.T,mm(V,ebar))[0,0] + (GK*K*K).sum()
                        J = abs(J)
                        # print(J)
                        return J
                    def gradminJ(K,A,B,C,Q,R,F,G,H,V,r0):
                        K = K.reshape(Kshape)
                        CKRKC = mm(C.T,mm(K.T,mm(R,mm(K,C))))
                        CKRKC = (CKRKC.T + CKRKC)/2.
                        QCKRKC = Q + CKRKC
                        Ac = A - mm(B,mm(K,C))
                        Bc = G - mm(B,mm(K,F))
                        P = co.lyap(Ac.T,QCKRKC)
                        # print(Ac)
                        Acinv = np.linalg.solve(Ac,np.eye(Ac.shape[0]))
                        xbar = -mm(Acinv,mm(Bc,r0))
                        ybar = mm(C,xbar) + mm(F,r0)
                        X = mm(xbar,xbar.T)
                        ebar = mm(1. + mm(H,mm(Acinv,Bc)),r0)
                        #
                        S = co.lyap(Ac,X)
                        #
                        J = 0.5*np.trace(mm(P,X)) \
                            + 0.5*mm(ebar.T,mm(V,ebar))[0,0]
                        J = abs(J)
                        #
                        dJdK = mm(R,mm(K,mm(C,mm(S,C.T)))) \
                            - mm(B.T,mm(P,mm(S,C.T))) \
                            + mm(B.T,mm(Acinv.T,mm(P \
                                + mm(H.T,mm(V,H)),mm(xbar,ybar.T)))) \
                            - mm(B.T,mm(Acinv.T,mm(H.T,mm(V,mm(r0,ybar.T)))))
                        dJdK = dJdK.reshape(Kflat)
                        # print(J)
                        # print("dJdK shape =",dJdK.shape)
                        return J,dJdK
                    res = minimize(minJGK, # minJ, # gradminJ, # 
                        K0,args=(A,B,C,Q,R,GK,F,G,H,V,r0), # F,G,H,V,r0), # 
                        jac=None, # True, # 
                        method="SLSQP") # "Nelder-Mead") # ) # 
                    K = res.x.reshape(Kshape)*1.0
                    print(res)
                    if not res.success:
                        raise ValueError("LQT optimization failed!!")
                    cl_evals,cl_evecs = np.linalg.eig(A - mm(B,mm(K,C)))
                    print("eval cl =",cl_evals)
                    self.KP_DI,self.KI_DI = K[:,0:3],K[:,3:6]
                    print(self.KI_DI)
                    print(self.KP_DI)
                    self.KC_LQT = np.matmul(K,C)
                    self.KF_LQT = np.matmul(K,F)
                    # quit()
                    if self.first_LQT_step:
                        self.first_LQT_step = False
                    # elif self.second_LQT_step and t >= 2.0:
                    #     self.second_LQT_step = False
                    
                    if False: # True: # 
                        # closed-loop simulation
                        # change plot text parameters
                        plt.rcParams["font.family"] = "Serif"
                        plt.rcParams["font.size"] = 8.0
                        plt.rcParams["axes.labelsize"] = 8.0
                        plt.rcParams['axes.xmargin'] = 0
                        plt.rcParams['lines.linewidth'] = 0.75 # 1.0
                        plt.rcParams["xtick.minor.visible"] = True
                        plt.rcParams["ytick.minor.visible"] = True
                        plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
                        plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
                        plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
                        plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
                        plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
                        plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
                        plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
                        plt.rcParams["mathtext.fontset"] = "dejavuserif"
                        plt.rcParams['figure.dpi'] = 300.0
                        x0 = np.zeros((6,))
                        #
                        x0[3] = -self._get_reference(0.0)[3]
                        def dyn(t,x):
                            dx = np.matmul(A-np.matmul(B,np.matmul(K,C)),x)
                            return dx
                        # simulate
                        ts  = np.linspace(0.0,10.0,num=1001)
                        ts0 = np.linspace(0.0, 2.0,num= 201)
                        xs0 = odeint(dyn,x0,ts0,tfirst=True).T
                        x1 = xs0[:,-1]
                        x1[3] = x1[3] - x0[3] - self._get_reference(2.1)[3]
                        x1[4] = x1[4] - x0[4] - self._get_reference(2.1)[4]
                        x1[5] = x1[5] - x0[5] - self._get_reference(2.1)[5]
                        ts1 = np.linspace(2.0,10.0,num= 801)
                        xs1 = odeint(dyn,x1,ts1,tfirst=True).T
                        xs = np.concatenate((xs0[:,:-1],xs1),axis=1)
                        # print(xs.shape)
                        xs = np.rad2deg(xs)
                        us = np.array([-np.matmul(np.matmul(K,C),xsi) for xsi in xs.T]).T
                        fgs,axs = plt.subplots(1,3,
                            figsize=(6.0,3.0),dpi=300.0,sharex=True,constrained_layout=True)
                        lss = ["-","--","-."]
                        names = ["p","q","r"]
                        cnms = [r"$\delta_a$",r"$\delta_e^B$",r"$\delta_B$"]
                        for i in range(3):#xs.shape[0]):
                            par = dict(c="k",ls=lss[i],lw=0.5)
                            axs[1].plot(ts,xs[i+3],label=r"$\int e_{"+names[i]+r"} \, dt$",**par)
                            axs[0].plot(ts,xs[i  ],label=     r"$e_{"+names[i]+r"}$"      ,**par)
                            axs[2].plot(ts,us[i  ],label=cnms[i],**par)
                        axs[1].set_xlim(ts[0],ts[-1])
                        axs[1].set_ylabel(r"integrator [$^\circ$]")
                        axs[0].set_ylabel(r"error [$^\circ$/s]")
                        axs[2].set_ylabel(r"control [$^\circ$]")
                        axs[0].legend()
                        axs[1].legend()
                        axs[2].legend()
                        plt.show()

                        quit()

                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                trim_slice = self.x_trim_euler[self.Lin_Model.Cslice]
                ref_slice = ref - trim_slice
                slices = [3,4,5] + self.xIi_eul[1:]
                x_slice = x_euler[slices] - self.x_trim_euler[slices]
                
                delta = - np.matmul(self.KC_LQT,x_slice) - np.matmul(self.KF_LQT,ref_slice)

                # convert delta
                if self.use_transform:
                    dm = delta[1]
                    dn = delta[2]
                    dB = atan2(dn,dm)
                    de = np.sign(dm/np.cos(dB))*(dm**2. + dn**2.)**0.5 # 
                    # print(t,np.rad2deg(dB),np.rad2deg(de))
                    # if dB < -np.pi/2.:
                    #     # # print("-np.pi/2.")
                    #     # e2s = e1s = abs(dB) // np.pi
                    #     # mult = +1.0
                    #     # dB += np.pi
                    #     de *= -1.0
                    # elif dB > +np.pi/2.:
                    #     # # print("+np.pi/2.")
                    #     # e2s = e1s = abs(dB) // np.pi
                    #     # mult = -1.0
                    #     # dB -= np.pi
                    #     de *= -1.0
                    # else: # if True:#
                    #     e2s = -1
                    #     e1s = 1
                    #     mult = +1.0
                    # dB += mult*(e2s + 1)*np.pi
                    # de = (-1.0)**(e1s + 1)*(dm**2. + dn**2.)**0.5 # 
                    # print(t,np.rad2deg(dB),np.rad2deg(de))
                    # print()

                    delta = np.array([delta[0],de,dB])
                
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((delta + self.u_trim[0:3],[tcom]))


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs


class LinearQuadraticRegulatorDynamicInversionAircraft(Aircraft):
    """A default class for calculating and containing the mass properties of a
    Cuboid.

    Parameters
    ----------
    input_vars : dict , optional
        Must be a python dictionary
    """
    def __init__(self,input_dict={}):

        # invoke init of parent
        Aircraft.__init__(self,input_dict,folder_prefix = "track")
        self.tracking = True
        self.first_LQDI_step = True # False # 
    
    def __del__(self):
        print("A =",self.A_tr)
        print("Binv =",self.Binv_LQRDI)
        print("KI =",self.KI_LQRDI)
        print("KP =",self.KP_LQRDI)

    def _get_control(self,t,x,is_controlled=True,given_control=False,u="o",
        force_control_to_inputs=False):
        # build control or pass through
        if not given_control:
            if is_controlled and (not(self.enforce_update_frequency) or 
                (self.enforce_update_frequency and self.can_update) ):
                if self.use_quaternions:
                    x_euler = self.quat2euler_state(x)
                else:
                    x_euler = x*1.
                    # reset angles
                    x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
                #
                ref = self._get_reference(t)[self.Lin_Model.Cslice]
                # per dave, full stick should be 270 deg/s in aileron
                # 120 deg/s in elevator
                # 60 deg/s in rudder
                #

                # dynamic inversion!!!
                if self.first_LQDI_step:
                    # state
                    self.b_prev = 0.0
                    self.b_counter = 0. # -1 # 
                    self.b_ref = np.deg2rad(+0.0020460560691862)
                    Vxb = self.x_trim[0]; Vyb = self.x_trim[1]; Vzb = self.x_trim[2]
                    V = (Vxb**2. + Vyb**2. + Vzb**2.)**0.5
                    u2w2 = Vxb**2. + Vzb**2.
                    den = V**2.*(V**2. - Vyb**2.)**0.5
                    T = np.zeros((3,3))
                    T[0,0] = 2.*Vxb/V
                    T[0,1] = 2.*Vyb/V
                    T[0,2] = 2.*Vzb/V
                    T[1,0] = -Vzb/u2w2
                    T[1,2] =  Vxb/u2w2
                    T[2,0] = -2.*Vxb*Vyb/den
                    T[2,1] = (V**2. - 2.*Vyb**2.)/den
                    T[2,2] = -2.*Vyb*Vzb/den
                    Z = np.zeros((3,3))
                    I = np.eye(3)
                    T = np.block([[T,Z],[Z,I]])
                    Tinv = np.linalg.solve(T,np.eye(6))
                    # build system
                    rows = [0,1,2,3,4,5]; cols = [0,1,2]
                    A_tr = (self.Lin_Model.A[rows])[:,rows]
                    B_tr = (self.Lin_Model.B[rows])[:,cols]
                    rows = [2,3,4,5]
                    A_tr = (np.matmul(T,np.matmul(A_tr,Tinv))[rows])[:,rows]
                    B_tr = np.matmul(T,B_tr)[rows]
                    self.A_tr = A_tr
                    # apply
                    self.Binv_LQRDI = np.linalg.pinv(B_tr)
                    # solve LQR problem
                    Z = np.zeros((4,4))
                    I = np.eye(4)
                    A = np.block([[Z,I],[Z,A_tr]])
                    B = np.block([[np.zeros((4,3))],[B_tr]])
                    # Q = np.diag([1.0e-2,1.0e-2,1.0e-2]+[1.0e+0,2.0e+0,1.0e+1])
                    ## vvv FROM SUMMER REPORT
                    Q = np.diag([1.0e-2]+[1.0e-1]*3+[1.0e-0,1.0e+1,1.0e+1,1.0e+1])
                    R = np.diag([1.0e+0,1.0e+0,1.0e+1])
                    ## ^^^ FROM SUMMER REPORT
                    # Q = np.diag([1.0e-2]+[5.0e-1]*3+[1.0e-0,1.0e+1,1.0e+1,1.0e+1])
                    # Q[1,3] = Q[3,1] = 5.0e-1
                    # Q[5,7] = Q[7,5] = 1.0e+1
                    # R = np.diag([1.0e+0,1.0e+0,1.0e+1]) # 5.0e-1]) # 
                    K,_,K_eigs = co.lqr(A,B,Q,R,method="scipy")
                    self.KI_LQRDI,self.KP_LQRDI = K[:,0:4],K[:,4:8]
                    print("KI =",self.KI_LQRDI)
                    print("KP =",self.KP_LQRDI)
                    print(K_eigs)
                    self.first_LQDI_step = False

                #-------------------#
                # STATE DEFINITIONS #
                #-------------------#
                Vxb     = x_euler[0]
                Vyb     = x_euler[1]
                Vzb     = x_euler[2]
                p       = x_euler[3]
                q       = x_euler[4]
                r       = x_euler[5]
                epI     = x_euler[self.xIi_eul[1]]
                eqI     = x_euler[self.xIi_eul[2]]
                erI     = x_euler[self.xIi_eul[3]]
                V = (Vxb**2. + Vyb**2. + Vzb**2.)**0.5
                a = atan2(Vzb,Vxb)
                b = asin(Vyb/V)
                w  = np.array([  p,  q,  r])
                eI = np.array([epI,eqI,erI])
                x_trim = self.x_trim
                dref = ref - x_trim[3:6]
                dref = np.concatenate(([0.0],dref))
                refdot = dref*0.0
                e = w - ref
                self.b_counter += 1
                bI = self.b_prev + (b - self.b_prev)*self.dt
                if self.b_counter % 5 == 0: self.b_prev = b
                # print(t,self.b_prev)
                # # vvv FROM SUMMER REPORT
                eI = np.concatenate(([0.0],eI))
                # # ^^^ FROM SUMMER REPORT
                # eI = np.concatenate(([bI],eI))
                e = np.concatenate(([b - self.b_ref],e))
                # if 0.0 <= t <= 0.0 + self.dt:
                #     refdot = e/self.dt
                # elif 2.0 <= t <= 2.0 + self.dt:
                #     refdot = e/self.dt

                nu = - np.matmul(self.KP_LQRDI,e) - np.matmul(self.KI_LQRDI,eI)
                ff = np.matmul(self.Binv_LQRDI,-(np.matmul(self.A_tr,dref) + refdot))
                nu += ff
                
                #
                # tcom = self.u_trim[3]
                tcom = self._get_V_tau_control(t,x_euler)
                #
                u = np.concatenate((nu + self.u_trim[0:3],[tcom]))# # 


                if self.order > 0:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
                # #
                self.u_til_next_update = u*1.
                self.can_update = False
            elif is_controlled and self.enforce_update_frequency and \
                not(self.can_update):
                u = self.u_til_next_update*1.
                if self.order > 0:
                    q = 1*self.use_quaternions
                    ## INTSTATE
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
            else:
                inputs = u = self.Lin_Model.uhat_eq*1.
        elif given_control:
            if u[0] == "o":
                raise TypeError("Control input required.")
            else:
                if self.order > 0 and not force_control_to_inputs:
                    q = 1*self.use_quaternions
                    inputs = x[12+q:16+q]*1.
                else:
                    inputs = u*1.
        
        # limit actuators
        # #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.integrator == "odeint":
            u = self._limit_input(u)
        # #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        inputs = self._limit_input(inputs)
        if self.order > 0:
            q = 1*self.use_quaternions
            x[12+q:16+q] = np.array(inputs)*1.
        # quantize actuators
        inputs = self._quantize_input(inputs)

        return u,inputs



def _float2str(value):
    # _num_format = ':.4g'
    # return f"{value:{_num_format}}"
    return "{:.4f}".format(value)

def _tf_factorized_polynomial_to_string(roots, gain=1, var='s'):
    """Convert a factorized polynomial to a string"""

    if roots.size == 0:
        return _float2str(gain)

    factors = []
    for root in sorted(roots, reverse=True):
        if np.isreal(root):
            if root == 0:
                factor = f"{var}"
                factors.append(factor)
            elif root > 0:
                factor = f"{var} - {_float2str(np.abs(root))}"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}"
                factors.append(factor)
        elif np.isreal(root * 1j):
            if root.imag > 0:
                factor = f"{var} - {_float2str(np.abs(root))}j"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}j"
                factors.append(factor)
        else:
            if root.real > 0:
                factor = f"{var} - ({_float2str(root)})"
                factors.append(factor)
            else:
                factor = f"{var} + ({_float2str(-root)})"
                factors.append(factor)

    multiplier = ''
    if round(gain, 4) != 1.0:
        multiplier = _float2str(gain) + " "

    if len(factors) > 1 or multiplier:
        factors = [f"({factor})" for factor in factors]

    return multiplier + " ".join(factors)

def print_tf(tf,num_name="",den_name="",var="s"):

    z,p,k = scipy_tf2zpk(tf.num[0][0],tf.den[0][0])

    numstr = _tf_factorized_polynomial_to_string(z, gain=k, var=var)
    denstr = _tf_factorized_polynomial_to_string(p, var=var)

    # Figure out the length of the separating line
    dashcount = max(len(numstr), len(denstr))
    dashes = '-' * dashcount

    # Center the numerator or denominator
    if len(numstr) < dashcount:
        numstr = ' ' * ((dashcount - len(numstr)) // 2) + numstr
    if len(denstr) < dashcount:
        denstr = ' ' * ((dashcount - len(denstr)) // 2) + denstr
    
    # add label
    namecount = max(len(num_name),len(den_name))
    numstr = "{:^{}s}   ".format(num_name,namecount) + numstr
    dashes = "-"*namecount + " = " + dashes
    denstr = "{:^{}s}   ".format(den_name,namecount) + denstr

    outstr = "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"

    print(outstr)

    return outstr

def build_bire_controller(bire_dict,save_folder):

    # build linearized model
    mrrr = [6,7,8,11]
    mrrc = [3]
    print("BIRE")
    bire_dict["controller"]["type"] = "none"
    bire = Aircraft(bire_dict)
    bire._report_trim_solution(bire.x_trim,bire.u_trim)
    # bire.use_quaternions = False
    _,Lin_Model = bire._build_controller(
        bire.x_trim_euler,bire.u_trim,
        report=False,save_matrices=False,
        mrrr=mrrr,mrrc=mrrc,drop_actrs=True,
        # use_VAB_format=True,
        # use_numerical_linearization=True,
        # numerical_dynamics=bire._nonlinear_euler_dynamics,
        include_stall_derivatives=False,skip_reporting=True,run_freq=False)
    # bire.use_quaternions = True
    aa = Lin_Model.A_min*1.
    print(np.linalg.eig(aa)[0])
    ba = Lin_Model.B_min*1.
    ##
    # trim values
    de_trim = bire.u_trim[1]*1.0; dB_trim = bire.u_trim[2]*1.0
    dm_trim = de_trim*cos(dB_trim); dn_trim = de_trim*sin(dB_trim)
    # transform
    dedm = 2.*dm_trim/(dm_trim**2. + dn_trim**2.)**0.5
    dedn = 2.*dn_trim/(dm_trim**2. + dn_trim**2.)**0.5
    dBdm =  - dn_trim/(dm_trim**2. + dn_trim**2.)
    dBdn =    dm_trim/(dm_trim**2. + dn_trim**2.)
    T = np.array([[1.,0.,0.],[0.,dedm,dedn],[0.,dBdm,dBdn]])
    # apply
    ba = np.matmul(ba,T)
    ##
    ca = np.block([np.zeros((3,3)),np.eye(3),np.zeros((3,2))])
    # add in integrator states # # p q r pI qI rI
    Z3 = np.zeros((aa.shape[0],3))
    I3 = np.eye(3)
    Zbf = np.zeros((3,3))
    Zaf = np.zeros((3,2))
    A = np.block([[aa,Z3],[Zbf,I3,Zaf,Zbf]])
    B = np.block([[ba],[Zbf]])
    print(B.shape)
    C = np.block([[ca,Zbf],[Z3.T,np.eye(3)]]) # np.block([ca,Z3]) # 
    D = 0.0
    # report
    print(np.linalg.eig(A)[0])
    report_latex(A,"A",decimals=5,predecimals=5,print_report=True)
    report_latex(B,"B",decimals=5,predecimals=5,print_report=True)
    report_latex(C,"C",decimals=5,predecimals=5,print_report=True)

    # longitudinal eigenvalues
    row_lon = [0,2,4,7]
    aa_lon = (A[row_lon,:])[:,row_lon]
    eval_lon,evec_lon = np.linalg.eig(aa_lon)
    en = eval_lon[:,np.newaxis]
    report_latex(en,r"\lambda_{lon}",decimals=5,predecimals=5,print_report=True)
    # lateral eigenvalues
    row_lat = [1,3,5,6]
    aa_lat = (A[row_lat,:])[:,row_lat]
    eval_lat,evec_lat = np.linalg.eig(aa_lat)
    et = eval_lat[:,np.newaxis]
    report_latex(et,r"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    
    # run through by loop
    # p
    t_c_max = 1.0
    s_max = 1./t_c_max
    #
    row = [3,6,8]
    kpP = -0.6 # -0.045
    acl = A - np.matmul(B,np.block([
        [np.array([0.0]*3 + [kpP] + [0.0]*7)],
        [np.zeros((2,A.shape[0]))]
    ]))
    sys = co.ss((acl[row,:])[:,row],B[row,0],C[3,row],D)
    tf = co.ss2tf(sys)
    # print(co.ssdata(sys))
    print_tf(tf,"p","da")
    k = -np.logspace(-1,1.0,1000)
    # print(k)
    kpI = -6.8 # -1.15
    kai = np.argmin(np.abs(k-kpI)) # 
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    # print(r.shape)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri))#,c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    # # plot wn,z
    # wn = 7.0 # rad/s
    # zt = 0.6
    # th = np.linspace(0.0,2.*np.pi,100)
    # wnx = 

    # axs.set_xlim((-1,1))
    # axs.set_ylim((-1,1))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # fig.savefig(save_folder + "/p_rlocus.png",dpi=300.0)
    show_p_rlocus = False # True # 
    if show_p_rlocus:
        plt.show()
    else:
        plt.close("all")
    # #
    # #
    # new closed loop system
    acl = acl - np.matmul(B,np.block([
        [np.array([0.0]*8 + [kpI] + [0.0]*2)],
        [np.zeros((2,A.shape[0]))]
    ]))
    # eigenvalues
    row_lat = [1,3,5,6,8,10]
    aa_lat = (acl[row_lat,:])[:,row_lat]
    eval_lat,evec_lat = np.linalg.eig(aa_lat)
    et = eval_lat[:,np.newaxis]
    report_latex(et,r"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    # #
    # #
    # #
    # q
    W = bire.inertia_model.W
    V = bire.V0
    rho = bire.rho0
    Sw = bire.Sw
    CW = W/0.5/rho/V**2./Sw
    CAP = np.array([0.28, 3.6])
    wn_sp_lim = (CAP*bire.aero_model.CLa/CW)**0.5
    zn_sp_lim = np.array([0.35, 1.3])
    # phugoid
    zt_ph_min = 0.04
    #
    row = [0,2,4,7,9]
    kqP = -1.0
    acl = A - np.matmul(B,np.block([
        [np.zeros((1,A.shape[0]))],
        [np.array([0.0]*4 + [kqP] + [0.0]*6)],
        [np.zeros((1,A.shape[0]))]
    ]))
    sys = co.ss((acl[row,:])[:,row],B[row,1],C[4,row],D)
    tf = co.ss2tf(sys)
    # print(tf)
    # print(co.ssdata(sys))
    print_tf(tf,"q","de")
    k = -np.logspace(-2,2,1000)
    # print(k)
    kqI = -5.0 # 0.01
    kai = np.argmin(np.abs(k-kqI)) # 
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    print("evals at goal =",r[kai])
    # report wn, zt
    rat = r[kai]
    sg = -np.real(rat)
    wd = np.abs(np.imag(rat))
    e0s = np.array([complex(-sg[j],wd[j]) for j in range(len(sg))])
    e1s = np.conjugate(e0s)
    wn = np.sqrt(e0s*e1s)
    zt = - (e0s + e1s)/2./np.sqrt(e0s*e1s)
    print("   wn at goal =",wn)
    print(" zeta at goal =",zt)
    print("sp    wn lims =",wn_sp_lim[0]," ",wn_sp_lim[1])
    print("sp  zeta lims =",zn_sp_lim[0]," ",zn_sp_lim[1])
    print("ph  zeta min  =",zt_ph_min)
    # print(r.shape)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri))#,c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    # plot limits
    # sg = np.outer(-wn,zn)
    # # print(sg)
    # wd = np.abs(np.outer(wn,(1. - np.array(zn,dtype=complex))**0.5))
    # # print(wd)
    # #
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # fig.savefig(save_folder + "/q_rlocus.png",dpi=300.0)
    show_q_rlocus = True # False # 
    if show_q_rlocus:
        plt.show()
    else:
        plt.close("all")
    # #
    # #
    # new closed loop system
    acl = acl - np.matmul(B,np.block([
        [np.zeros((1,A.shape[0]))],
        [np.array([0.0]*9 + [kqI] + [0.0]*1)],
        [np.zeros((1,A.shape[0]))]
    ]))
    # eigenvalues
    row_lon = [0,2,4,7,9]
    aa_lon = (acl[row_lon,:])[:,row_lon]
    eval_lon,evec_lon = np.linalg.eig(aa_lon)
    en = eval_lon[:,np.newaxis]
    report_latex(en,r"\lambda_{lon}",decimals=5,predecimals=5,print_report=True)
    # #
    # #
    # #
    # # r
    # # dutch roll lims, spiral
    # zn_dr_min = 0.4
    # sg_dr_min = 0.4
    # wn_dr_min = 1.0
    # #
    # row = [1,3,5,6,8,10]
    # krP = -0.8
    # acl = A - np.matmul(B,np.block([
    #     [np.zeros((2,A.shape[0]))],
    #     [np.array([0.0]*5 + [krP] + [0.0]*5)]
    # ]))
    # sys = co.ss((acl[row,:])[:,row],B[row,2],C[5,row],D)
    # tf = co.ss2tf(sys)
    # # print(tf)
    # # print(co.ssdata(sys))
    # print_tf(tf,"r","dr")
    # k = -np.logspace(-1,2,1000)
    # # print(k)
    # krI = -15.0
    # kai = np.argmin(np.abs(k-krI)) # 
    # # print(k[kai])
    # r,_ = co.rlocus(sys,kvect=k)
    # print("evals at goal =",r[kai])
    # # report wn, zt
    # rat = r[kai]
    # sg = -np.real(rat)
    # wd = np.abs(np.imag(rat))
    # e0s = np.array([complex(-sg[j],wd[j]) for j in range(len(sg))])
    # e1s = np.conjugate(e0s)
    # wn = np.sqrt(e0s*e1s)
    # zt = - (e0s + e1s)/2./np.sqrt(e0s*e1s)
    # t2d = -np.log(2.0)/sg
    # print("   wn at goal =",wn)
    # print(" zeta at goal =",zt)
    # print("t2dbl at goal =",t2d)
    # print("dr    wn min  =",wn_dr_min)
    # print("dr  zeta min  =",zn_dr_min)
    # print("dr    sg min  =",sg_dr_min)
    # print("sl t2dbl min  =",20.0)
    # # print(r.shape)
    # plt.close()
    # fig,axs = plt.subplots(figsize=(3.25,3.5),dpi=300.0,constrained_layout=True)
    # # print(r.shape,k.shape)
    # for i in range(r.shape[1]):
    #     ri = r[:,i]
    #     axs.plot(np.real(ri),np.imag(ri))#,c="k")
    #     axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
    #     axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    # #
    # axs.set_xlabel("Real [s]")
    # axs.set_ylabel("Imaginary [s]")
    # axs.set_title("Root Locus")
    # axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # fig.savefig(save_folder + "/r_rlocus.png",dpi=300.0)
    # show_r_rlocus = False
    # if show_r_rlocus:
    #     plt.show()
    # else:
    #     plt.close("all")
    # #
    # #
    # # new closed loop system
    # acl = acl - np.matmul(B,np.block([
    #     [np.zeros((2,A.shape[0]))],
    #     [np.array([0.0]*10 + [krI])]
    # ]))
    # # eigenvalues
    # row_lat = [1,3,5,6,8,10]
    # aa_lat = (acl[row_lat,:])[:,row_lat]
    # eval_lat,evec_lat = np.linalg.eig(aa_lat)
    # et = eval_lat[:,np.newaxis]
    # report_latex(et,r"\lambda_{lat}",decimals=5,predecimals=5,print_report=True)
    # # all eigvals
    # eval,evec = np.linalg.eig(acl)
    # et = eval[:,np.newaxis]
    # report_latex(et,r"\lambda_{cl}",decimals=5,predecimals=5,print_report=True)
    # print("kpP =",-kpP)
    # print("kpI =",-kpI)
    # print("kqP =",-kqP)
    # print("kqI =",-kqI)
    # print("krP =",-krP)
    # print("krI =",-krI)

    # # actual subsystem
    # rows = [3,4,5,8,9,10]
    # aa_cl = (acl[rows,:])[:,rows]
    # print(aa_cl)
    # eval,evec = np.linalg.eig(aa_cl)
    # et = eval[:,np.newaxis]
    # report_latex(et,r"\lambda_{cl}",decimals=5,predecimals=5,print_report=True)
    

    return


if __name__ == "__main__":

    # filenames 
    base_fs_file = "base_fs_in.json"
    bire_fs_file = "bire_fs_in.json"
    # base_rc_file = "base_rc_in.json"
    # bire_rc_file = "bire_rc_in.json"

    # read in json to ensure no file changes while running
    base_fs_dict = json.loads( open(base_fs_file).read() )
    bire_fs_dict = json.loads( open(bire_fs_file).read() )
    # base_rc_dict = json.loads( open(base_rc_file).read() )
    # bire_rc_dict = json.loads( open(bire_rc_file).read() )

    
    # # trim for BIRE, determine LQR for controller code example
    # V = 520.0
    # H = 19400.0
    # compr = False
    # stall = False
    # phi_trim = 45.0
    # #
    # bire_fs_dict["initial"].pop("mach")
    # bire_fs_dict["initial"]["airspeed[ft/s]"] = V
    # bire_fs_dict["initial"]["altitude[ft]"] = H
    # bire_fs_dict["initial"]["trim"]["bank_angle[deg]"] = phi_trim
    # bire_fs_dict["simulation"]["include_compressibility"] = compr
    # bire_fs_dict["simulation"]["include_stall"] = stall
    # bire_fs_dict["simulation"]["use_fitted_thrust_model"] = False
    # bire_fs_dict["initial"]["trim"]["type"] = "sct"
    # bire_fs_dict["initial"]["type"] = "trim"
    # bire = Aircraft(bire_fs_dict)
    # # print(bire.inertia_model.W)
    # # print(bire.cgshift)
    # bire._report_trim_solution()
    # # # build linearized system
    # # bire._build_controller(save_matrices=False,mrrr=[0,1,2,6,7,8,9,10,11],
    # #     mrrc=[3],drop_actrs=True,run_freq=False)
    # quit()

    # # build controller
    # build_bire_controller(bire_fs_dict,"FS_bire_control_design")
    # quit()

    plot_vars = {
        "show" : False,
        "plot_full" : True,
        "plot_delta" : True,
        "zoom_deltas" : True,
        # "zoom_fraction" : 0.05,
        "zoom_fraction" : 0.13333333333333333333333,
        "transparent" : False, # True, # 
        "format" : "pdf"
    }

    # bire FM
    bire_fs_FM_errs = [
        0.25  , # CL
        0.25  , # CS
        0.25  , # CD
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # base FM
    base_fs_FM_errs = [
        0.25  , # CL
        0.25  , # CS
        0.25  , # CD
        0.25  , # Cl
        0.25  , # Cm
        0.25   # Cn
    ]
    # RC
    # # bire FM
    # bire_rc_FM_errs = [
    #     0.25  , # CL
    #     0.25  , # CS
    #     0.25  , # CD
    #     0.25  , # Cl
    #     0.25  , # Cm
    #     0.25   # Cn
    # ]
    # # base FM
    # base_rc_FM_errs = [
    #     0.25  , # CL
    #     0.25  , # CS
    #     0.25  , # CD
    #     0.25  , # Cl
    #     0.25  , # Cm
    #     0.25   # Cn
    # ]

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
        0.5, 0.5, 0.5, # 20., 10., 10., # 
        1., 1., 50.,
        25., 10., 1.,
        5., 5., 5., 0.05
    ]

    run_base_fs = {
        "aircraft_class" : NonlinearDynamicInversionAircraft,
        "actr_warm_start" : False,
        "num" : 1000,
        "final_time" : 5., # 120., # 
        "track_check_time" : 1.,
        # "time_step" : 0.01,
        # "initial_velocity" : 100.,
        "initial_mach" : flight_conditions[f1]["m"],
        "initial_altitude" : flight_conditions[f1]["h"], # 4500., # 
        "trim_bank" :  0.0,
        "trim_climb" : 0.0,
        # "start_climbing" : False,
        # "end_gs_climbing" : False,
        # "final_mach" : flight_conditions[f1]["m"]*1., # f2]["m"]*1., # 
        # "final_altitude" : flight_conditions[f1]["h"]*1., # f2]["h"]*1., # 
        "t_gain_schedule" : 0.1, # 90., # 
        "gain_steps" : 2,
        "cut_mine" : True,
        "save_data" : True,
        "statistical" : True,
        "has_turbulence" : False,
        "turbulence_setting" : "light", # "moderate", # "severe", # 
        "has_model_error" : False,
        "FM_errors" : base_fs_FM_errs,
        "state_threshold" : state_threshold, # 64.0, # 
        "random_seed" : 13,
        "turbulence_random_seed" : 15, # 13, # 
        "error_random_seed" : 14, # 13, # 
        "rerandomize_turbulence" : True,
        "mrrr" : [0,1,2,6,7,8,9,10,11],
        "mrrc" : [3],
        "get_aero_FM" : True,
        "include_stall_derivatives" : False, # True, # 
        "skip_simulation" : False, # True, # 
        "skip_video" : True, # False, # 
        "plot_ul_bounds" : False,
        "name_end" : "_" + f1 + "_AA_1" # "_DI_1" # 
        # 4 -- incr wt on tau, decr wt on da,de
        # 5 -- decr wt on da
    }
    run_bire_fs = {**run_base_fs}
    run_bire_fs["FM_errors"] = bire_fs_FM_errs
    # run_base_rc = {**run_base_fs}
    # run_base_rc.pop("initial_mach")
    # run_base_rc["initial_velocity"] = 100.
    # run_base_rc["initial_altitude"] = 4500.
    # run_base_rc["FM_errors"] = base_rc_FM_errs
    # run_base_rc["name_end"] = "_" + "LGN" + run_base_fs["name_end"][3:]
    # run_bire_rc = {**run_base_rc}
    # run_bire_rc["FM_errors"] = bire_rc_FM_errs
    # run_bire_rc["mrrc"] = [3]

    bire_fs_dict["controller"] = {
        "enforce_update_frequency" : False,
        "update_frequency[hz]" : 100.0,
        "type" : "gains",
        "name" : "gains",
        "integral_states" : [0,3,4,5],
        "gains" : {
            "K" : [ [ -10.0,  0.0,  12.0],
                    [  0.0, -5.0, -4.0],
                    [  0.0,  4.0, 30.0]],
            "KI" :[ [ -1.0,  0.0,  0.0],
                    [  0.0, -5.0,  0.0],
                    [  0.0,  0.0,  5.0]]
        }
    }
    

    # per dave, max throws would be p=270deg/s,q=120deg/s,r=60deg/s
    # from 2nd to last flight test:
    # about 1/6 throw was max commanded in flight

    # run single case
    # # 
    plot_vars["plot_full"] = True # False # 
    plot_vars["plot_delta"] = False # True # 
    plot_vars["zoom_deltas"] = False
    plot_vars["format"] = "png" # "pdf" # 
    # plot_vars["format"] = "pdf" # "png" # 
    plot_vars["output_states"] = True # False # 
    plot_vars["plot_norm"] = False # True # 
    #
    di = [0.,0.,0.]
    # di = [5.,10.,7.] # see below
    run_base_fs["num"] = run_bire_fs["num"] = 1
    ##
    # # # # # # NDI_1
    # run_bire_fs["aircraft_class"] = NonlinearDynamicInversionAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_NDI_1_wS_wA" # " # nolim" # 
    # # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0]
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0
    # #
    # # # # TNDI_1
    # run_bire_fs["aircraft_class"] = TransformedNonlinearDynamicInversionAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_TNDI_2"
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 
    # #
    # run_bire_fs["aircraft_class"] = DynamicInversionAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_DI_2"
    # # DI_1
    # # zt_p,zt_q,zt_r =  0.7 , 0.7 , 0.7 
    # # wn_p,wn_q,wn_r = 10.0 ,10.0 ,10.0 
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 
    # # # #
    # bire_fs_dict["controller"]["gains"][ "K"] = np.array([
    #     [2.*zt_p*wn_p,         0.0,         0.0], #  -zt_r*wn_r], # 
    #     [         0.0,2.*zt_q*wn_q,         0.0], #  -zt_r*wn_r], # 
    #     [         0.0,         0.0,2.*zt_r*wn_r]
    # ]).tolist()
    # bire_fs_dict["controller"]["gains"]["KI"] = np.array([
    #     [wn_p**2.,     0.0,     0.0], # wn_r**2.], # 
    #     [     0.0,wn_q**2.,     0.0],
    #     [     0.0,     0.0,wn_r**2.]
    # ]).tolist()
    # # #
    # run_bire_fs["aircraft_class"] = DynamicInversionBacksteppingAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_DIB_1"
    # #
    # run_bire_fs["aircraft_class"] = DynamicInversionGainScheduledAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_DIGS_1"
    # # # 
    run_bire_fs["aircraft_class"] = ControlAllocationMomentAssignmentAircraft
    run_bire_fs["name_end"] = "_" + f1 + "_CAMA_sine" # 2" # 
    bire_fs_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0]
    # # # # # # 
    # # # run_bire_fs["aircraft_class"] = ControlAllocationMomentAssignmentActuatorsAircraft
    # # # run_bire_fs["name_end"] = "_" + f1 + "_CAMAA_1"
    # # # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # # # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 
    # # # #
    # bire_fs_dict["controller"]["gains"][ "K"] = np.array([
    #     [2.*zt_p*wn_p,         0.0,  -zt_r*wn_r], #         0.0], # 
    #     [         0.0,2.*zt_q*wn_q,  -zt_r*wn_r], #         0.0], # 
    #     [         0.0,         0.0,2.*zt_r*wn_r]
    # ]).tolist()
    # bire_fs_dict["controller"]["gains"]["KI"] = np.array([
    #     [wn_p**2.,     0.0,wn_r**2.], #      0.0], # 
    #     [     0.0,wn_q**2.,     0.0],
    #     [     0.0,     0.0,wn_r**2.]
    # ]).tolist()
    # # # # 
    # run_bire_fs["aircraft_class"] = TPIAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_TPI_1"
    # zt_p,zt_q,zt_r =  0.6 , 0.6 , 0.6
    # wn_p,wn_q,wn_r =  8.0 , 8.0 , 8.0 
    # # # # 
    # # # # 
    # run_bire_fs["aircraft_class"] = StabilityAugmentationircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_SA_1"
    # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0]
    # # # # 
    # # # # 
    # run_bire_fs["aircraft_class"] = ITPIAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_ITPI_noABup" # 1" # 
    # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0] # [0.0,0.0,0.0] # 
    # left_roll = True # False # 
    # run_bire_fs["time_step"] = 0.001
    # # run_bire_fs["initial_mach"] = run_bire_fs["final_mach"] = 0.614417991271374
    # # # # #
    # bire_fs_dict["controller"]["gains"][ "K"] = np.array([
    #     [2.*zt_p*wn_p,         0.0,         0.0], #   -zt_r*wn_r], # 
    #     [         0.0,2.*zt_q*wn_q,         0.0], #   -zt_r*wn_r], # 
    #     [         0.0,         0.0,2.*zt_r*wn_r]
    # ]).tolist()
    # bire_fs_dict["controller"]["gains"]["KI"] = np.array([
    #     [wn_p**2.,     0.0,     0.0], # wn_r**2.], # 
    #     [     0.0,wn_q**2.,     0.0],
    #     [     0.0,     0.0,wn_r**2.]
    # ]).tolist()
    #
    # run_bire_fs["aircraft_class"] = HinfDIAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_HIDI_1" # 
    # # # 
    # run_bire_fs["aircraft_class"] = TransformedDynamicInversionAircraft
    # run_bire_fs["name_end"] = "_" + f1 + "_TDI_1"
    # #
    # # 
    # run_bire_fs["aircraft_class"] = LinearQuadraticTrackingAircraft
    # # LQT_1
    # run_bire_fs["name_end"] = "_" + f1 + "_LQT_1"
    # # 
    # run_bire_fs["aircraft_class"] = TransformedLinearQuadraticRegulatorAircraft
    # # TLQR_1
    # run_bire_fs["name_end"] = "_" + f1 + "_TLQR_1"
    # # 
    # run_bire_fs["aircraft_class"] = LinearQuadraticRegulatorDynamicInversionAircraft
    # # LQR_1
    # run_bire_fs["name_end"] = "_" + f1 + "_LQRDI_1" # 2" # 
    # # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [+1.0,+0.0,0.0]
    # #
    # run_bire_fs["aircraft_class"] = LinearAdaptiveAircraft
    # # LAC_1
    # run_bire_fs["name_end"] = "_" + f1 + "_LAC_1"
    # run_bire_fs["state_threshold"] += [1.]*18
    # # 
    # run_bire_fs["aircraft_class"] = ModelReferenceAdaptiveAircraft
    # # MRAC_1
    # run_bire_fs["state_threshold"] += [1.]*21
    # # #
    # run_bire_fs["aircraft_class"] = LyapunovRegulationAircraft
    # # LyR_1
    # run_bire_fs["name_end"] = "_" + f1 + "_LyR_1"
    # bire_fs_dict["simulation"]["include_stall"] = False
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    # # #
    # run_bire_fs["aircraft_class"] = LyapunovTrackingAircraft
    # # LyT_1
    # run_bire_fs["name_end"] = "_" + f1 + "_LyT_1"
    # bire_fs_dict["simulation"]["include_stall"] = False
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    # # #
    # run_bire_fs["aircraft_class"] = SecondOrderTrackingAircraft
    # # SO_1
    # run_bire_fs["name_end"] = "_" + f1 + "_SO_1"
    # # # 
    # # #
    # # #
    # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 10 deg bank fullscale BIRE
    p_tr_deg = -0.0236847366216922
    q_tr_deg =  0.0886486340380570
    r_tr_deg =  0.5027513865539764
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 15 deg bank fullscale BIRE
    # p_tr_deg = -0.0361891562749016
    # q_tr_deg =  0.2007714630167870
    # r_tr_deg =  0.7492893006885849
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 20 deg bank fullscale BIRE
    # p_tr_deg = -0.0495920266927013
    # q_tr_deg =  0.3603497293338741
    # r_tr_deg =  0.9900527444514043
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # 25 deg bank fullscale BIRE
    # # (0) tail
    # p_tr_deg = -0.0638179984370310
    # q_tr_deg =  0.5703966435396264
    # r_tr_deg =  1.2232195495061524
    # #######################################################################
    # # 30 deg bank fullscale BIRE
    # p_tr_deg = -0.0820880039056245
    # q_tr_deg =  0.8352580178704386
    # r_tr_deg =  1.4467093243808735
    # (0) tail
    # p_tr_deg = -0.0800043056586719
    # q_tr_deg =  0.8353731041767737
    # r_tr_deg =  1.4469086597107013
    # # (+) tail
    # p_tr_deg = -0.0783041992237063
    # q_tr_deg =  0.8354699615635688
    # r_tr_deg =  1.4470764216257186
    # #######################################################################
    # # 35 deg bank fullscale BIRE
    # # (0) tail
    # p_tr_deg = -0.0982988942950006
    # q_tr_deg =  1.1619249236475548
    # r_tr_deg =  1.6594007636912391
    # #######################################################################
    # # # 40 deg bank fullscale BIRE
    # (0) tail
    p_tr_deg = -0.1195200902391827
    q_tr_deg =  1.5598776114662467
    r_tr_deg =  1.8589897474721753
    # #######################################################################
    # # 50 deg bank fullscale BIRE
    # (0) tail
    # p_tr_deg = -0.1755564353175651
    # q_tr_deg =  2.6372142861590873
    # r_tr_deg =  2.2128855348515439
    # #######################################################################
    # 60 deg bank fullscale BIRE
    # # (0) tail
    p_tr_deg = -0.2654216358834438
    q_tr_deg =  4.3218126454667702
    r_tr_deg =  2.4951996942473698
    # #######################################################################
    # # 10 deg bank RC scale BIRE w/o stall
    # p_tr_deg = -0.3294739663431505
    # q_tr_deg =  0.5582409457023837
    # r_tr_deg =  3.1659417263281258
    p_bfcm = 60.0 # 40.0 # 50.0 # 30.0 # 15.0 # 5.0 # 20.0 # 10.0 # 7.5 # 
    if "left_roll" in locals():
        p_bfcm = - p_bfcm
        p_tr_deg = - p_tr_deg
        r_tr_deg = - r_tr_deg
    r_comm = 0.0    # 
    p_comm = p_bfcm # 
    a_tr_rad =  np.deg2rad(2.6447774345355031)
    r_comm = p_bfcm*np.sin(a_tr_rad) # 
    p_comm = p_bfcm*np.cos(a_tr_rad) # 
    ###########################################################################
    t_zero = 0.0
    recover_time = 10.0
    transition_time = 1.0 # 2.0 # 
    p_time = t_zero + transition_time
    # p_time2 = p_time  + recover_time
    # p_time3 = p_time2 + transition_time
    t_end = 0.0 # 25.0 # 
    tf = 10.0 # 2.50 # 4.90 # 60.0 # 20.0 # 
    V_trim = 634.4133153512273111
    bire_fs_dict["reference"] = {
        "deg2rad_states" : [1,2,3,4,5],
        "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
        "3" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero, p_comm], [p_time, p_comm], [p_time, p_tr_deg], ], # [p_time2, p_tr_deg ], [p_time2, p_comm], [p_time3, p_comm], [p_time3, p_tr_deg2] ], # [p_time + recover_time, p_tr_deg], [p_time + recover_time, -p_comm], [p_time + recover_time + transition_time, -p_comm], [p_time + recover_time + transition_time, 0.0], ], # 
        "4" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero,    0.0], [p_time,    0.0], [p_time, q_tr_deg], ], # [p_time2, q_tr_deg ], [p_time2,    0.0], [p_time3,    0.0], [p_time3, q_tr_deg2] ], # [p_time + recover_time, q_tr_deg], [p_time + recover_time, -   0.0], [p_time + recover_time + transition_time, -   0.0], [p_time + recover_time + transition_time, 0.0], ], # 
        "5" : [ [0.0, 0.0], [t_zero, 0.0], [t_zero, r_comm], [p_time, r_comm], [p_time, r_tr_deg], ], # [p_time2, r_tr_deg ], [p_time2, r_comm], [p_time3, r_comm], [p_time3, r_tr_deg2] ], # [p_time + recover_time, r_tr_deg], [p_time + recover_time, -r_comm], [p_time + recover_time + transition_time, -r_comm], [p_time + recover_time + transition_time, 0.0], ], # 
        "sct_on_5" : False
    }
    run_bire_fs["track_check_time"] = run_bire_fs["final_time"] = tf # 200.0 # 10.0 # 
    # bire_fs_dict["simulation"]["include_stall"] = False
    # bire_fs_dict["simulation"]["include_compressibility"] = False
    bire_fs_dict["simulation"]["integrator"] = "rk4"
    # run_bire_fs["time_step"] = 0.001 # 0.0001 # 
    # #
    # bire_fs_dict["actuators"]["order"] = 0
    # run_bire_fs["state_threshold"] = run_bire_fs["state_threshold"][:-4]
    # run_bire_fs["name_end"] += "_noact"
    # #
    # bire_fs_dict["aircraft"]["CG_shift[ft]"] = [0.0, 0.0, 0.0] # [1.0, 2.0, -1.0] # 
    # #
    # # # # run_bire_fs["initial_mach"] = 0.2
    # bire_fs_dict["initial"]["trim_guess"] = {
    #         # tail zero
    #         "elevator[deg]" : 5.0,
    #         "BIRE[deg]" : -5.0
    #         # # tail +
    #         # "elevator[deg]" : 0.0,
    #         # "BIRE[deg]" : 5.0
    #     }
    # run_bire_fs["trim_bank"] = 35.0 # 25.0 # 40.0 # 30.0 # 20.0 # 
    # #
    # new_lag = 0.0495e-1 # +0 # 
    # bire_fs_dict["actuators"][ "aileron"]["lag[s]"] = new_lag
    # bire_fs_dict["actuators"]["elevator"]["lag[s]"] = new_lag
    # bire_fs_dict["actuators"][    "BIRE"]["lag[s]"] = new_lag
    # # # # # # 
    blm = 200.0 # 50.0 # 150.0 # 125.0 # 100.0 # 250.0 # 500.0 # 600.0 # 750.0 # 1000.0 # 1500.0 # 
    bire_fs_dict["actuators"][    "BIRE"]["rate_limits[deg/s]"] = [-blm,blm]
    # # # # # # 
    # elm = 50.0
    # bire_fs_dict["actuators"]["elevator"]["rate_limits[deg/s]"] = [-elm,elm]
    # # # # # # #

    # bire_fs_dict["simulation"][      "limit_input"] = False # True # 
    # bire_fs_dict["simulation"]["limit_input_rates"] = False # True # 
    # run_bire_fs["name_end"] += "_nolim"
    # # # # # #
    # bire_fs_dict["simulation"]["constant_density"] = True # False # 
    # # # # # 
    # run_bire_fs["has_turbulence"] = True # False # 
    # run_bire_fs["has_model_error"] = False # True # 
    # # #########################################################################
    # # # # zeros
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "3" : [[0.0]*2]*2, "4" : [[0.0]*2]*2, "5" : [[0.0]*2]*2, "sct_on_5" : False
    # }
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_bire_fs["trim_bank"] = 10.0 # 10.0 # 30.0 # 
    # # di = [0.7,0.0,0.0] # [0.7772,0.0,0.0] # [1.0,1.0,1.0] # [0.0,0.0,0.0] # [0.1,0.1,0.1] # [10.0,10.0,10.0] # 
    # di = [0.0, 35.0994612584134487, 0.0] # [0.0, 28.1437298697430229, 0.0] # 
    # # run_bire_fs[ "has_turbulence"] = True # False # 
    # # # run_bire_fs["has_model_error"] = False # True # 
    # # run_bire_fs["name_end"] += "_rt"
    run_single_simulation(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # run_single_simulation(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # run_single_simulation(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # run_single_simulation(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    quit()

    # # # # run monte carlo perturbation analysis
    # plot_vars["format"] = "pdf" # "png" # 
    # run_bire_fs["plot_ul_bounds"] = True
    # run_bire_fs["final_time"] = 10.0
    # run_base_fs["num"] = run_bire_fs["num"] = 1000
    # #########################################################################
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_bire_fs["trim_bank"] = 10.0 # 30.0 # 
    # # # 
    # di = [16.0, 2.0, 0.4] # DI_2
    # di = [ 3.0, 1.0, 0.1] # LQT_1
    # di = [30.0,15.0, 1.0] # LQRDI_1
    # di = [ 6.0, 1.0, 0.1] # TPI_1
    # # di = [ 5.0, 6.0, 0.1] # MFBL_1
    # # di = [12.0, 0.2, 0.3] # NDI_1
    # # # 
    # run_bire_fs["has_model_error"] = True # False # 
    # run_bire_fs["FM_errors"] = [0.06,0.25,0.25,0.25,0.25,0.25] # DI_2
    # run_bire_fs["FM_errors"] = [0.08,0.25,0.25,0.25,0.25,0.25] # LQT_1
    # run_bire_fs["FM_errors"] = [0.16,0.25,0.25,0.25,0.25,0.25] # LQRDI_1
    # run_bire_fs["FM_errors"] = [0.06,0.25,0.25,0.15,0.13,0.25] # TPI_1
    # # run_bire_fs["FM_errors"] = [0.02,0.25,0.25,0.25,0.25,0.25] # MFBL_1
    # # run_bire_fs["FM_errors"] = [0.01,0.25,0.25,0.25,0.25,0.25] # NDI_1
    # # # 
    # monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    # # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    # # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    # # # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()
    # #
    # single axis pqr dispersions
    ###########################################################################
    bire_fs_dict["reference"] = {
        "deg2rad_states" : [1,2,3,4,5],
        "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
        "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg],],
        "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg],],
        "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg],],
        "sct_on_5" : False
    }
    run_bire_fs["trim_bank"] = 10.0
    run_bire_fs["num"] = 1000 # 3 # 10 # 
    plot_vars["format"] = "pdf" # "png" # 
    run_bire_fs["plot_ul_bounds"] = True
    ###########################################################################
    # disa = [[ 25.,0.,0.],[0., 10.,0.],[0.,0.,  1.1]] # DI_2
    # disa = [[ 25.,0.,0.],[0.,  3.,0.],[0.,0.,  1.1]] # LQT_1
    # disa = [[100.,0.,0.],[0., 60.,0.],[0.,0.,  3.0]] # LQRDI_1
    # disa = [[ 10.,0.,0.],[0., 10.,0.],[0.,0.,  0.4]] # TPI_1
    # disa = [[  5.,0.,0.],[0., 20.,0.],[0.,0.,  0.2]] # MFBL_1
    disa = [[ 15.,0.,0.],[0., 20.,0.],[0.,0.,  1.1]] # NDI_1
    for i in [1]: # [2]: # [0]: # range(3): # 
        ds = disa[i]
        monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=ds,**run_bire_fs,**plot_vars)
        # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=ds,**run_base_fs,**plot_vars)
        # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=ds,**run_bire_rc,**plot_vars)
        # # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=ds,**run_base_rc,**plot_vars)
    quit()
    # #
    # # single FM error dispersions
    # names = ["CL","CS","CD","Cell","Cm","Cn"]
    # run_bire_fs["track_check_time"] = run_bire_fs["final_time"] = 10.0
    # run_base_fs["has_model_error"] = run_bire_fs["has_model_error"] = True
    # f1 = "C2"
    # di = [16.0, 2.0, 0.4] # DI_2
    # di = [ 3.0, 1.0, 0.1] # LQT_1
    # di = [30.0,15.0, 1.0] # LQRDI_1
    # di = [ 6.0, 1.0, 0.1] # TPI_1
    # # di = [ 5.0, 6.0, 0.1] # MFBL_1
    # # di = [12.0, 0.2, 0.3] # NDI_1
    # bire_fs_dict["reference"] = {
    #     "deg2rad_states" : [1,2,3,4,5],
    #     "0" : [[ 0.0,   V_trim],[ 2.0,   V_trim],],
    #     "3" : [[ 0.0, p_tr_deg],[ 2.0, p_tr_deg]],
    #     "4" : [[ 0.0, q_tr_deg],[ 2.0, q_tr_deg]],
    #     "5" : [[ 0.0, r_tr_deg],[ 2.0, r_tr_deg]],
    #     "sct_on_5" : False
    # }
    # run_bire_fs["trim_bank"] = 10.0
    # run_bire_fs["num"] = 1000 # 3 # 10 # 
    # plot_vars["format"] = "pdf" # "png" # 
    # run_bire_fs["plot_ul_bounds"] = True
    # current_name = run_bire_fs["name_end"]
    # for i in [4,5]: # [2,3]: # [0,1]: # [5]: # [4,5]: # [1,2]: # [0]: # range(len(names)): # 
    #     name = names[i]
    #     # create FM errors
    #     FM_error_list = np.zeros((6,))
    #     FM_error_list[i] = 0.25
    #     run_base_fs["FM_errors"] = run_bire_fs["FM_errors"] =FM_error_list*1.
    #     run_bire_fs["name_end"] = current_name + "_" + name
    #     monte_carlo_perturbations(bire_fs_dict,rtdst_1sg=di,**run_bire_fs,**plot_vars)
    #     # monte_carlo_perturbations(base_fs_dict,rtdst_1sg=di,**run_base_fs,**plot_vars)
    #     # # monte_carlo_perturbations(bire_rc_dict,rtdst_1sg=di,**run_bire_rc,**plot_vars)
    #     # # monte_carlo_perturbations(base_rc_dict,rtdst_1sg=di,**run_base_rc,**plot_vars)
    # quit()