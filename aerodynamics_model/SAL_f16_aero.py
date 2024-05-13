from math import sin,cos,pi,exp, copysign as SIGN
from numpy import sign, deg2rad, rad2deg
from numpy import array as nparray
from numpy import zeros as npzeros
from numpy import float32 as FLOAT
import numpy as np
import json
from SAL_thrust import Propulsion

class SALF16Aero:
    def __init__(self, inp_dir='./', SAL_interp=True, **kwargs):
        fn = kwargs.get('fn', 'SAL_f16_tables.json')
        coeffs_dict = json.load(open(inp_dir + fn))

        # damping coefficient data
        C_DAMP = coeffs_dict.get("DAMP",{})
        self.DAMP_A = nparray(C_DAMP.get("A",npzeros((9,12)))).T

        # CX coefficient data
        C_CX = coeffs_dict.get("CX",{})
        self.CX_A = nparray(C_CX.get("A",npzeros((5,12)))).T

        # CZ coefficient data
        C_CZ = coeffs_dict.get("CZ",{})
        self.CZ_A = nparray(C_CZ.get("A",npzeros((1,12)))).T

        # CM coefficient data
        C_CM = coeffs_dict.get("CM",{})
        self.CM_A = nparray(C_CM.get("A",npzeros((5,12)))).T

        # Cl coefficient data
        C_CL = coeffs_dict.get("CL",{})
        self.  CL_A = nparray(C_CL.get("A",npzeros((7,12)))).T
        self.DLDA_A = nparray(C_CL.get("DA",npzeros((7,12)))).T
        self.DLDR_A = nparray(C_CL.get("DR",npzeros((7,12)))).T

        # CN coefficient data
        C_CN = coeffs_dict.get("CN",{})
        self.  CN_A = nparray(C_CN.get("A",npzeros((7,12)))).T
        self.DNDA_A = nparray(C_CN.get("DA",npzeros((7,12)))).T
        self.DNDR_A = nparray(C_CN.get("DR",npzeros((7,12)))).T

        # initialize thrust model
        self.Prop = Propulsion(inp_dir=kwargs.get("thrust_dir",inp_dir),**kwargs)

        # store stall model characteristics
        stall_model = coeffs_dict.get("stall_model",{})
        self.S_M = stall_model.get("blending_rate",7.0)
        self.S_ab = deg2rad(stall_model.get("stall_transition[deg]",45.0))

        # add in properties for compressibility
        is_rc = kwargs.get("use_rc_thrust_model",False)
        if is_rc:
            fn_props = kwargs.get("fn_props", "f16_rc_props.json") # same for F16 / BIRE
        else:
            fn_props = kwargs.get("fn_props", "f16_props.json") # same for F16 / BIRE
        self.props_dict = json.load(open(inp_dir + fn_props))
        self.geom_coeffs = self.props_dict["geometry"]
        self.S_w = self.geom_coeffs["S_w"]
        self.b_w = self.geom_coeffs["b_w"]
        self.c_w = self.geom_coeffs["c_w"]
        self.l_h = self.geom_coeffs["l_h"]
        self.Lam_w = self.geom_coeffs["Lam_w"]
        self.RA_w = self.geom_coeffs["RA_w"]
        self.Lam_v = self.geom_coeffs["Lam_v"]
        self.RA_v = self.geom_coeffs["RA_v"]
        self.Lam_h = self.geom_coeffs["Lam_h"]
        self.RA_h = self.geom_coeffs["RA_h"]
        ## ERRATA => ## Weight is 25000 in 3rd edition


    def DAMP(self,ALPHA):
        # various damping coefficients
        D = npzeros((9,))
        A = self.DAMP_A
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        D[0] = A[K+2,0] + abs(DA)*(A[L+2,0] - A[K+2,0])
        D[1] = A[K+2,1] + abs(DA)*(A[L+2,1] - A[K+2,1])
        D[2] = A[K+2,2] + abs(DA)*(A[L+2,2] - A[K+2,2])
        D[3] = A[K+2,3] + abs(DA)*(A[L+2,3] - A[K+2,3])
        D[4] = A[K+2,4] + abs(DA)*(A[L+2,4] - A[K+2,4])
        D[5] = A[K+2,5] + abs(DA)*(A[L+2,5] - A[K+2,5])
        D[6] = A[K+2,6] + abs(DA)*(A[L+2,6] - A[K+2,6])
        D[7] = A[K+2,7] + abs(DA)*(A[L+2,7] - A[K+2,7])
        D[8] = A[K+2,8] + abs(DA)*(A[L+2,8] - A[K+2,8])

        # D1= CXq; D2= CYr; D3= CYp; D4= CZq; D5= Clr; D6= Clp
        # D7= Cmq; D8= Cnr; D9= Cnp
        return D


    def CX(self,ALPHA,EL):
        # x-axis aerodynamic force coeff.
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S = EL/12.0
        M = int(S)
        if M <= -2: M = -1
        if M >=  2: M =  1
        DE = S - FLOAT(M)
        N = M + int( SIGN(1.1,DE) )
        T = self.CX_A[K+2,M+2]
        U = self.CX_A[K+2,N+2]
        V = T + abs(DA)*(self.CX_A[L+2,M+2] - T)
        W = U + abs(DA)*(self.CX_A[L+2,N+2] - U)
        CX = V + (W - V)*abs(DE)
        return CX


    def CY(self,BETA,AIL,RDR):
        # sideforce coefficient
        CY = -0.02*BETA + 0.021*(AIL/20.0) + 0.086*(RDR/30.0)
        return CY


    def CZ(self,ALPHA,BETA,EL):
        # z-axis force coeff.
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S = self.CZ_A[K+2,0] + abs(DA)*(self.CZ_A[L+2,0] - self.CZ_A[K+2,0])
        CZ = S*(1. - (BETA/57.3)**2.) - 0.19*(EL/25.0)
        return CZ


    def CM(self,ALPHA,EL):
        # pitching moment coeff.
        A = self.CM_A
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S = EL/12.0
        M = int(S)
        if M <= -2: M = -1
        if M >=  2: M =  1
        DE = S - FLOAT(M)
        N = M + int( SIGN(1.1,DE) )
        T = A[K+2,M+2]
        U = A[K+2,N+2]
        V = T + abs(DA)*(A[L+2,M+2] - T)
        W = U + abs(DA)*(A[L+2,N+2] - U)
        CM = V + (W - V)*abs(DE)
        return CM


    def CL(self,ALPHA,BETA):
        # rolling moment coeff.
        A = self.CL_A
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S = 0.2*abs(BETA)
        M = int(S)
        if M == 0: M = 1
        if M >= 6: M = 5
        DB = S - FLOAT(M)
        N = M + int( SIGN(1.1,DB) )
        T = A[K+2,M]
        U = A[K+2,N]
        V = T + abs(DA)*(A[L+2,M] - T)
        W = U + abs(DA)*(A[L+2,N] - U)
        DUM = V + (W - V)*abs(DB)
        CL = DUM*SIGN(1.0,BETA) ## ERRATA => ## CL = DUM + SIGN(1.0,BETA)
        return CL


    def CN(self,ALPHA,BETA):
        # yawing moment coeff.
        A = self.CN_A
        #
        S = 0.2*ALPHA
        K = int(S)
        if K <= -2: K = -1
        if K >=  9: K =  8
        DA = S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S = 0.2*abs(BETA)
        M = int(S)
        if M == 0: M = 1
        if M >= 6: M = 5
        DB = S - FLOAT(M)
        N = M + int( SIGN(1.1,DB) )
        T = A[K+2,M]
        U = A[K+2,N]
        V = T + abs(DA)*(A[L+2,M] - T)
        W = U + abs(DA)*(A[L+2,N] - U)
        DUM = V + (W - V)*abs(DB)
        CN = DUM*SIGN(1.0,BETA) ## ERRATA carries over from CL equation (same)
        return CN


    def DLDA(self,ALPHA,BETA):
        # rolling mom. due to ailerons
        A = self.DLDA_A
        #
        S= 0.2 * ALPHA
        K= int(S)
        if K <= -2 : K= -1
        if K >=  9 : K=  8
        DA= S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S= 0.1 * BETA
        M= int(S)
        if M == -3 : M= -2
        if M >=  3 : M=  2
        DB= S - FLOAT(M)
        N= M + int( SIGN(1.1,DB) )
        T= A[K+2,M+3]
        U= A[K+2,N+3]
        V= T + abs(DA) * (A[L+2,M+3] - T)
        W= U + abs(DA) * (A[L+2,N+3] - U)
        DLDA = V + (W-V) * abs(DB)
        return DLDA
    

    def DLDR(self,ALPHA,BETA):
        # rolling mom. due to rudder
        A = self.DLDR_A
        #
        S= 0.2 * ALPHA
        K= int(S)
        if K <= -2 : K= -1
        if K >=  9 : K=  8
        DA= S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S= 0.1 * BETA
        M= int(S)
        if M == -3 : M= -2
        if M >=  3 : M=  2
        DB= S - FLOAT(M)
        N= M + int( SIGN(1.1,DB) )
        T= A[K+2,M+3]
        U= A[K+2,N+3]
        V= T + abs(DA) * (A[L+2,M+3] - T)
        W= U + abs(DA) * (A[L+2,N+3] - U)
        DLDR = V + (W-V) * abs(DB)
        return DLDR


    def DNDA(self,ALPHA,BETA):
        # yawing mom. due to ailerons
        A = self.DNDA_A
        #
        S= 0.2 * ALPHA
        K= int(S)
        if K <= -2 : K= -1
        if K >=  9 : K=  8
        DA= S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S= 0.1 * BETA
        M= int(S)
        if M == -3 : M= -2
        if M >=  3 : M=  2
        DB= S - FLOAT(M)
        N= M + int( SIGN(1.1,DB) )
        T= A[K+2,M+3]
        U= A[K+2,N+3]
        V= T + abs(DA) * (A[L+2,M+3] - T)
        W= U + abs(DA) * (A[L+2,N+3] - U)
        DNDA = V + (W-V) * abs(DB)
        return DNDA
    

    def DNDR(self,ALPHA,BETA):
        # yawing mom. due to rudder
        A = self.DNDR_A
        #
        S= 0.2 * ALPHA
        K= int(S)
        if K <= -2 : K= -1
        if K >=  9 : K=  8
        DA= S - FLOAT(K)
        L = K + int( SIGN(1.1,DA) )
        S= 0.1 * BETA
        M= int(S)
        if M == -3 : M= -2
        if M >=  3 : M=  2
        DB= S - FLOAT(M)
        N= M + int( SIGN(1.1,DB) )
        T= A[K+2,M+3]
        U= A[K+2,N+3]
        V= T + abs(DA) * (A[L+2,M+3] - T)
        W= U + abs(DA) * (A[L+2,N+3] - U)
        DNDR = V + (W-V) * abs(DB)
        return DNDR


    def _inc_aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dr):
        # convert reading angles to degrees
        ALPHA = rad2deg(alpha)*1.
        BETA = rad2deg(beta)*1.
        AIL = rad2deg(da)*1.
        EL = rad2deg(de)*1.
        RDR = rad2deg(dr)*1.

        # calculate body-fixed forces and moments
        CXT = self.CX (ALPHA,EL)
        CYT = self.CY (BETA,AIL,RDR)
        CZT = self.CZ (ALPHA,BETA,EL)
        DAIL= AIL/20.0; DRDR= RDR/30.0
        CLT = self.CL(ALPHA,BETA) + self.DLDA(ALPHA,BETA)*DAIL \
                                  + self.DLDR(ALPHA,BETA)*DRDR
        CMT = self.CM(ALPHA,EL)
        CNT = self.CN(ALPHA,BETA) + self.DNDA(ALPHA,BETA)*DAIL \
                                  + self.DNDR(ALPHA,BETA)*DRDR
        #
        D = self.DAMP(ALPHA)
        CXT= CXT + D[0]*qbar
        CYT= CYT + D[1]*rbar + D[2]*pbar
        CZT= CZT + D[3]*qbar
        CLT= CLT + D[4]*rbar + D[5]*pbar
        CMT= CMT + D[6]*qbar# + CZT * (XCGR-XCG) # handle this outside
        CNT= CNT + D[7]*rbar + D[8]*pbar# - CYT*(XCGR-XCG) * CBAR/B # handle outside

        # convert to wind frame
        CA = -CXT*1.
        CY =  CYT*1.
        CN = -CZT*1.
        ca = cos(alpha); sa = sin(alpha)
        cb = cos(beta);  sb = sin(beta)
        CL = CN*ca - CA*sa
        CS = CA*ca*sb + CY*cb + CN*sa*sb
        CD = CA*ca*cb - CY*sb + CN*sa*cb
        Cl = CLT
        Cm = CMT
        Cn = CNT

        return [CL, CS, CD, Cl, Cm, Cn]

   
    def _stall_correction(self,a,CL,CD,Cm):
        # determine flat plate forces and moment
        CLplate = 2. * sign(a) * sin(a)**2. * cos(a)
        CDplate = 2. * sin(abs(a))**1.5
        Cmplate = -0.8 * sin(a)

        # determine stall sigmoid
        expMmin = exp(-self.S_M*(a-self.S_ab))
        expMplu = exp(self.S_M*(a+self.S_ab))
        sig = (1. + expMmin + expMplu) / (1. + expMmin) / (1. + expMplu)

        # add stall effects
        CL = (1. - sig) * CL + sig * CLplate
        CD = (1. - sig) * CD + sig * CDplate
        Cm = (1. - sig) * Cm + sig * Cmplate

        return CL,CD,Cm


    def _Anderson_correction(self, coeff, Lambda, RA, M):
        num = coeff*cos(Lambda)
        denom_2 = num/(pi*RA)
        denom_1 = (1. - M**2*cos(Lambda)**2 + denom_2**2)**0.5
        denom = denom_1 + denom_2
        return num/denom


    def _Prandtl_Glauert_subsonic_correction(self, coeff, M):
        return coeff / (1. - M**2.)**0.5


    def _Prandtl_Glauert_supersonic_correction(self, coeff, M):
        return coeff / (M**2. - 1.)**0.5

    def aero_results(self, alpha, beta, pbar, qbar, rbar, da, de, dr, 
    compressible=False, M=113.0, use_Anderson=False, enforce_stall=False):
        params = alpha, beta, pbar, qbar, rbar, da, de, dr

        # run incompressible
        [CL, CS, CD, Cl, Cm, Cn] = self._inc_aero_results(*params)

        # stall
        if enforce_stall:

            # implement stall effects
            [CL,CD,Cm] = self._stall_correction(alpha,CL,CD,Cm)
        
        if not compressible:
            return [CL, CS, CD, Cl, Cm, Cn]
        else:
            # if not given mach number, throw error
            if M == 113.0:
                raise ValueError("Mach number not specified")
            elif M < 1.:
                if use_Anderson:
                    CL = self._Anderson_correction(CL,self.Lam_w,self.RA_w,M)
                    CS = self._Anderson_correction(CS,self.Lam_v,self.RA_v,M)
                    Cl = self._Anderson_correction(Cl,self.Lam_v,self.RA_v,M) # w w
                    Cm = self._Anderson_correction(Cm,self.Lam_w,self.RA_w,M)
                    Cn = self._Anderson_correction(Cn,self.Lam_v,self.RA_v,M)
                else:
                    CL = self._Prandtl_Glauert_subsonic_correction(CL,M)
                    CS = self._Prandtl_Glauert_subsonic_correction(CS,M)
                    Cl = self._Prandtl_Glauert_subsonic_correction(Cl,M)
                    Cm = self._Prandtl_Glauert_subsonic_correction(Cm,M)
                    Cn = self._Prandtl_Glauert_subsonic_correction(Cn,M)
            else:
                CL = self._Prandtl_Glauert_supersonic_correction(CL,M)
                CS = self._Prandtl_Glauert_supersonic_correction(CS,M)
                Cl = self._Prandtl_Glauert_supersonic_correction(Cl,M)
                Cm = self._Prandtl_Glauert_supersonic_correction(Cm,M)
                Cn = self._Prandtl_Glauert_supersonic_correction(Cn,M)
            
            # return
            return [CL, CS, CD, Cl, Cm, Cn]

    def get_thrust(self,POW,ALT,AMACH):
        return self.Prop.get_thrust(POW,ALT,AMACH)
    
    def aero_CG_offset_results(self, alpha, beta, pbar, qbar, rbar, da, de, dB, tau, 
                               V, H, rho_0, rho, cg_shift=[0., 0., 0.], compressible=True,
                               M=113.0, use_Anderson=True, enforce_stall=True, thrust_off = False):
        
        [CL, CS, CD, Cl, Cm, Cn] = self.aero_results(alpha, beta, pbar, qbar, rbar, da, de, dB, 
                                                     compressible, M, use_Anderson, enforce_stall)
        
        x_shift, y_shift, z_shift = cg_shift
        
        nondim_const = 0.5*rho*V*V*self.S_w
        # body fixed force and moment coefficients
        CX = -(CD*cos(alpha)*cos(beta) + CS*cos(alpha)*sin(beta) - CL*sin(alpha))
        CY = CS*cos(beta) - CD*sin(beta)
        CZ = -(CD*sin(alpha)*cos(beta) + CS*sin(alpha)*sin(beta) + CL*cos(alpha))

        if thrust_off == False:
            thrust = self.get_thrust(tau, H, V)
        else:
            thrust = 0.0

        Fx = CX*nondim_const + thrust
        Fy = CY*nondim_const
        Fz = CZ*nondim_const
        Mx = Cl*nondim_const*self.b_w - Fz*y_shift + Fy*z_shift
        My = Cm*nondim_const*self.c_w - Fx*z_shift + Fz*x_shift
        Mz = Cn*nondim_const*self.b_w - Fy*x_shift + Fx*y_shift
        FM = [Fx, Fy, Fz, Mx, My, Mz]
        return FM



if __name__ == "__main__":
    # test case
    F16 = SALF16Aero()

    # testing
    a  = 0.5 # -10.0 # 
    b  = -0.2 # -10.0 # 
    al = deg2rad(-15.0) # -10.0 # 
    el = deg2rad(20.0) # -10.0 # 
    rd = deg2rad(-20.0) # -10.0 # 
    V = 500.0
    cw = 11.32
    bw = 30.0
    p = 0.7
    q = -0.8
    r = 0.9
    pbar = p*bw/2./V
    qbar = q*cw/2./V
    rbar = r*bw/2./V
    head = "-"*20
    print(head)
    print("a    =",a ,", deg =",rad2deg(a))
    print("b    =",b ,", deg =",rad2deg(b))
    print("al   =",al,", deg =",rad2deg(al))
    print("el   =",el,", deg =",rad2deg(el))
    print("rd   =",rd,", deg =",rad2deg(rd))
    print("p    =",p ,", deg =",rad2deg(p),", pbar =",pbar)
    print("q    =",q ,", deg =",rad2deg(q),", qbar =",qbar)
    print("r    =",r ,", deg =",rad2deg(r),", rbar =",rbar)
    print(head)
    # D = F16.DAMP(a)
    # print("CXq  =",D[0])
    # print("CYr  =",D[1])
    # print("CYp  =",D[2])
    # print("Czq  =",D[3])
    # print("Clr  =",D[4])
    # print("Clp  =",D[5])
    # print("Cmq  =",D[6])
    # print("Cnr  =",D[7])
    # print("Cnp  =",D[8])
    # print(head)
    # print("CX   =",F16.CX(a,el))
    # print("CY   =",F16.CY(b,al,rd))
    # print("CZ   =",F16.CZ(a,b,el))
    # print("CM   =",F16.CM(a,el))
    # print("CL   =",F16.CL(a,b))
    # print("CN   =",F16.CN(a,b))
    # print(head)
    # print("DLDA =",F16.DLDA(a,b))
    # print("DLDR =",F16.DLDR(a,b))
    # print("DNDA =",F16.DNDA(a,b))
    # print("DNDR =",F16.DNDR(a,b))
    # print(head)
    arr = nparray([a,b,pbar,qbar,rbar,al,el,rd]).tolist()
    C = F16.aero_results(*arr)
    print("CL   =",C[0])
    print("CS   =",C[1])
    print("CD   =",C[2])
    print("Cl   =",C[3])
    print("Cm   =",C[4])
    print("Cn   =",C[5])

    # # timing
    # from time import time as tm
    # num = 100000
    # start = tm()
    # for i in range(num):
    #     F16.DAMP(np.random.random()*20.)
    # dur0 = tm() - start
    # print("dur0 =",dur0,"s")
    # start = tm()
    # for i in range(num):
    #     F16.DAMP_slo(np.random.random()*20.)
    # dur1 = tm() - start
    # print("dur1 =",dur1,"s")

    # # check to make sure my EOM match S&L
    # import sympy as sy
    # # symbolic
    # sym = sy.Symbol
    # igr = sy.integrate
    # simp = sy.simplify
    # exp = sy.expand
    # piecewise = sy.Piecewise
    # diff = sy.diff
    # sin = sy.sin
    # cos = sy.cos
    # tan = sy.tan
    # mat = sy.Matrix
    # pi = sy.pi
    # frac = sy.Rational
    # sqrt = sy.sqrt
    # # # declare variables
    # AXX = sym("AXX")
    # AYY = sym("AYY")
    # AZZ = sym("AZZ")
    # AXZ = sym("AXZ")
    # AXY = AYZ = 0
    # HX = sym("HX")
    # HY = HZ = 0
    # P = sym("P")
    # Q = sym("Q")
    # R = sym("R")
    # ROLL = sym("ROLL")
    # PITCH = sym("PITCH")
    # YAW = sym("YAW")

    # # my eqs
    # af_b2 = AXX*AZZ - AXZ**2
    # I_INV = mat([
    #     [AZZ/af_b2,0,AXZ/af_b2],
    #     [0,1/AYY,0],
    #     [AXZ/af_b2,0,AXX/af_b2]
    # ])
    # M = mat([[ROLL],[PITCH],[YAW]])
    # H = mat([
    #     [  0,-HZ, HY],
    #     [ HZ,  0,-HX],
    #     [-HY, HX,  0]
    # ])
    # W = mat([[P],[Q],[R]])
    # E = mat([
    #     [(AYY - AZZ)*Q*R + AYZ*(Q*Q - R*R) + AXZ*P*Q - AXY*P*R],
    #     [(AZZ - AXX)*P*R + AXZ*(R*R - P*P) + AXY*Q*R - AYZ*P*Q],
    #     [(AXX - AYY)*P*Q + AXY*(P*P - Q*Q) + AYZ*P*R - AXZ*Q*R]
    # ])
    # WDOT = I_INV*(M + H*W + E)

    # # SAL eqs
    # AXZS = AXZ**2
    # XPQ  = AXZ*(AXX - AYY + AZZ)
    # GAM  = AXX*AZZ - AXZ**2
    # XQR  = AZZ*(AZZ - AYY) + AXZS
    # ZPQ  = (AXX - AYY)*AXX + AXZS
    # YPR  = AZZ - AXX
    # PQ = P*Q
    # QR = Q*R
    # QHX = Q*HX
    # WDOT_SAL = mat([
    #     [ ( XPQ*PQ - XQR*QR + AZZ*ROLL + AXZ*(YAW + QHX) )/GAM ],
    #     [ ( YPR*P*R - AXZ*(P**2 - R**2) + PITCH - R*HX )/AYY   ],
    #     [ ( ZPQ*PQ - XPQ*QR + AXZ*ROLL + AXX*(YAW + QHX) )/GAM ]
    # ])

    # DELTA = simp( exp( exp(WDOT) - exp(WDOT_SAL) ) )
    # print("please be zero, please be zero,",DELTA)
    # # returns zeros!

    