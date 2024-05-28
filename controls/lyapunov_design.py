import sympy as sy
import numpy as np

def print_to_latex(expr):

    expr = str(expr)
    expr = expr.replace("**","^").replace("*"," ")

    print(expr)

    return expr


if __name__ == "__main__":
    # symbolic
    sym = sy.Symbol
    igr = sy.integrate
    simp = sy.simplify
    exp = sy.expand
    factor = sy.factor
    piecewise = sy.Piecewise
    diff = sy.diff
    sin = sy.sin
    cos = sy.cos
    tan = sy.tan
    sqrt = sy.sqrt
    mat = sy.Matrix
    pi = sy.pi
    frac = sy.Rational
    # declare variables
    a = sym(r"\alpha")
    b = sym(r"\beta")
    p = sym("p")
    q = sym("q")
    r = sym("r")
    #
    da = sym(r"\delta_a")
    de = sym(r"\delta_e")
    dB = sym(r"\delta_B")
    #
    V = sym("V")
    cw = sym("c_w")
    bw = sym("b_w")
    Sw = sym("S_w")
    rho = sym(r"\rho")
    #
    hx = sym("h_x")
    hy = sym("h_y")
    hz = sym("h_z")
    #
    Ixx = sym("Ixx")
    Iyy = sym("Iyy")
    Izz = sym("Izz")
    Ixy = sym("Ixy") # 0 # 
    Ixz = sym("Ixz") # 0 # 
    Iyz = sym("Iyz") # 0 # 
    # print("ASSUMING Ixy = Ixz = Iyz = 0")
    print("ASSUMING Ixx, Iyy, Izz fixed at dB = 0")
    # computed
    pbar = p*bw/2/V
    qbar = q*cw/2/V
    rbar = r*bw/2/V
    #
    
    # set to vals
    #
    use_numerical = False # True # 
    if use_numerical:
        # other
        bw = 30.0
        cw = 11.32
        Sw = 300.0
        pbar = p*bw/2/V
        qbar = q*cw/2/V
        rbar = r*bw/2/V
        # CL
        CL0_A   = -0.0144; CL0_z   =  0.0621
        CLa_A   =  0.1091; CLa_z   =  3.5469
        #
        CLb_A   = -0.7216; CLb_z   =       0
        CLp_A   =       0; CLp_z   =       0
        CLq_A   =  2.0262; CLq_z   =  1.5469
        CLr_A   =  0.6798; CLr_z   =       0
        CLda_A  =       0; CLda_z  = -0.0007
        CLde_A  =  0.7646; CLde_z  = -0.1822
        # CS
        CS0_A   = -0.0106; CS0_z   =       0
        CSa_A   =  0.1834; CSa_z   =       0
        CSb_A   =  0.6805; CSb_z   = -0.8493
        # 
        CSp_A   =       0; CSp_z   = -0.0022
        CSLp_A  =  0.0192; CSLp_z  =  0.2233
        CSq_A   =  1.9916; CSq_z   =       0
        CSr_A   = -0.6134; CSr_z   =  0.5976
        CSda_A  =  0.0015; CSda_z  = -0.0524
        CSde_A  =  0.7352; CSde_z  =       0
        # 
        CD0_A   =       0; CD0_z   =  0.0209
        CDL_A   =       0; CDL_z   = -0.0332
        CDL2_A  =  0.0047; CDL2_z  =  0.1767
        CDS_A   =  0.0255; CDS_z   = -0.0000
        CDS2_A  =  0.3082; CDS2_z  =  0.6364
        CDp_A   =       0; CDp_z   =       0
        CDSp_A  =       0; CDSp_z  =  0.0013
        CDq_A   =       0; CDq_z   =  0.0261
        CDLq_A  =  0.3883; CDLq_z  =  0.3700
        CDL2q_A =       0; CDL2q_z = -0.0303
        CDr_A   =       0; CDr_z   =       0
        CDSr_A  =       0; CDSr_z  = -0.1146
        CDda_A  = -0.0079; CDda_z  =  0.0000
        CDSda_A =  0.0492; CDSda_z = -0.0381
        CDde_A  = -0.0061; CDde_z  =  0.0015
        CDLde_A =  0.1830; CDLde_z =       0
        CDde2_A = -0.0950; CDde2_z =  0.4244
        # 
        Cl0_A   =  0.0002; Cl0_z   =       0
        Cla_A   = -0.0023; Cla_z   =       0
        Clb_A   =  0.0017; Clb_z   = -0.0283
        Clp_A   =  0.0040; Clp_z   = -0.3069
        Clq_A   =       0; Clq_z   =       0
        Clr_A   =       0; Clr_z   =  0.0062
        ClLr_A  =       0; ClLr_z  =  0.1104
        Clda_A  =  0.0140; Clda_z  = -0.1065
        Clde_A  =  0.0017; Clde_z  =       0
        # 
        Cm0_A   =  0.0164; Cm0_z   = -0.0218
        Cma_A   = -0.1381; Cma_z   =  0.2720
        Cmb_A   =  0.8299; Cmb_z   =       0
        Cmp_A   = -0.0102; Cmp_z   =       0
        Cmq_A   = -2.3551; Cmq_z   = -2.5457
        Cmr_A   = -0.7667; Cmr_z   =       0
        Cmda_A  =  0.0008; Cmda_z  = -0.0007
        Cmde_A  = -0.9115; Cmde_z  =  0.2914
        # 
        Cn0_A   =  0.0048; Cn0_z   =       0
        Cna_A   = -0.0929; Cna_z   =       0
        Cnb_A   = -0.3176; Cnb_z   =  0.2804
        Cnp_A   =       0; Cnp_z   =  0.0010
        CnLp_A  = -0.0074; CnLp_z  = -0.0621
        Cnq_A   = -0.9205; Cnq_z   =       0
        Cnr_A   =  0.2894; Cnr_z   = -0.2789
        Cnda_A  =       0; Cnda_z  =  0.0131
        CnLda_A = -0.0169; CnLda_z =  0.0411
        Cnde_A  = -0.3527; Cnde_z  =       0
    else:
        # CL
        CL0_A   = -sym(   "C_{L0_A}"); CL0_z   =  sym(   "C_{L0_z}")
        CLa_A   =  sym(  "C_{L,a_A}"); CLa_z   =  sym(  "C_{L,a_z}")
        #
        CLb_A   = -sym(  "C_{L,b_A}"); CLb_z   =  0             
        CLp_A   =  0                 ; CLp_z   =  0             
        CLq_A   =  sym(  "C_{L,q_A}"); CLq_z   =  sym(  "C_{L,q_z}")
        CLr_A   =  sym(  "C_{L,r_A}"); CLr_z   =  0             
        CLda_A  =  0                 ; CLda_z  = -sym( "C_{L,da_z}")
        CLde_A  =  sym( "C_{L,de_A}"); CLde_z  = -sym( "C_{L,de_z}")
        # CS
        CS0_A   = -sym(   "C_{S0_A}"); CS0_z   =  0             
        CSa_A   =  sym(  "C_{S,a_A}"); CSa_z   =  0             
        CSb_A   =  sym(  "C_{S,b_A}"); CSb_z   = -sym(  "C_{S,b_z}")
        # 
        CSp_A   =  0                 ; CSp_z   = -sym(  "C_{S,p_z}")
        CSLp_A  =  sym( "C_{S,Lp_A}"); CSLp_z  =  sym( "C_{S,Lp_z}")
        CSq_A   =  sym(  "C_{S,q_A}"); CSq_z   =  0             
        CSr_A   = -sym(  "C_{S,r_A}"); CSr_z   =  sym(  "C_{S,r_z}")
        CSda_A  =  sym( "C_{S,da_A}"); CSda_z  = -sym( "C_{S,da_z}")
        CSde_A  =  sym( "C_{S,de_A}"); CSde_z  =  0             
        # 
        CD0_A   =  0                 ; CD0_z   =  sym(  "C_{D,0_z}")
        CDL_A   =  0                 ; CDL_z   = -sym(  "C_{D,L_z}")
        CDL2_A  =  sym( "C_{D,L2_A}"); CDL2_z  =  sym( "C_{D,L2_z}")
        CDS_A   =  sym(  "C_{D,S_A}"); CDS_z   = -sym(  "C_{D,S_z}")
        CDS2_A  =  sym( "C_{D,S2_A}"); CDS2_z  =  sym( "C_{D,S2_z}")
        CDp_A   =  0                 ; CDp_z   =  0             
        CDSp_A  =  0                 ; CDSp_z  =  sym( "C_{D,Sp_z}")
        CDq_A   =  0                 ; CDq_z   =  sym(  "C_{D,q_z}")
        CDLq_A  =  sym( "C_{D,Lq_A}"); CDLq_z  =  sym( "C_{D,Lq_z}")
        CDL2q_A =  0                 ; CDL2q_z = -sym("C_{D,L2q_z}")
        CDr_A   =  0                 ; CDr_z   =  0             
        CDSr_A  =  0                 ; CDSr_z  = -sym( "C_{D,Sr_z}")
        CDda_A  = -sym( "C_{D,da_A}"); CDda_z  =  sym( "C_{D,da_z}")
        CDSda_A =  sym("C_{D,Sda_A}"); CDSda_z = -sym("C_{D,Sda_z}")
        CDde_A  = -sym( "C_{D,de_A}"); CDde_z  =  sym( "C_{D,de_z}")
        CDLde_A =  sym("C_{D,Lde_A}"); CDLde_z =  0             
        CDde2_A = -sym("C_{D,de2_A}"); CDde2_z =  sym("C_{D,de2_z}")
        # 
        Cl0_A   =  sym(   "C_{l0_A}"); Cl0_z   =  0             
        Cla_A   = -sym(  "C_{l,a_A}"); Cla_z   =  0             
        Clb_A   =  sym(  "C_{l,b_A}"); Clb_z   = -sym(  "C_{l,b_z}")
        Clp_A   =  sym(  "C_{l,p_A}"); Clp_z   = -sym(  "C_{l,p_z}")
        Clq_A   =  0                 ; Clq_z   =  0             
        Clr_A   =  0                 ; Clr_z   =  sym(  "C_{l,r_z}")
        ClLr_A  =  0                 ; ClLr_z  =  sym( "C_{l,Lr_z}")
        Clda_A  =  sym( "C_{l,da_A}"); Clda_z  = -sym( "C_{l,da_z}")
        Clde_A  =  sym( "C_{l,de_A}"); Clde_z  =  0             
        # 
        Cm0_A   =  sym(   "C_{m0_A}"); Cm0_z   = -sym(  "C_{m,0_z}")
        Cma_A   = -sym(  "C_{m,a_A}"); Cma_z   =  sym(  "C_{m,a_z}")
        Cmb_A   =  sym(  "C_{m,b_A}"); Cmb_z   =  0             
        Cmp_A   = -sym(  "C_{m,p_A}"); Cmp_z   =  0             
        Cmq_A   = -sym(  "C_{m,q_A}"); Cmq_z   = -sym(  "C_{m,q_z}")
        Cmr_A   = -sym(  "C_{m,r_A}"); Cmr_z   =  0             
        Cmda_A  =  sym( "C_{m,da_A}"); Cmda_z  = -sym( "C_{m,da_z}")
        Cmde_A  = -sym( "C_{m,de_A}"); Cmde_z  =  sym( "C_{m,de_z}")
        # 
        Cn0_A   =  sym(   "C_{n0_A}"); Cn0_z   =  0             
        Cna_A   = -sym(  "C_{n,a_A}"); Cna_z   =  0             
        Cnb_A   = -sym(  "C_{n,b_A}"); Cnb_z   =  sym(  "C_{n,b_z}")
        Cnp_A   =  0                 ; Cnp_z   =  sym(  "C_{n,p_z}")
        CnLp_A  = -sym( "C_{n,Lp_A}"); CnLp_z  = -sym( "C_{n,Lp_z}")
        Cnq_A   = -sym(  "C_{n,q_A}"); Cnq_z   =  0             
        Cnr_A   =  sym(  "C_{n,r_A}"); Cnr_z   = -sym(  "C_{n,r_z}")
        Cnda_A  =  0                 ; Cnda_z  =  sym( "C_{n,da_z}")
        CnLda_A = -sym("C_{n,Lda_A}"); CnLda_z =  sym("C_{n,Lda_z}")
        Cnde_A  = -sym( "C_{n,de_A}"); Cnde_z  =  0             
    #
    # coeffs
    if True:
        # CL
        CL0   = CL0_A  *cos(2*dB) + CL0_z  ; dCL0  = diff(CL0  ,dB)
        CLa   = CLa_A  *cos(2*dB) + CLa_z  ; dCLa  = diff(CLa  ,dB)
        CL1     = CL0 + CLa*a              ; dCL1  = diff(CL1  ,dB)
        CLb   = CLb_A  *sin(2*dB) + CLb_z  ; dCLb  = diff(CLb  ,dB)
        CLp   = CLp_A             + CLp_z  ; dCLp  = diff(CLp  ,dB)
        CLq   = CLq_A  *cos(2*dB) + CLq_z  ; dCLq  = diff(CLq  ,dB)
        CLr   = CLr_A  *sin(2*dB) + CLr_z  ; dCLr  = diff(CLr  ,dB)
        CLda  = CLda_A            + CLda_z ; dCLda = diff(CLda ,dB)
        CLde  = CLde_A *cos(1*dB) + CLde_z ; dCLde = diff(CLde ,dB)
        # CS
        CS0   = CS0_A  *sin(2*dB) + CS0_z  ; dCS0  = diff(CS0  ,dB)
        CSa   = CSa_A  *sin(2*dB) + CSa_z  ; dCSa  = diff(CSa  ,dB)
        CSb   = CSb_A  *cos(2*dB) + CSb_z  ; dCSb  = diff(CSb  ,dB)
        CS1     = CS0 + CSb*b              ; dCS1  = diff(CS1  ,dB)
        CSp   = CSp_A             + CSp_z  ; dCSp  = diff(CSp  ,dB)
        CSLp  = CSLp_A *cos(2*dB) + CSLp_z ; dCSLp = diff(CSLp ,dB)
        CSq   = CSq_A  *sin(2*dB) + CSq_z  ; dCSq  = diff(CSq  ,dB)
        CSr   = CSr_A  *cos(2*dB) + CSr_z  ; dCSr  = diff(CSr  ,dB)
        CSda  = CSda_A *cos(2*dB) + CSda_z ; dCSda = diff(CSda ,dB)
        CSde  = CSde_A *sin(1*dB) + CSde_z ; dCSde = diff(CSde ,dB)
        # CD
        CD0   = CD0_A             + CD0_z  ; dCD0  = diff(CD0  ,dB)
        CDL   = CDL_A             + CDL_z  ; dCDL  = diff(CDL  ,dB)
        CDL2  = CDL2_A *cos(4*dB) + CDL2_z ; dCDL2 = diff(CDL2 ,dB)
        CDS   = CDS_A  *sin(2*dB) + CDS_z  ; dCDS  = diff(CDS  ,dB)
        CDS2  = CDS2_A *cos(2*dB) + CDS2_z ; dCDS2 = diff(CDS2 ,dB)
        CDp   = CDp_A             + CDp_z  ; dCDp  = diff(CDp  ,dB)
        CDSp  = CDSp_A            + CDSp_z ; dCDSp = diff(CDSp ,dB)
        CDq   = CDq_A             + CDq_z  ; dCDq  = diff(CDq  ,dB)
        CDLq  = CDLq_A *cos(2*dB) + CDLq_z ; dCDLq = diff(CDLq ,dB)
        CDL2q = CDL2q_A           + CDL2q_z; dCDL2q= diff(CDL2q,dB)
        CDr   = CDr_A             + CDr_z  ; dCDr  = diff(CDr  ,dB)
        CDSr  = CDSr_A            + CDSr_z ; dCDSr = diff(CDSr ,dB)
        CDda  = CDda_A *sin(2*dB) + CDda_z ; dCDda = diff(CDda ,dB)
        CDSda = CDSda_A*cos(2*dB) + CDSda_z; dCDSda= diff(CDSda,dB)
        CDde  = CDde_A *cos(1*dB) + CDde_z ; dCDde = diff(CDde ,dB)
        CDLde = CDLde_A*cos(1*dB) + CDLde_z; dCDLde= diff(CDLde,dB)
        CDde2 = CDde2_A*cos(1*dB) + CDde2_z; dCDde2= diff(CDde2,dB)
        # Cl
        Cl0   = Cl0_A  *sin(2*dB) + Cl0_z  ; dCl0  = diff(Cl0  ,dB)
        Cla   = Cla_A  *sin(4*dB) + Cla_z  ; dCla  = diff(Cla  ,dB)
        Clb   = Clb_A  *cos(2*dB) + Clb_z  ; dClb  = diff(Clb  ,dB)
        Clp   = Clp_A  *cos(2*dB) + Clp_z  ; dClp  = diff(Clp  ,dB)
        Clq   = Clq_A             + Clq_z  ; dClq  = diff(Clq  ,dB)
        Clr   = Clr_A             + Clr_z  ; dClr  = diff(Clr  ,dB)
        ClLr  = ClLr_A            + ClLr_z ; dClLr = diff(ClLr ,dB)
        Clda  = Clda_A *cos(2*dB) + Clda_z ; dClda = diff(Clda ,dB)
        Clde  = Clde_A *sin(1*dB) + Clde_z ; dClde = diff(Clde ,dB)
        # Cm
        Cm0   = Cm0_A  *cos(2*dB) + Cm0_z  ; dCm0  = diff(Cm0  ,dB)
        Cma   = Cma_A  *cos(2*dB) + Cma_z  ; dCma  = diff(Cma  ,dB)
        Cmb   = Cmb_A  *sin(2*dB) + Cmb_z  ; dCmb  = diff(Cmb  ,dB)
        Cmp   = Cmp_A  *sin(2*dB) + Cmp_z  ; dCmp  = diff(Cmp  ,dB)
        Cmq   = Cmq_A  *cos(2*dB) + Cmq_z  ; dCmq  = diff(Cmq  ,dB)
        Cmr   = Cmr_A  *sin(2*dB) + Cmr_z  ; dCmr  = diff(Cmr  ,dB)
        Cmda  = Cmda_A *sin(2*dB) + Cmda_z ; dCmda = diff(Cmda ,dB)
        Cmde  = Cmde_A *cos(1*dB) + Cmde_z ; dCmde = diff(Cmde ,dB)
        # Cn
        Cn0   = Cn0_A  *sin(2*dB) + Cn0_z  ; dCn0  = diff(Cn0  ,dB)
        Cna   = Cna_A  *sin(2*dB) + Cna_z  ; dCna  = diff(Cna  ,dB)
        Cnb   = Cnb_A  *cos(2*dB) + Cnb_z  ; dCnb  = diff(Cnb  ,dB)
        Cnp   = Cnp_A             + Cnp_z  ; dCnp  = diff(Cnp  ,dB)
        CnLp  = CnLp_A *cos(2*dB) + CnLp_z ; dCnLp = diff(CnLp ,dB)
        Cnq   = Cnq_A  *sin(2*dB) + Cnq_z  ; dCnq  = diff(Cnq  ,dB)
        Cnr   = Cnr_A  *cos(2*dB) + Cnr_z  ; dCnr  = diff(Cnr  ,dB)
        Cnda  = Cnda_A            + Cnda_z ; dCnda = diff(Cnda ,dB)
        CnLda = CnLda_A*cos(2*dB) + CnLda_z; dCnLda= diff(CnLda,dB)
        Cnde  = Cnde_A *sin(1*dB) + Cnde_z ; dCnde = diff(Cnde ,dB)
    else:
        quit()
    # # what if?
    # ClLr  = CnLp  = CnLda = Cmda = Cmp = Cnp = Clr = 0
    # print("ASSUMING ClLr  = CnLp  = CnLda = Cmda = Cmp = Cnp = Clr = 0")
    # print()
    #
    #
    # full equations
    CL = CL1 + CLb*b + CLp*pbar + CLq*qbar + CLr*rbar + CLda*da + CLde*de
    CS = CS1 + CSa*a + (CSLp*CL1 + CSp)*pbar + CSq*qbar + CSr*rbar + CSda*da + CSde*de
    CD = CD0 + (CDL + CDL2*CL1)*CL1 + (CDS + CDS2*CS1)*CS1 \
        + (CDSp*CS1 + CDp)*pbar + ( (CDL2q*CL1 + CDLq)*CL1 + CDq)*qbar \
        + (CDSr*CS1 + CDr)*rbar \
        + (CDSda*CS1 + CDda)*da + (CDLde*CL1 + CDde + CDde2*de)*de
    Cl = Cl0 + Cla*a + Clb*b + Clp*pbar + Clq*qbar + (ClLr*CL1 + Clr)*rbar \
        + Clda*da + Clde*de
    Cm = Cm0 + Cma*a + Cmb*b + Cmp*pbar + Cmq*qbar + Cmr*rbar \
        + Cmda*da + Cmde*de
    Cn = Cn0 + Cna*a + Cnb*b + (CnLp*CL1 + Cnp)*pbar + Cnq*qbar + Cnr*rbar \
        + (CnLda*CL1 + Cnda)*da + Cnde*de

    # replace with symbols
    Cl_sym = sym("Cl")
    Cm_sym = sym("Cm")
    Cn_sym = sym("Cn")
    
    # dynamics
    M = frac(1,2)*rho*V**2*Sw*mat([ # 
        [bw, 0, 0],
        [ 0, cw, 0],
        [ 0, 0,bw]
    ])*mat([[Cl_sym],[Cm_sym],[Cn_sym]])#*0 + mat([[1],[1],[1]])
    # print(M)
    w = mat([[p],[q],[r]])
    hmat = mat([
        [  0,-hz, hy],
        [ hz,  0,-hx],
        [-hy, hx,  0]
    ])
    Sigma = mat([
            [(Iyy - Izz)*q*r + Iyz*(q**2 - r**2) + Ixz*p*q - Ixy*p*r],
            [(Izz - Ixx)*p*r + Ixz*(r**2 - p**2) + Ixy*q*r - Iyz*p*q],
            [(Ixx - Iyy)*p*q + Ixy*(p**2 - q**2) + Iyz*p*r - Ixz*q*r]
    ])
    det = ( Ixx*(Iyy*Izz - Iyz**2) - Ixy*Ixz*Iyz - (Ixy**2*Izz + Ixz**2*Iyy) )
    adj = mat([
        [ Iyy*Izz - Iyz**2 , Ixy*Izz + Ixz*Iyz, Ixy*Iyz + Ixz*Iyy],
        [ Ixy*Izz + Iyz*Ixz, Ixx*Izz - Ixz**2 , Ixx*Iyz + Ixy*Ixz],
        [ Ixy*Iyz + Ixz*Iyy, Ixx*Iyz + Ixz*Ixy, Ixx*Iyy - Ixy**2 ]
    ])
    Iinv = mat([
        [adj[0,0]/det, adj[0,1]/det, adj[0,2]/det],
        [adj[1,0]/det, adj[1,1]/det, adj[1,2]/det],
        [adj[2,0]/det, adj[2,1]/det, adj[2,2]/det]
    ])
    I = mat([ [ Ixx,-Ixy,-Ixz],[-Ixy, Iyy,-Iyz],[-Ixz,-Iyz, Izz] ])
    # I = I + Ixy*Ixz*Iyz*mat([[1,0,0],[0,1,0],[0,0,1]])
    # kp = sym("k_p"); kq = sym("k_q"); kr = sym("k_r")
    # I = I*mat([[kp,0,0],[0,kq,0],[0,0,kr]])
    wdot = Iinv*(M + hmat*w + Sigma)
    # wdot = Iinv*(w)

    # Lyapunov function
    # print(wdot.shape)
    # print(wdot[0])
    #
    Ip = Ixx # det#/(adj[0,0] + adj[1,0] + adj[2,0]) # 
    Iq = Iyy # det#/(adj[0,1] + adj[1,1] + adj[2,1]) # 
    Ir = Izz # det#/(adj[0,2] + adj[1,2] + adj[2,2]) # 
    # Ir = det/(Ixx*Iyy + Ixx*Iyz - Ixy**2 + Ixy*Ixz + Ixy*Iyz + Ixz*Iyy)
    # print(adj[0,0] + adj[1,0] + adj[2,0])
    # print(adj[0,1] + adj[1,1] + adj[2,1])
    # print(adj[0,2] + adj[1,2] + adj[2,2])    
    #
    V_fn = ( Ip*p**2 + Iq*q**2 + Ir*r**2 )#
    V_fn = frac(1,2)*(w.T*I*w)[0,0]*det/(det - Ixy*Ixz*Iyz)/frac(1,2)/rho/V**2/Sw
    print(V_fn)
    # quit()
    # V_fn = V_fn - p*r*Ixz*Iyy
    print("V_fn    =",V_fn)
    print()
    Vdot = diff(V_fn,p)*wdot[0] + diff(V_fn,q)*wdot[1] + diff(V_fn,r)*wdot[2]
    print("Vdot =",Vdot)
    print()
    Vdot = exp( sy.cancel( Vdot ) )
    print("Vdot =",Vdot)
    print()
    Vdot = Vdot.replace(Cl_sym,Cl).replace(Cm_sym,Cm).replace(Cn_sym,Cn)
    print("Vdot =",Vdot)
    print()
    # Vdot = exp(Vdot)
    # Vdot = sy.collect(sy.collect(sy.collect(Vdot,da),de),dB)
    # print("Vdot =",Vdot)
    # print()
    quit()
    # print("ASSUMING dB = 0")
    # Vdot = Vdot.replace(dB,0)
    # print("Vdot =",Vdot)
    # print()
    # print(sy.collect(Vdot,da,evaluate=False)[da])
    Vdot_da  = Vdot.coeff(da,1)
    Vdot_de  = Vdot.coeff(de,1)
    Vdot_base = exp( Vdot - Vdot_da*da - Vdot_de*de )
    # Vdot_base = sy.collect(Vdot_base,dB)
    Vdot_base = sy.collect(Vdot_base,p)
    Vdot_base = sy.collect(Vdot_base,q)
    Vdot_base = sy.collect(Vdot_base,r)
    Vdot_base = sy.collect(Vdot_base,b)
    Vdot_base = sy.collect(Vdot_base,a)
    print("-"*15)
    print("Vdot_da   =",Vdot_da)
    print("Vdot_de   =",Vdot_de)
    print("Vdot_base =",Vdot_base)
    print("-"*15)
    print()
    # assume alpha , beta <= +20 deg (somehow), V <= 634 ft/s
    Vdot_fbk = Vdot*1
    Vdot_fbk = Vdot_fbk.replace(V,634.0)
    Vdot_fbk = Vdot_fbk.replace(a,np.deg2rad(20.0)) # np.deg2rad(180.0)) # 
    Vdot_fbk = Vdot_fbk.replace(b,np.deg2rad(20.0)) # np.deg2rad(180.0)) # 
    # # insert control
    # s = sym("s"); t = sym("t"); v = sym("v")
    # # s = 2.0
    # # t = 0.7906510006650016*s - 0.008238647007969866
    # da_fbk = s*p
    # de_fbk = t*q
    # FDB = sym("F_{DB}")
    # dB_fbk = sy.asin(q*r/v) # sy.acos(- s*q - t*r) # 
    # Vdot_fbk = Vdot_fbk.replace(da,da_fbk)
    # Vdot_fbk = Vdot_fbk.replace(de,de_fbk)
    # # Vdot_fbk_pos = Vdot_fbk.replace(dB,sy.acos( s*q*r))
    # Vdot_fbk = Vdot_fbk.replace(dB,dB_fbk)
    # Vdot_fbk = simp( sy.expand( Vdot_fbk ) ) # sy.expand(
    # print("Vdot =",Vdot_fbk)
    # print()
    # Vdot_fbk = sy.collect(Vdot_fbk,p)
    # Vdot_fbk = sy.collect(Vdot_fbk,q)
    # Vdot_fbk = sy.collect(Vdot_fbk,r)
    # # Vdot_fbk = sy.collect(Vdot_fbk,b)
    # # Vdot_fbk = sy.collect(Vdot_fbk,a)
    # Vdot_fbk = sy.collect(Vdot_fbk,FDB)
    # print("Vdot =",Vdot_fbk)
    # print()
    # print_to_latex(Vdot_fbk)
    # print()
    # # print("Vdot - Vdotfbk+ =", simp( Vdot_fbk - Vdot_fbk_pos ) )
    # # print()

    # coeffs for 
    absa = abs(np.deg2rad(20.0))
    absb = abs(np.deg2rad(20.0))
    VT = sym("VT") # 
    VT = 634.0 # 200.0 # 
    latbar = bw/2/VT
    lonbar = cw/2/VT
    Ca = abs(Clb_A) + abs(Clb_z)
    Cb = ( abs(ClLr_z*CLa_A) + abs(ClLr_z*CLa_z) )*absa*latbar
    Dc = ( abs(Clp_A) - abs(Clp_z) )*latbar
    Cd = ( abs(ClLr_z*CL0_z) + abs(Clr_z) - abs(ClLr_z*CL0_A) )*latbar
    De = abs(Clda_A) - abs(Clda_z)
    Cf = ( abs(Cma_A) + abs(Cma_z) )*absa + abs(Cm0_A) - abs(Cm0_z)
    Dg = ( -abs(Cmq_z) - abs(Cmq_A) )*lonbar
    Dh = -abs(Cmda_z)
    Di = abs(Cmde_z) - abs(Cmde_A)
    Cj = ( abs(Cnb_A) + abs(Cnb_z) )*absb
    Ck = ( abs(CnLp_A*CLa_z) + abs(CnLp_A*CLa_A) + abs(CnLp_z*CLa_A) \
        + abs(CnLp_z*CLa_z) )*absa*latbar
    Cl = ( abs(CnLda_A*CLa_A) + abs(CnLda_A*CLa_z) + abs(CnLda_z*CLa_A) \
        + abs(CnLda_z*CLa_z) )*absa
    Dm = ( abs(CnLp_A*CL0_A) - abs(CnLp_A*CL0_z) + abs(CnLp_z*CL0_A) \
        - abs(CnLp_z*CL0_z) - abs(Cnp_z) )*latbar
    Cn = abs(CnLda_A*CL0_A) - abs(CnLda_A*CL0_z) - abs(CnLda_z*CL0_A) \
        + abs(CnLda_z*CL0_z) + abs(Cnda_z)
    Do = -abs(Cnq_z)*lonbar
    Cp = abs(Cnr_A) - abs(Cnr_z)
    Dq = -abs(Cnde_A)

    print("coeffs!")
    Ca *= bw; print("Ca =",Ca)
    Cb *= bw; print("Cb =",Cb)
    Dc *= bw; print("Dc =",Dc)
    Cd *= bw; print("Cd =",Cd)
    De *= bw; print("De =",De)
    Cf *= cw; print("Cf =",Cf)
    Dg *= cw; print("Dg =",Dg)
    Dh *= cw; print("Dh =",Dh)
    Di *= cw; print("Di =",Di)
    Cj *= bw; print("Cj =",Cj)
    Ck *= bw; print("Ck =",Ck)
    Cl *= bw; print("Cl =",Cl)
    Dm *= bw; print("Dm =",Dm)
    Cn *= bw; print("Cn =",Cn)
    Do *= bw; print("Do =",Do)
    Cp *= bw; print("Cp =",Cp)
    Dq *= bw; print("Dq =",Dq)

    # control gains
    Ap,Aq,Ar = sym("Ap"),sym("Aq"),sym("Ar")
    Ep,Eq,Er = sym("Ep"),sym("Eq"),sym("Er")
    #
    # 
    Ap =  1.35
    Aq =  0.0
    Ar =  0.2
    Ep =  0.0
    Eq =  0.5
    Er =  0.0
    # # from LQR # -1.0981
    # Ap = -0.2087809612206068
    # Aq =  -0.0212206578611206
    # Ar =  10.0433381629914411
    # Ep =  -0.0029887211662247
    # Eq =  -0.5342411709817626
    # Er =  -0.1570419293119688
    # # # -0.0962
    # Ap = 7.9
    # Aq = 0.0
    # Ar = 1.2
    # Ep = 0.0
    # Eq = 1.0
    # Er = -0.5
    # # # -0.1196
    # Ap = 10.0
    # Aq = 0.0
    # Ar = 1.5
    # Ep = 0.0
    # Eq = 1.3
    # Er = -0.7
    # # # -0.0509
    # Ap = 4.5
    # Aq = 0.0
    # Ar = 0.7
    # Ep = 0.0
    # Eq = 0.5
    # Er = -0.3
    #
    wval = -20.0
    Ip = (wval - abs(Ca))/De
    Lq = (wval - abs(Cf))/Di
    Ir = 0.0
    Lr = (wval - abs(Cj) - Cl*abs(Ir) - Cn*Ir)/Dq

    header = "*"*20
    print(header)
    print("Ap =",Ap)
    print("Aq =",Aq)
    print("Ar =",Ar)
    print("Ep =",Ep)
    print("Eq =",Eq)
    print("Er =",Er)
    print("Ip =",Ip)
    print("Lq =",Lq)
    print("Ir =",Ir)
    print("Lr =",Lr)
    print(header)

    psq = Dc + De*Ap
    qsq = Dg + Dh*Aq + Di*Eq
    rsq = Cp + Cl*abs(Ar) + Cn*Ar + Dq*Er
    print("p^2*    = {:> 7.4f}".format(psq))
    print("q^2*    = {:> 7.4f}".format(qsq))
    print("r^2*    = {:> 7.4f}".format(rsq))
    wsq = abs(Cb + Ck + Cl*abs(Ap) + De*Ip + Cl*abs(Ip) + Cn*Ip) \
        + abs(Dh*Ip) + abs(Cl*abs(Aq) + Dh*Ir + Di*Lr + Dq*Lq) \
        + abs(De*Aq + Dh*Ap + Di*Ep) + abs(Cd + Dm + De*Ar + Cn*Ap + Dq*Ep) \
        + abs(Dh*Ar + Di*Er + Cn*Aq + Dq*Eq)
    wsq *= 0.5
    print("wsq     = {:> 7.4f}".format(wsq))
    sqs = np.array([psq,qsq,rsq])
    msq = (np.min(np.abs(sqs[sqs<0])) + rsq)/2. + wsq
    print("msq     = {:> 7.4f}".format(msq))
    nsq = np.min(np.abs(sqs[sqs<0])) - msq
    print("is pos? = {:> 7.4f}".format(nsq))
    print("RoA : w <=",np.rad2deg(wval/nsq),"deg/s")


    print()
    quit()

    print("running cases")
    valnum = 15
    
    Aps = np.linspace( 1.0, 5.0,valnum)
    Aqs = np.linspace(-2.0, 2.0,valnum)
    Ars = np.linspace(-2.0, 2.0,valnum)
    Eps = np.linspace(-2.0, 2.0,valnum)
    Eqs = np.linspace(-1.0, 2.0,valnum)
    Ers = np.linspace(-2.0, 2.0,valnum)

    for i in range(valnum):
        Ap = Aps[i]
        for j in range(valnum):
            Aq = Aqs[j]
            for k in range(valnum):
                Ar = Ars[k]
                for l in range(valnum):
                    Ep = Eps[l]
                    for m in range(valnum):
                        Eq = Eqs[m]
                        for n in range(valnum):
                            Er = Ers[n]

                            # calc
                            psq = Dc + De*Ap
                            qsq = Dg + Dh*Aq + Di*Eq
                            rsq = Cp + Cl*abs(Ar) + Cn*Ar + Dq*Er
                            wsq = abs(Cb + Ck + Cl*abs(Ap)) + abs(Cl*abs(Aq)) \
                                + abs(De*Aq + Dh*Ap + Di*Ep) \
                                + abs(Cd + Dm + De*Ar + Cn*Ap + Dq*Ep) \
                                + abs(Dh*Ar + Di*Er + Cn*Aq + Dq*Eq)
                            wsq *= 0.5

                            # minval
                            msq = min(min(abs(psq),abs(qsq)),abs(rsq))
                            if msq > abs(wsq) and psq < 0 and qsq < 0 and rsq < 0:
                                print("yay",Ap,Aq,Ar,Ep,Eq,Er)
                                print(psq,qsq,rsq,wsq)
                                print()
                            
                            if psq < 0 and qsq < 0 and rsq == msq and rsq > 0 \
                                and (min(abs(psq),abs(qsq))+rsq)/2.+wsq < \
                                min(abs(psq),abs(qsq)):
                                print("other",Ap,Aq,Ar,Ep,Eq,Er)
                                print(psq,qsq,rsq,wsq)
                                print()

            
    quit()

    # Vd = p**2*(\
    #         q**2*r**2*(0.028*s + 0.000189274447949527) \
    #         - 0.1205*s - 0.00735567823343849) \
    #     + p*(q**4*r**5*(-0.0016009764972277*s - 1.65856299994657e-5) \
    #         + 0.00642281164733913*q**3*r**3*sqrt(-q**2*r**2 + 1) \
    #         + q**2*(r**3*(-0.0403990902446334*s - 0.000384562604166988) \
    #             + 0.00118682389135614*r**2 \
    #             + r*(-0.000603733333333333*s*sqrt(-q**2*r**2 + 1) \
    #                 + 0.000182119873817035*sqrt(-q**2*r**2 + 1)
    #             )\
    #         )\
    #         + q*(-0.00361140582366957*r*sqrt(-q**2*r**2 + 1) \
    #             - 0.000264133333333333*s + 0.0017*t*sqrt(-q**2*r**2 + 1)\
    #         ) \
    #         + r*(0.0871380776690708*s + 0.00185257298739554) \
    #         - 0.010471975511966\
    #     ) \
    #     + q**3*(-0.0240029234267829*r**2 + 0.343939333333333*r*t) \
    #     + q**2*(0.0136940063091483*r**4 - 0.22172662817336*r**3 \
    #         + 0.0301247697160883*r**2*sqrt(-q**2*r**2 + 1) \
    #         - 0.218619197430996*r*sqrt(-q**2*r**2 + 1) \
    #         + 0.109954933333333*t - 0.000642059449001052\
    #     ) \
    #     + q*(0.0552564350041093*r**2*sqrt(-q**2*r**2 + 1) \
    #         - 0.3527*r*t*sqrt(-q**2*r**2 + 1) + 0.0396018522471289\
    #     ) \
    #     + r**2*(-0.0158668857118822*q**4 - 0.0134455835962145) \
    #     + 0.208741378538522*r
    # ###########################################################################





    # dB = 0
    # Vd = p**2*(-0.0925*s - 0.00716640378548896) + \
    #     p*(-0.000264133333333333*q*s \
    #         + r*(0.0451380109272097*s + 0.00145142475322908) \
    #         - 0.00928515162060983)\
    #     + q**2*(-0.2339844*t - 0.0165089451608833) \
    #     + 0.015598928820346*q \
    #     + 0.000248422712933755*r**2 - 0.0129852496348378*r
    # ###########################################################################
    # Vd = p**2*(-0.0925*s - 0.00716640378548896) \
    #     + q**2*(-0.0925*s - 0.00716640378548896) \
    #     + r**2*(-0.0925*s - 0.00716640378548896)
    #     \
    #     + q**2*(-0.2339844*t - 0.0165089451608833 +0.0925*s + 0.00716640378548896) \
    #     + (+0.0925*s + 0.00716640378548896 + 0.000248422712933755)*r**2 \
    #     + p*(-0.000264133333333333*q*s \
    #         + r*(0.0451380109272097*s + 0.00145142475322908) \
    #         - 0.00928515162060983)\
    #     + 0.015598928820346*q \
    #     - 0.0129852496348378*r
    # ###########################################################################
    ### NOW LESS THAN OR EQUAL TO
    X = sym("X") # ||w||
    Vd = (-0.0925*s - 0.00716640378548896)*X**2\
        \
        + q**2*(0.2339844*t +0.0925*s +0.00934254137539434) \
        + (+0.0925*s + 0.007414826498422716)*r**2 \
        + p*(0.000264133333333333*q*s \
            + r*0.0451380109272097*s) \
        + p*r*0.00145142475322908 \
        + 0.00928515162060983*p \
        + 0.015598928820346*q \
        + 0.0129852496348378*r
    ###########################################################################
    # note |p*r| <= 1/2*X^2
    Vd = (0.2339844*t + 0.0925*s + 0.009590964088328095)*X**2\
        \
        + X**2*(0.022701072130271515*s + 0.00072571237661454) \
        + (0.03786933007579363)*X
    ###########################################################################
    # note X <= X^2
    Vd = (0.2339844*t + 0.0925*s + 0.009590964088328095)*X**2\
        \
        + (0.022701072130271515*s + 0.03859504245240817)*X**2
    ###########################################################################
    # note X <= X^2
    Vd = (0.2339844*t + 0.11520107213027152*s \
        + 0.048186006540736265)*X**2
    ###########################################################################
    # Choose 
    # 0.2339844*t + 0.11520107213027152*s + 0.048186006540736265 < 0
    # s < - 0.2339844/0.11520107213027152*t - 0.048186006540736265/0.11520107213027152
    # s < - 2.0310956805628173*t - 0.4182774140005106
    quit()
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    #
    #
    #
    #
    #
    #
    # Vdot = sy.expand_trig( Vdot )#- (w[0]*M[0] + w[1]*M[1] + w[2]*M[2])))
    # print("Vdot =",Vdot)
    # print()
    # Vdot = sy.trigsimp( Vdot )#- (w[0]*M[0] + w[1]*M[1] + w[2]*M[2])))
    # print("Vdot =",Vdot)
    # print()
    # Vdot = simp( Vdot )#- (w[0]*M[0] + w[1]*M[1] + w[2]*M[2])))
    # print("Vdot =",Vdot)
    

    # Linearization
    Dxcg, Dycg, Dzcg = sym("Dxcg"), sym("Dycg"), sym("Dzcg")

    # values for latter use
    #
    Ca = cos(a); Sa = sin(a)
    Cb = cos(b); Sb = sin(b)
    #
    Rlon = cw/2/V
    Rlat = bw/2/V
    #
    pbar = p*Rlat
    qbar = q*Rlon
    rbar = r*Rlat
    #
    g = sym("g")
    sos = sym("sos")
    #
    M = V / sos
    #
    Qdyn = frac(1,2)*rho*V**2*Sw
    Qlon = Qdyn*cw
    Qlat = Qdyn*bw
    #
    W = sym("W")
    m = W/g
    minv = g/W
    # derive Iinv symbolically
    # slapfix
    # Iinv = IM.inverse_tensor(dB)
    dIxx,dIyy,dIzz,dIxy,dIxz,dIyz = \
        sym("dIxx"),sym("dIyy"),sym("dIzz"),sym("dIxy"),sym("dIxz"),sym("dIyz")
    #######################################################################
    det = ( Ixx*(Iyy*Izz - Iyz**2) - Ixy*Ixz*Iyz - (Ixy**2*Izz + Ixz**2*Iyy) )
    
    ddet = ( dIxx*(Iyy*Izz - Iyz**2) + Ixx*(dIyy*Izz + Iyy*dIzz \
        - 2*Iyz*dIyz) - dIxy*Ixz*Iyz - Ixy*dIxz*Iyz - Ixy*Ixz*dIyz
        -(2*Ixy*dIxy*Izz + Ixy**2*dIzz + 2*Ixz*dIxz*Iyy + Ixz**2*dIyy))
    
    adj = mat([
        [ Iyy*Izz - Iyz**2 , Ixy*Izz + Ixz*Iyz, Ixy*Iyz + Ixz*Iyy],
        [ Ixy*Izz + Iyz*Ixz, Ixx*Izz - Ixz**2 , Ixx*Iyz + Ixy*Ixz],
        [ Ixy*Iyz + Ixz*Iyy, Ixx*Iyz + Ixz*Ixy, Ixx*Iyy - Ixy**2 ]
    ])
    
    
    dadj = mat([[0,0,0],[0,0,0],[0,0,0]])
    dadj[0,0] = dIyy*Izz + Iyy*dIzz - 2*Iyz*dIyz
    dadj[0,1] = dadj[1,0] = dIxy*Izz + Ixy*dIzz + dIxz*Iyz + Ixz*dIyz
    dadj[1,1] = dIxx*Izz + Ixx*dIzz - 2*Ixz*dIxz
    dadj[0,2] = dadj[2,0] = dIxy*Iyz + Ixy*dIyz + dIxz*Iyy + Ixz*dIyy
    dadj[2,2] = dIxx*Iyy + Ixx*dIyy - 2*Ixy*dIxy
    dadj[1,2] = dadj[2,1] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
    
    I = mat([
        [  Ixx, -Ixy, -Ixz],
        [ -Ixy,  Iyy, -Iyz],
        [ -Ixz, -Iyz,  Izz]
    ])

    Iinv = mat([
        [adj[0,0]/det, adj[0,1]/det, adj[0,2]/det],
        [adj[1,0]/det, adj[1,1]/det, adj[1,2]/det],
        [adj[2,0]/det, adj[2,1]/det, adj[2,2]/det]
    ])

    dIinv = (dadj - Iinv*ddet)/det
    #######################################################################
    # derive dIinv symbolically
    # slapfix
    
    # input aerodynamic force derivatives
    # evaluate derivatives wrt bire angle
    # evaluate_derivatives(dB)
    # for use
    CL1 = CL0 + CLa * a
    CS1 = CS0 + CSb * b
    dCL1 = dCL0 + dCLa * a
    dCS1 = dCS0 + dCSb * b
    # lift
    oCL_dB = dCL0 + dCLa*a + dCLb*b + dCLp*pbar + dCLq*qbar +\
        + dCLr*rbar + dCLda*da + dCLde*de
    # side
    oCS_dB = dCS0 + dCSa*a + dCSb*b + (dCSLp*CL1 + \
        + CSLp*dCL1 + dCSp)*pbar + dCSq*qbar + dCSr*rbar + \
        + dCSda*da + dCSde*de
    # drag
    oCD_da = CDSda*CS1 + CDda
    oCD_de = CDLde*CL1 + CDde + 2*CDde2*de
    oCD_dB = dCD0 + dCDL*CL1 + CDL*dCL1 + dCDL2*CL1**2 + \
        + 2*CDL2*CL1*dCL1 + dCDS*CS1 + CDS*dCS1 + \
        + dCDS2*CS1**2 + 2*CDS2*CS1*dCS1 + (dCDSp*CS1 + \
        + CDSp*dCS1 + dCDp)*pbar + (dCDL2q*CL1**2 + \
        + 2*CDL2q*CL1*dCL1 + dCDLq*CL1 + CDLq*dCL1 + \
        + dCDq)*qbar + (dCDSr*CS1 + CDSr*dCS1 + dCDr)*rbar + \
        + (dCDSda*CS1 + CDSda*dCS1 + dCDda)*da + \
        + (dCDLde*CL1 + CDLde*dCL1 + dCDde)*de + dCDde2*de**2
    # equated values
    oCL_da, oCL_de, oCS_da, oCS_de = CLda, CLde, CSda, CSde
    
    # input aerodynamic moment derivatives
    # roll
    oCl_dB = dCl0 + dCla*a + dClb*b + dClp*pbar + dClq*qbar +\
        + (dClLr*CL1 + ClLr*dCL1 + dClr)*rbar + dClda*da + \
        + dClde*de
    # pitch
    oCm_dB = dCm0 + dCma*a + dCmb*b + dCmp*pbar + dCmq*qbar +\
        + dCmr*rbar + dCmda*da + dCmde*de
    # yaw
    oCn_da = CnLda*CL1 + Cnda
    oCn_dB = dCn0 + dCna*a + dCnb*b + (dCnLp*CL1 + \
        + CnLp*dCL1 + dCnp)*pbar + dCnq*qbar + dCnr*rbar + \
        + (dCnLda*CL1 + CnLda*dCL1 + dCnda)*da + dCnde*de
    # equated values
    oCl_da, oCl_de, oCm_da, oCm_de = Clda, Clde, Cmda, Cmde
    oCn_de = Cnde
    
    CL_da,CL_de = oCL_da,oCL_de
    CS_da,CS_de = oCS_da,oCS_de
    CD_da,CD_de = oCD_da,oCD_de
    Cl_da,Cl_de = oCl_da,oCl_de
    Cm_da,Cm_de = oCm_da,oCm_de
    Cn_da,Cn_de = oCn_da,oCn_de
    # bire 
    CL_dB,CS_dB,CD_dB = oCL_dB,oCS_dB,oCD_dB
    Cl_dB,Cm_dB,Cn_dB = oCl_dB,oCm_dB,oCn_dB

    # body-fixed force derivatives wrt input
    Fx_da = Qdyn*(CL_da*Sa - CS_da*Ca*Sb - CD_da*Ca*Cb)
    Fx_de = Qdyn*(CL_de*Sa - CS_de*Ca*Sb - CD_de*Ca*Cb)
    Fx_dB = Qdyn*(CL_dB*Sa - CS_dB*Ca*Sb - CD_dB*Ca*Cb)
    #
    Fy_da = Qdyn*(CS_da*Cb - CD_da*Sb)
    Fy_de = Qdyn*(CS_de*Cb - CD_de*Sb)
    Fy_dB = Qdyn*(CS_dB*Cb - CD_dB*Sb)
    #
    Fz_da = Qdyn*(- CL_da*Ca - CS_da*Sa*Sb - CD_da*Sa*Cb)
    Fz_de = Qdyn*(- CL_de*Ca - CS_de*Sa*Sb - CD_de*Sa*Cb)
    Fz_dB = Qdyn*(- CL_dB*Ca - CS_dB*Sa*Sb - CD_dB*Sa*Cb)
    
    # body-fixed moment derivatives wrt input
    Mx_da = Qlat*Cl_da + Fy_da*Dzcg - Fz_da*Dycg
    Mx_de = Qlat*Cl_de + Fy_de*Dzcg - Fz_de*Dycg
    Mx_dB = Qlat*Cl_dB + Fy_dB*Dzcg - Fz_dB*Dycg
    #
    My_da = Qlon*Cm_da + Fz_da*Dxcg - Fx_da*Dzcg
    My_de = Qlon*Cm_de + Fz_de*Dxcg - Fx_de*Dzcg
    My_dB = Qlon*Cm_dB + Fz_dB*Dxcg - Fx_dB*Dzcg
    #
    Mz_da = Qlat*Cn_da + Fx_da*Dycg - Fy_da*Dxcg
    Mz_de = Qlat*Cn_de + Fx_de*Dycg - Fy_de*Dxcg
    Mz_dB = Qlat*Cn_dB + Fx_dB*Dycg - Fy_dB*Dxcg

    # evaluate at condtion for Mx, My, Mz
    # Fx = Qdyn*(CL*Sa - CS*Ca*Sb - CD*Ca*Cb) + T
    # Fy = Qdyn*(CS*Cb - CD*Sb)
    # Fz = Qdyn*(- CL*Ca - CS*Sa*Sb - CD*Sa*Cb)
    #
    Mx = Qlat*Cl # + Fy*Dzcg - Fz*Dycg
    My = Qlon*Cm # + Fz*Dxcg - Fx*Dzcg
    Mz = Qlat*Cn # + Fx*Dycg - Fy*Dxcg

    # set range values for ease of use
    ru = [0,1]
    r3 = 2
    
    B = mat([[0,0,0],[0,0,0],[0,0,0]])
    # # assemble components
    B[:,0:2] = Iinv*mat([
        [Mx_da, Mx_de],
        [My_da, My_de],
        [Mz_da, Mz_de]
    ])
    wdot = (
        mat([Mx,My,Mz]) +
        mat([
        [ 0., -hz,  hy],
        [ hz,  0., -hx],
        [-hy,  hx,  0.]
        ])*mat([p,q,r]) + 
        mat([
            ( Iyy- Izz)*q*r +  Iyz*(q**2-r**2) +  Ixz*p*q -  Ixy*p*r,
            ( Izz- Ixx)*p*r +  Ixz*(r**2-p**2) +  Ixy*q*r -  Iyz*p*q,
            ( Ixx- Iyy)*p*q +  Ixy*(p**2-q**2) +  Iyz*p*r -  Ixz*q*r
        ])
    )
    wdot_dB = (
        mat([Mx_dB,My_dB,Mz_dB]) +
        mat([
            (dIyy-dIzz)*q*r + dIyz*(q**2-r**2) + dIxz*p*q - dIxy*p*r,
            (dIzz-dIxx)*p*r + dIxz*(r**2-p**2) + dIxy*q*r - dIyz*p*q,
            (dIxx-dIyy)*p*q + dIxy*(p**2-q**2) + dIyz*p*r - dIxz*q*r
        ])
    )
    B[:,r3] = ( Iinv*wdot_dB + dIinv*wdot )

    # print(B)
    Ba = B[0,0]
    Bb = B[0,1]
    Bc = B[0,2]
    Bd = B[1,0]
    Be = B[1,1]
    Bf = B[1,2]
    Bg = B[2,0]
    Bh = B[2,1]
    Bi = B[2,2]

    B_det = Ba*(Be*Bi - Bf*Bh) - Bb*(Bd*Bi - Bf*Bg) + Bc*(Bd*Bh - Be*Bg)

    # simplifications
    print("|B| = ")
    print("ASSUME Dxcg = Dycg = Dzcg = 0")
    B_det = B_det.replace(Dxcg,0)
    B_det = B_det.replace(Dycg,0)
    B_det = B_det.replace(Dzcg,0)
    print("|B| = ",B_det)
    print("ASSUME Ixy = Ixz = Iyz = 0")
    B_det = B_det.replace(Ixy,0).replace(dIxy,0)
    B_det = B_det.replace(Ixz,0).replace(dIxz,0)
    B_det = B_det.replace(Iyz,0).replace(dIyz,0)
    print("|B| = ",B_det)
    print("ASSUME dIxx = dIyy = dIzz = 0")
    B_det = B_det.replace(dIxx,0) # .replace(Ixx,0) # 
    B_det = B_det.replace(dIyy,0) # .replace(Iyy,0) # 
    B_det = B_det.replace(dIzz,0) # .replace(Izz,0) # 
    print("|B| = ",B_det)
    print("ASSUME a = b = p = q = r = 0")
    B_det = B_det.replace(a,0)
    B_det = B_det.replace(b,0)
    B_det = B_det.replace(p,0)
    B_det = B_det.replace(q,0)
    B_det = B_det.replace(r,0)
    print("|B| = ",B_det)

    # B_det *= 2*Sw*V**4*bw**2*rho

    print("working on simp of |B|")
    # B_det = simp(B_det)
    print("    expand...")
    B_det = sy.expand(B_det)
    print("|B| =",B_det)
    print("    factor...")
    B_det = sy.factor(B_det)
    print("|B| =",B_det)
    print("    cancel...")
    B_det = sy.cancel(B_det)
    print("|B| =",B_det)

    print(B_det)

    B_det = Clde_A*Cnda_z*cw*(Ixx*p**2 + Iyy*q**2 + Izz*r**2)**6*(-2*Cm0_A*sin(2*dB) - 2*Cma_A*a*sin(2*dB) + 2*Cmb_A*b*cos(2*dB) + 2*Cmda_A*da*cos(2*dB) - Cmde_A*de*sin(dB) + Cmp_A*Sw*V**2*bw**2*p*rho*cos(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) - Cmq_A*Sw*V**2*bw*cw*q*rho*sin(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) + Cmr_A*Sw*V**2*bw**2*r*rho*cos(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2)\
        )*sin(dB)/(8*Ixx*Iyy*Izz*Sw**3*V**12*bw**4*rho**3) \
        \
        - Cnda_z*cw*(Cmde_A*cos(dB) + Cmde_z)*(Ixx*p**2 + Iyy*q**2 + Izz*r**2)**6*(2*Cl0_A*cos(2*dB) + 4*Cla_A*a*cos(4*dB) - 2*Clb_A*b*sin(2*dB) - 2*Clda_A*da*sin(2*dB) + Clde_A*de*cos(dB) - Clp_A*Sw*V**2*bw**2*p*rho*sin(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2)\
        )/(8*Ixx*Iyy*Izz*Sw**3*V**12*bw**4*rho**3) \
        \
        + (Clda_A*cos(2*dB) + Clda_z)*(-Cnde_A*cw*(Ixx*p**2 + Iyy*q**2 + Izz*r**2)**4*(-2*Cm0_A*sin(2*dB) - 2*Cma_A*a*sin(2*dB) + 2*Cmb_A*b*cos(2*dB) + 2*Cmda_A*da*cos(2*dB) - Cmde_A*de*sin(dB) + Cmp_A*Sw*V**2*bw**2*p*rho*cos(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) - Cmq_A*Sw*V**2*bw*cw*q*rho*sin(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) + Cmr_A*Sw*V**2*bw**2*r*rho*cos(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2))*sin(dB)/(4*Iyy*Izz*Sw**2*V**8*bw**3*rho**2) + cw*(Cmde_A*cos(dB) + Cmde_z)*(Ixx*p**2 + Iyy*q**2 + Izz*r**2)**4*(2*Cn0_A*cos(2*dB) - 2*CnLda_A*da*(CL0_A*cos(2*dB) + CL0_z + a*(CLa_A*cos(2*dB) + CLa_z))*sin(2*dB) - CnLp_A*Sw*V**2*bw**2*p*rho*(CL0_A*cos(2*dB) + CL0_z + a*(CLa_A*cos(2*dB) + CLa_z))*sin(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) + 2*Cna_A*a*cos(2*dB) - 2*Cnb_A*b*sin(2*dB) + Cnde_A*de*cos(dB) + Cnq_A*Sw*V**2*bw*cw*q*rho*cos(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2) - Cnr_A*Sw*V**2*bw**2*r*rho*sin(2*dB)/(Ixx*p**2 + Iyy*q**2 + Izz*r**2))/(4*Iyy*Izz*Sw**2*V**8*bw**3*rho**2)\
        )*(Ixx*p**2 + Iyy*q**2 + Izz*r**2)**2/(2*Ixx*Sw*V**4*bw*rho)