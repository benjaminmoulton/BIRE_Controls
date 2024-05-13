import numpy as np
from numpy import matmul as mm
import json
import sympy as sy
import control as co
from scipy.integrate import odeint
from scipy.signal import tf2zpk as scipy_tf2zpk
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from controller_simulation import Aircraft, report_latex, TGEAR
from tracking_control import TrackingAircraft
from quat import quat_mult, euler_2_quat, quat_2_euler

def test_code(input_json):
    # build aircraft
    base = Aircraft(input_json)

    # test dynamics
    t = 0.0
    u = np.array([
        np.deg2rad(-15.0), # aileron
        np.deg2rad(20.0), # elevator
        np.deg2rad(-20.0), # rudder
        0.9 # POW in this case
    ])
    PARAM = 0.40
    cw = 11.32
    xcg = (0.35 - PARAM)*cw
    base.cgshift = [xcg,0.0,0.0]
    V = 500.0
    a = 0.5
    b = -0.2
    Vxb = V*np.cos(a)*np.cos(b)
    Vyb = V*np.sin(b)
    Vzb = V*np.sin(a)*np.cos(b)
    x = np.array([
        Vxb,
        Vyb,
        Vzb,
        0.7, # p
        -0.8, # q
        0.9, # r
        1000.0, # xf
        900.0, # yf
        -10000.0, # zf (-ALT)
        -1.0, # phi
         1.0, # theta
        -1.0, # psi
        np.deg2rad(-15.0), # aileron
        np.deg2rad(20.0), # elevator
        np.deg2rad(-20.0), # rudder
        90.0 # throttle
    ])
    # dx
    dx_xyz = base._dynamics(t,x,True,True,u)
    dxnl = dx_xyz*1.
    xnl = x*1.
    # convert to Vab
    dum = xnl[0]*xnl[0] + xnl[2]*xnl[2]
    # # his math
    # dxnl[0] = (xnl[0]*dx_xyz[0] + xnl[1]*dx_xyz[1] + xnl[2]*dx_xyz[2])/V
    # dxnl[1] = (xnl[0]*dx_xyz[2] - xnl[2]*dx_xyz[0])/dum
    # dxnl[2] = (V*dx_xyz[1] - xnl[1]*dxnl[0])*np.cos(b)/dum
    # my math
    dxnl[0] = 1./V*(xnl[0]*dx_xyz[0] + xnl[1]*dx_xyz[1] + xnl[2]*dx_xyz[2])
    dxnl[2] = (dx_xyz[1]*V - xnl[1]*dxnl[0])/(V*(xnl[0]**2. + xnl[2]**2.)**0.5)
    dxnl[1] = (xnl[0]*dx_xyz[2] - dx_xyz[0]*xnl[2])/(xnl[0]**2. + xnl[2]**2.)
    dxnl[8] *= -1.0
    # convert to SAL order
    order = [0,1,2,  9,10,11,  3,4,5,  6,7,8,  12,13,14,15]
    dxnl = dxnl[order]
    dxnl_SAL = np.array([
        -75.23724, -0.8813491, -0.4759990,
        2.505734, 0.3250820, 2.145926,
        12.62679, 0.9649671, 0.5809759,
        342.4439, -266.7707, 248.1241,
        0.0, 0.0, 0.0, -58.68999
    ])
    trunc_vals = np.array([
        1.0e+5, 1.0e+7, 1.0e+7,
        1.0e+6, 1.0e+7, 1.0e+6,
        1.0e+5, 1.0e+7, 1.0e+7,
        1.0e+4, 1.0e+4, 1.0e+4,
        1.0e+0, 1.0e+0, 1.0e+0, 1.0e+5
    ])
    dxnl = np.round(dxnl*trunc_vals)/trunc_vals
    exps = np.array([
        1, -1, -1,
        0, -1, 0,
        1, -1, -1,
        2, 2, 2,
        0, 0, 0, 1
    ])
    diffs = (dxnl-dxnl_SAL)/10.0**exps
    # report_latex(dxnl,"xdot",decimals=7) # 
    report_latex(diffs,"xdot diff",decimals=7) # 

    # test trim algorithm against SAL Table 3.6-2
    # SAL DATA
    SAL_V = [ 
        130.0, 140.0, 150.0, 170.0, 200.0, 260.0, 300.0, 350.0,
        400.0, 440.0, 500.0, 540.0, 600.0, 640.0, 700.0, 800.0
        ]
    SAL_tau = [ 0.816, 0.736, 0.619, 0.464, 0.287, 0.148, 0.122, 0.107, 0.108,
                0.113, 0.137, 0.160, 0.200, 0.230, 0.282, 0.378]
    SAL_AOA = [ 45.6, 40.3, 34.6, 27.2, 19.7, 11.6, 8.49, 5.87, 4.16,
                3.19, 2.14, 1.63, 1.04, 0.742, 0.382, -0.045] # deg
    SAL_EL = [20.1, -1.36, 0.173, 0.621, 0.723, -0.090, -0.591, -0.539,-0.591,
                -0.671, -0.756, -0.798, -0.846, -0.871, -0.900, -0.943] # deg
    #
    # My trim
    # H = 0
    base.H0 = 0.0
    # cg = nominal
    PARAM = 0.35
    xcg = (0.35 - PARAM)*cw
    base.cgshift = [xcg,0.0,0.0]
    #
    base.trim_type == "sct"
    base.phi_trim = 0.0
    for i in range(len(SAL_V)):
        base.V0 = SAL_V[i]*1.
        base._initialize_state(no_report=True)
        tau_diff = np.round(base.u_trim[3],3) - SAL_tau[i]
        aoa = np.rad2deg(np.arctan2(base.x_trim[2],base.x_trim[0]))
        aoa_diff = np.round(aoa,3) - SAL_AOA[i]
        de_diff = np.round(base.u_trim_deg[1],3) - SAL_EL[i]
        print(("V = {:> 4.0f}, Dtau = {:> 6.3f}, " \
            + "Daoa = {:> 6.3f}, Dde = {:> 6.3f}").\
            format(SAL_V[i],tau_diff,aoa_diff,de_diff))
    # base._report_trim_solution(base.x_trim,base.u_trim,base.trim_iter)


    # test trim algorithm against SAL Table 3.6-3
    base.V0 = 502.0
    # Nominal
    base._initialize_state(no_report=True)
    base._report_trim_solution()#base.x_trim,base.u_trim,base.trim_iter)
    # Xcg = 0.3cbar
    PARAM = 0.3
    xcg = (0.35 - PARAM)*cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()#base.x_trim,base.u_trim,base.trim_iter)
    # Xcg = +0.38cbar
    PARAM = 0.38
    xcg = (0.35 - PARAM)*cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()#base.x_trim,base.u_trim,base.trim_iter)
    # Xcg = +0.3cbar, psidot = 0.3rad/s
    # (using phi = 1.367 rad = 78.323331 deg)
    PARAM = 0.3
    xcg = (0.35 - PARAM)*cw
    base.cgshift = [xcg,0.0,0.0]
    base.phi_trim = 1.3669
    base._initialize_state(no_report=True)
    base._report_trim_solution()#base.x_trim,base.u_trim,base.trim_iter)
    # base.trim_type = "spu"
    # base.q_trim = 0.3
    # base.phi_trim = 0.0
    # base.verbose_trim = True
    # PARAM = -0.3
    # xcg = (0.35 - PARAM)*base.cw
    # base.cgshift = [xcg,0.0,0.0]
    # base._initialize_state(no_report=True)
    # base._report_trim_solution()#base.x_trim,base.u_trim,base.trim_iter)
    # quit()

    return


def test_linearization(input_json):
    # build aircraft
    input_json["controller"]["LQR"]["Q"] = np.ones((12,)).tolist()
    base = Aircraft(input_json)

    # set trim
    spu_V = 502.0
    spu_a = 0.2485
    spu_b = 4.8e-4
    base.x_trim[0] = spu_V*np.cos(spu_a)*np.cos(spu_b)
    base.x_trim[1] = spu_V              *np.sin(spu_b)
    base.x_trim[2] = spu_V*np.sin(spu_a)*np.cos(spu_b)
    base.x_trim[3] = 0.0
    base.x_trim[4] = 0.3
    base.x_trim[5] = 0.0
    base.x_trim[6:9] = [0.0]*3
    base.x_trim[9:12] = [0.0,0.3006,0.0] # euler_2_quat()
    base.x_trim[12:16] = np.concatenate((np.deg2rad([-6.2e-4,-7.082,0.01655]),[102.3]))
    base.u_trim = base.x_trim[12:16]*1.
    base.u_trim[3] /= 100.0

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,mrrr=[],run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)
    
    # report matrices
    rows = [0,1,10,4,12  ,  2,9,3,5]
    cols = [3,1,0,2]
    report_latex((base.Lin_Model.A_min[rows,:])[:,rows],"A",
        decimals=5,predecimals=5,align=True,endln=True,print_report=True)
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    report_latex(np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg),"B",
        decimals=5,align=True,print_report=True)
    quit()


    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.3
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base.phi_trim = 1.3668
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,mrrr=[],run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)
    
    # report matrices
    rows = [0,1,10,4,12  ,  2,9,3,5]
    cols = [3,1,0,2]
    report_latex((base.Lin_Model.A_min[rows,:])[:,rows],"A",
        decimals=5,predecimals=5,align=True,endln=True,print_report=True)
    # val = np.deg2rad(1.0)
    # ch2deg = np.diag([1.0] + [val]*3)
    # report_latex(np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg),"B",
    #     decimals=5,align=True,print_report=True)


def output(self,x,u,is_VAB_format=False):
    # output
    o = np.zeros((3,))

    # an
    _,_,Fz,_,_,_,g = self._aerodynamics(x,u,is_trim=False,
        is_VAB_format=is_VAB_format)
    o[0] = -Fz/g

    # q
    o[1] = x[4]*1.

    # alpha
    if is_VAB_format:
        o[2] = x[1]
    else:
        o[2] = np.arctan2(x[2],x[0])

    return o


def EX_4_4_1(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))
    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[0,2,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)

    # build C matrix
    out = lambda x : output(base,x,base.u_trim,True)
    C = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    
    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    rows = [0,1,10,4]
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [2,1]
    chfmdeg = np.diag([1./val]*4)
    C = np.matmul((C[Crows,:])[:,rows],chfmdeg)
    D = np.zeros((2,1))
    report_latex(A,"A",
        decimals=5,predecimals=5,align=True,endln=True,print_report=True)
    report_latex(B,"B",
        decimals=5,align=True,endln=True,print_report=True)
    report_latex(C,"C",
        decimals=5,align=True,print_report=True)
    
    # create symbolic math
    s = sy.Symbol("s")
    # G = C*(sI - A)^-1 * B
    I = sy.Matrix(np.eye(A.shape[0]))
    Asym = sy.Matrix(A)
    Bsym = sy.Matrix(B)
    Csym = sy.Matrix(C)
    G = sy.simplify(sy.expand(Csym*(s*I - Asym)**-1*Bsym))
    # print(G)
    print("a/de =",sy.factor(G[0]))
    print()
    # num = -0.1232*(s + 75.0)*(s + complex(0.00982,0.09379))\
    #     *(s + complex(0.00982,-0.09379))
    # den = (s - 0.09755)*(s + 1.912)*(s + complex(0.1507,0.1153))\
    #     *(s + complex(0.1507,-0.1153))
    # G_SAL = sy.simplify(num/den)
    # print(G_SAL)
    # print()
    # print(sy.simplify(sy.expand(G[0]/G_SAL)))

    # root locus
    zcol = np.zeros((4,1))
    zcol10 = zcol*0.0; zcol10[1,0] = 10.0
    slow = False
    if slow:
        t_act = 0.1
    else:
        t_act = 0.0495
    s_act = 1./t_act
    aa = np.block([
        [A,-B,zcol], 
        [zcol.T,-s_act,0.0], 
        [zcol10.T,0.0,-10.0] 
    ])
    ba = np.array([
        [0.0]*4 + [s_act] + [0.0]
    ]).T
    ca = np.block([
        [C,np.zeros((2,2))],
        [np.zeros((5,)),1./val]
    ])
    da = np.zeros((1,1))
    sys = co.ss(aa,ba,ca[2,:],da)
    k = np.logspace(-2,1,2000)
    # print(k)
    kai = np.argmin(np.abs(k-0.5))
    # print(k[kai])
    r,_ = co.rlocus(sys,kvect=k)
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kai]),np.imag(ri[kai]),c="0.5",marker=".")
    axs.set_xlim((-21,1))
    axs.set_ylim((-10,10))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_1_a_rlocus"+slow*"_slow"+".png",
        dpi=300.0)
    if show_plots:
        plt.show()
    axs.set_xlim((-1.2,0.1))
    axs.set_ylim((-0.5,0.5))
    fig.savefig(save_folder + "/SAL_4_4_1_a_rlocus_zoomed"+slow*"_slow"+".png",
        dpi=300.0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # set ka = 0.5, compute as a function of kq
    ka = 0.5
    acl = aa - np.matmul(ba*ka,ca[2,np.newaxis,:])
    sysq = co.ss(acl,ba,ca[1,:],da)
    r,kqs = co.rlocus(sysq)
    kqi = np.argmin(np.abs(kqs-0.25))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kqi]),np.imag(ri[kqi]),c="0.5",marker=".")
    axs.set_xlim((-21,1))
    axs.set_ylim((-10,10))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_1_q_rlocus.png",dpi=300.0)
    axs.set_xlim((-1.2,0.1))
    axs.set_ylim((-0.5,0.5))
    fig.savefig(save_folder + "/SAL_4_4_1_q_rlocus_zoomed.png",dpi=300.0)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # set kq = 0.25
    kq = 0.25
    acl2 = acl - np.matmul(ba*kq,ca[1,np.newaxis,:])

    eigval_cl2,eigvec_cl2 = np.linalg.eig(acl2)
    print("2nd loop eigenvalues:\n",eigval_cl2)

    return


def EX_4_4_2(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[0,2,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)

    # build C matrix
    out = lambda x : output(base,x,base.u_trim,True)
    C = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    
    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    rows = [0,1,10,4]
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [2,1]
    chfmdeg = np.diag([1./val]*4)
    C = np.matmul((C[Crows,:])[:,rows],chfmdeg)
    D = np.zeros((2,1))
    report_latex(A,"A",
        decimals=5,predecimals=5,align=True,endln=True,print_report=True)
    report_latex(B,"B",
        decimals=5,align=True,endln=True,print_report=True)
    report_latex(C,"C",
        decimals=5,align=True,print_report=True)

    # root locus
    zcol = np.zeros((4,1))
    zcol10 = zcol*0.0; zcol10[1,0] = 10.0
    slow = False
    if slow:
        t_act = 0.1
    else:
        t_act = 0.0495
    s_act = 1./t_act
    aa = np.block([
        [A,-B,zcol], 
        [zcol.T,-s_act,0.0], 
        [zcol10.T,0.0,-10.0] 
    ])
    ba = np.array([
        [0.0]*4 + [s_act] + [0.0]
    ]).T
    ca = np.block([
        [C,np.zeros((2,2))],
        [np.zeros((5,)),1./val]
    ])
    da = np.zeros((1,1))

    # closed loop with ka = 0.1
    ka = 0.5
    acl = aa - np.matmul(ba*ka,ca[2,np.newaxis,:])
    qfb = co.ss(acl,ba,ca[1,:],da)
    z = 3.0
    p = 1.0
    lag = co.ss(-p,1,z-p,1)
    csys = co.series(lag,qfb)
    a,b,c,d = co.ssdata(csys)
    print("before lag")
    print("a =",acl)
    print("b =",ba)
    print("c =",ca[1,:])
    print("d =",da)
    print()
    print("after lag")
    print("a =",a)
    print("b =",b)
    print("c =",c)
    print("d =",d)
    print()
    k = np.logspace(-2,0,2000)
    r,_ = co.rlocus(csys,kvect=k)
    kqi = np.argmin(np.abs(k-0.2))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kqi]),np.imag(ri[kqi]),c="0.5",marker=".")
    axs.set_xlim((-21,1))
    axs.set_ylim((-10,10))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_2_q_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()

    return


def lat_output(self,x,u,is_VAB_format=False):
    # output
    o = np.zeros((2,))

    # p
    o[0] = x[3]*1.

    # r
    o[1] = x[5]*1.

    return o


def EX_4_4_3(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = (-np.ones((2,13))).tolist()
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 205.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[1,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)

    # build C matrix
    out = lambda x : lat_output(base,x,base.u_trim,True)
    C_full = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    
    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val]*2)
    rows = [2,9,11,3,5]
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [0,1]
    chfmdeg = np.diag([1./val]*5)
    C = np.matmul((C_full[Crows,:])[:,rows],chfmdeg)
    D = np.zeros((2,2))
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=1,endln=1,print_report=1)
    report_latex(C,"C",decimals=5,align=1,endln=1,print_report=1)
    report_latex(D,"D",decimals=5,align=1,print_report=1)
    # remove psi
    rows = [2,9,3,5]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [0,1]
    chfmdeg = np.diag([1./val]*4)
    C = np.matmul((C_full[Crows,:])[:,rows],chfmdeg)
    x0 = base.Lin_Model.xhat_eq[rows]
    
    # # create symbolic math
    # s = sy.Symbol("s")
    # # G = C*(sI - A)^-1 * B
    # I = sy.Matrix(np.eye(A.shape[0]))
    # Asym = sy.Matrix(A)
    # Bsym = sy.Matrix(B)
    # Csym = sy.Matrix(C)
    # G = sy.simplify(sy.expand(Csym*(s*I - Asym)**-1*Bsym))
    # # print(G)
    # print("p/da =",sy.factor(G[0,0]))
    # # print("p/dr =",sy.factor(G[0,1]))
    # # print("r/da =",sy.factor(G[1,0]))
    # # print("r/dr =",sy.factor(G[1,1]))
    # # print()
    # num,den = sy.fraction(G[0,0])
    # print(sy.factor(num))
    # print(sy.factor(den))

    # example
    print("creating control system...")
    ap = A; bp = B; cp = C; dp = D
    sig = 20.2
    aa = -np.diag([sig]*2); ba = -1.*aa # actuator
    ca = -np.eye(2); da = np.zeros((2,2)) # sign change
    actua = co.ss(aa,ba,ca,da) # u1=da, u2=dr
    plant = co.ss(ap,bp,cp,dp) # x1=beta, x2=phi, x3=p, x4=r
    sys1 = co.series(actua,plant) # y1=p, y2=r (degrees)

    # washout filter
    tw = 1.0
    aw = np.array([[-1./tw]]); bw = np.array([[0.0, 1./tw]])
    cw = np.array([[0.0],[-1.0]]); dw = np.eye(2) # y1=p, y2=washed-r
    wash = co.ss(aw,bw,cw,dw)
    sys2 = co.series(sys1,wash) # x1=wash, x2=beta, ... x6=ail, x7=rdr

    # root locus
    print("running root locus...")
    a,b,c,d = co.ssdata(sys2)
    k = np.linspace(0.0,0.9,3000)
    psys = co.ss(a,b[:,[0]],c[[0],:],0.0)
    r,_ = co.rlocus(psys,kvect=k)
    kpi = np.argmin(np.abs(k-0.2))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kpi]),np.imag(ri[kpi]),c="0.5",marker=".")
    axs.set_xlim((-12,1))
    axs.set_ylim((-5,5))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_3_p_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()
    # set kp = 0.2
    kp = 0.2
    acl1 = a - np.matmul(b[:,[0]]*kp,c[[0],:])
    rsys = co.ss(acl1,b[:,[1]],c[[1],:],0)
    r,kr = co.rlocus(rsys)#,kvect=k)
    kri = np.argmin(np.abs(kr-3.5))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kri]),np.imag(ri[kri]),c="0.5",marker=".")
    axs.set_xlim((-12,1))
    axs.set_ylim((-5,5))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_3_r_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()
    # setup final case
    acl2 = a - np.matmul(np.matmul(b,np.diag([0.2,3.5])),c)
    # set alternate case
    kp = 0.4
    acl3 = a - np.matmul(b[:,[0]]*kp,c[[0],:])
    rsys = co.ss(acl3,b[:,[1]],c[[1],:],0)
    r,kr = co.rlocus(rsys)#,kvect=k)
    kri = np.argmin(np.abs(kr-1.3))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kri]),np.imag(ri[kri]),c="0.5",marker=".")
    axs.set_xlim((-12,1))
    axs.set_ylim((-5,5))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus **")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_4_3_r2_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()
    
    # simulate time response of aileron doublet for secondary system
    x0 = np.concatenate((x0,np.zeros((3,))))
    print("run sim...")
    acl4 = a - np.matmul(np.matmul(b,np.diag([0.4,1.3])),c)
    ts = np.linspace(0.0,10.0,num=501)
    da_db = 1.8
    u_db = lambda t: -da_db if t <= 1.0 else +da_db
    u = lambda t: u_db(t) if t <= 2.0 else 0.0
    y = lambda x : np.matmul(c[[0],:],x)
    xdot = lambda t,x : np.matmul(acl4,x) + np.matmul(b[:,[0]],[u(t)])
    xs = odeint(xdot,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    ys = np.array([y(x) for x in xs.T])[:,0]
    us = np.array([u(t) for t in ts]).T
    da_db = 1.0
    xdot_ol = lambda t,x : np.matmul(a,x) + np.matmul(b[:,[0]],[u(t)])
    xs_ol = odeint(xdot_ol,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    ys_ol = np.array([y(x) for x in xs_ol.T])[:,0]
    us_ol = np.array([u(t) for t in ts]).T
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(ts,us_ol,c="0.5",lw=0.75,label="$\delta_a$, SAS off")
    axs.plot(ts,us   ,c="0.5"       ,label="$\delta_a$, SAS on")
    axs.plot(ts,ys_ol,c="k"  ,lw=0.75,label="$p$, SAS off")
    axs.plot(ts,ys   ,c="k"         ,label="$p$, SAS on")
    axs.set_xlim((ts[0],ts[-1]))
    axs.set_ylim((-3.0,4.0))
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Roll Rate [deg/s]")
    axs.set_title("Linear Simulation Response to Aileron Doublet")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    axs.legend()
    fig.savefig(save_folder + "/SAL_4_4_3_dblt_sim.png",dpi=300.0)
    if show_plots:
        plt.show()

    return


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

def EX_4_5_1(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[0,2,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)

    # build C matrix
    out = lambda x : output(base,x,base.u_trim,True)
    C = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    
    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    rows = [1,4]
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [2,1]
    chfmdeg = np.diag([1./val]*2)
    C = np.matmul((C[Crows,:])[:,rows],chfmdeg)
    D = np.zeros((2,1))
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=True,endln=True,print_report=True)
    report_latex(C,"C",decimals=5,align=True,print_report=True)
    x0 = base.Lin_Model.xhat_eq[rows]

    # define system
    ap = A; bp = B; cp = C; dp = D
    sysp = co.ss(ap,bp,cp,dp)
    sig = 20.2
    sysa = co.ss(-sig,sig,-1.0,0.0)
    sys1 = co.series(sysa,sysp)
    sysf = co.ss(-10.0,[10.0, 0.0],[[1.0],[0.0]],[[0.0, 0.0],[0.0, 1.0]])
    sys2 = co.series(sys1,sysf)
    a,b,c,d = co.ssdata(sys2)
    ka = 0.2
    acl = a - np.matmul(np.matmul(b,[[ka, 0]]),c)
    tf = co.ss2tf(co.ss(acl,b,c[[1],:],0))
    print_tf(tf,"q","u1")

    # add PI compensator
    sys3 = co.ss(acl,b,c,[[0.0],[0.0]])
    sysi = co.ss(0.0,3.0,1.0,1.0)
    sys4 = co.series(sysi,sys3)
    aa,bb,cc,dd = co.ssdata(sys4)
    k = np.linspace(0.0, 0.9,1000)
    r,_ = co.rlocus(co.ss(aa,bb,cc[1,:],0),kvect=k)
    kpi = np.argmin(np.abs(k-0.5))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kpi]),np.imag(ri[kpi]),c="0.5",marker=".")
    axs.set_xlim((-16,0))
    axs.set_ylim((-8,8))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_5_1_q_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()
    
    # different ka
    ka = 0.08
    acl = a - np.matmul(np.matmul(b,[[ka, 0]]),c)
    # tf = co.ss2tf(co.ss(acl,b,c[[1],:],0))
    # add PI compensator
    sys3 = co.ss(acl,b,c,[[0.0],[0.0]])
    sysi = co.ss(0.0,3.0,1.0,1.0)
    sys4 = co.series(sysi,sys3)
    aa,bb,cc,dd = co.ssdata(sys4)
    tf = co.ss2tf(aa,bb,cc,dd)
    # print_tf(tf,"q","r")
    k = np.linspace(0.0, 0.9,1000)
    r,_ = co.rlocus(co.ss(aa,bb,cc[[1],:],0),kvect=k)
    kpi = np.argmin(np.abs(k-0.5))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[kpi]),np.imag(ri[kpi]),c="0.5",marker=".")
    axs.set_xlim((-16,0))
    axs.set_ylim((-8,8))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_5_1_q_rlocus2.png",dpi=300.0)
    if show_plots:
        plt.show()
    
    # sim
    print("run sim...")
    acl2 = aa - np.matmul(bb,0.5*cc[[1],:])
    sys = co.ss(acl2,0.5*bb,cc[[1],:],0)
    aF,bF,cF,dF = co.ssdata(sys)
    aF_nz,bF_nz,cF_nz,dF_nz = co.ssdata(co.ss(acl,0.5*b,c[[1],:],0))
    # print(cF)
    # print(cF_nz)
    # print(aF)
    # print(aF_nz)
    x0_nz = np.concatenate((np.zeros((1,)),x0,np.zeros((1,))))
    x0 = np.concatenate((np.zeros((2,)),x0,np.zeros((1,))))
    # acl4 = a - np.matmul(np.matmul(b,np.diag([0.4,1.3])),c)
    ts = np.linspace(0.0,3.0,num=501)
    r    = lambda t : np.array([0.0,0.0,0.0,1.0,0.0])
    xdot = lambda t,x : np.matmul(aF,x-r(t))
    xs = odeint(xdot,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    r_nz = lambda t : np.array([0.0,0.0,1.0,0.0])
    xdot_nz = lambda t,x : np.matmul(aF_nz,x-r_nz(t))
    xs_nz = odeint(xdot_nz,x0_nz,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(ts,xs   [3],c="0.5",lw=0.75,label="With Zero")
    axs.plot(ts,xs_nz[2],c="k",lw=0.75,label="No Zero")
    axs.set_xlim((ts[0],ts[-1]))
    axs.set_ylim((0.0,1.4))
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Roll Rate [deg/s]")
    axs.set_title("Linear Simulation Response to Aileron Doublet")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    axs.legend()
    fig.savefig(save_folder + "/SAL_4_5_1_step_sim.png",dpi=300.0)
    if show_plots:
        plt.show()

    return


def lat_output_4_5_3(self,x,u,is_VAB_format=False):
    # output
    o = np.zeros((3,))

    # ny
    _,Fy,_,_,_,_,g = self._aerodynamics(x,u,is_trim=False,
        is_VAB_format=is_VAB_format)
    o[0] = Fy/self.inertia_model.W

    # p
    o[1] = x[3]*1.

    # r
    o[2] = x[5]*1.

    return o

def EX_4_5_3(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = (-np.ones((2,13))).tolist()
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[1,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True)

    # build C matrix
    out = lambda x : lat_output_4_5_3(base,x,base.u_trim,True)
    C_full = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    out2 = lambda u : lat_output_4_5_3(base,base.Lin_Model.xhat_eq,u,True)
    D_full = base.Lin_Model._calculate_jacobian(out2,base.u_trim)
    
    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val]*2)
    rows = [2,9,3,5]
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    Crows = [0,1,2]
    chfmdeg = np.diag([1./val]*len(rows))
    C = (C_full[Crows,:])[:,rows]
    chfmrows = [1,2]
    C[chfmrows,:] = np.matmul(C[chfmrows,:],chfmdeg)
    Dcol = [0,2]
    D = np.matmul( (D_full[Crows,:])[:,Dcol],ch2deg)
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=1,endln=1,print_report=1)
    report_latex(C,"C",decimals=5,align=1,endln=1,print_report=1)
    report_latex(D,"D",decimals=5,align=1,print_report=1)
    # # remove psi
    # rows = [2,9,3,5]
    # A = (base.Lin_Model.A_min[rows,:])[:,rows]
    # B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    # Crows = [0,1]
    # chfmdeg = np.diag([1./val]*len(rows))
    # C = np.matmul((C_full[Crows,:])[:,rows],chfmdeg)
    x0 = base.Lin_Model.xhat_eq[rows]

    # actuator dynamics
    ap = A; bp = B; cp = C; dp = D
    plant = co.ss(ap,bp,cp,dp)
    kari = 0.13*np.rad2deg(base.Lin_Model.xhat_eq[1]) - 0.7
    sig = 20.2
    aa = np.diag([-sig]*2) # two actuators
    ba = np.array([[sig,0.0],[kari*sig,sig]]) # inp-1=ail., inp-2=ARI & rdr
    ca = -np.eye(2) # sign change in C
    da = np.zeros((2,2))
    actua = co.ss(aa,ba,ca,da) # x1=beta, x2=phi, x3=p, x4=r
    sys1 = co.series(actua,plant) # x5=aileron, x6=rudder
    a1,b1,c1,d1 = co.ssdata(sys1)
    
    # washout
    km = base.Lin_Model.xhat_eq[1]
    aw = [[-1.0]]; bw = [[0.0, -km, 1.0]] # washout filter
    cw = [[0.0],[0.0],[1.0]] # outputs ay,p,rw
    dw = np.eye(3) # inputs ay,p,r
    dw[2,1] = -km 
    wash = co.ss(aw,bw,cw,dw)
    sys2 = co.series(sys1,wash) # x1=wash, x2=beta, etc
    a,b,c,d = co.ssdata(sys2) # complete augmented system
    # TF
    tf = co.ss2tf(co.ss(a1,b1[:,[0]],c1[[1],:],(d1[[1],:])[:,[0]]))
    # print(tf)
    print_tf(tf,"p","ua")

    # close p loop
    kp = 0.2
    acl = a - np.matmul(np.matmul(b,[[0,kp,0],[0,0,0]]),c)
    tf = co.ss2tf(co.ss(acl,b[:,[1]],c[[2],:],0))
    print_tf(tf,"rw","ur")

    # close r loop
    kr = 0.8
    acl = a - np.matmul(np.matmul(b,[[0,kp,0],[0,0,kr]]),c)
    tf = co.ss2tf(co.ss(acl,b[:,[1]],c[[0],:],0))
    print_tf(tf,"ay","rc")

    # root locus
    k = np.linspace(0.0, 100.,2000)
    r,_ = co.rlocus(co.ss(acl,b[:,[1]],c[[0],:],0),kvect=k)
    ki = np.argmin(np.abs(k-10.0))
    plt.close()
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # print(r.shape,k.shape)
    for i in range(r.shape[1]):
        ri = r[:,i]
        axs.plot(np.real(ri),np.imag(ri),c="k")
        axs.plot(np.real(ri[0]),np.imag(ri[0]),c="k",marker="x")
        axs.plot(np.real(ri[ki]),np.imag(ri[ki]),c="0.5",marker=".")
    axs.set_xlim((-23.5,0.5))
    axs.set_ylim((-12,12))
    axs.set_xlabel("Real [s]")
    axs.set_ylabel("Imaginary [s]")
    axs.set_title("Root Locus")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_4_5_3_ny_rlocus.png",dpi=300.0)
    if show_plots:
        plt.show()
    
    # close ny loop
    ka = 10.0
    acl = a - np.matmul(np.matmul(b,[[0,kp,0],[ka,0,kr]]),c)
    # tf = co.ss2tf(co.ss(acl,b,c,d))
    # print(tf)
    tf = co.ss2tf(co.ss(acl,b[:,[0]],c[[1],:],0))
    print_tf(tf,"p","pC")
    tf = co.ss2tf(co.ss(acl,b[:,[1]],c[[0],:],0))
    print_tf(tf,"ay","rC")

    return


def EX_4_7_1(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["simulation"]["limit_inputs"] = False
    input_json["simulation"]["limit_input_rates"] = False
    input_json["simulation"]["use_quaternions"] = True
    input_json["reference"] = {
        "deg2rad_states" : [4],
        "4" : [
            [ 0.0,  0.0 ],
            [10.0,  0.0 ],
            [10.0,  8.65], # 
            [20.0,  8.65], # 
            [20.0,  0.0 ],
            [50.0,  0.0 ],
            [50.0, 10.0 ]
        ],
        "sct_on_5" : False
    }
    input_json["controller"]["integral_states"] = [4]
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    input_json["controller"]["gains"]["KI"] = [[-1.0]]
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # add alpha filter state to dynamics
    a_trim = np.arctan2(base.x_trim[2],base.x_trim[0])
    x0 = np.concatenate((base.x_trim,[a_trim]))
    base.x_trim_euler_deg = np.concatenate((base.x_trim_euler_deg,[np.rad2deg(a_trim)]))
    dynamics = lambda t,x,i,g,u : \
        np.concatenate((base._get_dynamics(t,x,i,g,u,False)[:18],\
        [10.*(np.arctan2(x[2],x[0]) - x[18])]
    ))
    thtl = lambda t : 1.0
    r2d = np.deg2rad(1.0)
    base._get_control = lambda t,x,\
        is_controlled=True,given_control=False,u="o",\
        force_control_to_inputs=False : (np.array([
        base.u_trim[0],
        - 1.5*x[17] + 0.5*x[4] + 0.08*x[18],
        base.u_trim[2],
        base.u_trim[3] if t < 10. else thtl(t)
    ]), x[13:17])
    x0[17] = (0.5*x0[4] + 0.08*x0[18] - base.u_trim[1])/1.5 # set integrator state

    print((- 1.5*r2d*x0[17] + 0.5*r2d*x0[4] + 0.08*r2d*x0[18])/r2d)
    print(base.u_trim[1])
    print(x0[17]/r2d)
    # quit()

    # simulate
    base.tf = tf = 100.0
    dt = 0.001
    ts = np.linspace(0.0,tf,num=int(tf/dt + 1.))
    print("simulating...")
    xs = odeint(dynamics,x0,ts,args=(True,False,"o"),
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    # convert to euler angles
    xquat = xs*1.
    xnew = np.delete(xs,12,axis=0)
    xnew[9:12] = np.array([base._euler_angles(xs[:,i]) 
        for i in range(len(ts))]).T
    xs = xnew*1.
    
    # plot
    base.tarr = ts
    base.xarr = xs
    base.uarr = np.array([
        base._get_control(ts[i],xquat[:,i],True,False,"o",False)[0] \
        for i in range(len(ts))
    ]).T
    # calculate total velocity and aero angles
    Vxarr = (base.xarr[0]**2. + base.xarr[1]**2. + base.xarr[2]**2.)**0.5
    axarr = np.rad2deg(np.arctan2(base.xarr[2],base.xarr[0]))
    bxarr = np.rad2deg(np.arcsin(base.xarr[1]/Vxarr)) # experimental beta
    Mxarr = np.array([Vxarr[i]/base.stdatm(-base.xarr[8,i])[5] \
        for i in range(len(base.tarr))])
    base.aerox = np.array([Vxarr,Mxarr,axarr,bxarr])
    # convert to degrees
    xicnv = [3,4,5,9,10,11] + [12,13,14]*(base.order >=1) + [16,17]
    uicnv = [0,1,2]
    base.xarr[xicnv,:] = np.rad2deg(base.xarr[xicnv,:])
    base.uarr[uicnv,:] = np.rad2deg(base.uarr[uicnv,:])
    # convert POW state back to throttle
    POW2tau = lambda POW : POW/64.94 if POW < 50.0 else (POW + 117.38)/217.38
    base.xarr[15,:] = [POW2tau(base.xarr[15,i]) for i in range(len(ts))]
    temp = base.max_tau*1.,base.max_taudot*1.,base.min_taudot*1.
    base.max_tau = 1.0
    base.max_taudot =  1.0
    base.min_taudot = -1.0
    #
    plot_dict = dict(zoom_fraction=1.0,plot_full=True,plot_delta=False,
        plotting_directory=save_folder+"/",format="png",transparent=False)
    base.plot_results(**plot_dict)
    base.max_tau = temp[0]*1.
    base.max_taudot = temp[1]*1.
    base.min_taudot = temp[2]*1.

    # plot trajectory
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(xs[6],-xs[8],c="k")
    axs.set_xlim((0.0,25000.0))
    axs.set_ylim((0.0,25000.0))
    axs.set_xlabel("Distance north in ft.")
    axs.set_ylabel("Altitude in ft.")
    axs.set_title("Aircraft trajectory with pitch-rate CAS")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # axs.legend()
    fig.savefig(save_folder + "/full/base_position_view2.png",dpi=300.0)
    if show_plots:
        plt.show()

    return


def EX_4_7_2(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["simulation"]["limit_inputs"] = False
    input_json["simulation"]["limit_input_rates"] = False
    input_json["simulation"]["use_quaternions"] = True
    input_json["reference"] = {
        "deg2rad_states" : [3,4,5],
        "3" : [
            [  0.0,   0.0 ],
            [ 15.0,   0.0 ],
            [ 15.0, 150.0 ],
            [ 17.0, 150.0 ],
            [ 17.0,   0.0 ],
            [ 30.0,   0.0 ]
        ],
        "4" : [
            [ 0.0,  0.0 ],
            [ 5.0,  0.0 ],
            [ 5.0, 15.0 ],
            [30.0, 15.0 ]
        ],
        "5" : [
            [ 0.0,  0.0 ],
            [30.0,  0.0 ]
        ],
        "sct_on_5" : False
    }
    input_json["controller"]["integral_states"] = [4]
    input_json["controller"]["gains"]["K"] = -np.eye(3)
    input_json["controller"]["gains"]["KI"] = [[-1.0]]
    base = Aircraft(input_json)
    # set reference
    # qcom = np.deg2rad(15.0)
    # pcom = np.deg2rad(150.0)
    # def reference(t):
    #     r = np.zeros(22)
    #     if t > 5.0:
    #         r[4] = qcom
    #     if 15.0 <= t <= 17.0:
    #         r[3] = pcom
    #     print(t)
    #     return r
    # base._get_reference = reference

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # add alpha filter state to dynamics
    a_trim = np.arctan2(base.x_trim[2],base.x_trim[0])
    washout0 = 0.0
    x0 = np.concatenate((base.x_trim,[a_trim],[washout0]))
    base.x_trim_euler_deg = np.concatenate((base.x_trim_euler_deg,
        [np.rad2deg(a_trim)],[washout0]))
    d2r = np.deg2rad(1.0)
    r2d = 1./d2r
    washout = lambda x : (x[5]*r2d - x[3]*r2d*x[18] - x[19]*r2d)*d2r
    dynamics = lambda t,x,i,g,u : \
        np.concatenate((base._get_dynamics(t,x,i,g,u,False)[:18],\
        [10.*(np.arctan2(x[2],x[0]) - x[18])],
        [washout(x)]
    ))
    thtl = lambda t : 1.0
    def controller(t,x,is_controlled=True,given_control=False,u="o",\
        force_control_to_inputs=False):
        # initialize u
        u = x[:4]*0.0

        # aileron
        ua = 0.2*(base._get_reference(t)[3] - x[3])
        u[0] = -ua
        # elevator
        ue = 1.5*x[17] - 0.5*x[4] - 0.08*x[18]
        u[1] = -ue
        # rudder
        ari = (0.13*x[18]*r2d - 0.7)*ua*r2d
        xd6 = washout(x)
        err = base._get_reference(t)[5]*r2d - 0.8*xd6*r2d - 10.*base._SAL_ay
        # u[2] = -(err + ari)*d2r

        # throttle
        u[3] = base.u_trim[3] if t < 5. else thtl(t)
        # print(t,washout(x)*r2d,x[19]*r2d,u[1]*r2d,u[2]*r2d)
        print(t)

        return u,x[13:17]
    base._get_control = controller
    # base._get_control = lambda t,x,\
    #     is_controlled=True,given_control=False,u="o",\
    #     force_control_to_inputs=False : (np.array([
    #     -0.2*(base._get_reference(t)[3] - x[3]),
    #     - 1.5*x[18] + 0.5*x[4] + 0.08*x[20],
    #     -(
    #         base._get_reference(t)[5] - 0.8*(x[5] - x[3]*x[20] - x[21]) \
    #         - 10.*base._SAL_ay # err
    #     ) - (
    #         (0.13*x[20] - 0.7)*(0.2*(base._get_reference(t)[3] - x[3])) # ari
    #     ),
    #     base.u_trim[3] if t < 5. else thtl(t)
    # ]), x[13:17])
    x0[17] = (0.5*x0[4] + 0.08*x0[18] - base.u_trim[1])/1.5 # set integrator state

    # print((- 1.5*r2d*x0[17] + 0.5*r2d*x0[4] + 0.08*r2d*x0[18])/r2d)
    # print(base.u_trim[1])
    # print(x0[17]/r2d)
    # # quit()

    # simulate
    base.tf = tf = 30.0
    dt = 0.001
    ts = np.linspace(0.0,tf,num=int(tf/dt + 1.))
    print("simulating...")
    xs = odeint(dynamics,x0,ts,args=(True,False,"o"),
        # atol=1e-10,rtol=1e-10,
        tfirst=True).T
    # convert to euler angles
    xquat = xs*1.
    xnew = np.delete(xs,12,axis=0)
    xnew[9:12] = np.array([base._euler_angles(xs[:,i]) 
        for i in range(len(ts))]).T
    xs = xnew*1.
    
    # plot
    base.tarr = ts
    base.xarr = xs
    base.uarr = np.array([
        base._get_control(ts[i],xquat[:,i],True,False,"o",False)[0] \
        for i in range(len(ts))
    ]).T
    # calculate total velocity and aero angles
    Vxarr = (base.xarr[0]**2. + base.xarr[1]**2. + base.xarr[2]**2.)**0.5
    axarr = np.rad2deg(np.arctan2(base.xarr[2],base.xarr[0]))
    bxarr = np.rad2deg(np.arcsin(base.xarr[1]/Vxarr)) # experimental beta
    Mxarr = np.array([Vxarr[i]/base.stdatm(-base.xarr[8,i])[5] \
        for i in range(len(base.tarr))])
    base.aerox = np.array([Vxarr,Mxarr,axarr,bxarr])
    # convert to degrees
    xicnv = [3,4,5,9,10,11] + [12,13,14]*(base.order >=1) + [16,17]
    uicnv = [0,1,2]
    base.xarr[xicnv,:] = np.rad2deg(base.xarr[xicnv,:])
    base.uarr[uicnv,:] = np.rad2deg(base.uarr[uicnv,:])
    # convert POW state back to throttle
    POW2tau = lambda POW : POW/64.94 if POW < 50.0 else (POW + 117.38)/217.38
    base.xarr[15,:] = [POW2tau(base.xarr[15,i]) for i in range(len(ts))]
    temp = base.max_tau*1.,base.max_taudot*1.,base.min_taudot*1.
    base.max_tau = 1.0
    base.max_taudot =  1.0
    base.min_taudot = -1.0
    #
    plot_dict = dict(zoom_fraction=1.0,plot_full=True,plot_delta=False,
        plotting_directory=save_folder+"/",format="png",transparent=False)
    base.plot_results(**plot_dict)
    base.max_tau = temp[0]*1.
    base.max_taudot = temp[1]*1.
    base.min_taudot = temp[2]*1.

    # plot trajectory
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(xs[6],-xs[8],c="k")
    axs.set_xlim((0.0,6000.0))
    axs.set_ylim((0.0,6000.0))
    axs.set_xlabel("Distance north in ft.")
    axs.set_ylabel("Altitude in ft.")
    axs.set_title("Aircraft trajectory with pitch-rate CAS")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # axs.legend()
    fig.savefig(save_folder + "/full/base_position_view2.png",dpi=300.0)
    if show_plots:
        plt.show()

    return


def LQ(A,B,C,F,G,P,Q,R,r0,knum=10,use_pass=False,is_part_c=False,is_lat=False):
    print("running LQ...")
    # initialize    
    Kshape = (B.shape[1],C.shape[0])
    Kflatshape = (Kshape[0]*Kshape[1],)
    # K0 = np.ones(Kshape)
    print(np.linalg.eig(A)[0])
    if is_lat:
        p = np.array([complex(-0.1-j/10.,0.0) for j in range(A.shape[0])])
    else:
        if is_part_c:
            p = np.array(
                [complex(-3.26,2.83), complex(-3.26,-2.83), complex(-1.02,0.0),
                complex(-10.67,0.0),complex(-14.09)]
            )
        else:
            p = np.array(
                [complex(-8.67,9.72), complex(-8.67,-9.72), complex(-9.85,0.0),
                complex(-4.07,0.0),complex(-1.04,0.0)]
            )
    K1 = co.place(A,B,p)
    # print(A - mm(B,K1))
    # print(np.linalg.eig(A - mm(B,K1))[0])
    # print()
    CCT = mm(C, C.T)
    K1 = mm(mm(np.linalg.solve(CCT,np.eye(CCT.shape[0])),C),K1.T).T
    # first guess is now stabilizing
    K0 = K1*1.
    # print(A - mm(mm(B,K1),C))
    # print(np.linalg.eig(A - mm(mm(B,K1),C))[0])
    # print()
    # print(K1)
    # quit()
    # K0 = np.array([[-0.05, -1.08, 3.39]])
    flattenK = lambda Ko : np.reshape(Ko,Kflatshape)
    reformK  = lambda Ko : np.reshape(Ko,Kshape)
    Pshape = P.shape
    Pflatshape = (Pshape[0]*Pshape[1],)
    flattenP = lambda Po : np.reshape(Po,Pflatshape)
    reformP  = lambda Po : np.reshape(Po,Pshape)
    Sshape = (P.shape[1],C.shape[1])
    Sflatshape = (Sshape[0]*Sshape[1],)
    flattenS = lambda So : np.reshape(So,Sflatshape)
    reformS  = lambda So : np.reshape(So,Sshape)

    # initialize functions
    Ac = lambda A,B,C,K : A - np.matmul(np.matmul(B,K),C)
    Bc = lambda G,B,F,K : G - np.matmul(np.matmul(B,K),F)
    inv = lambda A : np.linalg.solve(A,np.eye(A.shape[0]))
    xbar = lambda Ac,Bc,r0 : -np.matmul(np.matmul(inv(Ac),Bc),r0)
    ybar = lambda C,xbar,F,r0 : np.matmul(C,xbar) + np.matmul(F,r0)
    X = lambda xbar : np.matmul(xbar,xbar.T)
    # optimiizer functions
    # g
    gfun  = lambda Ac,P1,P0 : np.matmul(Ac.T,P1) + np.matmul(P1,Ac) + P0
    gkfun = lambda Ac,P1,P0,k,Q,C,K,R : np.matmul(Ac.T,P1) + np.matmul(P1,Ac)+\
        np.math.factorial(k)*P0 + Q + mm(mm(mm(mm(C.T,K.T),R),K),C)
    # s
    sfun    = lambda Ac,S1,S0 : np.matmul(Ac,S0) + np.matmul(S0,Ac.T) + S1
    skm1fun = lambda Ac,S1,S0,k : np.matmul(Ac,S0) + np.matmul(S0,Ac.T) + \
        np.math.factorial(k)*S1

    # initialize arrays for optimizer use
    Ps = np.zeros((knum,) + Pshape)
    Ss = np.zeros((knum,) + Sshape)

    def opt_fun(K):
        # reform K
        K = reformK(K)

        # solve g equations
        P0 = P
        AcK = Ac(A,B,C,K)
        AcKinv = inv(AcK)
        BcK = Bc(G,B,F,K)
        for k in range(knum):
            if k < knum - 1:
                res = minimize(
                    lambda P1f : np.linalg.norm( gfun(AcK,reformP(P1f),P0) ),
                    flattenP(P0))
            else:
                res = minimize(
                    lambda P1f : np.linalg.norm( gkfun(AcK,reformP(P1f),P0,k,Q,C,K,R) ),
                    flattenP(P0))
            P0 = reformP(res.x)
            # assign
            Ps[k] = P0*1.
        
        if is_part_c:
            J = 0.5*mm(r0.T,mm(BcK.T,mm(Ps[k],mm(BcK,r0))))[0,0]
            print(K,J)
            print()
            return J
        else:
            pass
        
        # solve s equations
        xbarK = xbar(AcK,BcK,r0)
        ybarK = ybar(C,xbarK,F,r0)
        Sk = X(xbarK)
        for k in range(knum-1,-1,-1):
            if k == knum - 2:
                res = minimize(
                    lambda Skp1f : np.linalg.norm(sfun(AcK,Sk,reformS(Skp1f))),
                    flattenS(Sk))
                # print("  ",k)
            else:
                res = minimize(
                    lambda Skp1f : np.linalg.norm(\
                    skm1fun(AcK,Sk,reformS(Skp1f),k)), \
                    flattenS(Sk))
                # print(k)
            Sk = reformS(res.x)
            # assign
            Ss[k] = Sk*1.

        sums = 0.0
        for k in range(knum):
            sums += mm(Ps[k],Ss[k])
        
        # objective function
        XK = X(xbarK)
        J = 0.5 * np.trace(mm(Ps[k],XK))
        
        kf = knum-1
        grad = mm(mm(mm(mm(R,K),C),Ss[kf]),C.T) 
        grad += - mm(mm(B.T,sums),C.T) 
        grad += mm(mm(mm(mm(B.T,AcKinv.T),Ps[kf]),xbarK),ybarK.T)
        grad = grad[0,:]
        # print(grad)
        # print(grad.shape)
        print(K,J,grad)
        print()
        return J,grad

    # ,method="SLSQP",tol=2.0,method="Nelder-Mead"
    if use_pass:
        if is_part_c:
            K = np.array([[
                -0.08131105427860522, -0.47634547706436, 1.3765954017015933
            ]])
        else:
            K = np.array([[
                -0.046360270693492855, -1.0765616099150208, 3.4061347842829086
            ]])
    else:
        res = minimize(opt_fun,flattenK(K0),jac=not(is_part_c),options={"maxiter":25})
        print(res)
        K = reformK(res.x)
        print(K[0,0])
        print(K[0,1])
        print(K[0,2])

    return K


def EX_5_5_3(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[0,2,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True,
        drop_actrs=False)

    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    # build C matrix
    C = np.zeros((3,5))
    C[0,3] = C[1,1] = 1./val
    C[2,4] = 1.0
    #
    rows = [1,4,12]
    # print(base.Lin_Model.A_min)
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    A[0,2] *= val
    A[1,2] *= val
    A = np.block([[A,np.zeros((3,2))],[np.zeros((2,5))]])
    A[3,0] = 10.0; A[3,3] = -10.0; A[4,1] = -1./val
    B = base.Lin_Model.B_min[rows,:] # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    B = np.block([[B],[np.zeros((2,1))]])
    G = np.zeros((5,1)); G[4,0] = 1.0
    F = np.zeros((3,1))
    H = np.zeros((1,5)); H[0,1] = 1./val
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=True,endln=True,print_report=True)
    report_latex(G,"G",decimals=5,align=True,print_report=True)
    report_latex(C,"C",decimals=5,align=True,print_report=True)
    report_latex(F,"F",decimals=5,align=True,print_report=True)
    report_latex(H,"H",decimals=5,align=True,print_report=True)
    x0 = base.Lin_Model.xhat_eq[rows]; x0 = np.concatenate((x0,[x0[0],0.0]))
    x0[2] = np.rad2deg(x0[2])
    print(x0)

    # create K
    P = np.matmul(H.T,H)
    Q = np.zeros(A.shape)
    R = np.array([[1.0]])
    r0 = np.array([[1.0]])
    K = LQ(A,B,C,F,G,P,Q,R,r0,use_pass=True)

    # simulate
    print("run sim...")
    aF= A - mm(mm(B,K),C)
    ts = np.linspace(0.0,10.0,num=1001)
    r    = lambda t : np.array([0.0,1.0,0.0,0.0,0.0])
    xdot = lambda t,x : np.matmul(aF,x-r(t))
    xs = odeint(xdot,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(ts,xs   [1],c="k",lw=0.75)
    axs.set_xlim((ts[0],ts[-1]))
    axs.set_ylim((0.0,1.5))
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Output [deg/s]")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_5_5_3_step_sim_b.png",dpi=300.0)
    if show_plots:
        plt.show()

    
    # part c
    p = 10.0
    P = np.diag([0.0]*4 + [p])
    Q = np.diag([0.0]*2 + [1.0] + [0.0]*2)
    R = np.zeros((1,1))
    r0 = np.array([[1.0]])
    K = LQ(A,B,C,F,G,P,Q,R,r0,is_part_c=True,use_pass=True)

    # simulate
    print("run sim...")
    aF= A - mm(mm(B,K),C)
    ts = np.linspace(0.0,10.0,num=1001)
    r    = lambda t : np.array([0.0,1.0,0.0,0.0,0.0])
    xdot = lambda t,x : np.matmul(aF,x-r(t))
    xs = odeint(xdot,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(ts,xs   [1],c="k",lw=0.75)
    axs.set_xlim((ts[0],ts[-1]))
    axs.set_ylim((0.0,1.5))
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Output [deg/s]")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    fig.savefig(save_folder + "/SAL_5_5_3_step_sim_c.png",dpi=300.0)
    if show_plots:
        plt.show()


    return


def EX_5_5_3L(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["controller"]["gains"]["K"] = -np.ones((2,14))
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # test numerical linearization
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[13,15], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[1,3],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True,
        drop_actrs=False)

    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    # build C matrix
    C = np.zeros((5,9))
    C[0,5] = C[1,1] = C[2,2] = 1./val
    C[3,7] = C[4,8] = 1.0
    #
    rows = [2,3,5,9,12,13]
    # print(base.Lin_Model.A_min)
    # cols = [3,1,0,2]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    # print(A)
    A[0,4] *= val; A[1,4] *= val; A[2,4] *= val
    A[0,5] *= val; A[1,5] *= val; A[2,5] *= val
    A = np.block([[A,np.zeros((6,3))],[np.zeros((3,9))]])
    A[6,0] = 10.0; A[6,6] = -10.0; A[7,1] = -1./val; A[8,2] = -1./val
    B = base.Lin_Model.B_min[rows,:] # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    B = np.block([[B],[np.zeros((3,2))]])
    G = np.zeros((9,2)); G[7,0] = G[8,1] = 1.0
    F = np.zeros((5,2))
    H = np.zeros((2,9)); H[0,1] = H[1,2] = 1./val
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=True,endln=True,print_report=True)
    report_latex(G,"G",decimals=5,align=True,print_report=True)
    report_latex(C,"C",decimals=5,align=True,print_report=True)
    report_latex(F,"F",decimals=5,align=True,print_report=True)
    report_latex(H,"H",decimals=5,align=True,print_report=True)
    x0 = base.Lin_Model.xhat_eq[rows]; x0 = np.concatenate((x0,[x0[0],0.0,0.0]))
    x0[4] = np.rad2deg(x0[4])
    x0[5] = np.rad2deg(x0[5])
    print(x0)

    # create K
    P = np.matmul(H.T,H)
    Q = np.zeros(A.shape)
    R = np.eye(2)
    r0 = np.array([[1.0],[1.0]])
    # K = LQ(A,B,C,F,G,P,Q,R,r0,is_lat=True)#,use_pass=True)

    # # simulate
    # print("run sim...")
    # aF= A - mm(mm(B,K),C)
    # ts = np.linspace(0.0,10.0,num=1001)
    # r    = lambda t : np.array([0.0,1.0,0.0,0.0,0.0])
    # xdot = lambda t,x : np.matmul(aF,x-r(t))
    # xs = odeint(xdot,x0,ts,
    #     atol=1e-10,rtol=1e-10,
    #     tfirst=True).T
    # fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # axs.plot(ts,xs   [1],c="k",lw=0.75)
    # axs.set_xlim((ts[0],ts[-1]))
    # axs.set_ylim((0.0,1.5))
    # axs.set_xlabel("Time [s]")
    # axs.set_ylabel("Output [deg/s]")
    # axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # fig.savefig(save_folder + "/SAL_5_5_3L_step_sim_b.png",dpi=300.0)
    # if show_plots:
    #     plt.show()

    
    # # part c
    # p = 10.0
    # P = np.diag([0.0]*4 + [p])
    # Q = np.diag([0.0]*2 + [1.0] + [0.0]*2)
    # R = np.zeros((1,1))
    # r0 = np.array([[1.0]])
    # K = LQ(A,B,C,F,G,P,Q,R,r0,is_part_c=True,use_pass=True)

    # # simulate
    # print("run sim...")
    # aF= A - mm(mm(B,K),C)
    # ts = np.linspace(0.0,10.0,num=1001)
    # r    = lambda t : np.array([0.0,1.0,0.0,0.0,0.0])
    # xdot = lambda t,x : np.matmul(aF,x-r(t))
    # xs = odeint(xdot,x0,ts,
    #     atol=1e-10,rtol=1e-10,
    #     tfirst=True).T
    # fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    # axs.plot(ts,xs   [1],c="k",lw=0.75)
    # axs.set_xlim((ts[0],ts[-1]))
    # axs.set_ylim((0.0,1.5))
    # axs.set_xlabel("Time [s]")
    # axs.set_ylabel("Output [deg/s]")
    # axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # fig.savefig(save_folder + "/SAL_5_5_3L_step_sim_c.png",dpi=300.0)
    # if show_plots:
    #     plt.show()


    return


def EX_5_8_1(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["simulation"]["use_quaternions"] = True
    input_json["controller"]["gains"]["K"] = -np.block([np.eye(2),np.zeros((2,12))])
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = -0.3
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    V_trim = 502.0 # ft/s
    a_trim = 0.3006 # rad
    b_trim = 4.1e-5 # rad
    th_trim = 0.3006 # rad
    q_trim = 0.3
    tau_trim = 1.023
    el_trim = np.deg2rad(-7.082) # rad
    ail_trim = np.deg2rad(-6.2e-4) # rad
    rdr_trim = np.deg2rad(0.01655) # rad
    vx_trim = V_trim*np.cos(a_trim)*np.cos(b_trim)
    vy_trim = V_trim               *np.sin(b_trim)
    vz_trim = V_trim*np.sin(a_trim)*np.cos(b_trim)
    POW_trim = TGEAR(tau_trim)
    quat = euler_2_quat([0.0,th_trim,0.0])
    base.x_trim = np.array([
        vx_trim,vy_trim,vz_trim,
        0.0,q_trim,0.0,
        0.0,0.0,-base.H0] 
        + quat 
        + [ail_trim,el_trim,rdr_trim,POW_trim
    ])
    base.u_trim = np.array([
        ail_trim,el_trim,rdr_trim,tau_trim
    ])
    base.x_trim_euler = np.array([
        vx_trim,vy_trim,vz_trim,
        0.0,q_trim,0.0,
        0.0,0.0,-base.H0,
        0.0,th_trim,0.0,
        ail_trim,el_trim,rdr_trim,POW_trim
    ])
    base.x_trim_euler_deg = base.x_trim_euler*1.
    xro = [3,4,5,9,10,11,12,13,14]
    base.x_trim_euler_deg[xro] = np.rad2deg(base.x_trim_euler[xro])*1.
    uro = [0,1,2]
    base.u_trim_deg = base.u_trim*1.
    base.u_trim_deg[uro] = np.rad2deg(base.u_trim[uro])*1.
    base._report_trim_solution()

    # print(base.x_trim)

    # test numerical linearization
    base.use_quaternions = False
    base._build_controller(report=False,
        save_matrices=False,
        mrrr=[12,14], # [2,3,5,6,7,8,9,10,11,12],
        mrrc=[0,2],
        run_freq=False,
        use_numerical_linearization=True,
        numerical_dynamics=base._nonlinear_euler_dynamics_VAB,
        use_VAB_format=True, turn_off_warnings=True,
        skip_reporting=True,
        drop_actrs=False)
    base.use_quaternions = True
    base._initialize_state(no_report=True)

    # report matrices
    val = np.deg2rad(1.0)
    ch2deg = np.diag([1.0] + [val]*3)
    ch2deg = np.diag([val])
    # build C matrix
    C = np.zeros((3,5))
    C[0,3] = C[1,1] = 1./val
    C[2,4] = 1.0
    #
    rows = [0,1,10,4,12]
    # print(base.Lin_Model.A_min)
    cols = [0,]
    col_thtl = [-1,]
    A = (base.Lin_Model.A_min[rows,:])[:,rows]
    A[:-1,-1] = np.deg2rad(A[:-1,-1])
    # A[3,0] = 0.0
    # A[3,1] = -4.56
    # A[3,3] = -1.58
    # A[3,4] = -0.2
    #
    # A[0,2] *= val
    # A[1,2] *= val
    # A = np.block([[A,np.zeros((5,2))],[np.zeros((2,5))]])
    # A[3,0] = 10.0; A[3,3] = -10.0; A[4,1] = -1./val
    B_thtl = (base.Lin_Model.A_min[rows,:])[:,col_thtl]
    # print(B_thtl)
    B = (base.Lin_Model.B_min[rows,:])[:,cols] # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    #
    #
    A = np.array([
        [-0.1270, -235.0000, -32.2000, -9.5100, -0.2440],
        [0, -0.9690, 0, 0.9080, -0.0020],
        [0, 0, 0, 1.0000, 0],
        [0, -4.5600, 0, -1.5800, -0.2000],
        [0, 0, 0, 0, -20.2000]
    ])
    B = np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [20.2]
    ])
    #
    #
    report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    report_latex(B,"B",decimals=5,align=True,endln=True,print_report=True)
    # report_latex(C,"C",decimals=5,align=True,print_report=True)
    x0 = base.Lin_Model.xhat_eq[rows]#; x0 = np.concatenate((x0,[x0[0],0.0]))
    # x0[2] = np.rad2deg(x0[2])
    x0 = np.zeros((5,))
    print(x0)

    # nz
    nz  = np.array([[  0.004, 15.88  , 0.0, 1.481 ,  0.033   ]])
    nzp = np.array([[  0.004, 16.2620, 0.0, 0.9780, -0.0485  ]])
    Cstar = nzp + 12.4*np.array([[0.0]*3 + [1.0] + [0.0]])
    print("Cstar =",Cstar)
    print()

    # evaluate eigenvalues
    C = nz
    IA = np.eye(A.shape[0])
    Az = mm(IA - mm(B,mm(np.linalg.solve(mm(C,B),np.eye(B.shape[1])),C)),A)
    Az_eigs = np.linalg.eig( Az )[0]
    print("C = nz ; Az eigs =\n",Az_eigs)
    C = nzp
    IA = np.eye(A.shape[0])
    Az = mm(IA - mm(B,mm(np.linalg.solve(mm(C,B),np.eye(B.shape[1])),C)),A)
    Az_eigs = np.linalg.eig( Az )[0]
    print("C = nzp; Az eigs =\n",Az_eigs)
    C = Cstar
    IA = np.eye(A.shape[0])
    Az = mm(IA - mm(B,mm(np.linalg.solve(mm(C,B),np.eye(B.shape[1])),C)),A)
    Az_eigs = np.linalg.eig( Az )[0]
    print("C = C* ; Az eigs =\n",Az_eigs)
    C = Cstar - 0.014*np.array([[1.0] + [0.0]*4])
    IA = np.eye(A.shape[0])
    Az = mm(IA - mm(B,mm(np.linalg.solve(mm(C,B),np.eye(B.shape[1])),C)),A)
    Az_eigs = np.linalg.eig( Az )[0]
    print("C = C*-0.014vt; Az eigs =\n",Az_eigs)
    print()
    report_latex(C,"C",decimals=5,align=True,print_report=True)
    print()

    CBinv = np.linalg.solve(mm(C,B),np.eye(C.shape[0]))[0,0]

    # define controller
    def controller(t,x):
        x_VA = x*1.
        # elevator
        r = 1.0
        rdot = 0.0
        K = 20.0
        y = mm(C,x_VA)[0]
        e = r - y
        v = K*e
        w = rdot - mm(mm(C,A),x_VA)[0] + v
        ue = CBinv*w

        return ue
    
    def dynamics(t,x):
        u = controller(t,x)*20.2
        dx = mm(A,x) + np.array([0.0]*4+[u])
        # print(dx)
        return dx
    
    # simulate
    base.tf = tf = 5.0
    base._get_control = controller
    dt = 0.001
    ts = np.linspace(0.0,tf,num=int(tf/dt + 1.))
    print("simulating...")
    xs = odeint(dynamics,x0,ts,
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    # # convert to euler angles
    # xquat = xs*1.
    # xnew = np.delete(xs,12,axis=0)
    # xnew[9:12] = np.array([base._euler_angles(xs[:,i]) 
    #     for i in range(len(ts))]).T
    # xs = xnew*1.

    # print(np.rad2deg(xs[2]))
    
    # # plot
    # base.tarr = ts
    # base.xarr = xs
    # base.uarr = np.array([
    #     base._get_control(ts[i],xquat[:,i],True,False,"o",False)[0] \
    #     for i in range(len(ts))
    # ]).T
    # # calculate total velocity and aero angles
    # Vxarr = (base.xarr[0]**2. + base.xarr[1]**2. + base.xarr[2]**2.)**0.5
    # axarr = np.rad2deg(np.arctan2(base.xarr[2],base.xarr[0]))
    # bxarr = np.rad2deg(np.arcsin(base.xarr[1]/Vxarr)) # experimental beta
    # Mxarr = np.array([Vxarr[i]/base.stdatm(-base.xarr[8,i])[5] \
    #     for i in range(len(base.tarr))])
    # base.aerox = np.array([Vxarr,Mxarr,axarr,bxarr])
    # # convert to degrees
    # xicnv = [3,4,5,9,10,11] + [12,13,14]*(base.order >=1)
    # uicnv = [0,1,2]
    # base.xarr[xicnv,:] = np.rad2deg(base.xarr[xicnv,:])
    # base.uarr[uicnv,:] = np.rad2deg(base.uarr[uicnv,:])
    # # convert POW state back to throttle
    # POW2tau = lambda POW : POW/64.94 if POW < 50.0 else (POW + 117.38)/217.38
    # base.xarr[15,:] = [POW2tau(base.xarr[15,i]) for i in range(len(ts))]
    # temp = base.max_tau*1.,base.max_taudot*1.,base.min_taudot*1.
    # base.max_tau = 1.0
    # base.max_taudot =  1.0
    # base.min_taudot = -1.0
    
    # plot_dict = dict(zoom_fraction=1.0,plot_full=True,plot_delta=False,
    #     plotting_directory=save_folder+"/",format="png",transparent=False)
    # base.plot_results(**plot_dict)
    # base.max_tau = temp[0]*1.
    # base.max_taudot = temp[1]*1.
    # base.min_taudot = temp[2]*1.

    # plot
    y = np.array([mm(C,xs[:,i]) for i in range(len(ts))])
    fig,axs = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axs.plot(ts,y,c="k",lw=0.75)
    axs.set_xlim((ts[0],ts[-1]))
    axs.set_ylim((0.0,1.1))
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("modified C* CV")
    axs.grid(which="major",lw=0.6,ls="-",c="0.75")
    # axs.legend()
    fig.savefig(save_folder + "/SAL_5_8_1_sim_Cstar.png",dpi=300.0)
    fiq,axq = plt.subplots(figsize=(3.25,3.5),constrained_layout=True)
    axq.plot(ts,xs   [3],c="k",lw=0.75)
    axq.set_xlim((ts[0],ts[-1]))
    # axq.set_ylim((0.0,1.1))
    axq.set_xlabel("Time [s]")
    axq.set_ylabel("q [rad/s]")
    axq.grid(which="major",lw=0.6,ls="-",c="0.75")
    # axq.legend()
    fiq.savefig(save_folder + "/SAL_5_8_1_sim_q.png",dpi=300.0)
    if show_plots:
        plt.show()
    


    return


def EX_5_8_2star(input_json,save_folder=".",show_plots=False):

    print("running {}...".format(save_folder.split("/")[1]))

    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    # build aircraft
    input_json["reference"] = {
        "deg2rad_states" : [3,4,5],
        "3" : [
            [  0.0,  1.0 ],
            [100.0,  1.0 ],
        ],
        "4" : [
            [  0.0,  1.0 ],
            [100.0,  1.0 ],
        ],
        "5" : [
            [  0.0,  1.0 ],
            [100.0,  1.0 ],
        ],
        "sct_on_5" : False
    }
    input_json["controller"]["integral_states"] = [3,4,5]
    input_json["controller"]["gains"]["K"] = [[-1.0]]
    input_json["controller"]["gains"]["KI"] = [[-1.0]]
    input_json["simulation"]["use_quaternions"] = True
    # input_json["simulation"]["stevens_and_lewis"] = False
    base = Aircraft(input_json)

    base.H0 = 0.0
    base.V0 = 502.0
    PARAM = 0.35
    xcg = (0.35 - PARAM)*base.cw
    base.cgshift = [xcg,0.0,0.0]
    base._initialize_state(no_report=True)
    base._report_trim_solution()

    # # test numerical linearization
    # base._build_controller(report=False,
    #     save_matrices=False,
    #     # mrrr=[12,13,14], # [2,3,5,6,7,8,9,10,11,12],
    #     # mrrc=[0,2,3],
    #     run_freq=False,
    #     # use_numerical_linearization=True,
    #     numerical_dynamics=base._nonlinear_euler_dynamics,
    #     turn_off_warnings=True,
    #     skip_reporting=True)

    # # build C matrix
    # out = lambda x : output(base,x,base.u_trim,True)
    # C = base.Lin_Model._calculate_jacobian(out,base.Lin_Model.xhat_eq)
    
    # # report matrices
    # val = np.deg2rad(1.0)
    # ch2deg = np.diag([1.0] + [val]*3)
    # ch2deg = np.diag([val])
    # rows = [1,4]
    # # cols = [3,1,0,2]
    # A = (base.Lin_Model.A_min[rows,:])[:,rows]
    # B = np.matmul(base.Lin_Model.B_min[rows,:],ch2deg) # np.matmul((base.Lin_Model.B_min[rows,:])[:,cols],ch2deg)
    # Crows = [2,1]
    # chfmdeg = np.diag([1./val]*2)
    # C = np.matmul((C[Crows,:])[:,rows],chfmdeg)
    # D = np.zeros((2,1))
    # report_latex(A,"A",decimals=5,predecimals=5,align=1,endln=1,print_report=1)
    # report_latex(B,"B",decimals=5,align=True,endln=True,print_report=True)
    # report_latex(C,"C",decimals=5,align=True,print_report=True)
    # x0 = base.Lin_Model.xhat_eq[rows]

    # define reference
    # define control
    def control(t,state,is_controlled=True,given_control=False,u="o",\
        force_control_to_inputs=False):

        # other defs
        x_eq = base.x_trim
        u_eq = base.u_trim
        zeta = np.array([0.7]*3)
        wn = np.array([7.0]*3)
        KP = 2.*zeta*wn
        KI = wn**2.
        drfun = 0.0
        #-------------------#
        # STATE DEFINITIONS #
        #-------------------#
        V_xb    = state[0]
        V_yb    = state[1]
        V_zb    = state[2]
        p       = state[3]
        q       = state[4]
        r       = state[5]
        x_f     = state[6]
        y_f     = state[7]
        z_f     = state[8]
        # phi     = state[9]
        # theta   = state[10]
        # psi     = state[11]
        pI      = -state[17]
        qI      = -state[18]
        rI      = -state[19]
        # Derived Quantities
        V_tot   = np.sqrt(V_xb**2+V_yb**2+V_zb**2)
        V_xb_ss = x_eq[0]
        V_yb_ss = x_eq[1]
        V_zb_ss = x_eq[2]
        V_ss    = np.sqrt(V_xb_ss**2+V_yb_ss**2+V_zb_ss**2)
        aero = 0
        if aero == 0:
            alpha   = np.arctan2(V_zb,V_xb)
            beta    = np.sin(V_yb/V_tot)
            V = V_tot
        # elif aero == 1:
        #     alpha   = 0.0
        #     beta    = 0.0
        #     V = V_tot
        # elif aero == 2:
        #     alpha = np.arctan2(V_zb_ss/V_xb_ss)
        #     beta  = np.sin(V_yb_ss/V_ss)
        #     V     = V_tot
        # elif aero == 3:
        #     alpha = np.arctan2(V_zb_ss/V_xb_ss)
        #     beta  = np.sin(V_yb_ss/V_ss)
        #     V = V_xb
        # elif aero == 4:
        #     alpha = np.arctan2(V_zb_ss/V_xb_ss)
        #     beta  = np.sin(V_yb_ss/V_ss)
        #     V = V_ss

        _,_,_,_,rho,g = base.stdatm(float(-z_f))#;   rho = rho*0.001940320
        # other componets
        S_w = 300
        b_w = 30
        cbar_w = 11.32
        h_xb = 160
        h_yb = 0
        h_zb = 0
        hmat = np.array([[0, -h_zb, h_yb], [h_zb, 0, -h_xb], [-h_yb, h_xb, 0]])
        # Weight and inertia (See Table A.2)
        W = base.inertia_model.W
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = base.inertia_model.inertia_results(0.0)

        # define aero derivs
        C_ell_beta = -0.0786
        C_ell_pbar = -0.3182
        C_ell_rbar = 0.0469
        C_ell_Lrbar = 0.1067
        C_ell_delta_a = -0.0741
        C_ell_delta_r = 0.0257
        C_m_0 = -0.0097
        C_m_alpha = 0.1766
        C_m_qbar = -4.8503
        C_m_delta_e = -0.5881
        C_n_beta = 0.2426
        C_n_pbar = 0.0131
        C_n_Lpbar = -0.1005
        C_n_rbar = -0.1787
        C_n_delta_a = -0.0276
        C_n_Ldelta_a = 0.0
        C_n_delta_r = -0.0899
        C_L_0 = 0.0456
        C_L_alpha = 3.5791
        C_L_1 = C_L_0 + C_L_alpha*alpha
        pbar = p*b_w/(2*V)
        qbar = q*cbar_w/(2*V)
        rbar = r*b_w/(2*V)

        # define components
        I = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
        Iinv = np.linalg.solve(I,np.eye(3))
        G = 0.5*rho*V**2.*S_w*np.diag([b_w,cbar_w,b_w])
        Cstates = np.array([
            C_ell_beta*beta + C_ell_pbar*pbar + \
                (C_ell_Lrbar*C_L_1 + C_ell_rbar)*rbar,
            C_m_0 + C_m_alpha*alpha + C_m_qbar*qbar,
            C_n_beta*beta + (C_n_Lpbar*C_L_1+C_n_pbar)*pbar + C_n_rbar*rbar
        ])
        w  = np.array([ p, q, r])
        wI = np.array([pI,qI,rI])
        Imult = np.array([(Iyy-Izz)*q*r + Iyz*(q**2-r**2) + Ixz*p*q - Ixy*p*r,
                    (Izz-Ixx)*p*r + Ixz*(r**2-p**2) + Ixy*q*r - Iyz*p*q,
                    (Ixx-Iyy)*p*q + Ixy*(p**2-q**2) + Iyz*p*r - Ixz*q*r])
        Sigma = np.matmul(hmat,w) + Imult
        Ccontrol = np.array([
            [C_ell_delta_a, 0.0, C_ell_delta_r],
            [0.0, C_m_delta_e, 0.0],
            [(C_n_Ldelta_a*C_L_1 + C_n_delta_a), 0.0, C_n_delta_r]
        ])
        IinvGCcontrol = np.matmul(np.matmul(Iinv,G),Ccontrol)
        IinvGCcontrol_inv = np.linalg.solve(IinvGCcontrol,np.eye(3))

        # # control law design
        # wdot = I^-1*(G*Cstates + Sigma) + I^-1*G*Ccontrol*delta
        # delta = (I^-1*G*Ccontrol)^-1 ( v - I^-1*(G*Cstates + Sigma) )
        # v = - k_p*p - k_q*q - k_r*r

        # define controller
        v = - KP*w - KI*wI + drfun
        delta = np.matmul(IinvGCcontrol_inv, ( v - np.matmul(Iinv,(np.matmul(G,Cstates) + Sigma)) ) )
        u = np.concatenate((delta,[u_eq[3]]))

        return u,state[13:17]

    # simulate
    base._get_control = control
    base.tf = tf = 10.0
    dt = 0.001
    ts = np.linspace(0.0,tf,num=int(tf/dt + 1.))
    print("simulating...")
    xs = odeint(base._dynamics,base.x_trim,ts,args=(True,False,"o"),
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    # convert to euler angles
    xquat = xs*1.
    xnew = np.delete(xs,12,axis=0)
    xnew[9:12] = np.array([base._euler_angles(xs[:,i]) 
        for i in range(len(ts))]).T
    xs = xnew*1.
    
    # plot
    base.tarr = ts
    base.xarr = xs
    base.uarr = np.array([
        base._get_control(ts[i],xquat[:,i],True,False,"o",False)[0] \
        for i in range(len(ts))
    ]).T
    # calculate total velocity and aero angles
    Vxarr = (base.xarr[0]**2. + base.xarr[1]**2. + base.xarr[2]**2.)**0.5
    axarr = np.rad2deg(np.arctan2(base.xarr[2],base.xarr[0]))
    bxarr = np.rad2deg(np.arcsin(base.xarr[1]/Vxarr)) # experimental beta
    Mxarr = np.array([Vxarr[i]/base.stdatm(-base.xarr[8,i])[5] \
        for i in range(len(base.tarr))])
    base.aerox = np.array([Vxarr,Mxarr,axarr,bxarr])
    # convert to degrees
    xicnv = [3,4,5,9,10,11] + [12,13,14]*(base.order >=1)
    uicnv = [0,1,2]
    base.xarr[xicnv,:] = np.rad2deg(base.xarr[xicnv,:])
    base.uarr[uicnv,:] = np.rad2deg(base.uarr[uicnv,:])
    # convert POW state back to throttle
    POW2tau = lambda POW : POW/64.94 if POW < 50.0 else (POW + 117.38)/217.38
    base.xarr[15,:] = [POW2tau(base.xarr[15,i]) for i in range(len(ts))]
    temp = base.max_tau*1.,base.max_taudot*1.,base.min_taudot*1.
    base.max_tau = 1.0
    base.max_taudot =  1.0
    base.min_taudot = -1.0
    #
    plot_dict = dict(zoom_fraction=1.0,plot_full=True,plot_delta=False,
        plotting_directory=save_folder+"/",format="png",transparent=False)
    base.plot_results(**plot_dict)
    base.max_tau = temp[0]*1.
    base.max_taudot = temp[1]*1.
    base.min_taudot = temp[2]*1.

    return

if __name__ == "__main__":
    # filename
    base_SAL_file = "f16_SAL_fs_in.json"

    # read in json to ensure no file changes while running
    base_SAL_dict = json.loads( open(base_SAL_file).read() )

    # # test aero model / dynamics and trim
    # test_code(base_SAL_dict)

    # # test linearization
    # test_linearization(base_SAL_dict)

    # # Example 4.4-1
    # EX_4_4_1(base_SAL_dict,save_folder="SAL_plots/EX_4_4_1",show_plots=False)

    # # Example 4.4-2
    # EX_4_4_2(base_SAL_dict,save_folder="SAL_plots/EX_4_4_2",show_plots=False)

    # # Example 4.4-3
    # EX_4_4_3(base_SAL_dict,save_folder="SAL_plots/EX_4_4_3",show_plots=False)

    # # Example 4.5-1
    # EX_4_5_1(base_SAL_dict,save_folder="SAL_plots/EX_4_5_1",show_plots=False)

    # # Example 4.5-3
    # EX_4_5_3(base_SAL_dict,save_folder="SAL_plots/EX_4_5_3",show_plots=False)

    # # Example 4.7-1
    # EX_4_7_1(base_SAL_dict,save_folder="SAL_plots/EX_4_7_1",show_plots=False)

    # # Example 4.7-2
    # EX_4_7_2(base_SAL_dict,save_folder="SAL_plots/EX_4_7_2",show_plots=False)

    # # Example 5.5-3
    # EX_5_5_3(base_SAL_dict,save_folder="SAL_plots/EX_5_5_3",show_plots=False)

    # # Example 5.5-3L
    # EX_5_5_3L(base_SAL_dict,save_folder="SAL_plots/EX_5_5_3L",show_plots=False)

    # # Example 5.8-1
    # EX_5_8_1(base_SAL_dict,save_folder="SAL_plots/EX_5_8_1",show_plots=False)

    # Example 5.8-2star
    EX_5_8_2star(base_SAL_dict,save_folder="SAL_plots/EX_5_8_2star",show_plots=False)

