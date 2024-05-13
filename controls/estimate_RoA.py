import numpy as np
import json
import control as co
import sympy as sy
from controller_simulation import Aircraft as Aircraft
from linearization import linearization as lin


def get_control(self,t,x,is_controlled=True,given_control=False,u="o"):
    # build control or pass through
    if not given_control:
        if is_controlled:
            # if self.use_quaternions:
            #     x_euler = self.quat2euler_state(x)
            # else:
            #     x_euler = x*1.
            #     # reset angles
            #     ## INTSTATE
            #     x_euler[9:12] = quat_2_euler(euler_2_quat(x_euler[9:12]))
            x_euler = x
            ##################################
            x_tr = self.Lin_Model.xhat_eq*1.
            u_tr = self.Lin_Model.uhat_eq*1.
            K_tr = self.Lin_Model.K
            #
            u = self.u_trim*1.
            ###################################
            Dx = x_euler - x_tr
            u = u_tr - np.matmul(K_tr,Dx)#*0.
            if False: # self.order > 0:
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
            inputs = u*1.
    
    # # limit actuators
    # u = self._limit_input(u)
    # inputs = self._limit_input(inputs)

    return u,inputs


def aerodynamics(craft,x,u,Vg=[0.0,0.0,0.0],Wg=[0.0,0.0,0.0]):
    # aero conditions
    ## INTSTATE
    Vu,Vv,Vw = x[0]+Vg[0], x[1]+Vg[1], x[2]+Vg[2]
    a = sy.atan2(Vw,Vu)
    V = (Vu * Vu + Vv * Vv + Vw * Vw)**0.5
    b = sy.asin(Vv/V)
    _,g,_,_,rho,sos = craft.stdatm(-x[6])
    # ##############################
    # g = 32.12780074195162
    # ##############################
    M = V / sos

    # nondimensionalize rates
    ## INTSTATE
    pbar = (x[3]+Wg[0])*craft.bw/2./V
    qbar = (x[4]+Wg[1])*craft.cw/2./V
    rbar = (x[5]+Wg[2])*craft.bw/2./V

    # pass in controls state
    ail = u[0]
    ele = u[1]
    rud = u[2]
    thr = u[3]

    # use aircraft model
    aero_results = craft.aero_model.aero_results(*[
        a,b,pbar,qbar,rbar,ail,ele,rud,
        craft.is_compressible,M,craft.use_anderson,craft.has_stall
    ])
    # add in errors
    [CL, CS, CD, Cl, Cm, Cn] = [aero_results[i]*(1. + craft.FM_errors[i]) \
        for i in range(len(aero_results))]

    # thrust forces
    ## INTSTATE
    T = craft.aero_model.get_thrust(thr,-x[6],V)
    FP = T  * craft.T_dir
    MP = [
        FP[2] * craft.T_loc[1] - FP[1] * craft.T_loc[2],
        FP[0] * craft.T_loc[2] - FP[2] * craft.T_loc[0],
        FP[1] * craft.T_loc[0] - FP[0] * craft.T_loc[1]
    ]

    # aero forces
    ca = cos(a); sa = sin(a)
    cb = cos(b); sb = sin(b)
    dynF = 0.5 * rho * V*V * craft.Sw
    Fx = FP[0] + dynF * (  CL*sa - CS*ca*sb - CD*ca*cb)
    Fy = FP[1] + dynF * (  CS*cb - CD*sb)
    Fz = FP[2] + dynF * (- CL*ca - CS*sa*sb - CD*sa*cb)
    Mx = MP[0] + Cl * dynF * craft.bw
    My = MP[1] + Cm * dynF * craft.cw
    Mz = MP[2] + Cn * dynF * craft.bw

    # add in CG effects
    cg = craft.cgshift
    Mx -= Fz * cg[1] - Fy * cg[2]
    My -= Fx * cg[2] - Fz * cg[0]
    Mz -= Fy * cg[0] - Fx * cg[1]

    return Fx,Fy,Fz,Mx,My,Mz,g


def nonlinear_euler_dynamics(craft,t,x,
    is_controlled=True,given_control=False,u="o"):

    # get control
    u,inputs = get_control(craft,t,x,is_controlled,given_control,u)

    # disturbance model
    ## INTSTATE

    # get aero forces
    Fx,Fy,Fz,Mx,My,Mz,g = aerodynamics(craft,x,inputs)

    # read in mass properties
    W = craft.inertia_model.W
    Ixx,Iyy,Izz,Ixy,Ixz,Iyz = craft.inertia_model.inertia_results(u[3])
    Im1 = craft.inertia_model.inverse_tensor(u[3])
    hx,hy,hz = craft.inertia_model.angular_momentum_results()

    ## INTSTATE
    Vu = x[0]
    Vv = x[1]
    Vw = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    
    dx = x * 0.
    
    ## INTSTATE
    ph,th,ps = x[7],x[8],0.0 # craft._euler_angles(x) # 
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
    dx[6] = mat[2][0]*Vu + mat[2][1]*Vv + mat[2][2]*Vw

    
    # euler angles
    mat = [
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ]
    ## INTSTATE
    dx[ 7] = mat[0][0]*p + mat[0][1]*q + mat[0][2]*r
    dx[ 8] = mat[1][0]*p + mat[1][1]*q + mat[1][2]*r

    return dx



if __name__ == "__main__":

    # symbolic
    sym = sy.Symbol
    igr = sy.integrate
    simp = sy.simplify
    exp = sy.expand
    piecewise = sy.Piecewise
    diff = sy.diff
    sin = sy.sin
    cos = sy.cos
    tan = sy.tan
    mat = sy.Matrix
    pi = sy.pi
    frac = sy.Rational
    # declare variables
    x1 = sym("Vx")
    x2 = sym("Vy")
    x3 = sym("Vz")
    x4 = sym("p")
    x5 = sym("q")
    x6 = sym("r")
    x7 = sym("zf")
    x8 = sym("ph")
    x9 = sym("th")

    # filenames 
    base_file = "base_fs_in.json"
    bire_file = "bire_fs_in.json"

    # read in json to ensure no file changes while running
    base_dict = json.loads( open(base_file).read() )
    bire_dict = json.loads( open(bire_file).read() )


    # # test trim with cg forward
    # bire_dict["aircraft"]["CG_shift[ft]"] = [1.0,0.0,0.0]
    # bire_dict["initial"]["trim_guess"] = {}
    # bire_dict["initial"]["trim_guess"]["BIRE[deg]"] = 0.0
    bire_dict["initial"]["trim"]["bank_angle[deg]"] = 0.0
    # bire_dict["initial"]["trim"]["verbose_trim"] = True
    bire = Aircraft(bire_dict)
    # bire.run_trim()
    x0 = bire.x_trim_euler*1.
    u0 = bire.u_trim*1.
    bire._report_trim_solution(bire.x_trim,bire.u_trim)
    print()

    # build controller
    bire._build_controller(report=False,save_matrices=False,run_freq=False)

    # Lyapunov
    Q = np.eye(9)
    P = co.lyap(bire.Lin_Model.Acl.T,Q)
    P = bire.Lin_Model.P

    xrow = [0,1,2,3,4,5,8,9,10]
    x = x0[xrow]
    Vfun = lambda dx : np.matmul(np.matmul(dx.T,P),dx)
    f = nonlinear_euler_dynamics
    Vdotfun = lambda dx : np.matmul(np.matmul(f(bire,0.0,x+dx).T,P),dx) + \
        np.matmul(np.matmul(dx.T,P),f(bire,0.0,x+dx))
    
    # bire.use_quaternions = False
    # print(Vfun(x0[xrow]))
    # print(Vdotfun(x0[xrow]))

    # x = mat([x1,x2,x3,x4,x5,x6,x7,x8,x9])
    # print(x)
    # V = simp(x.T*P*x)
    # print(V)
    # Vd = 
    # f = nonlinear_euler_dynamics(bire,0.0,x)
    # print(f)

    state_threshold = np.array([
        10., 15., 15.,
        np.deg2rad(20.), np.deg2rad(10.), np.deg2rad(10.), # 
        50., np.deg2rad(25.), np.deg2rad(10.)
    ])
    num = 9
    xs = np.linspace(-state_threshold,state_threshold,num=num)
    c = 0.25
    Vc =    np.zeros(tuple([num]*3))
    Vdotc = np.zeros(tuple([num]*3))
    ic = 0
    for i in range(num):
        for j in range(num):
            for k in range(num):
                # for l in range(num):
                #     for m in range(num):
                #         for n in range(num):
                #             for o in range(num):
                #                 for p in range(num):
                #                     for q in range(num):
                #                         # create x var
                xf = x*0.
                # xf[0] += xs[i,0]
                # xf[1] += xs[j,1]
                # xf[2] += xs[k,2]
                xf[3] += xs[i,3] # l,3]
                xf[4] += xs[j,4] # m,4]
                xf[5] += xs[k,5] # n,5]
                # xf[6] += xs[o,6]
                # xf[7] += xs[p,7]
                # xf[8] += xs[q,8]
                
                # evaluate
                # Vc   [i][j][k][l][m][n][o][p][q] = \
                #     Vfun(xf)
                # Vdotc[i][j][k][l][m][n][o][p][q] = \
                #     Vdotfun(xf)
                Vc   [i][j][k] = Vfun(xf)
                Vdotc[i][j][k] = Vdotfun(xf)
                
                if ic != 0 and (ic+1) % 100 == 0:
                    print("completed cases: " + 
                        "{:> 7d}/{:> 7d}".format(
                        ic+1,num**3))
                
                ic += 1
    
    # breakpoint()


    # determine acceptable regions
    id = Vc <= c
    Vdot_max = np.max(Vdotc[id])
    print("Vdot_max =",Vdot_max)
    print("Vdot_min =",np.min(Vdotc))



