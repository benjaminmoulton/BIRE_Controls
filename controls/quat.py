from math import pi, sin, cos, asin, atan2
import numpy as np

def quat_mult(ea,eb):
    # pull out indexes, name them
    ea0,ea1,ea2,ea3 = ea
    eb0,eb1,eb2,eb3 = eb

    # calculate multiplied quaternion
    return [
        ea0*eb0 - ea1*eb1 - ea2*eb2 - ea3*eb3,
        ea0*eb1 + ea1*eb0 + ea2*eb3 - ea3*eb2,
        ea0*eb2 - ea1*eb3 + ea2*eb0 + ea3*eb1,
        ea0*eb3 + ea1*eb2 - ea2*eb1 + ea3*eb0
    ]


def euler_2_quat(a):
    # save 0.5 * angle
    a0h = a[0]*0.5
    a1h = a[1]*0.5
    a2h = a[2]*0.5

    # calculate cosines and sines
    Sf = sin(a0h); Cf = cos(a0h)
    St = sin(a1h); Ct = cos(a1h)
    Ss = sin(a2h); Cs = cos(a2h)

    # simplify calcs a hair
    CtCs = Ct*Cs
    StSs = St*Ss
    StCs = St*Cs
    CtSs = Ct*Ss

    # create quaternion
    return [
        Cf*CtCs + Sf*StSs,
        Sf*CtCs - Cf*StSs,
        Cf*StCs + Sf*CtSs,
        Cf*CtSs - Sf*StCs
    ]


def eulerdot_2_quatdot(e,euler,eulerdot):
    # get inverse of euler equation
    p,t,_ = euler
    cp = cos(p); sp = sin(p)
    ct = cos(t); st = sin(t)
    E = np.array([
        [1., sp/cp*st/ct + sp/ct*(st*cp + st*sp*sp/cp), -cp*(st*cp + st*sp*sp/cp)],
        [0., 1./cp - sp*sp/cp, sp*ct],
        [0., -sp, cp*ct]
    ])

    # 2d array of eulerdot
    eulerdot = np.array([
        [eulerdot[0]],
        [eulerdot[1]],
        [eulerdot[2]]
    ])

    # quat dot
    e0,ex,ey,ez = e
    edot = np.array([
        [-ex, -ey, -ez],
        [e0, -ez, ey],
        [ez, e0, -ex],
        [-ey, ex, e0]
    ])

    # calculate and return
    return (0.5 * np.matmul(np.matmul(edot,E),eulerdot)[:,0]).tolist()


def quatdot_2_eulerdot(e,euler,quatdot):
    # get euler equation
    p,t,_ = euler
    cp = cos(p); sp = sin(p)
    ct = cos(t); st = sin(t)
    E = np.array([
        [1., sp*st/ct, cp*st/ct],
        [0., cp, -sp],
        [0., sp/ct, cp/ct]
    ])

    # 2d array of eulerdot
    quatdot_T = np.array([
        [quatdot[0]],
        [quatdot[1]],
        [quatdot[2]],
        [quatdot[3]]
    ])

    # quat
    e0,ex,ey,ez = e
    edot = np.array([
        [-ex, e0, ez, -ey],
        [-ey, -ez, e0, ex],
        [-ez, ey, -ex, e0]
    ])

    # calculate and return
    return (0.5 * np.matmul(np.matmul(E,edot),quatdot_T)[:,0]).tolist()


def quat_norm(e):
    # pull out indices, name them
    e0,e1,e2,e3 = e

    # divide by magnitude
    m = (e0*e0 + e1*e1 + e2*e2 + e3*e3)**-0.5
    # m = 1.5 - 0.5 * (e0*e0 + e1*e1 + e2*e2 + e3*e3)
    return [e0*m,e1*m,e2*m,e3*m]


def quat_2_euler(e):
    # pull out indices, name them
    e0,e1,e2,e3 = e

    # determine measurement comparison
    compare = e0*e2 - e1*e3

    # if nose up
    if abs(compare) != 0.5: # not gimbal-locked
        e0sme2s = e0*e0 - e2*e2
        e3sme1s = e3*e3 - e1*e1
        return [
            atan2(2.*(e0*e1 + e2*e3),e0sme2s + e3sme1s),
            asin( 2.*compare),
            atan2(2.*(e0*e3 + e1*e2),e0sme2s - e3sme1s)
        ]
    else:
        if compare == 0.5:
            return [
                2. * asin(e1*1.4142135623730949E+00),
                1.5707963267948966E+00,
                0.0
            ]
        elif compare == -0.5: # if nose down
            return [
                2. * asin(e1*1.4142135623730949E+00),
                -1.5707963267948966E+00,
                0.0
            ]


def body_2_fixed(v0,e):
    # pull out indices, name them
    v00,v01,v02 = v0
    e0,e1,e2,e3 = e

    # calculate T quaternion
    T0 =  v00*e1 + v01*e2 + v02*e3
    Tx =  v00*e0 - v01*e3 + v02*e2
    Ty =  v00*e3 + v01*e0 - v02*e1
    Tz = -v00*e2 + v01*e1 + v02*e0

    # calculate final quaternion
    return [
        e0*Tx + e1*T0 + e2*Tz - e3*Ty,
        e0*Ty - e1*Tz + e2*T0 + e3*Tx,
        e0*Tz + e1*Ty - e2*Tx + e3*T0
    ]        


def fixed_2_body(v0,e):
    # pull out indices, name them
    v00,v01,v02 = v0
    e0,e1,e2,e3 = e

    # calculate T quaternion
    T0 = -v00*e1 - v01*e2 - v02*e3
    Tx =  v00*e0 + v01*e3 - v02*e2
    Ty = -v00*e3 + v01*e0 + v02*e1
    Tz =  v00*e2 - v01*e1 + v02*e0

    # calculate final quaternion
    return [
        e0*Tx - e1*T0 - e2*Tz + e3*Ty,
        e0*Ty + e1*Tz - e2*T0 - e3*Tx,
        e0*Tz - e1*Ty + e2*Tx - e3*T0
    ]


if __name__ == "__main__":
    import sympy as sy

    sym = sy.Symbol
    diff = sy.diff
    simp = sy.simplify

    u = sym("V_{x_b}")
    v = sym("V_{y_b}")
    w = sym("V_{z_b}")
    e0 = sym("e_0")
    ex = sym("e_x")
    ey = sym("e_y")
    ez = sym("e_z")

    quat = body_2_fixed([u,v,w],[e0,ex,ey,ez])
    print("e0 =", simp(quat[0]))
    print("e1 =", simp(quat[1]))
    print("e2 =", simp(quat[2]))
    print()
    print()
    dq0du = diff(quat[0],u)
    dq1du = diff(quat[1],u)
    dq2du = diff(quat[2],u)
    dq0dv = diff(quat[0],v)
    dq1dv = diff(quat[1],v)
    dq2dv = diff(quat[2],v)
    dq0dw = diff(quat[0],w)
    dq1dw = diff(quat[1],w)
    dq2dw = diff(quat[2],w)

    AxV = [
        [dq0du,dq0dv,dq0dw],
        [dq1du,dq1dv,dq1dw],
        [dq2du,dq2dv,dq2dw]
    ]

    for i in range(len(AxV)):
        print("{}{}".format("    ","    "),end="")
        for j in range(len(AxV[i])):
            print(str(AxV[i][j]).replace("**","^").replace("*"," "),end="")
            if j == len(AxV[i]) - 1:
                print(" \\\\")
            else:
                print(" & ",end="")


    dq0de0 = diff(quat[0],e0)
    dq1de0 = diff(quat[1],e0)
    dq2de0 = diff(quat[2],e0)
    dq0dex = diff(quat[0],ex)
    dq1dex = diff(quat[1],ex)
    dq2dex = diff(quat[2],ex)
    dq0dey = diff(quat[0],ey)
    dq1dey = diff(quat[1],ey)
    dq2dey = diff(quat[2],ey)
    dq0dez = diff(quat[0],ez)
    dq1dez = diff(quat[1],ez)
    dq2dez = diff(quat[2],ez)

    Axe = [
        [dq0de0,dq0dex,dq0dey,dq0dez],
        [dq1de0,dq1dex,dq1dey,dq1dez],
        [dq2de0,dq2dex,dq2dey,dq2dez]
    ]

    print()
    print()

    print("2*  VV")

    for i in range(len(Axe)):
        print("{}{}".format("    ","    "),end="")
        for j in range(len(Axe[i])):
            print(str(simp(Axe[i][j]/2)).replace("**","^").replace("*"," "),end="")
            if j == len(Axe[i]) - 1:
                print(" \\\\")
            else:
                print(" & ",end="")