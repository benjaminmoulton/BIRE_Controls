import sympy as sy
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
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
    print("declaring variables...")
    x1 = sym("x1")
    x2 = sym("x2")
    x3 = sym("x3")
    y1 = sym("y1")
    y2 = sym("y2")
    y3 = sym("y3")
    z1 = sym("z1")
    z2 = sym("z2")
    z3 = sym("z3")

    # big letters
    A = x2**2*y1**2 - x1**2*y2**2
    B = x1**2*y3**2 - x3**2*y1**2
    C = x3**2*z1**2 - x1**2*z3**2
    D = x1**2*z2**2 - x2**2*z1**2
    E = y1**2*z3**2 - y3**2*z1**2
    F = y2**2*z1**2 - y1**2*z2**2
    G = y3**2*z2**2 - y2**2*z3**2
    H = x2**2*z3**2 - x3**2*z2**2
    I = x3**2*y2**2 - x2**2*y3**2
    ACmBD = -A*C + B*D
    x2mx1 = x2**2 - x1**2
    x3mx1 = x3**2 - x1**2

    # unsimp
    c0 = ( (ACmBD)/(-B*x2mx1 - A*x3mx1) )**frac(1,2)
    b0 = ( (A)/(x2**2*(1 - z1**2/c0**2) - x1**2*(1 - z2**2/c0**2)) )**frac(1,2)
    a0 = ( (x1**2)/(1 - y1**2/b0**2 - z1**2/c0**2) )**frac(1,2)

    # simp
    c1 = c0
    b1 = ( (ACmBD)/(-C*x2mx1 - D*x3mx1) )**frac(1,2)
    a1 = ( (x1**2*ACmBD)/(ACmBD - x1**2*(E*x2mx1 + F*x3mx1) ) )**frac(1,2)

    # check that unsimp are correct
    ell = lambda x,y,z,a,b,c : (x/a)**2 + (y/b)**2 + (z/c)**2
    # print(simp(ell(x1,y1,z1,a0,b0,c0)),"== 1",simp(ell(x1,y1,z1,a0,b0,c0))==1)
    # print(simp(ell(x2,y2,z2,a0,b0,c0)),"== 1",simp(ell(x2,y2,z2,a0,b0,c0))==1)
    # print(simp(ell(x3,y3,z3,a0,b0,c0)),"== 1",simp(ell(x3,y3,z3,a0,b0,c0))==1)
    # print(simp(ell(x1,y1,z1,a1,b1,c1)),"== 1",simp(ell(x1,y1,z1,a1,b1,c1))==1)
    # print(simp(ell(x2,y2,z2,a1,b1,c1)),"== 1",simp(ell(x2,y2,z2,a1,b1,c1))==1)
    # print(simp(ell(x3,y3,z3,a1,b1,c1)),"== 1",simp(ell(x3,y3,z3,a1,b1,c1))==1)

    # check that simps are same as unsimp
    # print("a0 == a1",simp(a0**2-a1**2)==0)
    # print("b0 == b1",simp(b0**2-b1**2)==0)
    # print("c0 == c1",simp(c0**2-c1**2)==0)

    # report results
    print("a =",simp(exp(a1)))
    print("b =",simp(exp(b1)))
    print("c =",simp(exp(c1)))

    # # old
    # a = ( (x1**2*G + x2**2*E + x3**2*F )/(E + F + G) )**frac(1,2)
    # b = ( (x1**2*G + x2**2*E + x3**2*F )/(C + D + H) )**frac(1,2)
    # c = ( (x1**2*G + x2**2*E + x3**2*F )/(A + B + I) )**frac(1,2)
    # new
    a = ( (x1**2*G + x2**2*E + x3**2*F )/(E + F + G) )**frac(1,2)
    b = ( (y1**2*H + y2**2*C + y3**2*D )/(C + D + H) )**frac(1,2)
    c = ( (z1**2*I + z2**2*B + z3**2*A )/(A + B + I) )**frac(1,2)

    # check rearranged
    print("a0 == a",simp(a0**2-a**2)==0)
    print("b0 == b",simp(b0**2-b**2)==0)
    print("c0 == c",simp(c0**2-c**2)==0)
    print()


    # new new
    AA = y3**2*z2**2 - y2**2*z3**2
    BB = y1**2*z3**2 - y3**2*z1**2
    CC = y2**2*z1**2 - y1**2*z2**2
    DD = x2**2*z3**2 - x3**2*z2**2
    EE = x3**2*z1**2 - x1**2*z3**2
    FF = x1**2*z2**2 - x2**2*z1**2
    GG = x3**2*y2**2 - x2**2*y3**2
    HH = x1**2*y3**2 - x3**2*y1**2
    II = x2**2*y1**2 - x1**2*y2**2

    a_ = ( (x1**2*AA + x2**2*BB + x3**2*CC )/(AA + BB + CC) )**frac(1,2)
    b_ = ( (y1**2*DD + y2**2*EE + y3**2*FF )/(DD + EE + FF) )**frac(1,2)
    c_ = ( (z1**2*GG + z2**2*HH + z3**2*II )/(GG + HH + II) )**frac(1,2)

    print( simp(a0) )
    print( simp(b0) )
    print( simp(c0) )
    print()
    print( simp(a_) )
    print( simp(b_) )
    print( simp(c_) )
    print()

    print("a0 == a_",simp(a0**2-a_**2)==0)
    print("b0 == b_",simp(b0**2-b_**2)==0)
    print("c0 == c_",simp(c0**2-c_**2)==0)


