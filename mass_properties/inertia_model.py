#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:18:40 2021

@author: christian
"""

from numpy import array, matmul, pi, deg2rad, zeros
from math import sin, cos
import json

class InertiaModel:
    def __init__(self, inp_dir='./', **kwargs):
        is_bire = kwargs.get("is_bire",False)
        is_rc = kwargs.get("is_rc", False)
        is_SAL = kwargs.get("is_SAL", False)
        if is_SAL:
            fn_def = "f16_SAL_inertial_properties.json"
        else:
            if is_rc:
                if is_bire:
                    fn_def = "bire_rc_inertial_properties.json"
                else:
                    fn_def = "f16_rc_inertial_properties.json"
            else:
                if is_bire:
                    fn_def = "bire_inertial_properties.json"
                else:
                    fn_def = "f16_inertial_properties.json"
        fn = kwargs.get('fn', fn_def)
        self.model_coeffs_dict = json.load(open(inp_dir + fn))

        self.W = self.model_coeffs_dict["weight"]

        self.inertia_coeffs = self.model_coeffs_dict["inertia"]
        self.Ixx_coeffs = self.inertia_coeffs["Ixx"]
        self.Iyy_coeffs = self.inertia_coeffs["Iyy"]
        self.Izz_coeffs = self.inertia_coeffs["Izz"]
        self.Ixy_coeffs = self.inertia_coeffs["Ixy"]
        self.Ixz_coeffs = self.inertia_coeffs["Ixz"]
        self.Iyz_coeffs = self.inertia_coeffs["Iyz"]
        self.Ixx_A = self.Ixx_coeffs["A"]
        self.Ixx_w = self.Ixx_coeffs["w"]
        self.Ixx_p = self.Ixx_coeffs["phi"]
        self.Ixx_z = self.Ixx_coeffs["z"]
        self.Iyy_A = self.Iyy_coeffs["A"]
        self.Iyy_w = self.Iyy_coeffs["w"]
        self.Iyy_p = self.Iyy_coeffs["phi"]
        self.Iyy_z = self.Iyy_coeffs["z"]
        self.Izz_A = self.Izz_coeffs["A"]
        self.Izz_w = self.Izz_coeffs["w"]
        self.Izz_p = self.Izz_coeffs["phi"]
        self.Izz_z = self.Izz_coeffs["z"]
        self.Ixy_A = self.Ixy_coeffs["A"]
        self.Ixy_w = self.Ixy_coeffs["w"]
        self.Ixy_p = self.Ixy_coeffs["phi"]
        self.Ixy_z = self.Ixy_coeffs["z"]
        self.Ixz_A = self.Ixz_coeffs["A"]
        self.Ixz_w = self.Ixz_coeffs["w"]
        self.Ixz_p = self.Ixz_coeffs["phi"]
        self.Ixz_z = self.Ixz_coeffs["z"]
        self.Iyz_A = self.Iyz_coeffs["A"]
        self.Iyz_w = self.Iyz_coeffs["w"]
        self.Iyz_p = self.Iyz_coeffs["phi"]
        self.Iyz_z = self.Iyz_coeffs["z"]

        self.h_coeffs = self.model_coeffs_dict["angular_momentum"]
        self.hx = self.h_coeffs["hx"]
        self.hy = self.h_coeffs["hy"]
        self.hz = self.h_coeffs["hz"]

        if is_rc:
            self._Iyz = lambda dB : self.Iyz_A*sin(self.Iyz_w*dB + self.Iyz_p)\
                + self.Iyz_z
            self._dIyz = lambda dB : self.Iyz_A*self.Iyz_w*cos(\
                self.Iyz_w*dB + self.Iyz_p)

    def _Ixx(self, dB):
        Ixx = self.Ixx_A*sin(self.Ixx_w*dB + self.Ixx_p) + self.Ixx_z
        return Ixx

    def _dIxx(self, dB):
        dIxx = self.Ixx_A*self.Ixx_w*cos(self.Ixx_w*dB + self.Ixx_p)
        return dIxx

    def _Iyy(self, dB):
        Iyy = self.Iyy_A*sin(self.Iyy_w*dB + self.Iyy_p) + self.Iyy_z
        return Iyy

    def _dIyy(self, dB):
        dIyy = self.Iyy_A*self.Iyy_w*cos(self.Iyy_w*dB + self.Iyy_p)
        return dIyy

    def _Izz(self, dB):
        Izz = self.Izz_A*sin(self.Izz_w*dB + self.Izz_p) + self.Izz_z
        return Izz

    def _dIzz(self, dB):
        dIzz = self.Izz_A*self.Izz_w*cos(self.Izz_w*dB + self.Izz_p)
        return dIzz

    def _Ixy(self, dB):
        Ixy = self.Ixy_A*sin(self.Ixy_w*dB + self.Ixy_p) + self.Ixy_z
        return Ixy

    def _dIxy(self, dB):
        dIxy = self.Ixy_A*self.Ixy_w*cos(self.Ixy_w*dB + self.Ixy_p)
        return dIxy

    def _Ixz(self, dB):
        Ixz = self.Ixz_A*sin(self.Ixz_w*dB + self.Ixz_p) + self.Ixz_z
        return Ixz

    def _dIxz(self, dB):
        dIxz = self.Ixz_A*self.Ixz_w*cos(self.Ixz_w*dB + self.Ixz_p)
        return dIxz

    def _Iyz(self, dB):
        Iyz = self.Iyz_A*abs(sin(self.Iyz_w*dB + self.Iyz_p)) + self.Iyz_z
        return Iyz

    def _dIyz(self, dB):
        if dB // pi != 0.0:
            O = self.Iyz_w*dB + self.Iyz_p
            dIyz = self.Iyz_A*self.Iyz_w*sin(O)*cos(O)/abs(sin(O))
        else:
            dIyz = 0.0
        return dIyz
    
    def _determinant(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        det = ( Ixx*(Iyy*Izz - Iyz**2.) - Ixy*Ixz*Iyz \
            - (Ixy**2.*Izz + Ixz**2.*Iyy) )
        return det
    
    def _determinant_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        ddet = ( dIxx*(Iyy*Izz - Iyz**2.) + Ixx*(dIyy*Izz + Iyy*dIzz \
            - 2.*Iyz*dIyz) - dIxy*Ixz*Iyz - Ixy*dIxz*Iyz - Ixy*Ixz*dIyz
            -(2.*Ixy*dIxy*Izz + Ixy**2.*dIzz + 2.*Ixz*dIxz*Iyy + Ixz**2.*dIyy))
        return ddet
    
    def _adjoint(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [ Iyy*Izz - Iyz**2., Ixy*Izz + Ixz*Iyz, Ixy*Iyz + Ixz*Iyy],
            [ Ixy*Izz + Iyz*Ixz, Ixx*Izz - Ixz**2., Ixx*Iyz + Ixy*Ixz],
            [ Ixy*Iyz + Ixz*Iyy, Ixx*Iyz + Ixz*Ixy, Ixx*Iyy - Ixy**2.]
        ]
    
    def _adjoint_derivative(self, dB):
        # get values
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        # get derivatives
        dIxx = self._dIxx(dB)
        dIyy = self._dIyy(dB)
        dIzz = self._dIzz(dB)
        dIxy = self._dIxy(dB)
        dIxz = self._dIxz(dB)
        dIyz = self._dIyz(dB)
        # initialize and assign
        dadj = zeros((3,3))
        dadj[0,0] = dIyy*Izz + Iyy*dIzz - 2.*Iyz*dIyz
        dadj[0,1] = dadj[1,0] = dIxy*Izz + Ixy*dIzz + dIxz*Iyz + Ixz*dIyz
        dadj[1,1] = dIxx*Izz + Ixx*dIzz - 2.*Ixz*dIxz
        dadj[0,2] = dadj[2,0] = dIxy*Iyz + Ixy*dIyz + dIxz*Iyy + Ixz*dIyy
        dadj[2,2] = dIxx*Iyy + Ixx*dIyy - 2.*Ixy*dIxy
        dadj[1,2] = dadj[2,1] = dIxx*Iyz + Ixx*dIyz + dIxy*Ixz + Ixy*dIxz
        return dadj

    def inertia_results(self, dB):
        return [self._Ixx(dB), self._Iyy(dB), self._Izz(dB),
        self._Ixy(dB), self._Ixz(dB), self._Iyz(dB)]

    def inertia_derivative_results(self, dB):
        return [self._dIxx(dB), self._dIyy(dB), self._dIzz(dB),
        self._dIxy(dB), self._dIxz(dB), self._dIyz(dB)]

    def angular_momentum_results(self):
        return [self.hx, self.hy, self.hz]

    def inertia_tensor(self, dB):
        Ixx = self._Ixx(dB)
        Iyy = self._Iyy(dB)
        Izz = self._Izz(dB)
        Ixy = self._Ixy(dB)
        Ixz = self._Ixz(dB)
        Iyz = self._Iyz(dB)
        return [
            [  Ixx, -Ixy, -Ixz],
            [ -Ixy,  Iyy, -Iyz],
            [ -Ixz, -Iyz,  Izz]
        ]

    def inverse_tensor(self, dB):
        # return ( array(self._adjoint(dB))/self._determinant(dB) ).tolist()
        adj = self._adjoint(dB)
        det = self._determinant(dB)
        return [
            [adj[0][0]/det, adj[0][1]/det, adj[0][2]/det],
            [adj[1][0]/det, adj[1][1]/det, adj[1][2]/det],
            [adj[2][0]/det, adj[2][1]/det, adj[2][2]/det]
        ]

    def inverse_tensor_derivative(self, dB):
        # get pieces
        Iinv = array(self.inverse_tensor(dB))
        det = self._determinant(dB)
        dadj = array(self._adjoint_derivative(dB))
        ddet = self._determinant_derivative(dB)

        return ( (dadj - Iinv*ddet)/det ).tolist()

if __name__ == "__main__":
    case = InertiaModel(fn="test_inertial_properties.json")
    params = deg2rad(10.)
    [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] = case.inertia_results(params)
    print(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    a = array(case.inertia_tensor(params))
    print(a)
    print(array(case._determinant(params)))
    print(array(case._adjoint(params)))
    b = array(case.inverse_tensor(params))
    print(b)
    print(matmul(a,b))
