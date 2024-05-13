import json
import sys

aero_directory = '../aerodynamics_model/'
sys.path.insert(1, aero_directory)

mass_directory = '../mass_properties/'
sys.path.insert(1, mass_directory)

from inertia_model import InertiaModel
from atmospheric_functions import statee, gravity_english

class AircraftProperties:
    def __init__(self, V=200., H=15000., Gamma=0.1, bire=False, **kwargs):

        '''
        Parameters
        -----------
        V: float
            airspeed to trim at in ft/s
        H: float
            altitude in feet
        Gamma: float
            relaxtion factor for the fixed-point iteration method
        bire: boolean
            True for BIRE false for F-16
        '''

        fn = kwargs.get('filename', 'f16_props.json') # props the same for f16 vs BIRE
        prop_dict = json.load(open(aero_directory + fn))
        #parse values from JSON input
        self.BIRE = bire
        self.S_w = prop_dict["geometry"]["S_w"]
        self.b_w = prop_dict["geometry"]["b_w"]
        self.c_w = prop_dict["geometry"]["c_w"]
        self.l_h = prop_dict["geometry"]["l_h"]
        self.RA_w = prop_dict["geometry"]["RA_w"]
        self.Lam_w = prop_dict["geometry"]["Lam_w"]
        self.RA_v = prop_dict["geometry"]["RA_v"]
        self.Lam_v = prop_dict["geometry"]["Lam_v"]
        self.RA_h = prop_dict["geometry"]["RA_h"]
        self.Lam_h = prop_dict["geometry"]["Lam_h"]
        if bire:
            # create functions for BIRE inertias from given bire roation dB
            I_model = InertiaModel(inp_dir=mass_directory, is_bire=True)
            self.I_xx = lambda dB : I_model._Ixx(dB)
            self.I_yy = lambda dB : I_model._Iyy(dB)
            self.I_zz = lambda dB : I_model._Izz(dB)
            self.I_yz = lambda dB : I_model._Iyz(dB)
            self.I_xy = lambda dB : I_model._Ixy(dB)
            self.I_xz = lambda dB : I_model._Ixz(dB)
        else:
            I_model = InertiaModel(inp_dir=mass_directory, is_bire=False)
            self.Ixx = I_model._Ixx(0.0)
            # self.Ixy = 0.3*self.Ixx
            # self.Iyx = self.Ixy
            self.Ixy = I_model._Ixy(0.0)
            self.Iyx = I_model._Ixy(0.0)
            self.Ixz = I_model._Ixz(0.0)
            self.Izx = I_model._Ixz(0.0)
            self.Iyy = I_model._Iyy(0.0)
            self.Iyz = I_model._Iyz(0.0)
            self.Izy = I_model._Iyz(0.0)
            self.Izz = I_model._Izz(0.0)
        self.W = I_model.W
        self.hz = I_model.hz
        self.hy = I_model.hy
        self.hx = I_model.hx
        # some additional flight condition properties
        self.g = gravity_english(H)
        _, dummyz, dummyT, dummyp, self.rho, self.a = statee(H)
        _, dummyz, dummyT, dummyp, self.rho_0, self.a_0 = statee(0.0)
        self.nondim_const = 0.5*self.rho*V*V*self.S_w
        self.V = V
        self.H = H
        self.Gamma = Gamma
        self.M = self.V/self.a

    def calc_BIRE_inertia(self, dB):
        # calculate BIRE inertia from given BIRE angle
        self.Ixx = self.I_xx(dB)
        self.Ixy = self.I_xy(dB)
        self.Ixz = self.I_xz(dB)
        self.Iyy = self.I_yy(dB)
        self.Iyz = self.I_yz(dB)
        self.Izz = self.I_zz(dB)