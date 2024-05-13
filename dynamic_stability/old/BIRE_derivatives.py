import numpy as np
import json
import sys


aero_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/aerodynamics_model/'
trim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/trim/'
sim_directory = 'C:/Users/troya/Desktop/Aerolab/git_repos/BIRE/flight_simulation/'

sys.path.insert(1, aero_directory)
sys.path.insert(1, trim_directory)
sys.path.insert(1, sim_directory)


from bire_aero import BIREAero
from aero_trim import trim, AircraftProperties

from trim_functions import solve_trim

class BIRE_derivs:
    def __init__(self):
        temp = 0
        