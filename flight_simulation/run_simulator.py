from simulator_physics import simulator
import numpy as np

simulator_class = simulator(init_filename = 'simulator_input.json')
# simulator_class.dde = np.deg2rad(10.0)
# simulator_class.dda = np.deg2rad(10.0)
# simulator_class.ddr = np.deg2rad(1.0)
# simulator_class.ddq = np.deg2rad(5.0)
simulator_class.run_sim(plot_results=True)
simulator_class.normalize_states()