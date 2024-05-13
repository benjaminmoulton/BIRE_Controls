import machupX as mx 
import numpy as np 
import matplotlib.pyplot as plt 
import json 
import itertools
import os
import time 


class MU_tables:
    """"This class allows for the creation of tables for any aircraft that can be read into machup"""
    def __init__(self, input_file, input_range_json):
        self.input_file = input_file
        self.input_range_json = input_range_json
        self.load_json()
        self.my_scene = mx.Scene(scene_input = self.input_file)


    def load_json(self):
        """This function pulls in all the input values from the json"""
        with open(self.input_range_json, 'r') as json_handle:
            input_vals = json.load(json_handle)
            self.alpha_vals = input_vals["alphas[deg]"]
            self.beta_vals = input_vals["betas[deg]"]
            self.elevator_vals = input_vals["elevators[deg]"]
            self.aileron_vals = input_vals["ailerons[deg]"]
            self.rudder_vals = input_vals["rudders[deg]"]
            self.p_vals = input_vals["p_vals[deg/s]"]
            self.q_vals = input_vals["q_vals[deg/s]"]
            self.r_vals = input_vals["r_vals[deg/s]"]
            self.velocity_vals = input_vals["Velocity[ft/s]"]


    def parse_forces(self, forces_dictionary):
        """This function parses out the forces dictionary that machupX outputs"""
        Cx = forces_dictionary["total"]["Cx"]
        Cy = forces_dictionary["total"]["Cy"]
        Cz = forces_dictionary["total"]["Cz"]
        Cl = forces_dictionary["total"]["Cl"]
        Cm = forces_dictionary["total"]["Cm"]
        Cn = forces_dictionary["total"]["Cn"]
        Force_array = np.array([Cx,Cy,Cz,Cl,Cm,Cn])
        return Force_array


    def create_scene(self, velocity, alpha, beta, p, q, r, elevator, rudder, aileron):
        """This will create a scene based on the inputs and return the forces and moments of that scene"""
        ## initialize_scene
        # my_scene = mx.Scene(scene_input = self.input_file)
        ## Set aircraft states
        state = {
        "velocity" : velocity,
        "alpha" : alpha,
        "beta" : beta,
        "angular_rates" : [p, q, r],
        }
        control_state = {
        "elevator" : elevator,
        "rudder" : rudder,
        "aileron" : aileron
        }
        #set aircraft states
        self.my_scene.set_aircraft_state(state = state)
        #set aircraft control states
        self.my_scene.set_aircraft_control_state(control_state = control_state)
        # Get the forces
        forces = self.my_scene.solve_forces(dimensional = False, wind_frame = False)
        # print(json.dumps(forces["F16"], indent = 4))
        # parse the forces and get the total values
        parsed_forces = self.parse_forces(forces["F16"])

        self.my_scene.display_wireframe(show_vortices = False, show_legend = True)
        
        return parsed_forces


    def loop_and_create_tables(self, alpha_vals, beta_vals, elevator_vals, aileron_vals, rudder_vals, p_vals, q_vals, r_vals, velocity_vals):
        """This function takes in the entire range of each independent variable, calls the create scene function at each point, and creates a table with all the values together"""
        
        # Combine all input variable arrays into one list for easier iteration
        all_vars = [alpha_vals, beta_vals, elevator_vals, aileron_vals, rudder_vals, p_vals, q_vals, r_vals, velocity_vals]
        
        # Sort input variable arrays from shortest to longest
        all_vars.sort(key=len)

        # Map variable names to input variables
        input_variable_names = {
            "alpha": alpha_vals,
            "beta": beta_vals,
            "elevator": elevator_vals,
            "aileron": aileron_vals,
            "rudder": rudder_vals,
            "p": p_vals,
            "q": q_vals,
            "r": r_vals,
            "velocity": velocity_vals
        }

        # Extract input variable names in the same order as sorted all_vars
        sorted_variable_names = []
        for var in all_vars:
            for name, value in input_variable_names.items():
                if value is var:
                    sorted_variable_names.append(f"{name}")
                    break

        # Calculate total number of combinations
        total_combinations = 1
        for var in all_vars:
            total_combinations *= len(var)

        # Initialize new_table with appropriate dimensions
        new_table = np.empty((total_combinations + 2, len(all_vars) + 6), dtype=object)
        new_table[0][0] = "numberIndependentVariables"
        new_table[0][1] = "="
        new_table[0][2] = len(all_vars)
        new_table[1] = sorted_variable_names + ["Cx", "Cy", "Cz", "Cl", "Cm", "Cn"]
        
        # Create combinations of input variables and populate new_table
        index = 2
        for combo in itertools.product(*all_vars):
            new_table[index, :len(combo)] = combo
            index += 1
        
        # Create CSV file with independent variables
        create_csv(new_table, "test_file_independent_variables")
        
        # Initialize index for populating new_table with force values
        force_index = len(all_vars) + 2
        
        for i in range(2, new_table.shape[0]):
            # input_vars = new_table[i, :len(all_vars)]  # Extract input variables for the current combination
            input_vars = new_table[i,:len(all_vars)]  # Extract input variables for the current combination
            Cx, Cy, Cz, Cl, Cm, Cn = self.create_scene(*input_vars)  # Assuming velocity_vals is single-valued
            force_list = [Cx, Cy, Cz, Cl, Cm, Cn]
            
            # Find the index where independent variables end and forces begin
            independent_vars_end = len(all_vars) 
            
            # Loop through the indices where forces begin
            for j in range(independent_vars_end, min(len(new_table[i]), len(force_list)) + independent_vars_end):
                new_table[i, j] = force_list[j - independent_vars_end]
        
        ## Write to CSV file
        output_file_name = "C(x,y,z,l,m,n)(alpha,beta,de,da,dr,p,q,r,v)(full_table)"
        create_csv(new_table, output_file_name)
        print("Data written to '{}.csv' file.".format(output_file_name))
    

def create_csv(array, output_file_name):
        
        if os.path.isfile(output_file_name):
            try:
                with open(output_file_name, 'wt') as f:
                    pass
            except PermissionError:
                print("Error: File '{}' is likely open on your computer. Try closing the file and re-running the code.".format(output_file_name))
        np.savetxt(output_file_name + ".csv", array, delimiter=',', fmt='%s') # the fmt thing is just saying that it's printing strings out to the csv file. That way it doesn't get mad that there are floats and strings in the csv


if __name__=="__main__":

    vals = MU_tables("F16_RC_input.json", "F16_RC_inputranges.json")

    # vals.create_scene(vals.velocity_vals[0],vals.alpha_vals[0],vals.beta_vals[0],vals.p_vals[0],vals.q_vals[0],vals.r_vals[0],vals.elevator_vals[0],vals.rudder_vals[0],vals.aileron_vals[0])

    time_1 = time.perf_counter()
    
    # vals.loop_and_create_tables(vals.alpha_vals, vals.beta_vals, vals.elevator_vals, vals.aileron_vals, vals.rudder_vals, vals.p_vals, vals.q_vals, vals.r_vals, vals.velocity_vals)
    vals.create_scene(vals.velocity_vals[0], vals.alpha_vals[0], vals.beta_vals[0], vals.p_vals[0], vals.q_vals[0], vals.r_vals[0], vals.elevator_vals[0], vals.rudder_vals[0], vals.aileron_vals[0])

    time_2 = time.perf_counter()
    print(f"Plot_generate_time:{time_2-time_1:0.4f} seconds")




    