import numpy as np 

def three_layer_interp(alphas, alpha, betas, beta, dhs, dh, first, second, third, fourth):
    """This function runs one recursive interpolation"""
    up_right = np.interp(alpha, alphas, first)
    print("first alpha interp:", up_right)
    second_up_right = np.interp(alpha, alphas, second)
    print("second alpha interp:", second_up_right)
    bottom_right = np.interp(alpha, alphas, third)
    print("third alpha interp:", bottom_right)
    second_bottom_right = np.interp(alpha, alphas, fourth)
    print("fourth alpha interp:", second_bottom_right)

    vals_up_middle = [up_right, second_up_right]
    up_middle = np.interp(beta, betas, vals_up_middle)
    print("first beta interp:", up_middle)

    vals_down_middle = [bottom_right, second_bottom_right]
    down_middle = np.interp(beta, betas, vals_down_middle)
    print("second beta interp:", down_middle)

    final_interp_vals = [up_middle, down_middle]
    Interpolated = np.interp(dh, dhs, final_interp_vals)
    return Interpolated


if __name__ == "__main__":
    """This function takes in parameter bounds """
    alphas = [0, 5]
    alpha = 2.3

    betas = [4, 6]
    beta = 4.2

    dhs = [10, 25]
    dh = 22.0

    first = [-0.0592, -0.0195]
    second = [-0.0584, -0.0184]
    third = [-0.10610, -0.07440]
    fourth = [-0.10450, -0.07040]

    interpolated = three_layer_interp(alphas, alpha, betas, beta, dhs, dh, first, second, third, fourth)
    print("\n Total interpolation:", interpolated, "\n")






