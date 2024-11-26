import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def dynamics(t,x,P):
    zt = P["zeta"]
    wn = P["omega_n"]

    dx = x*0.0
    dx[0] = x[1]
    dx[1] = -2.0*zt*wn*x[1] - wn**2.0*x[0]

    return dx

if __name__ == "__main__":
    # sim
    P = {
        "zeta" : 0.2,
        "omega_n" : 8.0,
    }
    x0 = np.array([
        1.0, 1.0
    ])
    ts = np.linspace(0.0,5.0,500)
    xs = odeint(dynamics,x0,ts,args=(P,),
        atol=1e-10,rtol=1e-10,
        tfirst=True).T
    
    # plotting
    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 10.0#*scale_font_size
    plt.rcParams["axes.labelsize"] = 10.0#*scale_font_size
    # plt.rcParams['axes.xmargin'] = 0 # uncomment to have xaxis fit data
    plt.rcParams['lines.linewidth'] = 0.75 # 1.0
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

    # initialize plots
    plot_dict = dict(figsize=(3.25,3.5),dpi=300.0,sharex=True,
        constrained_layout=True)
    fig,axs = plt.subplots(2,1,**plot_dict)

    axs[0].plot(ts,xs[0],"k")
    axs[1].plot(ts,xs[1],"k")

    # labels
    axs[1].set_xlabel("Time, s")
    axs[0].set_ylabel("State")
    axs[1].set_ylabel("Derivative")

    plt.show()