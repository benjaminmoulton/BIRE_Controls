import numpy as np

from scipy.integrate import quad
import json
from matplotlib import pyplot as plt
from turbulence import VonKarmanTurbulence

if __name__ == "__main__":


    print("building model...")
    mdl = VonKarmanTurbulence(
        {"number_frequency_bins":100,"turbulence_intensity":"light",
        # "initial_altitude[ft]":200.0,
        "initial_altitude[ft]":15000.0,
        "random_seed" : 1,
        },
        # wingspan=3.03,V=100.,
        wingspan=30.,V=634.,
        # show_plot=True,
        )


    # quit()
    print("interpreting signal...")
    num = 10001
    t = np.linspace(0.,10.,num=num)
    V = 634.
    
    # get signal
    xs = V*t
    Vgus = xs*0.0
    Vgvs = xs*0.0
    Vgws = xs*0.0
    Wgps = xs*0.0
    Wgqs = xs*0.0
    Wgrs = xs*0.0
    for j in range(len(t)):
        Vgus[j] = mdl.Vgu(t[j])
        Vgvs[j] = mdl.Vgv(t[j])
        Vgws[j] = mdl.Vgw(t[j])
        Wgps[j] = mdl.Wgp(t[j])
        Wgqs[j] = mdl.Wgq(t[j])
        Wgrs[j] = mdl.Wgr(t[j])
        
    # integrating signal
    IVgus = xs*0.0
    IVgvs = xs*0.0
    IVgws = xs*0.0
    IWgps = xs*0.0
    IWgqs = xs*0.0
    IWgrs = xs*0.0
    for j in range(len(t)):
        IVgus[j] = np.trapz(Vgus[:j+1],xs[:j+1])
        IVgvs[j] = np.trapz(Vgvs[:j+1],xs[:j+1])
        IVgws[j] = np.trapz(Vgws[:j+1],xs[:j+1])
        IWgps[j] = np.trapz(Wgps[:j+1],xs[:j+1])
        IWgqs[j] = np.trapz(Wgqs[:j+1],xs[:j+1])
        IWgrs[j] = np.trapz(Wgrs[:j+1],xs[:j+1])
        
    # integrating signal gain
    IIVgus = xs*0.0
    IIVgvs = xs*0.0
    IIVgws = xs*0.0
    IIWgps = xs*0.0
    IIWgqs = xs*0.0
    IIWgrs = xs*0.0
    for j in range(len(t)):
        IIVgus[j] = np.trapz(IVgus[:j+1],xs[:j+1])
        IIVgvs[j] = np.trapz(IVgvs[:j+1],xs[:j+1])
        IIVgws[j] = np.trapz(IVgws[:j+1],xs[:j+1])
        IIWgps[j] = np.trapz(IWgps[:j+1],xs[:j+1])
        IIWgqs[j] = np.trapz(IWgqs[:j+1],xs[:j+1])
        IIWgrs[j] = np.trapz(IWgrs[:j+1],xs[:j+1])
    
    # plot
    # change plot text parameters
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["axes.labelsize"] = 8.0
    plt.rcParams['lines.linewidth'] = 1.0
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

    lwi = 0.75

    # plot
    fig,axs = plt.subplots(3,2,figsize=(3.25*2.0,3.5),sharex=True,constrained_layout=True)
    #
    axs[0][0].plot(t,Vgus,"k",ls= "-",lw=lwi,label=r"$V_{x_b \, D}$, ft/s")
    axs[0][0].plot(t,Vgvs,"k",ls="--",lw=lwi,label=r"$V_{y_b \, D}$, ft/s")
    axs[0][0].plot(t,Vgws,"k",ls="-.",lw=lwi,label=r"$V_{z_b \, D}$, ft/s")
    axs[0][0].set_ylabel(r"velocities, ft/s")
    axs[0][0].legend()
    #
    axs[1][0].plot(t,IVgus,"k",ls= "-",lw=lwi,label=r"$\int V_{x_b \, D} \, dt$, ft")
    axs[1][0].plot(t,IVgvs,"k",ls="--",lw=lwi,label=r"$\int V_{y_b \, D} \, dt$, ft")
    axs[1][0].plot(t,IVgws,"k",ls="-.",lw=lwi,label=r"$\int V_{z_b \, D} \, dt$, ft")
    axs[1][0].set_ylabel(r"integrated, ft")
    axs[1][0].legend()
    #
    axs[2][0].plot(t,IIVgus,"k",ls= "-",lw=lwi,label=r"$\int \int V_{x_b \, D} \, dt \, dt$, ft-s")
    axs[2][0].plot(t,IIVgvs,"k",ls="--",lw=lwi,label=r"$\int \int V_{y_b \, D} \, dt \, dt$, ft-s")
    axs[2][0].plot(t,IIVgws,"k",ls="-.",lw=lwi,label=r"$\int \int V_{z_b \, D} \, dt \, dt$, ft-s")
    axs[2][0].set_ylabel(r"double integrated, ft-s")
    axs[2][0].legend()
    axs[2][0].set_xlabel("Time $t$, sec")
    #
    axs[0][1].plot(t,np.rad2deg(Wgps),"k",ls= "-",lw=lwi,label=r"$p_D$, deg/s")
    axs[0][1].plot(t,np.rad2deg(Wgqs),"k",ls="--",lw=lwi,label=r"$q_D$, deg/s")
    axs[0][1].plot(t,np.rad2deg(Wgrs),"k",ls="-.",lw=lwi,label=r"$r_D$, deg/s")
    axs[0][1].set_ylabel(r"rates, deg/s")
    axs[0][1].legend()
    #
    axs[1][1].plot(t,np.rad2deg(IWgps),"k",ls= "-",lw=lwi,label=r"$\int p_D \, dt$, deg")
    axs[1][1].plot(t,np.rad2deg(IWgqs),"k",ls="--",lw=lwi,label=r"$\int q_D \, dt$, deg")
    axs[1][1].plot(t,np.rad2deg(IWgrs),"k",ls="-.",lw=lwi,label=r"$\int r_D \, dt$, deg")
    axs[1][1].set_ylabel(r"integrated, deg")
    axs[1][1].legend()
    #
    axs[2][1].plot(t,np.rad2deg(IIWgps),"k",ls= "-",lw=lwi,label=r"$\int \int p_D \, dt \, dt$, deg-s")
    axs[2][1].plot(t,np.rad2deg(IIWgqs),"k",ls="--",lw=lwi,label=r"$\int \int q_D \, dt \, dt$, deg-s")
    axs[2][1].plot(t,np.rad2deg(IIWgrs),"k",ls="-.",lw=lwi,label=r"$\int \int r_D \, dt \, dt$, deg-s")
    axs[2][1].set_ylabel(r"double integrated, deg-s")
    axs[2][1].legend()
    axs[2][1].set_xlabel("Time $t$, sec")

    plt.show()