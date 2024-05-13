from numpy import sin, pi, exp, linspace, inf, array, zeros, interp, flip
from numpy import arctan2, real, imag, rad2deg, logspace, log, log10, arcsin
from numpy import sum as npsum
from numpy import outer as npouter
from numpy import add as npadd
from numpy import max as npmax
from numpy import min as npmin
from numpy import abs as npabs
from numpy.random import default_rng
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d,interpn,RegularGridInterpolator
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import periodogram, welch
import json
from matplotlib import pyplot as plt

class ZeroTurbulence:
    """A base object for determining turbulence at a given condition.
    """

    def __init__(self,input_file={},wingspan=30.,V=634.,dt=0.01, tf=0.0):

        # # report
        # if isinstance(input_file,(str)):
        #     print("\ninitializing " + input_file + "...")
        # else:
        #     print("\ninitializing zero turbulence model...")

        # get input variables
        self._get_input_vars(input_file)


    def _get_input_vars(self,input_vars):
        # get info or raise error
        # determine if the input_vars is a file or a dictionary
        input_vars_type = type(input_vars)

        # dictionary
        if input_vars_type == dict:
            input_dict = input_vars
        
        # json file
        elif input_vars_type == str and input_vars.split(".")[-1] == "json":
            # import json file from file path
            json_string = open(input_vars).read()

            # save to vals dictionary
            input_dict = json.loads(json_string)

        # raise error
        else:
            raise IOError("input_vars must be json file path, or " + \
                "dictionary, not {0}".format(input_vars_type))
        
        # store simulation variables globally
        self.in_dict = input_dict
    

    def get_disturbance(self,t,V):
        return 0.,0.,0.,0.,0.,0.


    def get_precomputed_disturbance(self,t,V):
        return self.get_disturbance(t,V)


class DampedSinusoidGust(ZeroTurbulence):

    def __init__(self,input_file={},wingspan=30.,V=634.,dt=0.01, tf=0.0):
        # invoke init of parent
        ZeroTurbulence.__init__(self,input_file)

        # retrieve additional info
        self._get_input_vars(input_file)


    def _get_input_vars(self,input_vars):        
        # store simulation variables globally
        Aw_max = input_vars.get("amplitude[ft/s]")
        self.Aw = Aw_max * array(input_vars.get("directions(body-fixed)"))
        self.zetaw = input_vars.get("damping_rate[1/s]")
        self.ww = input_vars.get("frequency[rad/s]")
        self.t_gust = input_vars.get("init_time[s]")
    

    def get_disturbance(self,t,V):
        # gust dynamics
        if t >= self.t_gust:
            tau = t - self.t_gust
            Vg = self.Aw*exp(-self.zetaw*tau)*sin(self.ww*tau)
        else:
            Vg = [0.,0.,0.]
        return Vg[0],Vg[1],Vg[2],0.,0.,0.


class VonKarmanTurbulence(ZeroTurbulence):

    def __init__(self,input_file={},wingspan=30.,V=634.,dt=0.01,tf=15.,
        show_plot=False):
        # invoke init of parent
        ZeroTurbulence.__init__(self,input_file)

        # retrieve additional info
        self._get_input_vars(input_file)

        # build model
        # self.intensity = "4"
        # for i in range(41):
        #     self.H0 = float(i)*2000.
        #     self._build_model(wingspan)
        self._build_model(wingspan,show_plot=show_plot)

        # precompute disturbance
        t = linspace(0.,tf,int(tf/dt)+1)
        xs = V*t
        Vgu = array([self.Au*npsum(sin(self.u_freq*x + self.u_phas)) for x in xs])
        Vgv = array([self.Av*npsum(sin(self.v_freq*x + self.v_phas)) for x in xs])
        Vgw = array([self.Aw*npsum(sin(self.w_freq*x + self.w_phas)) for x in xs])
        Wgp = array([self.Ap*npsum(sin(self.p_freq*x + self.p_phas)) for x in xs])
        Wgq = array([self.Aq*npsum(sin(self.q_freq*x + self.q_phas)) for x in xs])
        Wgr = array([self.Ar*npsum(sin(self.r_freq*x + self.r_phas)) for x in xs])
        if show_plot:
            fsg,asg = plt.subplots(2,1,layout="constrained",sharex=True)
            asg[0].plot(t,Vgu,label="$u_d$")
            asg[0].plot(t,Vgv,label="$v_d$")
            asg[0].plot(t,Vgw,label="$w_d$")
            asg[1].plot(t,rad2deg(Wgp),label="$p_d$")
            asg[1].plot(t,rad2deg(Wgq),label="$q_d$")
            asg[1].plot(t,rad2deg(Wgr),label="$r_d$")
            asg[1].set_xlabel("Time [s]")
            asg[0].set_ylabel("Disturbance velocity [ft/s]")
            asg[1].set_ylabel("Disturbance rate [deg/s]")
            asg[0].legend()
            asg[1].legend()
            asg[0].set_xlim((0.,t[-1]))
            fsg.savefig("PSD_plots/disturbance.png",dpi=300.)#,transparent=True)
            plt.close()
            # report 'max' values
            Vgum =         max(abs(npmax(Vgu)),abs(npmin(Vgu)))
            Vgvm =         max(abs(npmax(Vgv)),abs(npmin(Vgv)))
            Vgwm =         max(abs(npmax(Vgw)),abs(npmin(Vgw)))
            Wgpm = rad2deg(max(abs(npmax(Wgp)),abs(npmin(Wgp))))
            Wgqm = rad2deg(max(abs(npmax(Wgq)),abs(npmin(Wgq))))
            Wgrm = rad2deg(max(abs(npmax(Wgr)),abs(npmin(Wgr))))
            # calculate max deltas
            DV = (Vgum**2. + Vgvm**2. + Vgwm**2.)**0.5
            Dau = rad2deg(arctan2( Vgwm,V-Vgum))
            Dal = rad2deg(arctan2(-Vgwm,V-Vgum))
            Dbu = rad2deg(arcsin( Vgvm/(V-DV)))
            Dbl = rad2deg(arcsin(-Vgvm/(V-DV)))
            print("{:> 8.3f} ft/s  < Delta   Velocity < {:> 8.3f} ft/s ".\
                format(-DV,DV))
            print("{:> 8.3f} deg   < Delta      Alpha < {:> 8.3f} deg  ".\
                format(Dal,Dau))
            print("{:> 8.3f} deg   < Delta       Beta < {:> 8.3f} deg  ".\
                format(Dbl,Dbu))
            print("{:> 8.3f} deg/s < Delta  Roll Rate < {:> 8.3f} deg/s".\
                format(-Wgpm,Wgpm))
            print("{:> 8.3f} deg/s < Delta Pitch Rate < {:> 8.3f} deg/s".\
                format(-Wgqm,Wgqm))
            print("{:> 8.3f} deg/s < Delta   Yaw Rate < {:> 8.3f} deg/s".\
                format(-Wgrm,Wgrm))
        # initialize interp
        self.Vgu = lambda tval : interp(tval,t,Vgu)#,period=tf)
        self.Vgv = lambda tval : interp(tval,t,Vgv)#,period=tf)
        self.Vgw = lambda tval : interp(tval,t,Vgw)#,period=tf)
        self.Wgp = lambda tval : interp(tval,t,Wgp)#,period=tf)
        self.Wgq = lambda tval : interp(tval,t,Wgq)#,period=tf)
        self.Wgr = lambda tval : interp(tval,t,Wgr)#,period=tf)
        # self.Vgu = interp1d(t,Vgu)#,period=tf)
        # self.Vgv = interp1d(t,Vgv)#,period=tf)
        # self.Vgw = interp1d(t,Vgw)#,period=tf)
        # self.Wgp = interp1d(t,Wgp)#,period=tf)
        # self.Wgq = interp1d(t,Wgq)#,period=tf)
        # self.Wgr = interp1d(t,Wgr)#,period=tf)

        # save for later use
        self.tf = tf
        self.dt = dt
        self.V = V
        self.t_signal = t*1.
        self.Vgu_signal = Vgu
        self.Vgv_signal = Vgv
        self.Vgw_signal = Vgw
        self.Wgp_signal = Wgp
        self.Wgq_signal = Wgq
        self.Wgr_signal = Wgr


    def _get_input_vars(self,input_vars):
        # store variables globally

        # turbulence model information
        self.H0 = input_vars.get("initial_altitude[ft]",15000.)
        self.t_turb = input_vars.get("init_time[s]",0.)
        self.n_bins = input_vars.get("number_frequency_bins",100)
        self.intensity = input_vars.get("turbulence_intensity","light")#"moderate")
        self.intensity_folder = input_vars.get("intensity_folder","./")
        random_seed = input_vars.get("random_seed",None)
        self.rng = default_rng(random_seed)

        # raise error if invalid intensity given
        possint = ["1","2","3","4","5","6","light","moderate","severe"]
        if self.intensity not in possint:
            raise ValueError("Turbulence intensity setting " + \
                "'{}' invalid. Must be one of {}".format(self.intensity,\
                possint))


    def _interpolate_intensity(self,H,intensity):
        # determine folder and curve to pull data from
        file_name = "intensity_line_data.json"
        if H >= 2000.:
            data_file = "high_altitude_turbulence_intensity/" + file_name
            if intensity.isnumeric():
                curve = "i1em" + intensity
            else:
                curve = intensity
            var_to_interpolate = H/1000.
        elif 1000. <= H < 2000.:
            data_file = "high_altitude_turbulence_intensity/" + file_name
            if intensity.isnumeric():
                curve = "i1em" + intensity
            else:
                curve = intensity
            var_to_interpolate = H/1000.
        else: # H < 1000.
            data_file =  "low_altitude_turbulence_intensity/" + file_name
            curve = "mean_wind_speed"
            if intensity == "light":
                intensity = "2"
            elif intensity == "moderate":
                intensity = "3"
            elif intensity == "severe":
                intensity = "5"
            var_to_interpolate = 10.0**(-float(intensity))

        # import data, interpolate on curve
        data = array( json.loads(open(self.intensity_folder + \
            data_file).read())[curve] )

        # interpolate data
        Vinterp = interp(var_to_interpolate,flip(data[:,1]),flip(data[:,0]))
        
        # modify interpolated value if necessary
        if H >= 2000.:
            Su = Vinterp*1.
            Sv = Su*1.
            Sw = Su*1.
        elif 1000. <= H < 2000.:
            Su = Vinterp*1.
            Sv = Su*1.
            Sw = Su*1.
        else: # H < 1000.
            Vinterp = Vinterp*1.6878098571011957 # *1.852*1000./3600./0.3048 # 
            Su = 0.1*Vinterp
            Sv = Su/(0.177 + 0.000823*H)**0.4
            Sw = Sv*1.

        return Su,Sv,Sw


    def _frequency_newton_solver(self,w_init,bin_area,fun,imax=1000,
        errmax=1.0e-6,relaxation=0.1):
        # initial parameters
        error = 1.0e10

        # run through
        min_fun = lambda w_end : abs(quad(fun,w_init,w_end,\
            epsabs=1.49e-12,epsrel=1.49e-12)[0] - bin_area)
        w_final = minimize_scalar(min_fun).x

        return w_final


    def _build_model(self,b=30.,H="o",show_plot=False):
        if H is "o":
            H = self.H0
        # initialize lengths and 1-sigma values based on the altitude
        # lengths
        if H >= 2000.:
            # fixed values
            Lu = 2500.
            Lv = 2500.
            Lw = 2500.
            Su,Sv,Sw = self._interpolate_intensity(H,self.intensity)
        elif 1000. <= H < 2000.:
            Lu = 2500.
            Lv = 2500.
            Lw = 2500.
            Su,Sv,Sw = self._interpolate_intensity(H,self.intensity)
        else: # H0 < 1000.
            Lw = H*1.
            Lu = H/(0.177 + 0.000823*H)**1.2
            Lv = Lu*1.
            Su,Sv,Sw = self._interpolate_intensity(H,self.intensity)

        # build spectral densities
        # velocities
        Luw = lambda w : ( 1.339*Lu*w )**2.
        Lvw = lambda w : ( 1.339*Lv*w )**2.
        Lww = lambda w : ( 1.339*Lw*w )**2.
        Phi_u = lambda w : Su**2.*2.*Lu/pi*1./(1. + Luw(w))**(5./6.)
        Phi_v = lambda w : Sv**2.*Lv/pi*(1. + 8./3.*Lvw(w))/\
            (1. + Lvw(w))**(11./6.)
        Phi_w = lambda w : Sw**2.*Lw/pi*(1. + 8./3.*Lww(w))/\
            (1. + Lww(w))**(11./6.)
        # rates
        Phi_p = lambda w : Sw**2./Lw*0.8*(pi*Lw/4./b)**(1./3.)/ \
            (1. + (4.*b*w/pi)**2.)
        Phi_q = lambda w : w**2./(1. + (4.*b*w/pi)**2.)*Phi_w(w)
        Phi_r = lambda w : w**2./(1. + (3.*b*w/pi)**2.)*Phi_v(w)
        # save spectral densities for later use
        self.Phi_u = Phi_u
        self.Phi_v = Phi_v
        self.Phi_w = Phi_w
        self.Phi_p = Phi_p
        self.Phi_q = Phi_q
        self.Phi_r = Phi_r

        # build velocity amplitudes
        self.Au = Su*(2./self.n_bins)**0.5
        self.Av = Sv*(2./self.n_bins)**0.5
        self.Aw = Sw*(2./self.n_bins)**0.5

        # build phase-changes
        self.u_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.v_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.w_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.p_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.q_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.r_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi

        # build unity-area spectral densities
        Phi_ua_u = lambda w : Phi_u(w)/Su**2.
        Phi_ua_v = lambda w : Phi_v(w)/Sv**2.
        Phi_ua_w = lambda w : Phi_w(w)/Sw**2.
        # rates
        c = 4.*b/pi
        Mp = 0.8*pi*Sw**2.*(Lw/c)**(1./3.)/(2.*Lw*c)
        # Mq = pi**2.*Sw**2.*(8.*b + 3.*pi*Lw)/8./b/(4.*b + pi*Lw)**2.
        # Mr = pi**2.*Sv**2.*(2.*b +    pi*Lv)/2./b/(3.*b + pi*Lv)**2.
        Mq,Mq_err = quad(Phi_q,0,inf,epsabs=1.49e-12,epsrel=1.49e-12)[0:2]
        Mr,Mr_err = quad(Phi_r,0,inf,epsabs=1.49e-12,epsrel=1.49e-12)[0:2]
        Phi_ua_p = lambda w : Phi_p(w)/Mp
        Phi_ua_q = lambda w : Phi_q(w)/Mq
        Phi_ua_r = lambda w : Phi_r(w)/Mr

        # build rate amplitudes
        self.Ap = (2.*Mp/self.n_bins)**0.5
        self.Aq = (2.*Mq/self.n_bins)**0.5
        self.Ar = (2.*Mr/self.n_bins)**0.5

        # initialize frequency arrays
        self.u_freq = zeros((self.n_bins - 1,))
        self.v_freq = zeros((self.n_bins - 1,))
        self.w_freq = zeros((self.n_bins - 1,))
        self.p_freq = zeros((self.n_bins - 1,))
        self.q_freq = zeros((self.n_bins - 1,))
        self.r_freq = zeros((self.n_bins - 1,))

        # run newton solver
        bin_frac = 1./self.n_bins
        solver = self._frequency_newton_solver
        self.u_freq[0] = solver(0.,bin_frac/2.,Phi_ua_u)
        self.v_freq[0] = solver(0.,bin_frac/2.,Phi_ua_v)
        self.w_freq[0] = solver(0.,bin_frac/2.,Phi_ua_w)
        self.p_freq[0] = solver(0.,bin_frac/2.,Phi_ua_p)
        self.q_freq[0] = solver(0.,bin_frac/2.,Phi_ua_q)
        self.r_freq[0] = solver(0.,bin_frac/2.,Phi_ua_r)
        # for remaining
        if self.n_bins > 2:
            for i in range(1,self.n_bins-1):
                self.u_freq[i] = solver(self.u_freq[i-1],bin_frac,Phi_ua_u)
                self.v_freq[i] = solver(self.v_freq[i-1],bin_frac,Phi_ua_v)
                self.w_freq[i] = solver(self.w_freq[i-1],bin_frac,Phi_ua_w)
                self.p_freq[i] = solver(self.p_freq[i-1],bin_frac,Phi_ua_p)
                self.q_freq[i] = solver(self.q_freq[i-1],bin_frac,Phi_ua_q)
                self.r_freq[i] = solver(self.r_freq[i-1],bin_frac,Phi_ua_r)

        if show_plot:
            # report frequency content
            header = "{:^4s}".format("frq#")
            for var in ["u","v","w","p","q","r"]:
                header += " {:^22s}".format(var)
            print(header)
            for i in range(self.u_freq.shape[0]):
                report = "{:^4d}".format(i)
                report += " {:> 22.15e}".format(self.u_freq[i])
                report += " {:> 22.15e}".format(self.v_freq[i])
                report += " {:> 22.15e}".format(self.w_freq[i])
                report += " {:> 22.15e}".format(self.p_freq[i])
                report += " {:> 22.15e}".format(self.q_freq[i])
                report += " {:> 22.15e}".format(self.r_freq[i])
                print(report)

            # initialize w
            # w = linspace(0.,6.0e-3,num=500)
            # x = linspace(0.,6.0e-1,num=500)
            # Phi_u_vals = Phi_u(w); Phi_u_vals /= np.max(Phi_u_vals)
            # Phi_v_vals = Phi_v(w); Phi_v_vals /= np.max(Phi_v_vals)
            # Phi_w_vals = Phi_w(w); Phi_w_vals /= np.max(Phi_w_vals)
            # Phi_p_vals = Phi_p(x); Phi_p_vals /= np.max(Phi_p_vals)
            # Phi_q_vals = Phi_q(x); Phi_q_vals /= np.max(Phi_q_vals)
            # Phi_r_vals = Phi_r(x); Phi_r_vals /= np.max(Phi_r_vals)
            Phi_u_vals = Phi_u(self.u_freq)#; Phi_u_vals /= np.max(Phi_u_vals)
            Phi_v_vals = Phi_v(self.v_freq)#; Phi_v_vals /= np.max(Phi_v_vals)
            Phi_w_vals = Phi_w(self.w_freq)#; Phi_w_vals /= np.max(Phi_w_vals)
            Phi_p_vals = Phi_p(self.p_freq)#; Phi_p_vals /= np.max(Phi_p_vals)
            Phi_q_vals = Phi_q(self.q_freq)#; Phi_q_vals /= np.max(Phi_q_vals)
            Phi_r_vals = Phi_r(self.r_freq)#; Phi_r_vals /= np.max(Phi_r_vals)

            fvs, avs = plt.subplots(constrained_layout=True)
            avs.plot(self.u_freq,Phi_u_vals,label="$u$")
            avs.plot(self.v_freq,Phi_v_vals,label="$v$")
            avs.plot(self.w_freq,Phi_w_vals,label="$w$")
            fws, aws = plt.subplots(constrained_layout=True)
            # freq = self.r_freq
            # Phi = Phi_r
            # Phi_binned = Phi(freq)
            # for i in range(len(Phi_binned)):
            #     aws.plot([freq[i],freq[i]],[0.,Phi_binned[i]],c="k")
            aws.plot(self.p_freq,Phi_p_vals,label="$p$")
            aws.plot(self.q_freq,Phi_q_vals,label="$q$")
            aws.plot(self.r_freq,Phi_r_vals,label="$r$")
            # avs.set_xscale("log")
            # aws.set_xscale("log")
            avs.legend()
            aws.legend()
            avs.set_ylabel("Spectral Density [(ft/s)^2/(rad/ft)]")
            aws.set_ylabel("Spectral Density [(ft/s)^2*(rad/ft)]")
            avs.set_xlabel("Spatial Frequency [rad/ft]")
            aws.set_xlabel("Spatial Frequency [rad/ft]")
            avs.set_xlim((0.,\
                np.max([max(self.u_freq),max(self.v_freq),max(self.w_freq)])))
            aws.set_xlim((0.,\
                np.max([max(self.p_freq),max(self.q_freq),max(self.r_freq)])))
            avs.set_yscale("log")
            aws.set_yscale("log")
            fvs.savefig("PSD_plots/PSD_uvw.png",dpi=300.)#,transparent=True)
            fws.savefig("PSD_plots/PSD_pqr.png",dpi=300.)#,transparent=True)
            plt.close()
            # plt.show(block=False)
            # plt.pause(0.5)
            # plt.close()
            # plt.show()


    def rebuild_turbulence_phases(self,update_precomputed=True):
        # build phase-changes
        self.u_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.v_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.w_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.p_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.q_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi
        self.r_phas = (self.rng.random(self.n_bins - 1)*2. - 1.)*pi


        if update_precomputed:
            # precompute disturbance
            xs = self.V*self.t_signal
            Vgu = self.Au*npsum(sin(npadd(npouter(self.u_freq, xs).T, self.u_phas).T), axis=0)
            Vgv = self.Av*npsum(sin(npadd(npouter(self.v_freq, xs).T, self.v_phas).T), axis=0)
            Vgw = self.Aw*npsum(sin(npadd(npouter(self.w_freq, xs).T, self.w_phas).T), axis=0)
            Wgp = self.Ap*npsum(sin(npadd(npouter(self.p_freq, xs).T, self.p_phas).T), axis=0)
            Wgq = self.Aq*npsum(sin(npadd(npouter(self.q_freq, xs).T, self.q_phas).T), axis=0)
            Wgr = self.Ar*npsum(sin(npadd(npouter(self.r_freq, xs).T, self.r_phas).T), axis=0)
            # initialize interp
            self.Vgu = lambda tval : interp(tval,self.t_signal,Vgu)
            self.Vgv = lambda tval : interp(tval,self.t_signal,Vgv)
            self.Vgw = lambda tval : interp(tval,self.t_signal,Vgw)
            self.Wgp = lambda tval : interp(tval,self.t_signal,Wgp)
            self.Wgq = lambda tval : interp(tval,self.t_signal,Wgq)
            self.Wgr = lambda tval : interp(tval,self.t_signal,Wgr)


    def get_disturbance(self,t,V):
        # gust dynamics
        # if t >= self.t_turb:
        x = V*t
        Vgu = self.Au*npsum(sin(self.u_freq*x + self.u_phas))
        Vgv = self.Av*npsum(sin(self.v_freq*x + self.v_phas))
        Vgw = self.Aw*npsum(sin(self.w_freq*x + self.w_phas))
        Wgp = self.Ap*npsum(sin(self.p_freq*x + self.p_phas))
        Wgq = self.Aq*npsum(sin(self.q_freq*x + self.q_phas))
        Wgr = self.Ar*npsum(sin(self.r_freq*x + self.r_phas))
        #
            # numpy method faster than vvv
            # vu = mdl.Au*sum([sin(mdl.u_freq[j]*x[i] + mdl.u_phas[j]) \
            #     for j in range(mdl.n_bins)])
        # else:
        #     Vgu = 0.
        #     Vgv = 0.
        #     Vgw = 0.
        #     Wgp = 0.
        #     Wgq = 0.
        #     Wgr = 0.
        return Vgu,Vgv,Vgw,Wgp,Wgq,Wgr


    def get_precomputed_disturbance(self,t,V):
        Vgu = self.Vgu(t)
        Vgv = self.Vgv(t)
        Vgw = self.Vgw(t)
        Wgp = self.Wgp(t)
        Wgq = self.Wgq(t)
        Wgr = self.Wgr(t)
        return Vgu,Vgv,Vgw,Wgp,Wgq,Wgr


import numpy as np
def signal_decomp(x,y):
    '''uses FFT to find the frequency of the largest component based on power spectrum'''
    N = len(y) # number of data points
    dt = (x[-1] - x[0])/N # average step size of the data
    num_freqs = N//2 # Nyquist frequency, highest frequency that can be measured in the given data
    fourier_transform = np.fft.fft(y) # discrete Fourier transform
    fourier_zero = fourier_transform[0]
    amp_zero = fourier_zero/N
    fourier_oneside = fourier_transform[1:num_freqs] # slice only the positive frequency terms, exclude the first zero freq term
    fourier_oneside[:] = 2*fourier_oneside # doubles the non zero frequency terms. I think this only matters for the power calculation?
    amp_spec_one = np.absolute(fourier_oneside)/N # this looks like its missing the sqrt(2) that his paper has?
    phase_spec_one = np.pi/2.0 + np.arctan2(fourier_oneside.imag,
                                            fourier_oneside.real) # why the pi/2.0?
    # phase_spec_one = np.arctan2(fourier_oneside.imag, fourier_oneside.real)
    amplitude = max(amp_spec_one).real
    freq_spec = np.fft.fftfreq(N, d=dt) # frequency bin centers in cycles per unit of the sample spacing (with zero at the start)
    freq_spec_one = freq_spec[1:num_freqs]*2.0*np.pi # radians per unit of sample spacing, ignore the zero frequency index
    power_spec = (np.abs(fourier_oneside)**2)*((dt)**2)
    highest_power_index = power_spec.argmax()
    frequency = freq_spec_one[highest_power_index]
    return frequency

if __name__ == "__main__":


    mdl = VonKarmanTurbulence(
        {"number_frequency_bins":100,"turbulence_intensity":"light",
        "initial_altitude[ft]":200.0,
        "random_seed" : 1
        },
        wingspan=3.03,V=100.,
        # wingspan=30.,V=634.,
        show_plot=True)


    quit()
    num = 100001
    t = linspace(0.,100.,num=num)
    V = 634.
    # mdl.u_phas = zeros((mdl.n_bins - 1,))
    # mdl.v_phas = zeros((mdl.n_bins - 1,))
    # mdl.w_phas = zeros((mdl.n_bins - 1,))
    # mdl.p_phas = zeros((mdl.n_bins - 1,))
    # mdl.q_phas = zeros((mdl.n_bins - 1,))
    # mdl.r_phas = zeros((mdl.n_bins - 1,))
    # T0 = array([mdl.get_disturbance(t[i],V) for i in range(len(t))])
    # num_zeros = 100
    # T0 = np.block([[T0],[np.zeros((num_zeros,6))]])
    # t = np.concatenate((t,np.zeros((num_zeros,))))
    # mdl.rebuild_turbulence_phases(update_precomputed=False)
    # T1 = array([mdl.get_disturbance(t[i],V) for i in range(len(t))])
    # T2s = []
    # for j in range(100):
    #     print(j,end="\r")
    #     mdl.rebuild_turbulence_phases(update_precomputed=False)
    #     T2 = array([mdl.get_disturbance(t[i],V) for i in range(len(t))])
    #     T2s.append(T2)
    # print("done")

    # find fastest 1d interp
    xs = V*t
    from time import time as tm
    start = tm()
    for j in range(len(t)):
        mdl.Vgu(t[j])
        mdl.Vgv(t[j])
        mdl.Vgw(t[j])
        mdl.Wgp(t[j])
        mdl.Wgq(t[j])
        mdl.Wgr(t[j])
    dur_0 = tm() - start
    print(mdl.Vgu(25.0),mdl.Vgv(25.0),mdl.Vgw(25.0),mdl.Wgp(25.0),mdl.Wgq(25.0),mdl.Wgr(25.0))

    dists = [
        mdl.Vgu_signal,
        mdl.Vgv_signal,
        mdl.Vgw_signal,
        mdl.Wgp_signal,
        mdl.Wgq_signal,
        mdl.Wgr_signal
    ]
    intp = interp1d(mdl.t_signal,dists,axis=1,copy=False,kind="linear",assume_sorted=True)
    start = tm()
    for j in range(len(t)):
        intp(t[j])
    dur_1 = tm() - start
    print(intp(25.0))

    # dists = np.array([
    #     mdl.Vgu_signal,
    #     mdl.Vgv_signal,
    #     mdl.Vgw_signal,
    #     mdl.Wgp_signal,
    #     mdl.Wgq_signal,
    #     mdl.Wgr_signal
    # ]).T
    # intpRGI = RegularGridInterpolator(mdl.t_signal,dists,method="linear")
    # start = tm()
    # for j in range(len(t)):
    #     intpRGI(t[j])
    # dur_3 = tm() - start
    # print(intpRGI(25.0))

    start = tm()
    ranger = np.array(range(len(mdl.t_signal)))
    def my_interp(tvalue):
        truths = mdl.t_signal >= tvalue
        i0 = (ranger[truths])[0]
        dx = mdl.t_signal[i0+1] - mdl.t_signal[i0]
        dy = mdl.Vgu_signal[i0+1] - mdl.Vgu_signal[i0]
        vals = [
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Vgu_signal[i0+1]-mdl.Vgu_signal[i0]) + mdl.Vgu_signal[i0],
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Vgv_signal[i0+1]-mdl.Vgv_signal[i0]) + mdl.Vgv_signal[i0],
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Vgw_signal[i0+1]-mdl.Vgw_signal[i0]) + mdl.Vgw_signal[i0],
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Wgp_signal[i0+1]-mdl.Wgp_signal[i0]) + mdl.Wgp_signal[i0],
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Wgq_signal[i0+1]-mdl.Wgq_signal[i0]) + mdl.Wgq_signal[i0],
            (tvalue - mdl.t_signal[i0])/dx*(mdl.Wgr_signal[i0+1]-mdl.Wgr_signal[i0]) + mdl.Wgr_signal[i0],
        ]
        return vals
    for j in range(len(t)):
        my_interp(t[j])
    dur_4 = tm() - start
    print(my_interp(25.0))

    print("numpy interp                  {:> 10.6f}".format(dur_0))
    print("scipy interp1d                {:> 10.6f}".format(dur_1))
    # print("scipy RegularGridInterpolator {:> 10.6f}".format(dur_3))
    print("my invention                  {:> 10.6f}".format(dur_4))
    quit()

    # print("numpy interp   {:> 10.6f}".format(dur_0))
    # print("scipy interp1d {:> 10.6f}".format(dur_1))
    # quit()

    # find avg method
    # vas = []
    # vbs = []
    # vcs = []
    # vds = []
    # ves = []
    # for i in range(len(T0[0])):
    #     if i >= 3:
    #         fun = np.rad2deg
    #     else:
    #         fun = lambda t : t
    #     vals = fun(T0[:,i])
    #     a = np.average(vals)
    #     b = np.average(np.abs(vals))
    #     c = np.average(np.abs(vals),weights=np.abs(vals)/np.max(np.abs(vals)))
    #     d = np.average(np.abs(vals),weights=np.abs(vals)**2./np.max(np.abs(vals))**2.)
    #     e = np.average(np.abs(vals),weights=np.abs(vals)**3./np.max(np.abs(vals))**3.)
    #     print(i,a,b,c,d,e)
    #     vas.append(a)
    #     vbs.append(b)
    #     vcs.append(c)
    #     vds.append(d)
    #     ves.append(e)
    # print("norm a's   ",np.linalg.norm(vas))
    # print("norm b's   ",np.linalg.norm(vbs))
    # print("norm c's   ",np.linalg.norm(vcs))
    # print("norm c's*2.",np.linalg.norm(vcs)*2.)
    # print("norm d's   ",np.linalg.norm(vds))
    # print("norm e's   ",np.linalg.norm(ves))
    # # quit()
        
    

    dx = (t[1] - t[0])*V
    fs = 1./dx
    # print(dx,fs)
    # (fp,Sp) = periodogram(T0[:,0],fs,scaling="density")
    n = len(t)
    # n = mdl.n_bins
    funs = [
        mdl.Phi_u, mdl.Phi_v, mdl.Phi_w,
        mdl.Phi_p, mdl.Phi_q, mdl.Phi_r
    ]
    plt.close()
    print("plotting")
    names = ["$u$","$v$","$w$","$p$","$q$","$r$"]
    freqs = [mdl.u_freq,mdl.v_freq,mdl.w_freq,mdl.p_freq,mdl.q_freq,mdl.r_freq]
    fpl,apl = plt.subplots(3,3,figsize=(16.0,12.0),layout="constrained")
    for i in range(6):
        f = linspace(-n/2.,n/2.,n)*fs/n
        y = fft(T0[:,i])#,n=n)
        f = fftfreq(n,d=dx)
        # f = fftshift(f)
        # y = fftshift(y)
        power  = npabs(y)**2./n
        power2 = npabs(y)**2./dx**2.
        power3 = npabs(y)**2.#/fs**2.
        wf,wP = welch(T0[:,i],fs=fs,scaling="spectrum")
        # (Sm,fm) = plt.psd(T0[:,0],Fs=fs)
        ir = i // 3
        ic = i % 3
        f_PSD = logspace(-5.,log10(npmax(f)),num=n*10)
        # print(f_PSD)
        apl[ir,ic].plot(f_PSD,funs[i](f_PSD),label="PSD")
        for j in range(mdl.n_bins-1):
            apl[ir,ic].plot([freqs[i][j]]*2,[0.,funs[i](freqs[i][j])],c="k",lw=0.5)
        # apl[ir,ic].plot(f_PSD,funs[i](f_PSD)/f_PSD,label="PSD/f")
        # apl[ir,ic].plot(f,power,lw=0.5,label="1")
        # apl[ir,ic].plot(f,power2,lw=0.5,label="mag 1")
        apl[ir,ic].plot(f[:int(n/2)+1],power3[:int(n/2)+1],lw=0.5,label="mag 2")
        # z = fft(T1[:,i])
        # Pz = npabs(z)**2./fs**2.
        # apl[ir,ic].plot(f[:int(n/2)+1],Pz[:int(n/2)+1],lw=0.5,label="mag 2,rnd")
        # Pqavg = T2s[0][:,i]*0.
        # for j in range(len(T2s)):
        #     q = fft(T2s[j][:,i])
        #     Pq = npabs(q)**2./fs**2.
        #     Pqavg = Pqavg + Pq
        # Pqavg = Pqavg/float(len(T2s))
        # apl[ir,ic].plot(f[:int(n/2)+1],Pqavg[:int(n/2)+1],lw=1.0,c="k",label="mag 2,rnd")


        # apl[ir,ic].plot(wf,wP,lw=0.5,label="welch")
        # apl[ir,ic].plot(f3,power[-int(n/2):],label="Harris3")
        # apl[ir,ic].plot(fp[1:],Sp[1:],label="periodogram")
        # apl[ir,ic].plot(fm[1:],Sm[1:],label="matplotlib")
        # apl[ir,ic].set_xscale("log")
        apl[ir,ic].set_yscale("log")
        apl[ir,ic].set_xlabel("spatial frequency [rad/ft]")
        apl[ir,ic].set_ylabel("PSD, " + names[i])
        apl[ir,ic].set_xlim((0,freqs[i][-1]))#)
        apl[ir,ic].legend()
    
    apl[2,0].plot(t[:num],T0[:num,0],label="$u$")
    apl[2,0].plot(t[:num],T0[:num,1],label="$v$")
    apl[2,0].plot(t[:num],T0[:num,2],label="$w$")
    apl[2,0].set_xlabel("Time $t$ [s]")
    apl[2,0].set_ylabel("Turbulent Velocity [ft/s]")
    apl[2,0].legend()
    apl[2,1].plot(t[:num],rad2deg(T0[:num,3]),label="$p$")
    apl[2,1].plot(t[:num],rad2deg(T0[:num,4]),label="$q$")
    apl[2,1].plot(t[:num],rad2deg(T0[:num,5]),label="$r$")
    apl[2,1].set_xlabel("Time $t$ [s]")
    apl[2,1].set_ylabel("Turbulent Rotation Rate [deg/s]")
    apl[2,1].legend()

    # apl[2,2].plot(t,np.abs(T0[:,0]),label="$u$")
    # apl[2,2].plot(t,rad2deg(np.abs(T0[:,3])),label="$p$")


    # ranphs = lambda x : (rand(x)*2. - 1.)*pi
    # x = np.linspace(0.,1000.,num=1001)
    # dx = x[1] - x[0]
    # f = fftfreq(len(x),d=dx)
    # n = 100
    # Pavg = x*0.
    # for i in range(n):
    #     P = 2.*sin(0.25*x + ranphs(1)) + 2.*sin(0.8*x + ranphs(1))
    #     y = fft(P)
    #     magy = npabs(y)**2.*dx**2.
    #     apl[2,2].plot(f[:int(len(x)/2)+1],magy[:int(len(x)/2)+1],lw=0.5)
    #     Pavg = Pavg + magy
    # Pavg = Pavg / float(n)
    # apl[2,2].plot(f[:int(len(x)/2)+1],Pavg[:int(len(x)/2)+1],lw=2.0,c="k")
    # apl[2,2].set_yscale("log")
    # apl[2,2].set_xlim(left=0.)
    
    plt.show()
    

    ## simplify turbulence intensity inputs

    # folder = "high_altitude_turbulence_intensity/"
    # # folder = "low_altitude_turbulence_intensity/"
    # file = "high_altitude_turbulence_intensity.json"
    # # file = "low_altitude_turbulence_intensity.json"
    # json_string = open(folder + file.split(".")[0]+"_ugly.json").read()
    # # save to vals dictionary
    # input_dict = json.loads(json_string)

    # pretty = json.dumps(input_dict,indent=4)

    # with open(folder + file,"w") as f:
    #     f.write(pretty)
    #     f.close()

    # new_data = {}
    # import numpy as np
    # ftb, atb = plt.subplots()

    # # create truncated json file
    # for data_dict in input_dict["datasetColl"]:
    #     name = data_dict["name"]
    #     new_data[name] = []
    #     line = []
    #     for point in data_dict["data"]:
    #         line.append( point["value"] )
    #     # numpy-ify
    #     line = array(line)

    #     # sort to find max value
    #     line = line[np.flip(np.argsort(line[:,1]))]
    #     inds = list(range(len(line)))
        
    #     # for curvy lines
    #     if name[0:4] == "i1em" or name == "mean_wind_speed":
    #         new_inds = [0]
    #         old_inds = list(range(1,len(line)))
    #         # run through points, rearrange so closest point is next
    #         for i in range(len(line)-1):
    #             pt = line[new_inds[-1]]
    #             diffs = ( (pt[0]-line[old_inds,0])**2. + \
    #                 (pt[1]-line[old_inds,1])**2. )**0.5
    #             sort_ind = np.argsort(diffs)
    #             next_ind = old_inds[sort_ind[0]]
    #             new_inds.append( old_inds.pop(sort_ind[0]) )

    #             # plt.plot(line[:,0],line[:,1],"bo")
    #             # plt.plot(line[new_inds,0],line[new_inds,1],"r")
    #             # if folder[0:3] == "low":
    #             #     plt.yscale("log")
    #             # plt.show(block=False)
    #             # plt.pause(0.2)
    #             # plt.close()
            
    #         line = line[new_inds]
        
    #     new_data[name] = line.tolist()

    #     if name[0:4] == "i1em":
    #         lbl = "10^-" + name[4:]
    #     else:
    #         lbl = name
        
    #     atb.plot(line[:,0],line[:,1],label=lbl)
    # if folder[0:3] == "low":
    #     atb.set_yscale("log")
    #     atb.set_ylabel("Turbulence Intensity")
    #     atb.set_xlabel("Mean Wind Speed at 20 ft [kts]")
    # else:
    #     atb.set_ylabel("Altitude [thousands ft]")
    #     atb.set_xlabel("RMS Turbulence Amplitude [ft/s]")
    #     atb.legend()
    # ftb.savefig(folder + "turbulence_intensity_plotted.png",dpi=300.)
    # plt.close()

    # json_obj = json.dumps(new_data,indent=4)

    # with open(folder + "intensity_line_data.json","w") as f:
    #     f.write(json_obj)
    #     f.close()



    ## symbolic integration

    # import sympy as sy
    # # determine symbolic integrals to infinity!
    # sym = sy.Symbol
    # igr = sy.integrate
    # simp = sy.simplify
    # spi = sy.pi
    # oo = sy.oo
    # frac = sy.Rational

    # # declare variables
    # print("declaring variables...")
    # x = sym("x",real=True,positive=True)
    # b = sym("b",real=True,positive=True,finite=True)
    # Su = sym("Su",real=True,positive=True,finite=True)
    # Sv = sym("Sv",real=True,positive=True,finite=True)
    # Sw = sym("Sw",real=True,positive=True,finite=True)
    # Lu = sym("Lu",real=True,positive=True,finite=True)
    # Lv = sym("Lv",real=True,positive=True,finite=True)
    # Lw = sym("Lw",real=True,positive=True,finite=True)

    # # create bounds
    # print("creating bounds...")
    # x_bnd = (x,0,oo)
    # # create functions
    # f_u = Su**2*(2*Lu/spi)*1/(1+(1.339*Lu*x)**2)**(frac(5,6))
    # f_v = Sv**2*(Lv/spi)*(1+frac(8,3)*(1.339*Lv*x)**2)/(1+(1.339*Lv*x)**2)**(frac(11,6))
    # f_w = Sw**2*(Lw/spi)*(1+frac(8,3)*(1.339*Lw*x)**2)/(1+(1.339*Lw*x)**2)**(frac(11,6))
    # f_p = Sw**2/Lw*0.8*(pi*Lw/4/b)**frac(1,3)/(1+(4*b*x/pi)**2)
    # f_q = x**2/(1+(4*b*x/pi)**2)*f_w
    # f_r = x**2/(1+(3*b*x/pi)**2)*f_v

    # # solve
    # print("solving...")
    # i_u = simp( igr(f_u,x_bnd) )
    # print("i_u =",i_u)
    # print("     ",i_u.evalf())
    # i_v = simp( igr(f_v,x_bnd) )
    # print("i_v =",i_v)
    # print("     ",i_v.evalf())
    # i_w = simp( igr(f_w,x_bnd) )
    # print("i_w =",i_w)
    # print("     ",i_w.evalf())
    # i_p = simp( igr(f_p,x_bnd) )
    # print("i_p =",i_p)
    # print("     ",i_p.evalf())
    # i_q = simp( igr(f_q,x_bnd) )
    # print("i_q =",i_q)
    # print("     ",i_q.evalf())
    # i_r = simp( igr(f_r,x_bnd) )
    # print("i_r =",i_r)
    # print("     ",i_r.evalf())
    # quit()
        


