import numpy as np
import scipy
import astropy.constants as astroconst
from astropy import units as u
import scipy.constants as const
import scipy.integrate
import scipy.optimize
from scipy import interpolate
import utils
import os
from numba import jitclass, jit
import pickle
from aux_numba import *

# check backend to import appropiate progress bar #
def tqdm_dump(array):
    return array
backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    tqdm = tqdm_dump

class wind:
    """
    A class used to represent the global properties of the wind, i.e, the accretion disc and black hole properties as well as attributes shared among streamlines.
    """
    
    ## Physical Constants ##
    G = astroconst.G.cgs.value
    Ms = astroconst.M_sun.cgs.value
    c = astroconst.c.cgs.value
    m_p = astroconst.m_p.cgs.value
    k_B = astroconst.k_B.cgs.value
    Ryd = u.astrophys.Ry.cgs.scale
    sigma_sb = astroconst.sigma_sb.cgs.value
    sigma_t = const.physical_constants['Thomson cross section'][0] * 1e4
    year = u.yr.cgs.scale

    def __init__(self, r_init = 236.84, M = 2e8, mdot = 0.5, spin=0.,eta=0.06, fx = 0.15, Rmin=6, Rmax=1400, T=2e6, mu = 1, modes =[], rho_shielding = 2e9, intsteps=1.,save_dir="Results"):
        """
        Parameters
        ----------
        r_init : float
            Radius of the first streamline to launch, in Rg units.
        M : float
            Black Hole Mass in solar mass units.
        mdot : float
            Accretion rate (mdot = L / Ledd)
        spin : float
            Spin black hole parameter between [0,1]
        eta : float
            Accretion efficiency (default is for scalar black hole).
        fx : float
            Ratio of luminosity in X-Rays, fx = Lx / Lbolumetric
        Rmin : float
            Minimum radius of acc. disc, default is ISCO for scalar black hole.
        Rmax : float
            Maximum radius of acc. disc.
        T : float
            Temperature of the disc atmosphere. Wind is assumed to be isothermal.
        mu : float
            Mean molecular weight ( 1 = pure hydrogen)
        modes : list 
            List of modes for debugging purposes. Available modes are:
                - 'oldint': Non adaptive disc integration (much faster but convergence is unreliable.)
                - 'altopts': Alternative opacities (experimental)
                - 'gravityonly': Disable radiation force, very useful for debugging.
        rho_shielding : float
            Initial density of the shielding material.
        intsteps : int
            If oldint mode enabled, this refined the integration grid.
        save_dir : str
            Directory to save results.
        """
        
        # array containing different modes for debugging #
        self.modes = modes
        # black hole and disc variables #
        self.M = M * self.Ms
        self.mdot = mdot
        self.spin = spin
        self.mu = mu
        self.fx = fx
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.r_init = r_init
        
        self.thetamax = 0 # saves maximum latitude achieved by a streamline.
        
       
        self.eta = eta
        self.Rg = self.G * self.M / (self.c ** 2) # gravitational radius
        self.rho_shielding = rho_shielding
        self.GE = 4 * np.pi * self.m_p * self.c**3 / (self.sigma_t) # useful normalisation quantity
        self.SE = self.GE / self.Rg
        self.norm = 3 * self.mdot * self.mu / (8. * np.pi * self.eta) # normalisation factor for ionisation parameter #
        self.Ledd = self.EddingtonLuminosity() #Edd luminosity
        self.Lbol = self.BolLuminosity() #Bol luminosity

        # create directory if it doesnt exist. Warning, this overwrites previous outputs.
        self.save_dir = save_dir
        try:
            os.mkdir(save_dir)
        except BaseException:
            pass

        ## aux variables for integration  ##
        self.phids = np.linspace(0, np.pi, intsteps * 100 + 1)
        self.deltaphids = np.asarray(
            [self.phids[i + 1] - self.phids[i] for i in range(0, len(self.phids) - 1)])
        self.rds = np.geomspace(self.Rmin, self.Rmax, intsteps * 250 + 1)
        self.deltards = np.asarray(
            [self.rds[i + 1] - self.rds[i] for i in range(0, len(self.rds) - 1)])

        # interpolate xray optical depths and ionisation parameter from tables #
        self.tau_xray_interp = pickle.load(open("Cloudy_tables/tau_interp_r_rho.pkl","rb")) # density first, r second#
        self.xi_log_interp = pickle.load(open("Cloudy_tables/xi_log_interp_r_rho.pkl","rb"))
        
        self.reff_hist = [0] # for debugging
        self.lines = [] # list of streamline objects

    def get_index(self, array, value):
        """
        Returns index of closest element on array to value.
        
        Parameters
        -----------
        array : list or numpy array
        value : float
        """
        
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def Tau_X(self,d):   
        """
        Computes X-Ray optical depth at a given spherical radius d.
        
        Parameters
        -----------
        d : float
            spherical radial distance.
        """
        
        return self.tau_xray_interp(self.rho_shielding,d)

    def MassAccretionRate(self):
        """ 
        Gives Mass accretion rate in units of solar mass per year.
        """
        
        aux = self.mdot * self.Ledd / (self.eta * self.c**2)
        aux = aux * u.g / u.s
        return aux.to(u.M_sun / u.year)

    def v_kepler(self, r ):
        """
        Keplerian tangential velocity in units of c.
        """
        
        return np.sqrt(1. / (r) )

    def v_esc(self,d):
        """
        Escape velocity in units of c.
        
        Parameters
        -----------
        d : float
            spherical radial distance.
        """
        
        return np.sqrt(2. / d)

    def EddingtonLuminosity(self):
        """ 
        Returns the Eddington Luminosity. 
        """
        
        return self.GE * self.Rg

    def BolLuminosity(self):
        """ 
        Bolumetric Luminosity 
        """
        
        return self.mdot * self.Ledd

    def T4(self, r):
        """
        Returns disc radiance assuming SS model ( F = sigma T4 -> T4 = F / sigma_t).
        """
        
        rel = (1. - np.sqrt(6. / r)) / r**3
        return self.norm * rel

    def Radiance(self, r):
        """
        Computes Disc Radiance assuming stantard SS disc.
        Radius in Rg units
        """
        
        return self.sigma_sb * self.T4(r)

    def RadianceNorm(self, r):
        """
        Radiance normalised wrt SE.
        """
        
        return self.sigma_sb * self.T4(r) / self.SE

    def ThermalVelocity(self, T):
        """
        Thermal velocity for gas with molecular weight mu and temperature T
        """
        
        return np.sqrt(self.k_B * T / (self.mu * self.m_p)) / self.c

    def Opacity(self, rho, T, uvorxray = 0):
        """
        Standard opacity ( thomson scattering ). 'altopts is experimental'.
        """
        
        if('altops' in self.modes):
            return (6.4e24 * rho * self.m_p * T**(-3.5) + 0.4) * self.m_p
        else:
            electron_scattering = self.sigma_t
            return electron_scattering

    def NormIonParameter(self):
        """
        Normalised ionisation parameter. 
        """
        
        return self.GE * self.sigma_t / (4 * np.pi * self.Ryd * self.c * self.Rg)

    def Tau_dr(self, opacity, density):
        """ 
        Differential optical depth.
        
        Parameters
        -----------
        opacity : float
            opacity of the material.
        density : float
            shielding density.
        """
        
        return opacity * self.mu * density * self.Rg
    
    def Xi(self,d, theta):
        """
        Computes ionisation paramter at a distance d and lattitude theta.
        If theta is bigger than thetamax, then there is no obscuration and X-Rays just diminish as 1/d^2.
        Otherwise, interpolates ionis
        
        Parameters
        -----------
        d : float
            spherical radial distance.
        theta : float
            Latitute given by theta = z / r.
        """
        
        if(theta > self.thetamax):
            return (self.fx * self.mdot * self.Ledd) / (self.rho_shielding * (d * self.Rg)**2)
        else:
            return 10**(self.xi_log_interp(self.rho_shielding, d))
        
    def K(self, xi):
        """
        Auxiliary function required for computing force multiplier.
        
        Parameters
        -----------
        xi: float
            Ionisation Parameter.
        """
        
        return 0.03 + 0.385 * np.exp(-1.4 * xi**(0.6))

    def Etamax(self, xi):
        """
        Auxiliary function required for computing force multiplier.
        
        Parameters
        -----------
        xi: float
            Ionisation Parameter.
        """
        
        if(np.log10(xi) < 0.5):
            aux = 6.9 * np.exp(0.16 * xi**(0.4))
            return 10**aux
        else:
            aux = 9.1 * np.exp(-7.96e-3 * xi)
            return 10**aux

    def EffectiveOpticalDepth(self, tau_dr, dv_dr, T):
        """
        Returns differential optical depth times a factor that compares thermal velocity with spatial velocity gradient.
        Required by ForceMultiplier.
        
        Parameters
        -----------
        tau_dr : float
            Differential optical depth.
        dv_dr : float
            Velocity spatial gradient.
        T : float
            Wind temperature.
        """
        
        dr_e = self.ThermalVelocity(T) / np.abs(dv_dr)
        return tau_dr * dr_e 

    def ForceMultiplier(self, t, xi):
        """
        Computes Force multiplier.
        
        Parameters
        -----------
        t : float
            Effective Optical Depth.
        xi : float
            Ionisation Parameter.
        """
        
        xi = xi 
        K = self.K(xi)
        etamax = self.Etamax(xi)
        taumax = t * etamax
        alpha = 0.6
        if (taumax < 0.001):
            aux = (1. - alpha) * (taumax ** alpha)
        else:
            aux = ((1. + taumax)**(1. - alpha) - 1.) / \
                ((taumax) ** (1. - alpha))
        return K * t**(-alpha) * aux

    
    def line(self,
            r_0=375.,
            z_0=1.,
            rho_0=2e8,
            T=2e6,
            v_r_0=0.,
            v_z_0=5e7,
            dt=4.096 / 10):
        """
        Initialises a streamline object.
        
        Parameters
        -----------
        r_0 : float
            Initial radius in Rg units.
        z_0: float
            Initial height in Rg units.
        rho_0 : float
            Initial number density. Units of 1/cm^3.
        T : float
            Initial stramline temperature.
        v_r_0 : float
            Initial radial velocity in units of cm/s.
        v_z_0 : float
            Initial vertical velocity in units of cm/s.
        dt : float
            Timestep in units of Rg/c.
        """

        from streamline import streamline
        return streamline(
            parent = self,
            r_0 = r_0,
            z_0 = z_0,
            rho_0 = rho_0,
            T = T,
            v_r_0 = v_r_0,
            v_z_0 = v_z_0,
            dt = dt)
    
    def StartLines(self, nr = 10, v_z_0 = 5e7, niter=5000):        
        """
        Starts and evolves a set of equally spaced streamlines.
        
        Parameters
        -----------
        nr : int 
            Number of streamlines.
        v_z_0 : float
            Initial vertical velocity.
        niter : int 
            Number of timesteps.
        """
        
        line_range = np.linspace(self.r_init, 1000, nr)
        for r in line_range:
            self.lines.append(self.line(r_0=r))
        for line in self.lines:
            line.iterate(niter=niter)
        return self.lines
