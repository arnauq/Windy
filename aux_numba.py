"""
Auxiliary file for Numba functions.
"""
import numpy as np
import scipy
import astropy.constants as astroconst
from astropy import units as u
import scipy.constants as const
import scipy.integrate
import scipy.optimize
from scipy import interpolate
#from profilehooks import profile
import numba as nb
from numba import jitclass, jit, int32, float32
import pickle
import h5py


@jit(nopython=True, parallel = True)
def _integrate_dblquad_kernel_r(r_d, phi_d, r, z, tau_dr, tau_dr_0, DeltaR, r_0):
    """
    Radial part of the radiation force integral kernel.
    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d 
    abs_uv = np.exp(- (tau_dr * delta ) )
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * cos_gamma  

@jit(nopython=True, parallel = True)
def _integrate_dblquad_kernel_z(r_d, phi_d, r, z, tau_dr, tau_dr_0, DeltaR, r_0):
    """
    Z part of the radiation force integral kernel.
    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d
    abs_uv = np.exp(- (tau_dr * delta ) )
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * sin_gamma


def integrate_dblquad(r, z, tau_dr, tau_dr_0, DeltaR,r_0, Rmin, Rmax):
    """
    Dblquad integration of the disc radiation field.
    """
    r_int = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_r,
        0,
        np.pi,
        Rmin,
        Rmax,
        args=(
            r,
            z,
            tau_dr,
            tau_dr_0,
            DeltaR,
            r_0))[0]
    z_int = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_z,
        0,
        np.pi,
        Rmin,
        Rmax,
        args=(
            r,
            z,
            tau_dr,
            tau_dr_0,
            DeltaR,
            r_0))[0]
    return [r_int, z_int]


@jit(nopython=True)
def _Distance_gas_disc(r_d, phi_d, r, z):
    """
    Distance between gas blob and disc element.
    
    Parameters
    ----------
    r_d : float
        Disc element radius in Rg.
    phi_d : float
        Disc element polar angle in radians.
    r : float
        Gas blob radial position in Rg.
    phi : float
        Gas blob angular position in radians.
    z : float
        Gas blob z position in Rg.
    """
    return np.sqrt(
        r ** 2. +
        r_d ** 2. +
        z ** 2. -
        2. *
        r *
        r_d *
        np.cos(phi_d))


@jit(nopython=True)
def _Force_integral_kernel(r_d, phi_d, deltaphi, deltar, r, z, tau_dr):
    """
    Combined radial and Z kernel for old integration method.
    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    abs_uv = np.exp(- tau_dr * delta) 
    darea = r_d * deltar * deltaphi
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return [ff * cos_gamma, ff * sin_gamma]


@jit(nopython=True)
def integration(rds, phids, deltards, deltaphids, r, z, tau_dr):
    """
    Old integration method. Much faster, but poor convergence near the disc.
    """
    integral = [0., 0.]
    for i in range(0, len(deltards)):
        for j in range(0, len(deltaphids)):
            aux = _Force_integral_kernel(
                rds[i], phids[j], deltaphids[j], deltards[i], r, z, tau_dr)
            integral[0] += aux[0]
            integral[1] += aux[1]
    return integral