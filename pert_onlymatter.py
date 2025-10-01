# ================
# Import libraries
# ================

import numpy as np
from astropy import constants as const


# ===========================
# Set cosmological parameters
# ===========================

cosmo_parameters = {
    'H0': 67.,
    'h' : .67,
    'Omega_m0': 1.,
    'w0' : -1.,
    'wa': -0.3,
    'cs2': 1.
}


c_kms = const.c.to('km/s').value


# ==================================
# Define general cosmology functions.
# ==================================
def EHubble(a, pars=cosmo_parameters):
    """
    Dimensionless Hubble parameter as a function of
    the scale factor. Cosmological parameters must
    be provided as a dictionary.
    """
    Omm = pars['Omega_m0']
    return np.sqrt(Omm*a**-3)


def Hubble(a, pars=cosmo_parameters):
    """
    Hubble parameter as a function of
    the scale factor. Cosmological parameters must
    be provided as a dictionary. Units are (1/Mpc).
    """
    return pars['H0']*EHubble(a, pars)/c_kms


def Omega_m(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_m0'] * a**-3 / (EHubble(a, pars))**2


def phi(a, k, X, pars=cosmo_parameters):
    """
    Gravitational potential as a function of the scale factor
    and wave-number. Perturbations must also be provided.
    """
    delta_m, theta_m = X
    H = Hubble(a,pars=pars)
    factor = -1.5 * (a*H/k)**2
    matter_term = (delta_m + 3*a*H/k**2 * theta_m)
    return factor * matter_term


def dphida(a, k, X, pars=cosmo_parameters):
    """
    Derivative of gravitational potential wrt the scale factor,
    as a function of the scale factor and wave-number. Perturbations 
    must also be provided.
    """
    delta_m, theta_m = X
    H = Hubble(a,pars=pars)
    factor = 1.5* H/k**2
    matter_term = Omega_m(a, pars=pars)*theta_m
    return factor*matter_term - phi(a, k, X, pars=pars)/a


def rhs_pert(a, X, k, pars=cosmo_parameters):
    """
    This function gives the rhs of the system of equations to be solved.
    """
    H = Hubble(a,pars)
    delta_m, theta_m = X

    phi_pot = phi(a, k, X, pars=pars)
    dphi_potda = dphida(a, k, X, pars=pars)

    output = [-theta_m/(a**2 * H) + 3*dphi_potda,
                -theta_m/a + k**2 * phi_pot/(a**2 * H)]
        
    return np.array(output)