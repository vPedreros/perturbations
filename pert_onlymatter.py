# ================
# Import libraries
# ================

import numpy as np
from astropy import constants as const


# ===========================
# Set cosmological parameters
# ===========================

c_kms = const.c.to('km/s').value

cosmo_parameters = {
    'H0 (km/s/Mpc)': 67.,
    'H0 (1/Mpc)': 67. / c_kms ,
    'h' : .67,
    'Omega_m0': 1.,
    'w0' : -1.,
    'wa': 0.,
    'cs2': 1.
}


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


def Hubble(a, pars=cosmo_parameters, units='1/Mpc'):
    """
    Hubble parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    You can choose units by setting the flag units to '1/Mpc' or 
    'km/s/Mpc'
    """
    if units == '1/Mpc':
        return pars['H0 (1/Mpc)'] * EHubble(a, pars=pars)
    elif units in ['km/s/Mpc', 'Km/s/Mpc']:
        return pars['H0 (km/s/Mpc)'] * EHubble(a, pars=pars)
    else: 
        raise ValueError('Invalid selection of units, please choose "(1/Mpc)" or "(km/s/Mpc)"')



def Omega_m(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_m0'] * a**(-3) / (EHubble(a, pars))**2


def k2phi(a, k, X, pars=cosmo_parameters):
    """
    Gravitational potential as a function of the scale factor
    and wave-number. Perturbations must also be provided.
    """
    delta_m, theta_m = X
    H = Hubble(a,pars=pars)
    factor = -1.5 * (pars['H0 (1/Mpc)'])**2
    matter_term = pars['Omega_m0']*(delta_m/a + 3*H/k**2 * theta_m)
    return factor * matter_term


def k2dphida(a, k, X, pars=cosmo_parameters, method='num', delta=1e-6):
    """
    Derivative of gravitational potential wrt the scale factor,
    as a function of the scale factor and wave-number. Perturbations 
    must also be provided. The flag "method" allows for the user to 
    choose how the derivative is considered, ie, numeric diff or analytic.
    For the numeric diff, user must provide the array with
    scale factors. 
    """
    delta_m, theta_m = X
    if method == 'anl':
        factor = 1.5 * pars['H0 (1/Mpc)'] / EHubble(a, pars)
        matter_term = pars['Omega_m0'] * theta_m / (a**3)
        return factor*matter_term - k2phi(a, k, X, pars)/a
    elif method == 'num':
        da = delta * a
        phi_plus = k2phi(a + da, k, X, pars)
        phi_minus = k2phi(a - da, k, X, pars)
        return (phi_plus - phi_minus) / (2 * da)


def rhs_pert(a, X, k, pars=cosmo_parameters, method='num'):
    """
    This function gives the rhs of the system of equations to be solved.
    """
    H = Hubble(a,pars)
    delta_m, theta_m = X

    k2phi_pot = k2phi(a, k, X, pars=pars)
    k2dphi_potda = k2dphida(a, k, X, pars=pars, method=method)

    output = [-theta_m/(a**2 * H) + 3*k2dphi_potda/(k**2),
                -theta_m/a + k2phi_pot/(a**2 * H)]
        
    return np.array(output)