import numpy as np
from astropy import constants as const


# ===========================
# Set cosmological parameters
# ===========================

cosmo_parameters = {
    'H0': 67.,
    'h' : .67,
    'Omega_c0': 0.25,
    'Omega_b0': 0.05,
    'Omega_m0': 0.3,
    'Omega_de0': 0.7,
    'w_de' : -1.,
    'w_de_prime': 0.,
    'cs2': 1.
}


sol = const.c.to('km/s').value


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
    Omde = pars['Omega_de0']
    w = pars['w_de']
    return np.sqrt(Omm*a**-3 + Omde*a**(-3*(1+w)))


def Hubble(a, pars=cosmo_parameters):
    """
    Hubble parameter as a function of
    the scale factor. Cosmological parameters must
    be provided as a dictionary. Units are (1/Mpc).
    """
    return pars['H0']*EHubble(a, pars) / sol


def Omega_m(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_m0'] * a**-3 / (EHubble(a, pars))**2


def Omega_de(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_de0'] * a **(-3*(1+pars['w_de'])) / (EHubble(a, pars))**2


def phi(a, k, X, pars=cosmo_parameters):
    """
    Gravitational potential as a function of the scale factor
    and wave-number. Perturbations must also be provided.
    """
    delta_m, theta_m = X
    H = Hubble(a,pars=pars)
    factor = -1.5* (a*H/k)**2
    matter_term = Omega_m(a, pars=pars) * (delta_m + 3*a*H/k**2 * theta_m)

    return factor * (matter_term)


def dphida(a, k, X, pars=cosmo_parameters):
    """
    Derivative of gravitational potential wrt the scale factor,
    as a function of the scale factor and wave-number. Perturbations 
    must also be provided.
    """
    delta_m, theta_m = X
    H = Hubble(a,pars=pars)
    factor = -1.5* H/k**2
    matter_term = Omega_m(a, pars=pars)*theta_m
    return factor*(matter_term) - phi(a, k, X, pars=pars)/a


def rhs_pert(a, X, k=None, pars=cosmo_parameters):
    """
    This function gives the rhs of the system of equations to be solved.
    """
    if k is None:
        k = pars['H0']

    w_de = pars['w_de']
    cs2 = pars['cs2']
    w_de_prime = pars['w_de_prime']

    H = Hubble(a,pars)
    delta_m, theta_m = X

    phi_pot = phi(a, k, X, pars=pars)
    dphi_potda = dphida(a, k, X, pars=pars)

    output = [-theta_m/(a**2 * H) + 3*dphi_potda,
                  -theta_m/a + k**2 * phi_pot/(a**2 * H)]

    return np.array(output)