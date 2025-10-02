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
    'Omega_c0': 0.25,
    'Omega_b0': 0.05,
    'Omega_m0': 0.3,
    'Omega_de0': 0.7,
    'w0' : -1.,
    'wa': 0.,
    'cs2': 1.
}




# ==================================
# Define general cosmology functions.
# ==================================

def w_de(a, pars=cosmo_parameters):
    if pars['wa'] == 0:
        return pars['w0']
    else:
        return pars['w0'] + pars['wa']*(1-a)
w_de = np.vectorize(w_de)


def eff_w_de(a, pars=cosmo_parameters):
    if pars['wa'] == 0:
        return w_de(a, pars=pars)
    else:
        return pars['w0'] + pars['wa'] *(1 - (a-1)/np.log(a))


def EHubble(a, pars=cosmo_parameters):
    """
    Dimensionless Hubble parameter as a function of
    the scale factor. Cosmological parameters must
    be provided as a dictionary.
    """
    Omm = pars['Omega_m0']
    Omde = pars['Omega_de0']
    return np.sqrt(Omm*a**-3 + Omde*a**(-3*(1+eff_w_de(a, pars=pars))))


def Hubble(a, pars=cosmo_parameters, units='1/Mpc'):
    """
    Hubble parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    You can choose units by setting the flag units to '1/Mpc' or 
    'km/s/Mpc'
    """
    if units == '1/Mpc':
        return pars['H0 (1/Mpc)'] * EHubble(a, pars=pars)
    elif units == 'km/s/Mpc' or 'Km/s/Mpc':
        return pars['H0 (km/s/Mpc)'] * EHubble(a, pars=pars)
    else: 
        raise('Invalid selection of units, please choose between "(1/Mpc)" and "(km/s/Mpc)"')

def Omega_m(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_m0'] * a**(-3) / (EHubble(a, pars))**2


def Omega_de(a, pars=cosmo_parameters):
    """
    Density parameter as a function of the scale factor.
    Cosmological parameters must be provided as a dictionary.
    """
    return pars['Omega_de0'] * a **(-3*(1+eff_w_de(a, pars=pars))) / (EHubble(a, pars))**2


def phi(a, k, X, pars=cosmo_parameters):
    """
    Gravitational potential as a function of the scale factor
    and wave-number. Perturbations must also be provided.
    """
    delta_m, theta_m, delta_de, theta_de = X
    H = Hubble(a,pars)
    factor = -1.5* (pars['H0 (1/Mpc)']/k)**2
    matter_term = pars['Omega_m0'] * (delta_m/a + 3*H/k**2 * theta_m)
    if pars['w0'] == -1 and pars['wa'] == 0:
        de_term = 0
    else:
        print('not In LambdaCDM')
        de_term = pars['Omega_de0'] * a ** (-3*eff_w_de(a, pars)) \
                    * (delta_de/a + 3*H/k**2 * (1+w_de(a, pars))* theta_de)
    return factor * (matter_term + de_term)


def dphida(a, k, X, pars=cosmo_parameters):
    """
    Derivative of gravitational potential wrt the scale factor,
    as a function of the scale factor and wave-number. Perturbations 
    must also be provided.
    """
    delta_m, theta_m, delta_de, theta_de = X
    factor = -1.5*pars['H0 (1/Mpc)']/(EHubble(a, pars)*k**2)
    matter_term = pars['Omega_m0'] * a**(-3)*theta_m
    if pars['w0'] == -1 and pars['wa'] == 0:
        de_term = 0
    else:
        print('not In LambdaCDM')
        de_term = pars['Omega_de0'] * a **(-3*(1+eff_w_de(a, pars=pars)))\
                    * theta_de*(1+w_de(a, pars=pars))
    return factor*(matter_term+de_term) - phi(a, k, X, pars=pars)/a


def rhs_pert(a, X, k, pars=cosmo_parameters):
    """
    This function gives the rhs of the system of equations to be solved.
    """

    w = w_de(a, pars=pars)
    cs2 = pars['cs2']
    w_de_prime = -pars['wa']

    H = Hubble(a,pars)
    delta_m, theta_m, delta_de, theta_de = X

    phi_pot = phi(a, k, X, pars=pars)
    dphi_potda = dphida(a, k, X, pars=pars)
    if pars['w0'] == -1 and pars['wa'] == 0:
        output = [-theta_m/(a**2 * H) + 3*dphi_potda,
                  -theta_m/a + k**2 * phi_pot/(a**2 * H),
                  0,
                  0]
    else: 
        print('not In LambdaCDM')
        output = [-theta_m/(a**2 * H) + 3*dphi_potda,
                  -theta_m/a + k**2 * phi_pot/(a**2 * H),
                  -(1+w)*(theta_de/(a**2 * H) - 3*dphi_potda) - 3/a*(cs2-w)*delta_de,
                  -1/a*(1-3*w)*theta_de-w_de_prime/(1+w)*theta_de + (k**2)/(H*(a**2))*(cs2/(1+w)*delta_de + phi_pot)]
        
    return np.array(output)

