import numpy as np
from math import exp, log, sqrt, pi
from scipy.integrate import solve_ivp

# ----- Parámetros y constantes -----
om0ref = 0.35
href   = 0.65
H0     = href / 3000.0
acut = 1.0e-3
a0   = 1.0
wmref = 0.0           # w_m(a) = 0
w0ref = -1.5
w1ref =  1.0
# Ojo: en el original "omref" se usa para la DE de hoy
omref = 0.65          # Ω_DE,0  (tal como aparece en el notebook)
G     = 1.0
cs = 1.0

# ----- EoS y escalado de la energía oscura -----
def wl(a):
    # w_l(a) = w0 + w1 (1 - a)
    return w0ref + w1ref * (1.0 - a)
def wde_factor(a):
    """
    Factor de escalado de ρ_DE(a):
    exp[-3 ∫_1^a (1 + w_l(x))/x dx] = a^{-3(1+w0+w1)} * exp(3 w1 (a - 1))
    """
    return (a ** (-3.0 * (1.0 + w0ref + w1ref))) * exp(3.0 * w1ref * (a - 1.0))
# ----- Densidades -----
rhoc = (3.0 * H0**2) / (8.0 * pi * G)
def rhol(a):
    # ρ_DE(a) = ρ_c * Ω_DE,0 * wde_factor(a)
    return rhoc * omref * wde_factor(a)
def rhom(a):
    # ρ_m(a) = ρ_c * Ω_m,0 * a^{-3}
    return rhoc * om0ref * (a ** -3.0)
# ----- H(a) -----
def hub(a, om):
    """
    H(a) = H0 * sqrt( om * a^{-3} + omref * wde_factor(a) )
    At a=1: hub(1, om0ref) = H0 * sqrt(om0ref + omref) = H0 * sqrt(1) = H0.
    """
    return H0 * sqrt(om * (a ** -3.0) + omref * wde_factor(a))
# ----- Sistema de EDOs -----
def rhs(a, y, k):
    """
    y = [dm, vm, dl, vl, g]
    Ecuaciones (idénticas a las de Mathematica):
      dm' = -vm/(H a^2) + 3 g'
      vm' = -vm/a + (k^2) g/(H a^2)
      dl' = -vl/(H a^2) + 3(1+w_l) g' - 3 (cs - w_l) dl / a
      vl' = -(1 - 3 w_l) vl / a + (k^2) cs dl/(H a^2) + (1+w_l) (k^2) g/(H a^2)
      g'  + [(k^2)/(3 a^2 H^2) + 1] g / a = -(ρ_m dm + ρ_l dl)/(2 a (ρ_l + ρ_m))
    """
    dm, vm, dl, vl, g = y
    H   = hub(a, om0ref)
    wl_a = wl(a)
    rho_m = rhom(a)
    rho_l = rhol(a)
    # g' a partir de la ecuación de restricción
    coeff = ((k**2) / (3.0 * a**2 * H**2) + 1.0) * (g / a)
    source = (rho_m * dm + rho_l * dl) / (2.0 * a * (rho_l + rho_m))
    gprime = - coeff - source
    dmprime = - vm / (H * a**2) + 3.0 * gprime
    vmprime = - vm / a + (k**2) * g / (H * a**2)
    dlprime = - vl / (H * a**2) + 3.0 * (1.0 + wl_a) * gprime - 3.0 * (cs**2 - wl_a) * dl / a
    vlprime = - (1.0 - 3.0 * wl_a) * vl / a \
              + (k**2) * cs**2 * dl / (H * a**2) \
              + (1.0 + wl_a) * (k**2) * g / (H * a**2)
    return [dmprime, vmprime, dlprime, vlprime, gprime]
def bothpre(k, a_max=0.995, rtol=1e-8, atol=1e-12):
    """
    Integra el sistema desde a=acut hasta a=a_max (por defecto 0.995).
    Devuelve el objeto de solución de solve_ivp con salida densa.
    """
    # Condiciones iniciales (a = acut)
    dm0 = (3.0/4.0) * 1.0e-5
    vm0 = 0.0
    dl0 = 0.0
    vl0 = 0.0
    g0  = -0.5e-5
    y0 = [dm0, vm0, dl0, vl0, g0]
    sol = solve_ivp(
        fun=lambda a, y: rhs(a, y, k),
        t_span=(acut, a_max),
        y0=y0,
        method="RK45",         # Puedes cambiar a "Radau" si sospechas rigidez
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=np.inf
    )
    return sol
# ----- Ejemplo de uso -----
if __name__ == "__main__":
    k_example = 0.1  # elige el k que necesites
    sol = bothpre(k_example)
    # Evaluar en una rejilla de a
    a_grid = np.linspace(acut, 0.995, 500)
    dm, vm, dl, vl, g = sol.sol(a_grid)
    # dm, vm, dl, vl, g están listos para analizar o graficar
    print(dm[-1], g[-1])