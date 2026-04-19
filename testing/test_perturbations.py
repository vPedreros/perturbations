"""
Tests for perturbation equations: growth rates, ODE solutions, and physical consistency.

Note on horizon scale: for k in Mpc^{-1} and H0~2.2e-4 Mpc^{-1}, the comoving
Hubble radius is aH = H0*sqrt(Omega_m0)*a^{-1/2}.  A mode enters the Hubble radius
(becomes sub-horizon) when k = aH.  We use k=0.1 Mpc^{-1} for growth tests;
this mode enters the Hubble radius at a ~ 5e-6, so it is safely sub-Hubble at
a_ini = 1e-4 used in these tests.  Super-Hubble modes (k~1e-3) do NOT show the
simple δ ∝ a growth and are excluded from the growth-rate tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import VP_pert as vp
import pert_onlymatter as pm


# ------------------------------------------------------------------ helpers
def lcdm_pars():
    p = dict(vp.cosmo_parameters)
    p['w0'] = -1.0
    p['wa'] = 0.0
    p['Omega_m0'] = 0.3
    return p

def matter_only_pars_vp():
    p = dict(vp.cosmo_parameters)
    p['w0'] = -1.0
    p['wa'] = 0.0
    p['Omega_m0'] = 1.0
    return p

def matter_only_pars_pm():
    p = dict(pm.cosmo_parameters)
    p['Omega_m0'] = 1.0
    return p

def solve_vp(pars, k, a_ini=1e-4, a_end=1.0):
    phi_ini = -1e-9
    H_ini = vp.Hubble(a_ini, pars)
    H_conf = a_ini * H_ini
    delta_ini = -2.0 * phi_ini * (1.0 + k**2 / (3.0 * H_conf**2))
    vm_ini = (2.0 * k**2 / (3.0 * H_conf)) * phi_ini
    X0 = [delta_ini, vm_ini, 0.0, 0.0, phi_ini]
    return solve_ivp(vp.rhs_pert, (a_ini, a_end), X0,
                     args=(k, pars), method='RK45',
                     dense_output=True, rtol=1e-8, atol=1e-11)

def solve_pm(pars, k, a_ini=1e-4, a_end=1.0):
    delta_ini = 1e-4
    X0 = [delta_ini, -pm.Hubble(a_ini, pars)*a_ini*delta_ini]
    return solve_ivp(lambda a, X: pm.rhs_pert(a, X, k, pars),
                     (a_ini, a_end), X0, method='RK45',
                     dense_output=True, rtol=1e-8, atol=1e-11)


# ------------------------------------------------------------------ rhs shape
class TestRhsShape:
    def test_vp_rhs_returns_5(self):
        p = lcdm_pars()
        X = [1e-4, 1e-5, 0.0, 0.0, -1e-9]
        out = vp.rhs_pert(0.5, X, k=0.1, pars=p)
        assert len(out) == 5

    def test_pm_rhs_returns_2(self):
        p = matter_only_pars_pm()
        X = [1e-4, 1e-5]
        out = pm.rhs_pert(0.5, X, k=0.1, pars=p)
        assert len(out) == 2

    def test_vp_rhs_lcdm_de_components_zero(self):
        """In LCDM the rhs of DE perturbations is zero (they stay frozen)."""
        p = lcdm_pars()
        X = [1e-4, 1e-5, 0.0, 0.0, -1e-9]
        out = vp.rhs_pert(0.5, X, k=0.1, pars=p)
        assert out[2] == 0.0
        assert out[3] == 0.0


# ------------------------------------------------------------------ k2phi / k2dphida
class TestPotentialFunctions:
    def test_k2phi_accepts_5element_X(self):
        p = lcdm_pars()
        X = [1e-4, 1e-5, 0.0, 0.0, -1e-9]
        result = vp.k2phi(0.5, 0.1, X, pars=p)
        assert np.isfinite(result)

    def test_k2dphida_accepts_5element_X(self):
        """k2dphida must accept a 5-element X (bug was it crashed with 5 or 4 elements)."""
        p = lcdm_pars()
        X = [1e-4, 1e-5, 0.0, 0.0, -1e-9]
        result = vp.k2dphida(0.5, 0.1, X, pars=p)
        assert np.isfinite(result)

    def test_k2phi_negative_for_positive_delta(self):
        """Positive density contrast → negative gravitational potential."""
        p = lcdm_pars()
        X = [1e-3, 0.0, 0.0, 0.0, -1e-9]
        phi_val = vp.k2phi(0.5, 0.1, X, pars=p)
        assert phi_val < 0

    def test_k2phi_scales_with_k2(self):
        """k2phi = k^2 * phi, so k2phi/k^2 should be constant (phi doesn't depend on k explicitly)."""
        p = lcdm_pars()
        X = [1e-3, 0.0, 0.0, 0.0, -1e-9]
        a = 0.5
        k2phi_large = vp.k2phi(a, 10.0, X, pars=p)
        k2phi_small = vp.k2phi(a, 0.01, X, pars=p)
        # The delta_m/a term in k2phi doesn't depend on k; the 3H/k^2*vm term does
        # With vm=0, k2phi should be independent of k
        assert abs(k2phi_large / k2phi_small - 1.0) < 1e-10


# ------------------------------------------------------------------ ODE solve
class TestODESolve:
    def test_vp_lcdm_solve_success(self):
        p = lcdm_pars()
        sol = solve_vp(p, k=0.1)
        assert sol.success, sol.message

    def test_vp_matter_only_solve_success(self):
        p = matter_only_pars_vp()
        sol = solve_vp(p, k=0.1)
        assert sol.success, sol.message

    def test_pm_solve_success(self):
        p = matter_only_pars_pm()
        sol = solve_pm(p, k=0.1)
        assert sol.success, sol.message

    def test_delta_m_grows_sub_hubble(self):
        """Sub-Hubble matter density contrast should grow from early to late times."""
        p = matter_only_pars_pm()
        sol = solve_pm(p, k=0.1)   # sub-Hubble mode
        assert sol.success
        delta = sol.y[0]
        assert delta[-1] > delta[0], f"delta went from {delta[0]:.3e} to {delta[-1]:.3e}"

    def test_delta_m_positive_throughout(self):
        """For sub-Hubble growing-mode IC, delta_m should stay positive."""
        p = matter_only_pars_pm()
        sol = solve_pm(p, k=0.1)
        assert sol.success
        assert np.all(sol.y[0] > 0)


# ------------------------------------------------------------------ growth rate
class TestMatterDominatedGrowth:
    """
    In a matter-dominated universe on sub-Hubble scales, δ_m ∝ a (growing mode).
    We use k=0.1 Mpc^{-1} which is sub-Hubble for a ≥ 1e-4 (entry at a~5e-6).
    """
    def test_pm_delta_scales_linearly_in_a(self):
        p = matter_only_pars_pm()
        sol = solve_pm(p, k=0.1, a_ini=1e-4, a_end=0.5)
        assert sol.success
        aa = np.linspace(1e-4, 0.5, 200)
        d = sol.sol(aa)[0]
        assert np.all(d > 0), "delta went negative — initial conditions may be wrong"
        slope = np.polyfit(np.log(aa), np.log(d), 1)[0]
        assert abs(slope - 1.0) < 0.05, f"δ_m ∝ a^{slope:.3f}, expected ~1"

    def test_vp_matter_only_growth_rate(self):
        """VP_pert matter-only should also show δ_m ∝ a on sub-Hubble scales."""
        p = matter_only_pars_vp()
        sol = solve_vp(p, k=0.1, a_ini=1e-4, a_end=0.5)
        assert sol.success
        aa = np.linspace(1e-4, 0.5, 200)
        d = sol.sol(aa)[0]
        assert np.all(d > 0)
        slope = np.polyfit(np.log(aa), np.log(d), 1)[0]
        assert abs(slope - 1.0) < 0.05, f"δ_m ∝ a^{slope:.3f}, expected ~1"


# ------------------------------------------------------------------ phi conservation
class TestPhiConservation:
    """
    On super-Hubble scales (k << aH), the gravitational potential is nearly
    constant during matter domination.
    """
    def test_phi_nearly_constant_super_hubble(self):
        p = matter_only_pars_vp()
        k = 1e-4  # super-Hubble throughout
        sol = solve_vp(p, k=k, a_ini=1e-4, a_end=0.5)
        assert sol.success
        phi = sol.y[4]
        variation = (np.max(phi) - np.min(phi)) / abs(phi[0])
        assert variation < 0.2, f"phi varies by {100*variation:.1f}%"


# ------------------------------------------------------------------ vp vs pm consistency
class TestVPvsPMConsistency:
    """
    VP_pert and pert_onlymatter should produce the same growth factor D(a) in a
    matter-only universe.  We compare δ(a₂)/δ(a₁) from both codes — the ratio
    should be the same regardless of initial conditions after the decaying mode
    has died away.
    """
    def test_growth_factor_ratio_matches(self):
        k = 0.1
        a_ref, a_end = 0.01, 0.5

        p_vp = matter_only_pars_vp()
        sol_vp = solve_vp(p_vp, k=k)
        assert sol_vp.success

        p_pm = matter_only_pars_pm()
        sol_pm = solve_pm(p_pm, k=k)
        assert sol_pm.success

        # Growth factor from a_ref to a_end
        vp_ref = sol_vp.sol(a_ref)[0]
        vp_end = sol_vp.sol([a_end])[0][0]
        pm_ref = sol_pm.sol(a_ref)[0]
        pm_end = sol_pm.sol([a_end])[0][0]

        D_vp = vp_end / vp_ref
        D_pm = pm_end / pm_ref

        assert abs(D_vp / D_pm - 1.0) < 0.005, \
            f"Growth factor VP={D_vp:.4f}, PM={D_pm:.4f}, ratio={D_vp/D_pm:.5f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
