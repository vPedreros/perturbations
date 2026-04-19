"""
Tests for background cosmology functions in VP_pert.py and pert_onlymatter.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import VP_pert as vp
import pert_onlymatter as pm


# ------------------------------------------------------------------ helpers
def pars_lcdm():
    p = dict(vp.cosmo_parameters)
    p['w0'] = -1.0
    p['wa'] = 0.0
    return p

def pars_w0wa():
    p = dict(vp.cosmo_parameters)
    p['w0'] = -0.7
    p['wa'] = 0.2
    return p


# ------------------------------------------------------------------ VP_pert
class TestEHubble:
    def test_unity_at_a1_lcdm(self):
        E = vp.EHubble(1.0, pars=pars_lcdm())
        assert abs(E - 1.0) < 1e-10, f"E(a=1) = {E}, expected 1"

    def test_unity_at_a1_w0wa(self):
        E = vp.EHubble(1.0, pars=pars_w0wa())
        assert abs(E - 1.0) < 1e-10, f"E(a=1) = {E}, expected 1"

    def test_matter_dominated_scaling(self):
        """At small a in matter-dominated universe: E ~ sqrt(Omega_m0) * a^{-3/2}"""
        p = pars_lcdm()
        p['Omega_m0'] = 1.0
        a = 1e-3
        E = vp.EHubble(a, pars=p)
        expected = np.sqrt(p['Omega_m0']) * a**(-1.5)
        assert abs(E / expected - 1.0) < 1e-6

    def test_array_input(self):
        aa = np.array([0.1, 0.5, 1.0])
        E = vp.EHubble(aa, pars=pars_lcdm())
        assert E.shape == (3,)
        assert abs(E[-1] - 1.0) < 1e-10

    def test_decreases_toward_present(self):
        """E(a) should decrease as a increases (expanding universe decelerates in H)."""
        aa = np.linspace(0.1, 1.0, 50)
        E = vp.EHubble(aa, pars=pars_lcdm())
        assert np.all(np.diff(E) <= 0)


class TestHubble:
    def test_H0_at_a1(self):
        p = pars_lcdm()
        H = vp.Hubble(1.0, pars=p, units='1/Mpc')
        assert abs(H / p['H0 (1/Mpc)'] - 1.0) < 1e-10

    def test_units_ratio(self):
        """H in km/s/Mpc = H in 1/Mpc * c_kms."""
        from astropy import constants as const
        c_kms = const.c.to('km/s').value
        p = pars_lcdm()
        a = 0.5
        H_inv = vp.Hubble(a, pars=p, units='1/Mpc')
        H_kms = vp.Hubble(a, pars=p, units='km/s/Mpc')
        assert abs(H_kms / (H_inv * c_kms) - 1.0) < 1e-10

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError):
            vp.Hubble(1.0, units='furlongs/fortnight')


class TestOmegaSum:
    def test_flat_universe_at_a1(self):
        """Omega_m(1) + Omega_de(1) should be 1 (flat universe)."""
        p = pars_lcdm()
        total = vp.Omega_m(1.0, pars=p) + vp.Omega_de(1.0, pars=p)
        assert abs(total - 1.0) < 1e-10

    def test_flat_universe_array(self):
        aa = np.linspace(0.1, 1.0, 100)
        total = vp.Omega_m(aa) + vp.Omega_de(aa)
        assert np.allclose(total, 1.0, atol=1e-10)

    def test_w0wa_flat(self):
        p = pars_w0wa()
        aa = np.linspace(0.1, 1.0, 50)
        total = vp.Omega_m(aa, pars=p) + vp.Omega_de(aa, pars=p)
        assert np.allclose(total, 1.0, atol=1e-10)


class TestEffWDE:
    def test_lcdm_constant(self):
        """For Lambda (w0=-1, wa=0), eff_w_de = -1 everywhere."""
        p = pars_lcdm()
        for a in [0.1, 0.5, 1.0]:
            assert abs(vp.eff_w_de(a, pars=p) + 1.0) < 1e-12

    def test_continuity_at_a1(self):
        """eff_w_de must equal w0 at a=1 and be continuous."""
        p = pars_w0wa()
        eps = 1e-6
        left  = vp.eff_w_de(1.0 - eps, pars=p)
        right = vp.eff_w_de(1.0 + eps, pars=p)
        center = vp.eff_w_de(1.0, pars=p)
        assert abs(center - p['w0']) < 1e-8
        assert abs(left - right) < 1e-4

    def test_array_input(self):
        p = pars_w0wa()
        aa = np.array([0.3, 0.6, 1.0])
        result = vp.eff_w_de(aa, pars=p)
        assert hasattr(result, '__len__') and len(result) == 3

    def test_formula_at_specific_a(self):
        """eff_w_de formula matches explicit calculation at a=0.5."""
        p = pars_w0wa()
        a = 0.5
        expected = p['w0'] + p['wa'] * (1.0 - (a - 1.0) / np.log(a))
        val = vp.eff_w_de(a, pars=p)
        assert abs(val - expected) < 1e-12


class TestDensities:
    def test_rho_m_scales_as_a3(self):
        """rho_m ∝ a^{-3}."""
        a1, a2 = 0.5, 1.0
        r = vp.rho_m(a1) / vp.rho_m(a2)
        assert abs(r - (a2/a1)**3) < 1e-8

    def test_rho_de_constant_for_lcdm(self):
        """For w=-1 (cosmological constant): rho_de is constant."""
        a1, a2 = 0.3, 0.8
        r = vp.rho_de(a1) / vp.rho_de(a2)
        assert abs(r - 1.0) < 1e-8

    def test_rho_cr_positive(self):
        for a in [0.1, 0.5, 1.0]:
            assert vp.rho_cr(a) > 0


# ---------------------------------------------------------------- pert_onlymatter
class TestPertOnlyMatter:
    def test_EHubble_unity_at_a1(self):
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        E = pm.EHubble(1.0, pars=p)
        assert abs(E - 1.0) < 1e-10

    def test_Omega_m_unity_matter_only(self):
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        aa = np.linspace(0.1, 1.0, 50)
        Om = pm.Omega_m(aa, pars=p)
        assert np.allclose(Om, 1.0, atol=1e-10)

    def test_k2phi_negative_for_positive_delta(self):
        """k^2 phi should be negative for positive density perturbation."""
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        k2phi_val = pm.k2phi(0.5, 0.1, [1e-3, 0.0], pars=p)
        assert k2phi_val < 0

    def test_k2dphida_anl_is_finite(self):
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        anl = pm.k2dphida(0.5, 0.1, [1e-3, 5e-5], pars=p, method='anl')
        assert np.isfinite(anl)

    def test_k2dphida_num_is_finite(self):
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        num = pm.k2dphida(0.5, 0.1, [1e-3, 5e-5], pars=p, method='num')
        assert np.isfinite(num)

    def test_k2dphida_num_matches_manual_finite_diff(self):
        """Numeric method should reproduce a manual central-difference calculation."""
        p = dict(pm.cosmo_parameters)
        p['Omega_m0'] = 1.0
        a, k, delta_step = 0.5, 0.1, 1e-6
        X = [1e-3, 5e-5]
        da = delta_step * a
        manual = (pm.k2phi(a+da, k, X, pars=p) - pm.k2phi(a-da, k, X, pars=p)) / (2*da)
        num = pm.k2dphida(a, k, X, pars=p, method='num')
        assert abs((num - manual) / (manual + 1e-30)) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
