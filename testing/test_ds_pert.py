"""
Tests for DS_pert.py background and perturbation functions.

DS_pert uses unusual EoS parameters (w0=-1.5, w1=1.0) so the effective w at
early times is w0+w1 = -0.5.  Standard δ_m ∝ a growth applies only when matter
strictly dominates, which here is partially broken by the unusual DE component.
Tests focus on background normalization, finite perturbations, and general
growth (not the exact δ_m ~ a slope).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from math import sqrt, pi
from scipy.integrate import solve_ivp
import DS_pert as ds


# ------------------------------------------------------------------ background
class TestHubNormalization:
    def test_hub_unity_at_a1(self):
        """hub(1, om0ref) / H0 must equal 1 (correct normalization after bug fix)."""
        E = ds.hub(1.0, ds.om0ref) / ds.H0
        assert abs(E - 1.0) < 1e-10, f"hub(1)/H0 = {E:.6f}, expected 1"

    def test_hub_positive_for_valid_a(self):
        for a in [0.01, 0.1, 0.5, 1.0]:
            assert ds.hub(a, ds.om0ref) > 0

    def test_hub_decreases_toward_present(self):
        """H(a) should be larger at smaller a (universe was denser in the past)."""
        aa = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
        H_vals = np.array([ds.hub(a, ds.om0ref) for a in aa])
        # H_vals should be decreasing (larger at small a, smaller at large a)
        assert np.all(np.diff(H_vals) <= 0), \
            f"H values {H_vals} are not monotonically decreasing"

    def test_omega_sum_at_a1(self):
        """At a=1: Omega_m + Omega_DE = (rho_m + rho_DE) / rho_c = 1."""
        total = (ds.rhom(1.0) + ds.rhol(1.0)) / ds.rhoc
        assert abs(total - 1.0) < 1e-10, f"Omega_tot(1) = {total:.6f}"


class TestWdeFactorAndDensities:
    def test_wde_factor_at_a1(self):
        """wde_factor(1) = 1 by construction (integral anchored at a=1)."""
        assert abs(ds.wde_factor(1.0) - 1.0) < 1e-12

    def test_rhoc_positive(self):
        assert ds.rhoc > 0

    def test_rhom_scales_as_a3(self):
        """rho_m ∝ a^{-3}."""
        r = ds.rhom(0.5) / ds.rhom(1.0)
        assert abs(r - 0.5**(-3)) < 1e-10


class TestWlEquationOfState:
    def test_wl_at_a1_equals_w0(self):
        assert abs(ds.wl(1.0) - ds.w0ref) < 1e-12

    def test_wl_at_a0_is_w0_plus_w1(self):
        assert abs(ds.wl(0.0) - (ds.w0ref + ds.w1ref)) < 1e-12

    def test_wl_is_linear_in_a(self):
        """w_l(a) = w0 + w1*(1-a) is linear; slope must be -w1."""
        a1, a2 = 0.2, 0.8
        slope = (ds.wl(a2) - ds.wl(a1)) / (a2 - a1)
        assert abs(slope + ds.w1ref) < 1e-12


# ------------------------------------------------------------------ perturbations
class TestRhsDS:
    def test_rhs_returns_five_components(self):
        y = [1e-4, 0.0, 0.0, 0.0, -1e-9]
        out = ds.rhs(0.5, y, k=0.1)
        assert len(out) == 5

    def test_rhs_finite(self):
        y = [1e-4, 1e-6, 1e-6, 1e-7, -1e-9]
        out = ds.rhs(0.5, y, k=0.1)
        assert all(np.isfinite(v) for v in out)

    def test_rhs_zero_perturbations(self):
        """With all perturbations = 0, all derivatives should be 0."""
        y = [0.0, 0.0, 0.0, 0.0, 0.0]
        out = ds.rhs(0.5, y, k=0.1)
        assert all(abs(v) < 1e-30 for v in out)

    def test_rhs_different_k(self):
        """rhs output should change when k changes (k appears in equations)."""
        y = [1e-4, 1e-6, 1e-6, 1e-7, -1e-9]
        out1 = ds.rhs(0.5, y, k=0.01)
        out2 = ds.rhs(0.5, y, k=1.0)
        assert not np.allclose(out1, out2)


class TestODESolveDS:
    def test_bothpre_solves_successfully(self):
        sol = ds.bothpre(k=0.1)
        assert sol.success, sol.message

    def test_delta_matter_grows_overall(self):
        """Matter density contrast should grow from acut to a_max."""
        sol = ds.bothpre(k=0.1)
        assert sol.success
        assert sol.y[0][-1] > sol.y[0][0], \
            f"dm went from {sol.y[0][0]:.3e} to {sol.y[0][-1]:.3e}"

    def test_solution_finite(self):
        sol = ds.bothpre(k=0.1)
        assert sol.success
        assert np.all(np.isfinite(sol.y))

    def test_dense_output_works(self):
        sol = ds.bothpre(k=0.1)
        a_grid = np.linspace(ds.acut, 0.9, 50)
        vals = sol.sol(a_grid)
        assert vals.shape == (5, 50)
        assert np.all(np.isfinite(vals))

    def test_multiple_k_values(self):
        """Solver should succeed for a range of k values."""
        for k in [0.01, 0.1, 0.5]:
            sol = ds.bothpre(k=k)
            assert sol.success, f"k={k}: {sol.message}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
