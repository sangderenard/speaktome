from __future__ import annotations

import math
import unittest

import numpy as np
import sympy as sp

from openrgb_dt_spirograph import (
    RADIAL_DIRECTIONS,
    adaptive_path_phases,
    bake_contribution_lut,
    contribution_matrix,
    hypotrochoid_point,
    hypotrochoid_repeat_period,
    radial_contributions,
    symbolic_wrapped_phase,
)
from openrgb_spectral_field import symbolic_radial_kernel


class RadialContributionTests(unittest.TestCase):
    def test_center_is_dim_and_uniform(self) -> None:
        np.testing.assert_allclose(radial_contributions(0.0, 0.0), np.full(8, 0.25))

    def test_perimeter_alignment_has_half_intensity_direct_neighbors(self) -> None:
        point = tuple(RADIAL_DIRECTIONS[5])  # installed right side
        row = radial_contributions(*point)
        self.assertAlmostEqual(row[5], 1.0)
        self.assertAlmostEqual(row[4], 0.5)
        self.assertAlmostEqual(row[6], 0.5)
        self.assertAlmostEqual(row[3], 0.0)
        self.assertAlmostEqual(row[7], 0.0)

    def test_opposite_light_falls_off_sharply_near_a_light(self) -> None:
        direction = RADIAL_DIRECTIONS[5]
        row = radial_contributions(*(0.82 * direction))
        self.assertEqual(row[5], max(row))
        self.assertLess(row[1], 1.0e-12)

    def test_larger_falloff_power_sharpens_the_neighbor_lobe(self) -> None:
        point = tuple(RADIAL_DIRECTIONS[5])
        quadratic = radial_contributions(*point, falloff_power=2.0)
        quartic = radial_contributions(*point, falloff_power=4.0)
        self.assertAlmostEqual(quadratic[5], 1.0)
        self.assertAlmostEqual(quartic[5], 1.0)
        self.assertLess(quartic[4], quadratic[4])

    def test_distance_kernel_retains_center_control_without_uniform_baseline(self) -> None:
        np.testing.assert_allclose(
            radial_contributions(0.0, 0.0, center_contribution=0.1),
            np.full(8, 0.1),
        )
        direction = RADIAL_DIRECTIONS[5]
        near_opposite = radial_contributions(*(0.4 * direction))[1]
        old_uniform_baseline = 0.25 * (1.0 - 0.4)
        self.assertLess(near_opposite, old_uniform_baseline / 100.0)

    def test_symbolic_and_numeric_distance_kernels_agree(self) -> None:
        x, y, dx, dy, center = sp.symbols("x y dx dy center")
        evaluate = sp.lambdify(
            (x, y, dx, dy, center),
            symbolic_radial_kernel(x, y, dx, dy, center),
            modules="numpy",
        )
        point = (0.31, -0.47)
        expected = radial_contributions(*point)
        actual = np.asarray([
            evaluate(*point, *direction, 0.25)
            for direction in RADIAL_DIRECTIONS
        ])
        np.testing.assert_allclose(actual, expected, atol=1.0e-14)

    def test_arbitrary_sites_generate_float_matrix(self) -> None:
        matrix = contribution_matrix(((0.0, 0.0), (1.0, 0.0), (0.2, -0.3)))
        self.assertEqual(matrix.shape, (3, 8))
        self.assertTrue(np.issubdtype(matrix.dtype, np.floating))
        self.assertTrue(np.all((0.0 <= matrix) & (matrix <= 1.0)))

    def test_outside_point_keeps_unit_direction_without_lobe_broadening(self) -> None:
        np.testing.assert_allclose(
            radial_contributions(10.0, 0.0),
            radial_contributions(1.0, 0.0),
        )

    def test_adaptive_sites_respect_eight_bit_step(self) -> None:
        phases = adaptive_path_phases(color_bits=8)
        points = [hypotrochoid_point(float(phase)) for phase in phases]
        matrix = contribution_matrix(points)
        maximum_step = float(np.max(np.abs(np.diff(matrix, axis=0))))
        self.assertLessEqual(maximum_step, 1.0 / 255.0 + 1.0e-12)
        self.assertGreater(len(phases), 256)
        self.assertAlmostEqual(float(phases[0]), 0.0)
        self.assertAlmostEqual(float(phases[-1]), 6.0 * math.pi)

    def test_lut_closes_at_repeat_epsilon(self) -> None:
        self.assertAlmostEqual(hypotrochoid_repeat_period(), 6.0 * math.pi)
        lut = bake_contribution_lut(color_bits=8, repeat_epsilon=1.0e-9)
        self.assertLessEqual(lut.repeat_error, 1.0e-9)
        self.assertTrue(lut.saturated)
        self.assertIsNone(lut.cutoff_reason)
        np.testing.assert_allclose(lut.contributions[0], lut.contributions[-1], atol=1.0e-9)

    def test_empirical_period_respects_degenerate_circle_geometry(self) -> None:
        # With the pen at the rolling-circle center, the position is simply a
        # base circle even though the unused rolling ratio is 2/3.
        self.assertAlmostEqual(
            hypotrochoid_repeat_period(pen_offset=0.0),
            2.0 * math.pi,
        )

    def test_symbolic_phase_uses_empirical_period_in_both_directions(self) -> None:
        period = hypotrochoid_repeat_period()
        phase, speed, dt, repeat = sp.symbols("phase speed dt repeat", real=True)
        evaluate = sp.lambdify(
            (phase, speed, dt, repeat),
            symbolic_wrapped_phase(phase, speed, dt, repeat),
            modules="numpy",
        )
        self.assertAlmostEqual(
            float(evaluate(2.0 * math.pi - 0.01, 1.0, 0.02, period)),
            2.0 * math.pi + 0.01,
        )
        self.assertAlmostEqual(
            float(evaluate(period - 0.01, 1.0, 0.02, period)),
            0.01,
        )
        self.assertAlmostEqual(
            float(evaluate(0.01, -1.0, 0.02, period)),
            period - 0.01,
        )

    def test_lut_returns_complete_cycle_when_dynamic_budget_cuts_off(self) -> None:
        lut = bake_contribution_lut(color_bits=12, max_sites=65)
        self.assertFalse(lut.saturated)
        self.assertEqual(lut.cutoff_reason, "site_budget")
        self.assertAlmostEqual(float(lut.phases[0]), 0.0)
        self.assertAlmostEqual(float(lut.phases[-1]), 6.0 * math.pi)
        self.assertLessEqual(lut.repeat_error, 1.0e-9)


if __name__ == "__main__":
    unittest.main()
