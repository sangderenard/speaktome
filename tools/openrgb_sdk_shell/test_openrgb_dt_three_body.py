from __future__ import annotations

import math
import unittest

import numpy as np

from openrgb_dt_three_body import (
    _grb_wire_frame,
    _logical_domain_frame,
    _thermal_velocity_scale,
    torus_face_projection,
)
from openrgb_torus_green import (
    build_torus_green_kernel,
    evaluate_green,
    evaluate_green_gradient,
    symbolic_torus_domain,
    symbolic_torus_laplace_beltrami,
)
from openrgb_spectral_field import (
    CMY_SPECTRA,
    WHITE_SPECTRUM,
    accumulate_spectral_field,
    raytracer_display_matrix,
    spectral_field_to_linear_rgb,
)


class ThreeBodyColorTests(unittest.TestCase):
    def test_cmy_are_nonnegative_spectra_in_raytracer_basis(self) -> None:
        self.assertEqual(CMY_SPECTRA.shape, (3, 8))
        self.assertTrue(np.all(CMY_SPECTRA >= 0.0))
        linear_rgb = CMY_SPECTRA @ raytracer_display_matrix()
        np.testing.assert_allclose(linear_rgb[0, 0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(linear_rgb[1], (1.0, 0.0, 1.0), atol=1.0e-12)
        np.testing.assert_allclose(linear_rgb[2, 2], 0.0, atol=1.0e-12)

    def test_spectra_add_before_display_projection(self) -> None:
        kernel = np.asarray(((0.25, 0.5, 0.75), (1.0, 0.0, 0.5)))
        field = accumulate_spectral_field(kernel, CMY_SPECTRA)
        direct = spectral_field_to_linear_rgb(field)
        expected = kernel @ (CMY_SPECTRA @ raytracer_display_matrix())
        np.testing.assert_allclose(direct, expected, atol=1.0e-12)

    def test_composite_magenta_retains_two_spectral_lines(self) -> None:
        active = np.flatnonzero(CMY_SPECTRA[1] > 0.0)
        np.testing.assert_array_equal(active, (1, 7))

    def test_spirograph_white_is_a_spectrum_not_three_rgb_constants(self) -> None:
        self.assertGreater(np.count_nonzero(WHITE_SPECTRUM), 1)
        np.testing.assert_allclose(
            WHITE_SPECTRUM @ raytracer_display_matrix(),
            (1.0, 1.0, 1.0),
            atol=1.0e-12,
        )

    def test_grb_wire_compensation_swaps_red_and_green(self) -> None:
        self.assertEqual(_grb_wire_frame([(10, 20, 30)]), [(20, 10, 30)])

    def test_view_reads_selected_slots_not_full_frame_prefix(self) -> None:
        from openrgb_dt_walk import AddressDomain
        from openrgb_shell import DomainAddress

        domain = AddressDomain(tuple(
            DomainAddress(i, i + 4, 100 + i, 2, "ring", i)
            for i in range(8)
        ), frame_count=12)
        frame = [(0, 0, 0)] * 4 + [(i, i + 1, i + 2) for i in range(8)]
        self.assertEqual(
            _logical_domain_frame(domain, frame),
            [(i, i + 1, i + 2) for i in range(8)],
        )

    def test_temperature_ratio_scales_kinetic_energy_in_kelvin(self) -> None:
        previous = 40.0
        current = 80.0
        scale = _thermal_velocity_scale(previous, current)
        self.assertAlmostEqual(
            scale**2,
            (current + 273.15) / (previous + 273.15),
            places=14,
        )
        self.assertGreater(scale, 1.0)
        self.assertLess(_thermal_velocity_scale(current, previous), 1.0)

    def test_parametric_domain_derives_exact_torus_metric(self) -> None:
        import sympy as sp

        domain, (u, v, major, minor) = symbolic_torus_domain()
        self.assertEqual(
            sp.simplify(domain.metric()),
            sp.diag((major + minor * sp.cos(v)) ** 2, minor**2),
        )

    def test_repository_laplace_beltrami_annihilates_constants(self) -> None:
        import sympy as sp

        self.assertEqual(symbolic_torus_laplace_beltrami(sp.Integer(1)), 0)

    def test_abstract_tensor_green_solve_is_symmetric_and_invertible(self) -> None:
        kernel = build_torus_green_kernel()
        np.testing.assert_allclose(kernel.stiffness, kernel.stiffness.T, atol=2e-13)
        np.testing.assert_allclose(
            kernel.stiffness @ kernel.inverse_stiffness,
            np.eye(kernel.basis_count),
            atol=2e-12,
        )
        self.assertGreater(float(np.linalg.eigvalsh(kernel.stiffness)[0]), 0.0)

    def test_green_kernel_is_symmetric_and_exactly_periodic(self) -> None:
        q = (0.37, -1.14)
        p = (-2.22, 2.73)
        expected = evaluate_green(*q, *p)
        self.assertAlmostEqual(expected, evaluate_green(*p, *q), places=13)
        self.assertAlmostEqual(
            expected, evaluate_green(q[0] + 2 * math.pi, q[1], *p), places=13
        )
        self.assertAlmostEqual(
            expected, evaluate_green(q[0], q[1] - 2 * math.pi, *p), places=13
        )

    def test_green_force_has_no_principal_wrap_discontinuity(self) -> None:
        epsilon = 1.0e-8
        source = (0.4, 0.2)
        left = evaluate_green_gradient(source[0] + math.pi - epsilon, 0.7, *source)
        right = evaluate_green_gradient(source[0] - math.pi + epsilon, 0.7, *source)
        np.testing.assert_allclose(left, right, atol=2e-7)

    def test_local_green_force_points_toward_source(self) -> None:
        gradient_u, _gradient_v = evaluate_green_gradient(0.1, 0.0, 0.0, 0.0)
        self.assertLess(gradient_u, 0.0)

    def test_face_projection_is_an_annulus_strictly_inside_light_ring(self) -> None:
        outer = torus_face_projection(0.0, 0.0, 1.5, 0.5, 0.82)
        inner = torus_face_projection(0.0, math.pi, 1.5, 0.5, 0.82)
        self.assertAlmostEqual(float(outer[0]), 0.82, places=14)
        self.assertAlmostEqual(float(inner[0]), 0.41, places=14)
        self.assertEqual(float(outer[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
