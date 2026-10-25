from __future__ import annotations

import colorsys
from types import SimpleNamespace
import unittest

import numpy as np

from openrgb_dt_walk import AddressDomain
from openrgb_field_studio import (
    StudioState,
    _send_all_off,
    apply_spirograph_ratio,
    hue_spectrum,
    remap_full_frame,
    remap_ring_colors,
    spectral_source_frame,
    spirograph_geometry_from_ratio,
    spirograph_speed_limit,
)
from openrgb_shell import DomainAddress
from openrgb_spectral_field import RADIAL_DIRECTIONS, raytracer_display_matrix


class FieldStudioTests(unittest.TestCase):
    def test_ratio_controls_map_to_hypotrochoid_geometry(self) -> None:
        self.assertEqual(spirograph_geometry_from_ratio(2, 3), (5.0, 3.0))

    def test_ratio_change_refreshes_empirical_period_and_bounds_phase(self) -> None:
        state = SimpleNamespace(
            fixed_radius=np.full(8, 5.0),
            rolling_radius=np.full(8, 3.0),
            pen_offset=np.full(8, 5.0),
            path_radius=np.full(8, 0.9),
            repeat_period=np.full(8, 6.0 * np.pi),
            phase=np.full(8, 17.0 * np.pi),
        )
        studio = StudioState(
            spiro_ratio_numerators=[1, 3, 4],
            spiro_ratio_denominators=[2, 4, 5],
        )
        apply_spirograph_ratio(studio, SimpleNamespace(state=state), 0)
        np.testing.assert_allclose(state.fixed_radius, 3.0)
        np.testing.assert_allclose(state.rolling_radius, 2.0)
        np.testing.assert_allclose(state.repeat_period, 4.0 * np.pi)
        self.assertTrue(np.all((0.0 <= state.phase) & (state.phase < state.repeat_period)))

    def test_spiro_speed_limit_scales_with_refresh_and_fastest_harmonic(self) -> None:
        studio = StudioState(refresh_rate=120.0)
        self.assertAlmostEqual(spirograph_speed_limit(studio), 30.0 * np.pi)
        studio.spiro_ratio_numerators[0] = 12
        studio.spiro_ratio_denominators[0] = 1
        self.assertAlmostEqual(spirograph_speed_limit(studio), 2.5 * np.pi)

    def test_general_eager_spiro_sources_accumulate_before_display(self) -> None:
        domain = AddressDomain(tuple(
            DomainAddress(i, i, 20 + i, 1, "ring", i) for i in range(8)
        ), frame_count=8)
        spirographs = [
            SimpleNamespace(state=SimpleNamespace(x=np.full(8, x), y=np.full(8, y)))
            for x, y in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        ]
        studio = StudioState(
            spiro_count=4,
            spiro_hues=[0.0, 90.0, 180.0, 270.0],
            spiro_hue_rates=[0.0] * 4,
            spiro_ratio_numerators=[2] * 4,
            spiro_ratio_denominators=[3] * 4,
        )
        points = [
            (float(spirograph.state.x[0]), float(spirograph.state.y[0]))
            for spirograph in spirographs
        ]
        frame = spectral_source_frame(
            domain, points, studio.spiro_hues, RADIAL_DIRECTIONS,
            studio, 96, 3.0,
        )
        self.assertEqual(len(frame), 8)
        self.assertTrue(any(color != (0, 0, 0) for color in frame))

    def test_all_off_uses_one_full_controller_black_frame(self) -> None:
        controller = SimpleNamespace(
            blank_frame=lambda: [(0, 0, 0)] * 5,
            sent=[],
        )
        controller.send_frame = controller.sent.append
        _send_all_off(controller)
        self.assertEqual(controller.sent, [[(0, 0, 0)] * 5])

    def test_offset_places_logical_zero_at_requested_physical_site(self) -> None:
        logical = [(index, 0, 0) for index in range(8)]
        mapped = remap_ring_colors(logical, offset=3, direction=1)
        self.assertEqual(mapped[3], logical[0])
        self.assertEqual(mapped[4], logical[1])

    def test_reverse_order_changes_neighbors_without_changing_offset(self) -> None:
        logical = [(index, 0, 0) for index in range(8)]
        mapped = remap_ring_colors(logical, offset=3, direction=-1)
        self.assertEqual(mapped[3], logical[0])
        self.assertEqual(mapped[2], logical[1])

    def test_full_frame_remap_preserves_addresses_outside_selected_zone(self) -> None:
        domain = AddressDomain(tuple(
            DomainAddress(i, i + 2, 20 + i, 1, "ring", i)
            for i in range(8)
        ), frame_count=12)
        original = [(90, 91, 92), (93, 94, 95)] + [(0, 0, 0)] * 10
        mapped = [(i, i, i) for i in range(8)]
        result = remap_full_frame(domain, original, mapped)
        self.assertEqual(result[:2], original[:2])
        self.assertEqual(result[2:10], mapped)

    def test_hue_fit_is_nonnegative_and_matches_representable_orange(self) -> None:
        hue = 30.0
        spectrum = hue_spectrum(hue)
        self.assertTrue(np.all(spectrum >= 0.0))
        expected = np.asarray(colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0))
        np.testing.assert_allclose(
            spectrum @ raytracer_display_matrix(), expected, atol=1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
