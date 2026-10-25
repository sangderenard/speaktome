from __future__ import annotations

import unittest
from importlib.util import find_spec
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from openrgb_dt_walk import (
    AddressDomain,
    ManagedLinearWalk,
    address_domain_for,
    compile_linear_walk_piece,
)
from openrgb_shell import DomainAddress


class FakeWalkPiece:
    entry = "fake_walk"
    batch = 3
    argument_names = (
        "address_count",
        "cursor",
        "domain_index",
        "red",
        "green",
        "blue",
        "speed",
        "white_level",
        "dt",
    )
    output_names = (
        "cursor_next",
        "red_next",
        "green_next",
        "blue_next",
        "dt_limit",
        "max_vel",
    )

    def instantiate(self, columns, outputs=None):
        self.columns = columns
        self.outputs = outputs or {}

    def __call__(
        self,
        address_count,
        cursor,
        domain_index,
        red,
        green,
        blue,
        speed,
        white_level,
        dt,
    ):
        advanced = cursor + speed * dt
        wrapped = np.where(advanced >= address_count, advanced - address_count, advanced)
        white = np.where(np.abs(domain_index - wrapped) < 0.5, white_level, 0.0)
        return wrapped, white, white, white, 1.0 / np.abs(speed), np.abs(speed)


class ManagedLinearWalkTests(unittest.TestCase):
    @staticmethod
    def arbitrary_domain() -> AddressDomain:
        return AddressDomain((
            DomainAddress(0, 2, 900, 1, "Region B", 0),
            DomainAddress(1, 0, 40, 0, "Region A", 0),
            DomainAddress(2, 1, 7, 0, "Region A", 1),
        ))

    def test_dt_walk_preserves_arbitrary_system_mapping_and_wraps(self) -> None:
        domain = self.arbitrary_domain()
        walk = ManagedLinearWalk(
            domain,
            frames_per_second=4.0,
            white_level=255,
            piece=FakeWalkPiece(),
        )
        self.assertEqual(walk.advance_frame(), [(0, 0, 0), (0, 0, 0), (255, 255, 255)])
        self.assertEqual(walk.advance_frame(), [(255, 255, 255), (0, 0, 0), (0, 0, 0)])
        self.assertEqual(walk.advance_frame(), [(0, 0, 0), (255, 255, 255), (0, 0, 0)])
        self.assertEqual(walk.advance_frame(), [(0, 0, 0), (0, 0, 0), (255, 255, 255)])

    def test_subset_domain_preserves_full_controller_frame_slots(self) -> None:
        domain = AddressDomain(
            (
                DomainAddress(0, 4, 165, 4, "Addressable Header 3/Audio", 0),
                DomainAddress(1, 1, 166, 4, "Addressable Header 3/Audio", 1),
            ),
            frame_count=6,
        )
        self.assertEqual(
            domain.frame_from_channels(
                np.array([10.0, 20.0]),
                np.array([30.0, 40.0]),
                np.array([50.0, 60.0]),
            ),
            [
                (0, 0, 0),
                (20, 40, 60),
                (0, 0, 0),
                (0, 0, 0),
                (10, 30, 50),
                (0, 0, 0),
            ],
        )

    def test_zone_projection_filters_before_assigning_dense_indices(self) -> None:
        source = (
            DomainAddress(0, 0, 0, 0, "Other", 0),
            DomainAddress(1, 3, 165, 4, "Header 3", 0),
            DomainAddress(2, 5, 166, 4, "Header 3", 1),
        )
        controller = SimpleNamespace(
            address_domain=source,
            led_count=6,
            zone=lambda selector: SimpleNamespace(index=4),
        )
        domain = address_domain_for(controller, "Header 3", position_count=2)
        self.assertEqual(tuple(address.domain_index for address in domain.addresses), (0, 1))
        self.assertEqual(tuple(address.frame_index for address in domain.addresses), (3, 5))
        self.assertEqual(domain.frame_count, 6)

    def test_position_count_requires_a_zone(self) -> None:
        controller = SimpleNamespace(
            address_domain=self.arbitrary_domain().addresses,
            led_count=3,
        )
        with self.assertRaisesRegex(ValueError, "requires a selected zone"):
            address_domain_for(controller, position_count=2)

    def test_walk_can_start_on_a_selected_domain_position(self) -> None:
        walk = ManagedLinearWalk(
            self.arbitrary_domain(),
            frames_per_second=4.0,
            white_level=255,
            start_position=1,
            piece=FakeWalkPiece(),
        )
        self.assertEqual(walk.advance_frame(), [(255, 255, 255), (0, 0, 0), (0, 0, 0)])
        self.assertEqual(walk.advance_frame(), [(0, 0, 0), (255, 255, 255), (0, 0, 0)])

    @unittest.skipUnless(find_spec("llvmlite"), "llvmlite is required for the native route")
    def test_boolean_ite_law_compiles_to_llvm_and_advances_under_dt(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            piece = compile_linear_walk_piece(3, directory=directory)
            walk = ManagedLinearWalk(
                self.arbitrary_domain(),
                frames_per_second=4.0,
                white_level=255,
                piece=piece,
            )
            self.assertEqual(
                [walk.advance_frame() for _ in range(4)],
                [
                    [(0, 0, 0), (0, 0, 0), (255, 255, 255)],
                    [(255, 255, 255), (0, 0, 0), (0, 0, 0)],
                    [(0, 0, 0), (255, 255, 255), (0, 0, 0)],
                    [(0, 0, 0), (0, 0, 0), (255, 255, 255)],
                ],
            )


if __name__ == "__main__":
    unittest.main()
