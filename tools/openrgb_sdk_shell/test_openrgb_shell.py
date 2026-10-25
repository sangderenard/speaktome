from __future__ import annotations

import unittest
from types import SimpleNamespace

from openrgb_shell import FullFrameController, OpenRGBShellError, connect, validate_rgb


class FakeDevice:
    def __init__(self) -> None:
        self.name = "ASRock Test Board"
        self.leds = [SimpleNamespace(id=index) for index in range(5)]
        self.zones = [
            SimpleNamespace(name="PCB", leds=self.leds[:2]),
            SimpleNamespace(name="Addressable Header", leds=self.leds[2:]),
        ]
        self.mode_calls: list[str] = []
        self.frame_calls: list[tuple[list[tuple[int, int, int]], bool]] = []

    def set_mode(self, mode: str) -> None:
        self.mode_calls.append(mode)

    def set_colors(self, colors, fast: bool = False) -> None:
        self.frame_calls.append((list(colors), fast))


class FakeClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.devices = [FakeDevice()]


class FullFrameControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = FakeDevice()
        self.controller = FullFrameController(None, self.device, lambda r, g, b: (r, g, b))

    def test_layout_and_pixel_frame_use_global_offsets(self) -> None:
        self.assertEqual(self.controller.zones[1].start, 2)
        self.assertEqual(
            self.controller.pixel_frame("Addressable Header", 1, (64, 0, 0)),
            [(0, 0, 0), (0, 0, 0), (0, 0, 0), (64, 0, 0), (0, 0, 0)],
        )

    def test_noncontiguous_system_addresses_map_to_dense_domain(self) -> None:
        leds = [SimpleNamespace(id=value) for value in (40, 7, 900)]
        device = SimpleNamespace(
            name="Arbitrary Address Device",
            leds=leds,
            zones=[
                SimpleNamespace(name="Region B", leds=[leds[2]]),
                SimpleNamespace(name="Region A", leds=leds[:2]),
            ],
        )
        controller = FullFrameController(None, device, lambda r, g, b: (r, g, b))
        self.assertEqual(
            [
                (
                    address.domain_index,
                    address.frame_index,
                    address.system_address,
                    address.region_name,
                    address.region_position,
                )
                for address in controller.address_domain
            ],
            [
                (0, 2, 900, "Region B", 0),
                (1, 0, 40, "Region A", 0),
                (2, 1, 7, "Region A", 1),
            ],
        )
        self.assertEqual(
            controller.pixel_frame("Region B", 0, (64, 0, 0)),
            [(0, 0, 0), (0, 0, 0), (64, 0, 0)],
        )

    def test_send_frame_requires_exact_length(self) -> None:
        with self.assertRaises(OpenRGBShellError):
            self.controller.send_frame([(0, 0, 0)])

    def test_send_frame_uses_one_full_device_update(self) -> None:
        frame = self.controller.solid_frame((32, 32, 32))
        self.controller.send_frame(frame)
        self.assertEqual(self.device.frame_calls, [(frame, True)])

    def test_explicit_mode_selection(self) -> None:
        self.controller.select_mode("Direct")
        self.assertEqual(self.device.mode_calls, ["Direct"])

    def test_rgb_validation(self) -> None:
        self.assertEqual(validate_rgb((0, 128, 255)), (0, 128, 255))
        with self.assertRaises(ValueError):
            validate_rgb((0, 0, 256))

    def test_remote_host_requires_explicit_permission(self) -> None:
        with self.assertRaises(OpenRGBShellError):
            connect(
                host="192.0.2.1",
                client_factory=FakeClient,
                color_factory=lambda r, g, b: (r, g, b),
            )


if __name__ == "__main__":
    unittest.main()
