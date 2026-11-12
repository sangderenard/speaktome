"""Native dedicated-source regression. Set NODUS_SCOPE_DLL and NODUS_FFT_DLL."""
import ctypes as c
import math
import os
import unittest


@unittest.skipUnless(os.environ.get("NODUS_SCOPE_DLL") and os.environ.get("NODUS_FFT_DLL"), "native DLLs not configured")
class ExternalScopeFeedTests(unittest.TestCase):
    def test_external_source_is_finite_and_excludes_loopback(self):
        dll = c.CDLL(os.environ["NODUS_SCOPE_DLL"])
        dll.openrgb_audio_scope_create.argtypes = [c.c_wchar_p, c.c_size_t, c.c_size_t, c.c_size_t, c.c_float, c.c_float]
        dll.openrgb_audio_scope_create.restype = c.c_void_p
        handle = dll.openrgb_audio_scope_create(os.environ["NODUS_FFT_DLL"], 1024, 4, 5, 40, 18000)
        self.assertTrue(handle)
        dll.openrgb_audio_scope_free.argtypes = [c.c_void_p]
        try:
            feed = dll.openrgb_audio_scope_feed_stereo
            feed.argtypes = [c.c_void_p, c.POINTER(c.c_float), c.c_size_t, c.c_uint32]
            samples = (c.c_float * 2048)(*[.2 * math.sin(2 * math.pi * 440 * (i // 2) / 48000) for i in range(2048)])
            self.assertEqual(feed(handle, samples, 1024, 48000), 1)
            self.assertEqual(feed(handle, samples, 1024, 44100), 0)
            bad = (c.c_float * 2)(float("nan"), 0)
            self.assertEqual(feed(handle, bad, 1, 48000), 0)
            dll.openrgb_audio_scope_start.argtypes = [c.c_void_p]
            self.assertEqual(dll.openrgb_audio_scope_start(handle), 0)
            snapshot = dll.openrgb_audio_scope_snapshot
            snapshot.argtypes = [c.c_void_p, c.c_float, c.POINTER(c.c_float), c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_size_t, c.POINTER(c.c_float), c.POINTER(c.c_uint32)]
            snapshot.restype = c.c_size_t
            xy, power, freq = (c.c_float * 10)(), (c.c_float * 5)(), (c.c_float * 5)()
            rms, rate = c.c_float(), c.c_uint32()
            self.assertEqual(snapshot(handle, 1, xy, power, freq, 5, c.byref(rms), c.byref(rate)), 5)
            self.assertEqual(rate.value, 48000)
            self.assertGreater(rms.value, 0)
            self.assertTrue(all(math.isfinite(x) for x in power))
        finally:
            dll.openrgb_audio_scope_free(handle)


if __name__ == "__main__":
    unittest.main()
