"""Native dedicated-source regression. Set NODUS_SCOPE_DLL and NODUS_FFT_DLL."""
import ctypes as c
import math
import os
import unittest


@unittest.skipUnless(os.environ.get("NODUS_SCOPE_DLL"), "native scope DLL not configured")
class CaptureOnlyCursorTests(unittest.TestCase):
    def test_raw_pcm_cursor_and_overrun_without_fft(self):
        dll = c.CDLL(os.environ["NODUS_SCOPE_DLL"])
        dll.openrgb_audio_scope_create_capture.restype = c.c_void_p
        dll.openrgb_audio_scope_free.argtypes = [c.c_void_p]
        feed = dll.openrgb_audio_scope_feed_stereo
        feed.argtypes = [c.c_void_p, c.POINTER(c.c_float), c.c_size_t, c.c_uint32]
        read = dll.openrgb_audio_scope_read_stereo
        read.argtypes = [c.c_void_p, c.POINTER(c.c_uint64), c.POINTER(c.c_float),
                         c.c_size_t, c.POINTER(c.c_uint32), c.POINTER(c.c_uint64)]
        read.restype = c.c_size_t
        handle = dll.openrgb_audio_scope_create_capture()
        self.assertTrue(handle)
        try:
            values = (c.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
            self.assertEqual(feed(handle, values, 4, 48000), 1)
            cursor, rate, lost = c.c_uint64(), c.c_uint32(), c.c_uint64()
            out = (c.c_float * 4)()
            self.assertEqual(read(handle, c.byref(cursor), out, 2,
                                  c.byref(rate), c.byref(lost)), 2)
            self.assertEqual((cursor.value, rate.value, lost.value), (2, 48000, 0))
            self.assertEqual(list(out), [1, 2, 3, 4])
            self.assertEqual(read(handle, c.byref(cursor), out, 2,
                                  c.byref(rate), c.byref(lost)), 2)
            self.assertEqual((cursor.value, lost.value), (4, 0))
            self.assertEqual(list(out), [5, 6, 7, 8])
            self.assertEqual(read(handle, c.byref(cursor), out, 2,
                                  c.byref(rate), c.byref(lost)), 0)
            chunk = (c.c_float * 8192)()
            for _ in range(65):
                self.assertEqual(feed(handle, chunk, 4096, 48000), 1)
            self.assertEqual(read(handle, c.byref(cursor), out, 2,
                                  c.byref(rate), c.byref(lost)), 2)
            self.assertEqual(lost.value, 4096)
            self.assertEqual(cursor.value, 4102)
        finally:
            dll.openrgb_audio_scope_free(handle)


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
