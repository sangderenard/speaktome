import numpy as np
from queue import Queue
import logging

class FFTVisualizer:
    def __init__(self, sample_rate=44100, fft_window_size=1024, time_window=1.0, band_bins=9):
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size
        self.time_window = time_window
        self.band_bins = band_bins
        self.cached_fft_data = []

    def update(self, dt, local_state):
        if "fft_in" not in local_state or not local_state["fft_in"]:
            logging.warning("No digital FFT data available for visualization.")
            return {}

        fft_data = local_state["fft_in"]
        fft_magnitudes = np.abs(fft_data[:self.fft_window_size])
        fft_phases = np.angle(fft_data[:self.fft_window_size])
        rebinned_magnitudes = fft_magnitudes.reshape(-1, self.band_bins).mean(axis=1)
        rebinned_phases = fft_phases.reshape(-1, self.band_bins).mean(axis=1)

        self.cached_fft_data.append((rebinned_magnitudes, rebinned_phases))
        if len(self.cached_fft_data) > int(self.sample_rate * self.time_window / self.fft_window_size):
            self.cached_fft_data.pop(0)

        blended_magnitudes = np.mean([data[0] for data in self.cached_fft_data], axis=0)
        blended_phases = np.mean([data[1] for data in self.cached_fft_data], axis=0)

        return {
            "vbo_in": {
                "frequencies": np.fft.rfftfreq(self.fft_window_size, 1 / self.sample_rate)[:len(blended_magnitudes)],
                "magnitudes": blended_magnitudes,
                "phases": blended_phases,
            }
        }

    def stop(self):
        logging.info("FFTVisualizer stopped.")


from baseplugin import BasePlugin
class Plugin(BasePlugin):
    def __init__(self):
        self.instance = FFTVisualizer()

    def start(self):
        return self.instance.start()

    def update(self, dt, local_state):
        return self.instance.update(dt, local_state)

    def stop(self):
        return self.instance.stop()

    def get_class_name(self):
        return self.instance.__class__.__name__