import numpy as np
from scipy.signal import butter, lfilter
from queue import Queue
import logging


class FFTVisualizer:
    def __init__(self, sample_rate=44100, fft_window_size=1024, time_window=1.0, band_bins=9):
        """
        Base class for an FFT visualizer.

        Args:
            sample_rate (int): Sampling rate for FFT processing.
            fft_window_size (int): Size of the FFT window.
            time_window (float): Time window for blending FFT samples (in seconds).
            band_bins (int): Number of frequency bands for re-binning.
        """
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size
        self.time_window = time_window
        self.band_bins = band_bins

        # Buffers
        self.fft_input_buffer = Queue()
        self.vbo_output_buffer = Queue()

        # Filters
        self.low_pass_cutoff = None
        self.high_pass_cutoff = None

        # Cached data
        self.cached_fft_data = []

    def set_filters(self, low_pass_cutoff=None, high_pass_cutoff=None):
        """
        Set low-pass and high-pass filters.

        Args:
            low_pass_cutoff (float): Low-pass filter cutoff frequency in Hz.
            high_pass_cutoff (float): High-pass filter cutoff frequency in Hz.
        """
        self.low_pass_cutoff = low_pass_cutoff
        self.high_pass_cutoff = high_pass_cutoff

    def apply_filters(self, data):
        """
        Apply low-pass and high-pass filters to the input data.

        Args:
            data (numpy.ndarray): Input audio data.

        Returns:
            numpy.ndarray: Filtered audio data.
        """
        nyquist = 0.5 * self.sample_rate
        if self.low_pass_cutoff:
            low = self.low_pass_cutoff / nyquist
            b, a = butter(5, low, btype='low')
            data = lfilter(b, a, data)
        if self.high_pass_cutoff:
            high = self.high_pass_cutoff / nyquist
            b, a = butter(5, high, btype='high')
            data = lfilter(b, a, data)
        return data

    def update(self, local_fft_data):
        """
        Update the FFT visualizer by processing local FFT data and returning a VBO object.

        Args:
            local_fft_data (numpy.ndarray): FFT data retrieved from the game locality.

        Returns:
            dict: VBO object containing formatted visualization data.
        """
        if local_fft_data is None or len(local_fft_data) < self.fft_window_size:
            logging.warning("Insufficient FFT data for visualization.")
            return None

        # Re-bin FFT data into bands
        fft_magnitudes = np.abs(local_fft_data[:self.fft_window_size])
        fft_phases = np.angle(local_fft_data[:self.fft_window_size])
        rebinned_magnitudes = fft_magnitudes.reshape(-1, self.band_bins).mean(axis=1)
        rebinned_phases = fft_phases.reshape(-1, self.band_bins).mean(axis=1)

        # Blend FFT data over time window
        self.cached_fft_data.append((rebinned_magnitudes, rebinned_phases))
        if len(self.cached_fft_data) > int(self.sample_rate * self.time_window / self.fft_window_size):
            self.cached_fft_data.pop(0)

        blended_magnitudes = np.mean([data[0] for data in self.cached_fft_data], axis=0)
        blended_phases = np.mean([data[1] for data in self.cached_fft_data], axis=0)

        # Generate VBO object
        vbo_object = {
            "frequencies": np.fft.rfftfreq(self.fft_window_size, 1 / self.sample_rate)[:len(blended_magnitudes)],
            "magnitudes": blended_magnitudes,
            "phases": blended_phases
        }

        self.vbo_output_buffer.put(vbo_object)
        return vbo_object

    def render(self, vbo_object):
        """
        Render the visualization data from a VBO object.

        Args:
            vbo_object (dict): The VBO object containing frequencies, magnitudes, and phases.

        Returns:
            str: Formatted textual output for debugging.
        """
        if not vbo_object:
            return "No VBO data to render."

        frequencies = vbo_object["frequencies"]
        magnitudes = vbo_object["magnitudes"]
        phases = vbo_object["phases"]

        rendered_output = "\n".join(
            [f"Freq: {freq:.2f} Hz, Magnitude: {mag:.2f}, Phase: {phase:.2f}"
             for freq, mag, phase in zip(frequencies, magnitudes, phases)]
        )
        logging.info(f"Rendered VBO:\n{rendered_output}")
        return rendered_output