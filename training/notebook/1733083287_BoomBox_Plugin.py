import sounddevice as sd
import numpy as np
from queue import Queue
import logging
import time

class BoomBox:
    def __init__(self, sample_rate=44100, fft_window_size=1024):
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size
        self.aux_in_buffer = Queue()
        self.fft_queue = Queue()
        self.local_wave_in = Queue()
        self._last_update_time = time.time()

    def start_aux_in(self):
        def audio_callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio Input Warning: {status}")
            self.aux_in_buffer.put(indata[:, 0])

        self.aux_in_stream = sd.InputStream(
            callback=audio_callback, samplerate=self.sample_rate, channels=1
        )
        self.aux_in_stream.start()
        logging.info("BoomBox aux-in started.")

    def stop_aux_in(self):
        if hasattr(self, 'aux_in_stream') and self.aux_in_stream.active:
            self.aux_in_stream.stop()
            self.aux_in_stream.close()
            logging.info("BoomBox aux-in stopped.")

    def process_fft(self):
        if not self.aux_in_buffer.empty():
            data = self.aux_in_buffer.get()
            if len(data) >= self.fft_window_size:
                fft_result = np.fft.rfft(data[:self.fft_window_size])
                self.fft_queue.put(fft_result)

    def update(self, dt, local_state):
        current_time = time.time()
        if current_time - self._last_update_time >= dt:
            self._last_update_time = current_time
            self.process_fft()

        if "fft_wave_in" in local_state and local_state["fft_wave_in"] is not None:
            self.local_wave_in.put(local_state["fft_wave_in"])

        return {"fft_out": self.fft_queue.get() if not self.fft_queue.empty() else None}

    def stop(self):
        self.stop_aux_in()

from baseplugin import BasePlugin
import torch

class Plugin(BasePlugin):
    def __init__(self, dt, local_state, stream_interceptor=None):
        self.dt_draw = 1.0
        self.battery_level = .2
        self.instance = BoomBox(dt, local_state)

        super().__init__(dt, local_state, stream_interceptor=stream_interceptor, autostart=False, needs_power=True, has_battery=True, battery_level=self.battery_level, dt_draw=1.0)

    def _start_plugin(self, dt, local_state):
        self.instance.start_aux_in()

    def _update_plugin(self, dt, local_state):
        return self.instance.update(dt, local_state)

    def _stop_plugin(self, dt, local_state):
        self.instance.stop()

    def get_class_name(self):
        return self.instance.__class__.__name__
