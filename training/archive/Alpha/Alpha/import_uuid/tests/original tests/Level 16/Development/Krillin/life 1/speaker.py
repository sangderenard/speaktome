import numpy as np
import logging

class Speaker:
    def __init__(self, sample_rate=44100, fft_window_size=1024):
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size

    def update(self, dt, local_state):
        if "fft_out" not in local_state or local_state["fft_out"] is None:
            logging.warning("No digital FFT data available for Speaker.")
            return {}

        fft_data = local_state["fft_out"]
        pressure_wave_data = np.fft.irfft(fft_data, n=self.fft_window_size)
        return {"fft_wave_in": pressure_wave_data}

    def stop(self):
        logging.info("Speaker stopped.")


from baseplugin import BasePlugin
class Plugin(BasePlugin):
    def __init__(self):
        self.instance = Speaker()


    def start(self):
        return self.instance.start()

    def update(self, dt, local_state):
        return self.instance.update(dt, local_state)

    def stop(self):
        return self.instance.stop()

    def get_class_name(self):
        return self.instance.__class__.__name__