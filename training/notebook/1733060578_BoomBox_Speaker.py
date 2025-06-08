import sounddevice as sd
import numpy as np
from queue import Queue
from scipy.signal import butter, lfilter
import time

class BoomBox:
    def __init__(self, sample_rate=44100, fft_window_size=1024, num_speakers=2):
        """
        Initialize the BoomBox with system audio in/out, FFT processing, and audio mixing.
        
        Args:
            sample_rate (int): Sampling rate for audio processing.
            fft_window_size (int): Size of the FFT window.
            num_speakers (int): Number of speakers for the boombox.
        """
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size
        
        # Audio Buffers
        self.aux_in_buffer = Queue()
        self.aux_out_buffer = Queue()
        self.fft_queue = Queue()
        
        # Boombox components
        self.microphone_enabled = False
        self.microphone_data = Queue()  # Placeholder for future game audio data
        self.speakers = [BoomBox.Speaker() for _ in range(num_speakers)]
        
        # Low-pass and high-pass filter design
        self.low_pass_cutoff = 5000.0  # Hz
        self.high_pass_cutoff = 100.0  # Hz
        self.b, self.a = self.design_filters()

    def design_filters(self):
        """
        Design low-pass and high-pass filters.
        """
        nyquist = 0.5 * self.sample_rate
        low = self.low_pass_cutoff / nyquist
        high = self.high_pass_cutoff / nyquist
        
        b_low, a_low = butter(5, low, btype='low')
        b_high, a_high = butter(5, high, btype='high')
        
        return (b_low, a_low), (b_high, a_high)
    
    def apply_filters(self, data):
        """
        Apply low-pass and high-pass filters to the audio data.
        
        Args:
            data (numpy.ndarray): Input audio data.
            
        Returns:
            numpy.ndarray: Filtered audio data.
        """
        # Apply low-pass filter
        low_filtered = lfilter(self.b[0], self.b[1], data)
        # Apply high-pass filter
        filtered_data = lfilter(self.a[0], self.a[1], low_filtered)
        return filtered_data

    def start_aux_in(self):
        """
        Start capturing system audio as aux in.
        """
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio Input Warning: {status}")
            filtered_data = self.apply_filters(indata[:, 0])  # Mono input
            self.aux_in_buffer.put(filtered_data)

        self.aux_in_stream = sd.InputStream(
            callback=audio_callback,
            samplerate=self.sample_rate,
            channels=1
        )
        self.aux_in_stream.start()
        print("Aux In started.")

    def stop_aux_in(self):
        """
        Stop capturing system audio as aux in.
        """
        if hasattr(self, 'aux_in_stream') and self.aux_in_stream.active:
            self.aux_in_stream.stop()
            self.aux_in_stream.close()
            print("Aux In stopped.")

    def start_aux_out(self):
        """
        Start streaming audio to system audio as aux out.
        """
        def playback_callback(outdata, frames, time, status):
            if status:
                print(f"Audio Output Warning: {status}")
            if not self.aux_out_buffer.empty():
                outdata[:, 0] = self.aux_out_buffer.get_nowait()[:frames]
            else:
                outdata.fill(0)  # Silence if no data available

        self.aux_out_stream = sd.OutputStream(
            callback=playback_callback,
            samplerate=self.sample_rate,
            channels=1
        )
        self.aux_out_stream.start()
        print("Aux Out started.")

    def stop_aux_out(self):
        """
        Stop streaming audio to system audio as aux out.
        """
        if hasattr(self, 'aux_out_stream') and self.aux_out_stream.active:
            self.aux_out_stream.stop()
            self.aux_out_stream.close()
            print("Aux Out stopped.")

    def process_fft(self):
        """
        Perform FFT on the most recent aux_in data and store it in the FFT queue.
        """
        if not self.aux_in_buffer.empty():
            data = self.aux_in_buffer.get()
            if len(data) >= self.fft_window_size:
                data = data[:self.fft_window_size]
                fft_result = np.abs(np.fft.rfft(data))  # Magnitude of FFT
                self.fft_queue.put(fft_result)

    class Speaker:
        """
        Speaker class to simulate in-game audio hashing and processing.
        """
        def __init__(self):
            self.location = (0, 0, 0)  # Placeholder for in-game coordinates
            self.audio_data = Queue()

        def receive_audio(self, data):
            """
            Receive audio data for the speaker.
            
            Args:
                data (numpy.ndarray): Audio data to process.
            """
            self.audio_data.put(data)

    def toggle_microphone(self, enabled):
        """
        Enable or disable the microphone.
        
        Args:
            enabled (bool): True to enable, False to disable.
        """
        self.microphone_enabled = enabled
        print(f"Microphone {'enabled' if enabled else 'disabled'}.")

    def run(self):
        """
        Example run loop to process FFT and simulate audio flow.
        """
        try:
            while True:
                self.process_fft()  # Continuously process FFT
                if not self.fft_queue.empty():
                    fft_data = self.fft_queue.get()
                    print(f"FFT Data: {fft_data[:10]}")  # Example FFT output
        except KeyboardInterrupt:
            print("Stopping BoomBox...")
    def update(self, dt):
        """
        Perform FFT updates at a timed interval.
        
        Args:
            dt (float): Time interval in seconds between updates.
        """
        current_time = time.time()
        if not hasattr(self, "_last_update_time"):
            self._last_update_time = current_time
        
        # Perform update only if the interval has elapsed
        if current_time - self._last_update_time >= dt:
            self._last_update_time = current_time
            self.process_fft()
# Example usage
if __name__ == "__main__":
    boombox = BoomBox()
    boombox.start_aux_in()
    boombox.start_aux_out()
    try:
        boombox.run()
    finally:
        boombox.stop_aux_in()
        boombox.stop_aux_out()
