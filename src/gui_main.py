import tkinter as tk
from tkinter import ttk
import queue
import sounddevice as sd
import nidaqmx
import numpy as np
import sys
from scipy import signal
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Configuration
SAMPLE_RATE = 20000
CHUNK_SIZE = 2048
QUEUE_SIZE = 20 

# Global State for Audio Processing
audio_state = {
    "gain": 1.2,
    "high_pass_cutoff": 80,
    "high_pass_enabled": True,
    "band_stop_low": 4000,
    "band_stop_high": 5000,
    "band_stop_enabled": False,
    # Filter coefficients (initialized later)
    "hp_b": None, "hp_a": None,
    "bs_b": None, "bs_a": None,
    "latest_chunk": None  # For visualization
}

# Queue
audio_queue = queue.Queue(maxsize=QUEUE_SIZE)

def update_filters():
    """Recalculate filter coefficients based on current state."""
    nyquist = SAMPLE_RATE / 2
    
    # High Pass
    if nyquist > audio_state["high_pass_cutoff"]:
        norm_cutoff = audio_state["high_pass_cutoff"] / nyquist
        # Ensure cutoff is valid
        if 0 < norm_cutoff < 1:
            audio_state["hp_b"], audio_state["hp_a"] = signal.butter(4, norm_cutoff, btype='high', analog=False)
        else:
             audio_state["hp_b"], audio_state["hp_a"] = None, None
    
    # Band Stop
    if nyquist > audio_state["band_stop_high"] and audio_state["band_stop_low"] < audio_state["band_stop_high"]:
        low = audio_state["band_stop_low"] / nyquist
        high = audio_state["band_stop_high"] / nyquist
        if 0 < low < high < 1:
            audio_state["bs_b"], audio_state["bs_a"] = signal.butter(4, [low, high], btype='bandstop', analog=False)
        else:
            audio_state["bs_b"], audio_state["bs_a"] = None, None
    else:
        # Invalid range or out of bounds
        audio_state["bs_b"], audio_state["bs_a"] = None, None

def audio_callback(outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    
    try:
        data = audio_queue.get(block=False)
        
        # --- Processing ---
        processed = data
        
        # 1. High Pass
        if audio_state["high_pass_enabled"] and audio_state["hp_b"] is not None:
             processed = signal.filtfilt(audio_state["hp_b"], audio_state["hp_a"], processed)
        
        # 2. Band Stop
        if audio_state["band_stop_enabled"] and audio_state["bs_b"] is not None:
             processed = signal.filtfilt(audio_state["bs_b"], audio_state["bs_a"], processed)

        # 3. Gain
        processed = processed * audio_state["gain"]
        
        # 4. Clipping
        processed = np.clip(processed, -1.0, 1.0)
        
        # Free run update for visualization (thread-safe assignment of reference)
        audio_state["latest_chunk"] = processed
        
        # --- Output ---
        if len(processed) < frames:
             outdata[:len(processed)] = processed.reshape(-1, 1)
             outdata[len(processed):] = 0
        else:
            outdata[:] = processed.reshape(-1, 1)
            
    except queue.Empty:
        outdata[:] = 0

def daq_thread_func(stop_event):
    """Producer thread that reads from DAQ."""
    print("DAQ Thread Started")
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=CHUNK_SIZE * 10)
        
        while not stop_event.is_set():
            try:
                data = task.read(number_of_samples_per_channel=CHUNK_SIZE)
                np_data = np.array(data, dtype=np.float32)
                try:
                    audio_queue.put(np_data, block=True, timeout=1)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"DAQ Error: {e}")
                break
    print("DAQ Thread Stopped")

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Audio Equalizer")
        
        # Initialize filters
        update_filters()
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Spectrum Plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Frequency Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_xlim(0, SAMPLE_RATE / 2)
        self.ax.set_ylim(-60, 40) # Adjust range as needed
        self.ax.grid(True)
        
        self.line, = self.ax.plot([], [], lw=1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls Frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)

        # Gain
        ttk.Label(controls_frame, text="Gain").pack(anchor=tk.W)
        self.gain_val = tk.DoubleVar(value=audio_state["gain"])
        scale_gain = ttk.Scale(controls_frame, from_=0.0, to=5.0, variable=self.gain_val, command=self.on_gain_change)
        scale_gain.pack(fill=tk.X, pady=(0, 10))
        
        # High Pass
        hp_frame = ttk.LabelFrame(controls_frame, text="High Pass Filter (Low Cut)")
        hp_frame.pack(fill=tk.X, pady=5)
        
        self.hp_enabled = tk.BooleanVar(value=audio_state["high_pass_enabled"])
        ttk.Checkbutton(hp_frame, text="Enable", variable=self.hp_enabled, command=self.on_hp_toggle).pack(anchor=tk.W)
        
        ttk.Label(hp_frame, text="Cutoff Freq (Hz)").pack(anchor=tk.W)
        self.hp_freq = tk.DoubleVar(value=audio_state["high_pass_cutoff"])
        ttk.Scale(hp_frame, from_=10, to=500, variable=self.hp_freq, command=self.on_hp_change).pack(fill=tk.X)
        self.hp_label = ttk.Label(hp_frame, text=f"{int(self.hp_freq.get())} Hz")
        self.hp_label.pack(anchor=tk.E)

        # Band Stop
        bs_frame = ttk.LabelFrame(controls_frame, text="Band Stop Filter (Notch)")
        bs_frame.pack(fill=tk.X, pady=5)
        
        self.bs_enabled = tk.BooleanVar(value=audio_state["band_stop_enabled"])
        ttk.Checkbutton(bs_frame, text="Enable", variable=self.bs_enabled, command=self.on_bs_toggle).pack(anchor=tk.W)
        
        # Range Slider simulation with two sliders
        ttk.Label(bs_frame, text="Low Freq (Hz)").pack(anchor=tk.W)
        self.bs_low = tk.DoubleVar(value=audio_state["band_stop_low"])
        ttk.Scale(bs_frame, from_=100, to=SAMPLE_RATE//2 - 100, variable=self.bs_low, command=self.on_bs_change).pack(fill=tk.X)
        self.bs_low_label = ttk.Label(bs_frame, text=f"{int(self.bs_low.get())} Hz")
        self.bs_low_label.pack(anchor=tk.E)

        ttk.Label(bs_frame, text="High Freq (Hz)").pack(anchor=tk.W)
        self.bs_high = tk.DoubleVar(value=audio_state["band_stop_high"])
        ttk.Scale(bs_frame, from_=100, to=SAMPLE_RATE//2, variable=self.bs_high, command=self.on_bs_change).pack(fill=tk.X)
        self.bs_high_label = ttk.Label(bs_frame, text=f"{int(self.bs_high.get())} Hz")
        self.bs_high_label.pack(anchor=tk.E)

        # Status
        self.status_label = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Start Audio
        self.start_audio()
        
        # Start Animation Loop
        self.update_plot()

    def update_plot(self):
        """Fetch latest audio data and update the plot."""
        chunk = audio_state["latest_chunk"]
        if chunk is not None:
            # Calculate FFT
            N = len(chunk)
            yf = np.fft.fft(chunk)
            xf = np.fft.fftfreq(N, 1 / SAMPLE_RATE)
            
            # Take positive half
            xf = xf[:N//2]
            magnitude = 2.0/N * np.abs(yf[:N//2])
            
            # Use log scale (dB)
            # Avoid log(0)
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-6))
            
            self.line.set_data(xf, magnitude_db)
            self.canvas.draw_idle()
        
        # Schedule next update (e.g. 50ms = 20fps)
        self.root.after(50, self.update_plot)


    def on_gain_change(self, val):
        audio_state["gain"] = float(val)

    def on_hp_toggle(self):
        audio_state["high_pass_enabled"] = self.hp_enabled.get()

    def on_hp_change(self, val):
        freq = float(val)
        self.hp_label.config(text=f"{int(freq)} Hz")
        audio_state["high_pass_cutoff"] = freq
        update_filters()

    def on_bs_toggle(self):
        audio_state["band_stop_enabled"] = self.bs_enabled.get()

    def on_bs_change(self, val):
        low = self.bs_low.get()
        high = self.bs_high.get()
        
        # Ensure Low < High
        if low >= high:
            # simple contraint logic
            pass 
        
        self.bs_low_label.config(text=f"{int(low)} Hz")
        self.bs_high_label.config(text=f"{int(high)} Hz")
        
        audio_state["band_stop_low"] = low
        audio_state["band_stop_high"] = high
        update_filters()

    def start_audio(self):
        self.stop_event = threading.Event()
        
        # Start DAQ thread
        self.daq_thread = threading.Thread(target=daq_thread_func, args=(self.stop_event,), daemon=True)
        self.daq_thread.start()
        
        # Start SD stream
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=audio_callback)
        self.stream.start()
        
        self.status_label.config(text="Audio Running")

    def stop_audio(self):
        self.stop_event.set()
        self.stream.stop()
        self.stream.close()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_audio)
    root.mainloop()
