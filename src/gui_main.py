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
import time
from nidaqmx.constants import AcquisitionType

import config as config
from beamformer import DelayAndSumBeamformer, MVDRBeamformer

# Configuration (Driven by config.py now)
SAMPLE_RATE = config.SAMPLE_RATE
CHUNK_SIZE = config.CHUNK_SIZE
QUEUE_SIZE = 20 

# Global State for Audio Processing
audio_state = {
    "gain": 1.2,
    "input_gains": [1.0] * len(config.MIC_CHANNELS), # Input calibration gains
    "target_angle": 0.0, # Degrees
    "beamformer": None,  # Instance
    "high_pass_cutoff": 80,
    "high_pass_enabled": True,
    # Filter coefficients
    "hp_b": None, "hp_a": None,
    
    # List of Band Stop Filters
    # Each item: {"enabled": bool, "low": float, "high": float, "b": array, "a": array, "id": int}
    "band_stop_filters": [],
    
    "latest_chunk": None,  # For visualization (Output)
    "latest_input_chunk": None, # For visualization (Input)
    "input_levels": [0.0] * len(config.MIC_CHANNELS) # RMS levels for UI
}

# Queue
audio_queue = queue.Queue(maxsize=QUEUE_SIZE)

def update_filters():
    """Recalculate filter coefficients and beamformer weights based on current state."""
    
    # Beamforming
    if audio_state["beamformer"] is not None:
        audio_state["beamformer"].update_steering_vector(audio_state["target_angle"])

    nyquist = SAMPLE_RATE / 2
    
    # High Pass
    if nyquist > audio_state["high_pass_cutoff"]:
        norm_cutoff = audio_state["high_pass_cutoff"] / nyquist
        if 0 < norm_cutoff < 1:
            audio_state["hp_b"], audio_state["hp_a"] = signal.butter(4, norm_cutoff, btype='high', analog=False)
        else:
             audio_state["hp_b"], audio_state["hp_a"] = None, None
    
    # Band Stop Filters
    for bs_filter in audio_state["band_stop_filters"]:
        if nyquist > bs_filter["high"] and bs_filter["low"] < bs_filter["high"]:
            low = bs_filter["low"] / nyquist
            high = bs_filter["high"] / nyquist
            if 0 < low < high < 1:
                bs_filter["b"], bs_filter["a"] = signal.butter(4, [low, high], btype='bandstop', analog=False)
            else:
                bs_filter["b"], bs_filter["a"] = None, None
        else:
            bs_filter["b"], bs_filter["a"] = None, None

def audio_callback(outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    
    try:
        data = audio_queue.get(block=False)
        
        # --- Processing ---
        
        # Apply Input Gains first
        if data.ndim == 2:
            # Broadcast multiply: (samples, channels) * (channels,)
            gains = np.array(audio_state["input_gains"])
            if len(gains) == data.shape[1]:
                data = data * gains
        
        # Calculate Input Levels (RMS) for UI
        if data.ndim == 2:
            rms_levels = np.sqrt(np.mean(data**2, axis=0))
            audio_state["input_levels"] = rms_levels.tolist()

        # Store raw input for visualization (ALL channels)
        if data.ndim == 2:
            audio_state["latest_input_chunk"] = data
        else:
            audio_state["latest_input_chunk"] = None

        # 0. Beamforming (Multi-channel -> Single-channel)
        if audio_state["beamformer"] is not None:
            processed = audio_state["beamformer"].apply(data) # Apply handles dimension checks/padding
        else:
             processed = data[:, 0] if data.ndim == 2 else data
        
        # 1. High Pass
        if audio_state["high_pass_enabled"] and audio_state["hp_b"] is not None:
             processed = signal.filtfilt(audio_state["hp_b"], audio_state["hp_a"], processed)
        
        # 2. Band Stop Filters
        for bs_filter in audio_state["band_stop_filters"]:
            if bs_filter["enabled"] and bs_filter["b"] is not None:
                processed = signal.filtfilt(bs_filter["b"], bs_filter["a"], processed)

        # 3. Gain
        processed = processed * audio_state["gain"]
        
        # 4. Clipping
        processed = np.clip(processed, -1.0, 1.0)
        
        # Free run update for visualization
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
    try:
        with nidaqmx.Task() as task:
            # Add all configured channels
            for channel in config.MIC_CHANNELS:
                try:
                    task.ai_channels.add_ai_voltage_chan(channel)
                except Exception as e:
                     print(f"DAQ Setup Error (Channel {channel}): {e}")
                     return # Exit thread if setup fails

            task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=CHUNK_SIZE * 10, sample_mode=AcquisitionType.CONTINUOUS)
            
            # Explicitly start
            try:
                task.start()
            except Exception as e:
                print(f"DAQ Start Error: {e}")
                return

            print("DAQ Task Running")
            
            while not stop_event.is_set():
                try:
                    avail = task.in_stream.avail_samp_per_chan
                    if avail < CHUNK_SIZE:
                        time.sleep(0.001)
                        continue

                    # Read multi-channel data
                    data = task.read(number_of_samples_per_channel=CHUNK_SIZE, timeout=10.0)
                    
                    # Convert to numpy array (float32) and Transpose to (samples, channels)
                    np_data = np.array(data, dtype=np.float32).T
                    
                    try:
                        audio_queue.put(np_data, block=True, timeout=1)
                    except queue.Full:
                        pass
                except Exception as e:
                    print(f"DAQ Read Error: {e}")
                    break
    except Exception as e:
        print(f"DAQ Initialization Error (Task creation): {e}")
    
    print("DAQ Thread Stopped")

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Audio Equalizer")
        self.root.geometry("600x950")
        
        # --- Dark Mode Colors ---
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"
        self.panel_bg = "#3c3f41"
        self.plot_input_colors = ["#00bfa5", "#ffab00", "#d50000", "#6200ea"] # Colors for Ch0, Ch1...
        
        self.root.configure(bg=self.bg_color)
        
        # Configure ttk Styles
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure(".", background=self.bg_color, foreground=self.fg_color, fieldbackground=self.panel_bg)
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TLabelframe", background=self.bg_color, foreground=self.fg_color, bordercolor=self.panel_bg)
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background=self.panel_bg, foreground=self.fg_color, borderwidth=1)
        style.map("TButton", background=[("active", self.accent_color)])
        style.configure("Horizontal.TScale", background=self.bg_color, troughcolor=self.panel_bg, sliderthickness=15)
        style.configure("TRadiobutton", background=self.bg_color, foreground=self.fg_color, indicatorbackground=self.bg_color, selectcolor=self.panel_bg)
        style.configure("TProgressbar", thickness=15, background=self.accent_color, troughcolor=self.panel_bg)
        
        # Custom style for scrollable frame
        style.configure("Panel.TFrame", background=self.panel_bg)

        # Initialize filter ID counter
        self.filter_id_counter = 0

        # Initialize Beamformer (Default: DelayAndSum)
        audio_state["beamformer"] = DelayAndSumBeamformer()
        update_filters()

        # --- GUI Layout ---
        # Main Container
        main_container = ttk.Frame(root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 1. Plots (Updated to have 2 subplots)
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor=self.bg_color)
        
        # Subplot 1: Frequency Spectrum (Output)
        self.ax_spec = self.fig.add_subplot(211)
        self.ax_spec.set_title("Output Frequency Spectrum", color=self.fg_color)
        self.ax_spec.set_ylabel("dB", color=self.fg_color)
        self.ax_spec.set_facecolor(self.bg_color)
        self.ax_spec.tick_params(colors=self.fg_color)
        for spine in self.ax_spec.spines.values():
            spine.set_edgecolor(self.panel_bg)
        self.ax_spec.set_xlim(0, SAMPLE_RATE / 2)
        self.ax_spec.set_ylim(-60, 40)
        self.ax_spec.grid(True, color=self.panel_bg, linestyle='--', alpha=0.5)
        self.line_spec, = self.ax_spec.plot([], [], lw=1, color=self.accent_color)

        # Subplot 2: Waveform (Input vs Output)
        self.ax_wave = self.fig.add_subplot(212)
        self.ax_wave.set_title("Waveform (Input vs Output)", color=self.fg_color)
        self.ax_wave.set_ylabel("Amplitude", color=self.fg_color)
        self.ax_wave.set_xlabel("Time (s)", color=self.fg_color)
        self.ax_wave.set_facecolor(self.bg_color)
        self.ax_wave.tick_params(colors=self.fg_color)
        for spine in self.ax_wave.spines.values():
            spine.set_edgecolor(self.panel_bg)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.grid(True, color=self.panel_bg, linestyle='--', alpha=0.5)
        
        # Time axis
        self.time_axis = np.linspace(0, CHUNK_SIZE/SAMPLE_RATE, CHUNK_SIZE)
        
        # Prepare lines for Inputs
        self.line_waves_in = []
        for i, ch in enumerate(config.MIC_CHANNELS):
            color = self.plot_input_colors[i % len(self.plot_input_colors)]
            line, = self.ax_wave.plot([], [], lw=1, color=color, alpha=0.6, label=f"In {i}")
            self.line_waves_in.append(line)

        # Prepare line for Output
        self.line_wave_out, = self.ax_wave.plot([], [], lw=1, color=self.accent_color, label="Output")
        
        self.ax_wave.legend(loc='upper right', frameon=False, labelcolor=self.fg_color)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 2. Controls - Scrollable Area using Canvas
        canvas_frame = ttk.Frame(main_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_scroll = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas_scroll.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas_scroll)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )
        
        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=560) 
        
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Content inside Scrollable Frame
        
        # --- Input Calibration Control ---
        cal_frame = ttk.LabelFrame(self.scrollable_frame, text="Input Calibration (Gain & Monitor)", padding="10")
        cal_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.input_bars = []
        
        for i in range(len(config.MIC_CHANNELS)):
            ch_frame = ttk.Frame(cal_frame)
            ch_frame.pack(fill=tk.X, pady=2)
            
            # Label
            ttk.Label(ch_frame, text=f"Ch {i}").pack(side=tk.LEFT, padx=5)
            
            # Gain Slider
            def make_gain_callback(index):
                def cb(val):
                    audio_state["input_gains"][index] = float(val)
                return cb
            
            gain_var = tk.DoubleVar(value=audio_state["input_gains"][i])
            ttk.Scale(ch_frame, from_=0.0, to=5.0, variable=gain_var, command=make_gain_callback(i)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Level Meter
            bar = ttk.Progressbar(ch_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=0.5) 
            bar.pack(side=tk.RIGHT)
            self.input_bars.append(bar)

        # --- Beamforming Control ---
        bf_frame = ttk.LabelFrame(self.scrollable_frame, text="Beamforming", padding="10")
        bf_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Algorithm Selection
        algo_frame = ttk.Frame(bf_frame)
        algo_frame.pack(fill=tk.X, pady=5)
        ttk.Label(algo_frame, text="Algorithm: ").pack(side=tk.LEFT)
        self.bf_algo_var = tk.StringVar(value="DelayAndSum")
        
        def on_algo_change():
            algo = self.bf_algo_var.get()
            current_angle = audio_state["target_angle"]
            current_spacing = 0.76 # default, ideally track this
            if algo == "DelayAndSum":
                audio_state["beamformer"] = DelayAndSumBeamformer()
            else:
                audio_state["beamformer"] = MVDRBeamformer()
            
            # Restore state
            audio_state["beamformer"].update_steering_vector(current_angle)
            if hasattr(self, 'spacing_val'):
                 audio_state["beamformer"].update_geometry(self.spacing_val.get())

        ttk.Radiobutton(algo_frame, text="Delay & Sum", variable=self.bf_algo_var, value="DelayAndSum", command=on_algo_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(algo_frame, text="MVDR", variable=self.bf_algo_var, value="MVDR", command=on_algo_change).pack(side=tk.LEFT, padx=5)

        # Angle Control
        self.angle_val = tk.DoubleVar(value=audio_state["target_angle"])
        
        # Angle Label with callback
        self.angle_label = ttk.Label(bf_frame, text=f"Angle: {int(self.angle_val.get())}°")
        self.angle_label.pack(anchor=tk.W)
        
        def on_angle_change(val):
            angle = float(val)
            self.angle_label.config(text=f"Angle: {int(angle)}°")
            audio_state["target_angle"] = angle
            # Only update if beamformer exists
            if audio_state["beamformer"]:
                 audio_state["beamformer"].update_steering_vector(angle)
                 
        ttk.Scale(bf_frame, from_=-90, to=90, variable=self.angle_val, command=on_angle_change).pack(fill=tk.X)

        # --- Mic Spacing Control ---
        spacing_frame = ttk.LabelFrame(self.scrollable_frame, text="Mic Spacing (m)", padding="10")
        spacing_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Default from config (calculate from positions if possible, or just default)
        initial_spacing = 0.76
        self.spacing_val = tk.DoubleVar(value=initial_spacing)
        
        self.spacing_label = ttk.Label(spacing_frame, text=f"{self.spacing_val.get():.2f} m")
        self.spacing_label.pack(anchor=tk.E)
        
        def on_spacing_change(val):
            spacing = float(val)
            self.spacing_label.config(text=f"{spacing:.2f} m")
            # Update beamformer geometry
            if audio_state["beamformer"]:
                 audio_state["beamformer"].update_geometry(spacing)

        ttk.Scale(spacing_frame, from_=0.01, to=1.0, variable=self.spacing_val, command=on_spacing_change).pack(fill=tk.X)

        # --- Global Gain ---
        gain_frame = ttk.LabelFrame(self.scrollable_frame, text="Master Gain", padding="10")
        gain_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.gain_val = tk.DoubleVar(value=audio_state["gain"])
        ttk.Scale(gain_frame, from_=0.0, to=5.0, variable=self.gain_val, command=self.on_gain_change).pack(fill=tk.X)
        
        # --- High Pass Filter ---
        hp_frame = ttk.LabelFrame(self.scrollable_frame, text="High Pass Filter", padding="10")
        hp_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.hp_enabled = tk.BooleanVar(value=audio_state["high_pass_enabled"])
        ttk.Checkbutton(hp_frame, text="Enable", variable=self.hp_enabled, command=self.on_hp_toggle).pack(anchor=tk.W)
        
        ttk.Label(hp_frame, text="Cutoff Frequency").pack(anchor=tk.W, pady=(5,0))
        self.hp_freq = tk.DoubleVar(value=audio_state["high_pass_cutoff"])
        self.hp_label = ttk.Label(hp_frame, text=f"{int(self.hp_freq.get())} Hz")
        self.hp_label.pack(anchor=tk.E)
        ttk.Scale(hp_frame, from_=10, to=1000, variable=self.hp_freq, command=self.on_hp_change).pack(fill=tk.X)

        # --- Band Stop Filters Section ---
        bs_header_frame = ttk.Frame(self.scrollable_frame)
        bs_header_frame.pack(fill=tk.X, pady=(20, 5), padx=5)
        ttk.Label(bs_header_frame, text="Band Stop Filters", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(bs_header_frame, text="+ Add Filter", command=self.add_filter_ui).pack(side=tk.RIGHT)
        
        self.filters_container = ttk.Frame(self.scrollable_frame)
        self.filters_container.pack(fill=tk.BOTH, expand=True)
        
        # Initial Filter
        self.add_filter_ui(initial_freqs=(4000, 5000))

        # Status Bar
        self.status_label = ttk.Label(root, text="Ready", relief=tk.FLAT, anchor=tk.W, background=self.panel_bg, padding=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Start Audio
        self.start_audio()
        self.update_plot()

    def add_filter_ui(self, initial_freqs=None):
        """Add a new Band Stop Filter UI block."""
        fid = self.filter_id_counter
        self.filter_id_counter += 1
        
        # Create Data Model
        low_hz = initial_freqs[0] if initial_freqs else 1000
        high_hz = initial_freqs[1] if initial_freqs else 2000
        
        new_filter = {
            "id": fid,
            "enabled": True,
            "low": low_hz,
            "high": high_hz,
            "b": None, "a": None
        }
        audio_state["band_stop_filters"].append(new_filter)
        update_filters()
        
        # Create UI
        frame = ttk.LabelFrame(self.filters_container, text=f"Filter #{fid+1}", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Top Row: Enable Checkbox and Delete Button
        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X)
        
        enabled_var = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(top_row, text="Enable", variable=enabled_var, 
                             command=lambda: self.on_filter_toggle(fid, enabled_var))
        cb.pack(side=tk.LEFT)
        
        del_btn = ttk.Button(top_row, text="Remove", command=lambda: self.remove_filter(fid, frame))
        del_btn.pack(side=tk.RIGHT)
        
        # Low Frequency
        ttk.Label(frame, text="Low Frequency").pack(anchor=tk.W, pady=(5,0))
        low_label = ttk.Label(frame, text=f"{low_hz} Hz")
        low_label.pack(anchor=tk.E)
        
        low_var = tk.DoubleVar(value=low_hz)
        low_scale = ttk.Scale(frame, from_=100, to=SAMPLE_RATE//2 - 100, variable=low_var)
        low_scale.pack(fill=tk.X)
        
        # High Frequency
        ttk.Label(frame, text="High Frequency").pack(anchor=tk.W, pady=(5,0))
        high_label = ttk.Label(frame, text=f"{high_hz} Hz")
        high_label.pack(anchor=tk.E)
        
        high_var = tk.DoubleVar(value=high_hz)
        high_scale = ttk.Scale(frame, from_=100, to=SAMPLE_RATE//2, variable=high_var)
        high_scale.pack(fill=tk.X)
        
        # Callbacks (using closure to capture specific widgets and IDs)
        def on_change(_):
            l = low_var.get()
            h = high_var.get()
            low_label.config(text=f"{int(l)} Hz")
            high_label.config(text=f"{int(h)} Hz")
            
            # Find filter data
            for f in audio_state["band_stop_filters"]:
                if f["id"] == fid:
                    f["low"] = l
                    f["high"] = h
                    break
            update_filters()
            
        low_scale.configure(command=on_change)
        high_scale.configure(command=on_change)

    def remove_filter(self, fid, frame_widget):
        # Remove from data
        audio_state["band_stop_filters"] = [f for f in audio_state["band_stop_filters"] if f["id"] != fid]
        # Remove from UI
        frame_widget.destroy()
        # Recalculate
        update_filters()

    def on_filter_toggle(self, fid, var):
        for f in audio_state["band_stop_filters"]:
            if f["id"] == fid:
                f["enabled"] = var.get()
                break
    
    def on_gain_change(self, val):
        audio_state["gain"] = float(val)

    def on_hp_toggle(self):
        audio_state["high_pass_enabled"] = self.hp_enabled.get()

    def on_hp_change(self, val):
        freq = float(val)
        self.hp_label.config(text=f"{int(freq)} Hz")
        audio_state["high_pass_cutoff"] = freq
        update_filters()

    # ... (start_audio, update_plot, stop_audio methods remain the same) ...

    def start_audio(self):
        self.stop_event = threading.Event()
        self.daq_thread = threading.Thread(target=daq_thread_func, args=(self.stop_event,), daemon=True)
        self.daq_thread.start()
        
        try:
            self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=audio_callback)
            self.stream.start()
            self.status_label.config(text="Audio Running")
        except Exception as e:
            print(f"Audio Output Error: {e}")
            self.status_label.config(text="Audio Output Failed (Check Console)")
            # Do NOT exit, allow GUI to run for inspection

    def stop_audio(self):
        self.stop_event.set()
        if hasattr(self, 'stream'):
             try:
                self.stream.stop()
                self.stream.close()
             except:
                 pass
        self.root.quit()
        # Force exit to kill threads immediately if they hang
        sys.exit(0)

    def update_plot(self):
        """Fetch latest audio data and update the plot."""
        chunk = audio_state["latest_chunk"]
        chunk_in = audio_state["latest_input_chunk"] # Raw input
        
        # Update Input Level Meters
        levels = audio_state["input_levels"]
        if len(levels) == len(self.input_bars):
            for i, level in enumerate(levels):
                try:
                     self.input_bars[i]['value'] = level
                except:
                     pass

        if chunk is not None:
            # 1. Update Spectrum
            N = len(chunk)
            yf = np.fft.fft(chunk)
            xf = np.fft.fftfreq(N, 1 / SAMPLE_RATE)
            
            # Take positive half
            xf = xf[:N//2]
            magnitude = 2.0/N * np.abs(yf[:N//2])
            
            # Use log scale (dB) with safe log
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-6))
            
            self.line_spec.set_data(xf, magnitude_db)
            
            # 2. Update Waveform
            # Output
            self.line_wave_out.set_data(self.time_axis, chunk)
            
            # Input
            if chunk_in is not None and chunk_in.ndim == 2:
                 # Update all input lines
                 for i, line in enumerate(self.line_waves_in):
                     if i < chunk_in.shape[1]:
                         line.set_data(self.time_axis, chunk_in[:, i])
            
            self.canvas.draw_idle()
        
        self.root.after(50, self.update_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_audio)
    root.mainloop()
