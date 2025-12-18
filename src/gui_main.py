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
    # Filter coefficients
    "hp_b": None, "hp_a": None,
    
    # List of Band Stop Filters
    # Each item: {"enabled": bool, "low": float, "high": float, "b": array, "a": array, "id": int}
    "band_stop_filters": [],
    
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
        processed = data
        
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
        self.root.geometry("600x800")
        
        # --- Dark Mode Colors ---
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#007acc"
        self.panel_bg = "#3c3f41"
        
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
        
        # Custom style for scrollable frame
        style.configure("Panel.TFrame", background=self.panel_bg)

        # Initialize filter ID counter
        self.filter_id_counter = 0

        # --- GUI Layout ---
        # Main Container
        main_container = ttk.Frame(root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 1. Spectrum Plot (Dark Mode)
        matplotlib.style.use('dark_background')
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Frequency Spectrum", color=self.fg_color)
        self.ax.set_xlabel("Frequency (Hz)", color=self.fg_color)
        self.ax.set_ylabel("Magnitude (dB)", color=self.fg_color)
        self.ax.set_facecolor(self.bg_color)
        self.ax.tick_params(colors=self.fg_color)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.panel_bg)
            
        self.ax.set_xlim(0, SAMPLE_RATE / 2)
        self.ax.set_ylim(-60, 40)
        self.ax.grid(True, color=self.panel_bg, linestyle='--', alpha=0.5)
        
        self.line, = self.ax.plot([], [], lw=1, color=self.accent_color)
        
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
        
        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=560) # Fixed width adjustment
        
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Content inside Scrollable Frame
        
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
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=audio_callback)
        self.stream.start()
        self.status_label.config(text="Audio Running")

    def stop_audio(self):
        self.stop_event.set()
        self.stream.stop()
        self.stream.close()
        self.root.quit()
        # Force exit to kill threads immediately if they hang
        sys.exit(0)

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
            
            # Use log scale (dB) with safe log
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-6))
            
            self.line.set_data(xf, magnitude_db)
            self.canvas.draw_idle()
        
        self.root.after(50, self.update_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_audio)
    root.mainloop()
