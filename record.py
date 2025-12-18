import nidaqmx
import numpy as np
import wave
import sys
import time
import os

# Configuration
SAMPLE_RATE = 20000
CHUNK_SIZE = 2048  # Number of samples per chunk
OUTPUT_FILE = "output.wav"

def record_audio():
    print(f"Starting Recording Mode...")
    print(f"Sample Rate: {SAMPLE_RATE}, Chunk Size: {CHUNK_SIZE}")
    print(f"Saving to: {os.path.abspath(OUTPUT_FILE)}")
    print("Press Ctrl+C to stop recording.")

    recorded_frames = []
    
    # Initialize Input Task (Producer)
    with nidaqmx.Task() as task:
        # Configure Channel
        task.ai_channels.add_ai_voltage_chan("Dev10/ai0")
        
        # Configure Timing
        # continuous sampling
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=CHUNK_SIZE)
        
        start_time = time.time()
        try:
            while True:
                # Read data from DAQ
                # blocks until CHUNK_SIZE samples are available
                data = task.read(number_of_samples_per_channel=CHUNK_SIZE)
                
                # Convert to numpy array
                np_data = np.array(data, dtype=np.float32)
                recorded_frames.append(np_data)
                
                # Show duration
                duration = time.time() - start_time
                print(f"\rRecording... {duration:.1f}s", end="", flush=True)

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        except Exception as e:
            print(f"\nError during recording: {e}")

    # Save to WAV file
    if recorded_frames:
        print(f"Saving {len(recorded_frames) * CHUNK_SIZE} samples to '{OUTPUT_FILE}'...")
        
        # Concatenate all chunks
        full_recording = np.concatenate(recorded_frames)
        
        # Clip to valid range
        audio_data = np.clip(full_recording, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        pcm_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(OUTPUT_FILE, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes = 16 bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data.tobytes())
        print("Done.")
    else:
        print("No data recorded.")

if __name__ == "__main__":
    record_audio()
