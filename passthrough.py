import queue
import sounddevice as sd
import nidaqmx
import numpy as np
import sys

# Configuration
SAMPLE_RATE = 20000
CHUNK_SIZE = 2048
# Buffer size in chunks. 
# If latency is too high, decrease this. If audio cuts out, increase this.
QUEUE_SIZE = 20 

# Communication queue between DAQ and Audio Output
audio_queue = queue.Queue(maxsize=QUEUE_SIZE)

def callback(outdata, frames, time, status):
    """
    Sounddevice callback function.
    It pulls data from the queue and writes it to the output buffer.
    """
    if status:
        print(status, file=sys.stderr)
    
    try:
        # Try to get data from the queue without blocking for too long
        data = audio_queue.get(block=False)
        
        # If the data size doesn't match the requested frames (rare but possible with resizing)
        if len(data) < frames:
             # Pad with zeros if we have partial data (shouldn't happen with fixed chunk size)
             outdata[:len(data)] = data.reshape(-1, 1)
             outdata[len(data):] = 0
             print("Buffer underrun (partial)", file=sys.stderr)
        else:
            outdata[:] = data.reshape(-1, 1)
            
    except queue.Empty:
        # Buffer underflow: Queue is empty, output silence
        print("Buffer underflow: Outputting silence", file=sys.stderr)
        outdata[:] = 0

def run_passthrough():
    print(f"Starting Audio Passthrough...")
    print(f"Sample Rate: {SAMPLE_RATE}, Chunk Size: {CHUNK_SIZE}")
    print("Press Ctrl+C to stop.")

    # Initialize Output Stream (Consumer)
    # We start the stream first; it will pull silence until data arrives
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, 
                         blocksize=CHUNK_SIZE, callback=callback):
        
        # Initialize Input Task (Producer)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
            task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=CHUNK_SIZE * 10)
            
            try:
                while True:
                    # Read data from DAQ
                    # nidaqmx returns a list of floats by default
                    data = task.read(number_of_samples_per_channel=CHUNK_SIZE)
                    
                    # Convert to numpy array (float32 is standard for audio)
                    np_data = np.array(data, dtype=np.float32)
                    
                    # Put data into the queue
                    # If queue is full, this will block, naturally throttling the loop 
                    # to the playback speed (or causing overflow if input is faster)
                    try:
                        audio_queue.put(np_data, block=True, timeout=1)
                    except queue.Full:
                        print("Queue full: Dropping data (Latency too high?)", file=sys.stderr)
                        pass

            except KeyboardInterrupt:
                print("\nStopping...")

if __name__ == "__main__":
    run_passthrough()
