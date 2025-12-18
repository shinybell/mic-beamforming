import queue
import sounddevice as sd
import nidaqmx
import numpy as np
import sys
from scipy import signal

# Configuration
SAMPLE_RATE = 20000
CHUNK_SIZE = 2048
# Buffer size in chunks. 
# If latency is too high, decrease this. If audio cuts out, increase this.
QUEUE_SIZE = 20 

# Communication queue between DAQ and Audio Output
audio_queue = queue.Queue(maxsize=QUEUE_SIZE)

# 音質向上のためのパラメータ
NOISE_GATE_THRESHOLD = 0.01  # ノイズゲートの閾値
GAIN = 1.2  # ゲイン調整
HIGH_PASS_CUTOFF = 80  # ハイパスフィルタのカットオフ周波数（Hz）

# ハイパスフィルタの設計（低周波ノイズ除去）
nyquist = SAMPLE_RATE / 2
if nyquist > HIGH_PASS_CUTOFF:
    normalized_cutoff = HIGH_PASS_CUTOFF / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
    FILTER_ENABLED = True
else:
    FILTER_ENABLED = False

# エコーキャンセレーション用のバッファ
echo_buffer = np.zeros(CHUNK_SIZE * 2)


def enhance_audio_quality(audio_data):
    """
    音質向上処理
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        入力音声データ（電圧値）
    
    Returns:
    --------
    numpy.ndarray
        処理済み音声データ
    """
    # 1. ハイパスフィルタ適用（低周波ノイズ除去）
    if FILTER_ENABLED:
        filtered = signal.filtfilt(b, a, audio_data)
    else:
        filtered = audio_data
    
    # 2. ノイズゲート（小さなノイズを除去）
    rms = np.sqrt(np.mean(filtered**2))
    if rms < NOISE_GATE_THRESHOLD:
        filtered = filtered * 0.1  # ノイズを大幅に減衰
    
    # 3. ゲイン調整
    filtered = filtered * GAIN
    
    # 4. クリッピング防止（オーバーフロー対策）
    filtered = np.clip(filtered, -1.0, 1.0)
    
    # 5. 簡単なエコーキャンセレーション（オプション）
    # 出力された音がマイクに戻ってくるのを軽減
    global echo_buffer
    if len(echo_buffer) >= len(filtered):
        echo_reduction = echo_buffer[:len(filtered)] * 0.1
        filtered = filtered - echo_reduction
    
    # エコーバッファを更新
    echo_buffer = np.roll(echo_buffer, -len(filtered))
    echo_buffer[-len(filtered):] = filtered
    
    return filtered


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
        
        # 音質向上処理を適用
        processed_data = enhance_audio_quality(data)
        
        # If the data size doesn't match the requested frames (rare but possible with resizing)
        if len(processed_data) < frames:
             # Pad with zeros if we have partial data (shouldn't happen with fixed chunk size)
             outdata[:len(processed_data)] = processed_data.reshape(-1, 1)
             outdata[len(processed_data):] = 0
             print("Buffer underrun (partial)", file=sys.stderr)
        else:
            outdata[:] = processed_data.reshape(-1, 1)
            
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
