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
LOW_PASS_CUTOFF = 8000  # ローパスフィルタのカットオフ周波数（Hz、高周波ノイズ除去）

# スペクトラルサブトラクション用のパラメータ
NOISE_PROFILE = None  # ノイズプロファイル（FFT結果）
NOISE_LEARNING_FRAMES = 20  # ノイズ学習フレーム数
NOISE_LEARNING_COUNT = 0  # ノイズ学習カウンタ
NOISE_FLOOR = 0.01  # ノイズフロア（最小レベル）
OVERSUBTRACTION_FACTOR = 2.5  # オーバーサブトラクション係数（大きいほど積極的にノイズ除去）
SPECTRAL_FLOOR = 0.02  # スペクトラルフロア（残存ノイズの最小レベル）

# VAD（Voice Activity Detection）用のパラメータ
VAD_THRESHOLD = 0.02  # VAD閾値（これ以下を無音と判定）
VAD_SMOOTHING = 0.9  # VAD平滑化係数
vad_energy = 0.0  # VADエネルギー（平滑化済み）

# マルチバンドノイズゲート用のパラメータ
NUM_BANDS = 8  # バンド数
BAND_THRESHOLDS = np.array([0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025])  # 各バンドの閾値

# ハイパスフィルタの設計（低周波ノイズ除去）
nyquist = SAMPLE_RATE / 2
if nyquist > HIGH_PASS_CUTOFF:
    normalized_cutoff = HIGH_PASS_CUTOFF / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
    FILTER_ENABLED = True
else:
    FILTER_ENABLED = False

# ローパスフィルタの設計（高周波ノイズ除去）
if nyquist > LOW_PASS_CUTOFF:
    normalized_lowpass = LOW_PASS_CUTOFF / nyquist
    b_low, a_low = signal.butter(4, normalized_lowpass, btype='low', analog=False)
    LOW_PASS_ENABLED = True
else:
    LOW_PASS_ENABLED = False

# バンドストップフィルタ（特定の帯域を除去）
# 例: 4000Hz〜5000Hzを除去したい場合
BAND_STOP_LOW = 4000
BAND_STOP_HIGH = 5000
if nyquist > BAND_STOP_HIGH:
    # 帯域阻止フィルタの設計
    low = BAND_STOP_LOW / nyquist
    high = BAND_STOP_HIGH / nyquist
    b_stop, a_stop = signal.butter(4, [low, high], btype='bandstop', analog=False)
    BAND_STOP_ENABLED = True
else:
    BAND_STOP_ENABLED = False

# エコーキャンセレーション用のバッファ
echo_buffer = np.zeros(CHUNK_SIZE * 2)

# マルチバンドフィルタの設計
band_edges = np.linspace(HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF, NUM_BANDS + 1)
band_filters = []
for i in range(NUM_BANDS):
    low = band_edges[i] / nyquist
    high = band_edges[i + 1] / nyquist
    if high < 1.0:
        b_band, a_band = signal.butter(2, [low, high], btype='band', analog=False)
        band_filters.append((b_band, a_band))
    else:
        band_filters.append(None)


def spectral_subtraction(audio_data, noise_profile):
    """
    スペクトラルサブトラクションによるノイズ除去
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        入力音声データ
    noise_profile : numpy.ndarray or None
        ノイズプロファイル（FFT結果）
    
    Returns:
    --------
    numpy.ndarray
        ノイズ除去後の音声データ
    """
    if noise_profile is None:
        return audio_data
    
    # FFTで周波数領域に変換
    fft_signal = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    
    # ノイズプロファイルからノイズを減算
    noise_magnitude = noise_profile * OVERSUBTRACTION_FACTOR
    clean_magnitude = magnitude - noise_magnitude
    
    # スペクトラルフロアを適用（負の値を防止し、残存ノイズを最小化）
    clean_magnitude = np.maximum(clean_magnitude, magnitude * SPECTRAL_FLOOR)
    
    # 位相を保持してIFFTで時間領域に戻す
    clean_fft = clean_magnitude * np.exp(1j * phase)
    clean_signal = np.fft.irfft(clean_fft, len(audio_data))
    
    return clean_signal


def voice_activity_detection(audio_data):
    """
    VAD（Voice Activity Detection）による音声区間の検出
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        入力音声データ
    
    Returns:
    --------
    bool
        音声が検出された場合True
    """
    global vad_energy
    
    # RMSエネルギーを計算
    rms = np.sqrt(np.mean(audio_data**2))
    
    # 指数移動平均で平滑化
    vad_energy = VAD_SMOOTHING * vad_energy + (1 - VAD_SMOOTHING) * rms
    
    # 閾値と比較
    return vad_energy > VAD_THRESHOLD


def multiband_noise_gate(audio_data):
    """
    マルチバンドノイズゲート
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        入力音声データ
    
    Returns:
    --------
    numpy.ndarray
        処理後の音声データ
    """
    filtered = audio_data.copy()
    
    # 各バンドで処理
    for i, (b_band, a_band) in enumerate(band_filters):
        if b_band is None:
            continue
        
        # バンドパスフィルタで帯域を抽出
        band_signal = signal.filtfilt(b_band, a_band, audio_data)
        
        # バンドのRMSを計算
        band_rms = np.sqrt(np.mean(band_signal**2))
        
        # 閾値以下なら減衰
        if band_rms < BAND_THRESHOLDS[i]:
            # 該当バンドを減衰
            filtered -= band_signal * (1 - 0.1)  # 90%減衰
    
    return filtered


def enhance_audio_quality(audio_data):
    """
    音質向上処理（強化版ノイズキャンセル）
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        入力音声データ（電圧値）
    
    Returns:
    --------
    numpy.ndarray
        処理済み音声データ
    """
    global NOISE_PROFILE, NOISE_LEARNING_COUNT
    
    # 0. VAD（Voice Activity Detection）で音声区間を検出
    is_voice = voice_activity_detection(audio_data)
    
    # 0.5 ノイズプロファイルの学習（無音時）
    if not is_voice and NOISE_LEARNING_COUNT < NOISE_LEARNING_FRAMES:
        # FFTで周波数領域に変換
        fft_signal = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_signal)
        
        # ノイズプロファイルを更新（指数移動平均）
        if NOISE_PROFILE is None:
            NOISE_PROFILE = magnitude.copy()
        else:
            alpha = 0.9  # 学習率
            NOISE_PROFILE = alpha * NOISE_PROFILE + (1 - alpha) * magnitude
        
        NOISE_LEARNING_COUNT += 1
    
    # 1. スペクトラルサブトラクション（周波数領域でのノイズ除去）
    if NOISE_PROFILE is not None:
        filtered = spectral_subtraction(audio_data, NOISE_PROFILE)
    else:
        filtered = audio_data.copy()
    
    # 2. ハイパスフィルタ適用（低周波ノイズ除去）
    if FILTER_ENABLED:
        filtered = signal.filtfilt(b, a, filtered)
    
    # 2.5 ローパスフィルタ適用（高周波ノイズ除去）
    if LOW_PASS_ENABLED:
        filtered = signal.filtfilt(b_low, a_low, filtered)
    
    # 2.7 バンドストップフィルタ適用
    if BAND_STOP_ENABLED:
        filtered = signal.filtfilt(b_stop, a_stop, filtered)
    
    # 3. マルチバンドノイズゲート（帯域ごとのノイズ除去）
    filtered = multiband_noise_gate(filtered)
    
    # 4. ノイズゲート（全体の小さなノイズを除去）
    rms = np.sqrt(np.mean(filtered**2))
    if rms < NOISE_GATE_THRESHOLD:
        filtered = filtered * 0.1  # ノイズを大幅に減衰
    
    # 5. ゲイン調整
    filtered = filtered * GAIN
    
    # 6. クリッピング防止（オーバーフロー対策）
    filtered = np.clip(filtered, -1.0, 1.0)
    
    # 7. エコーキャンセレーション
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
    print("=" * 60)
    print("Starting Audio Passthrough (Enhanced Noise Cancellation)...")
    print("=" * 60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz, Chunk Size: {CHUNK_SIZE}")
    print("\n強化されたノイズキャンセル機能:")
    print(f"  - スペクトラルサブトラクション: 有効")
    print(f"  - ノイズプロファイル学習: {NOISE_LEARNING_FRAMES}フレーム")
    print(f"  - ハイパスフィルタ: {HIGH_PASS_CUTOFF} Hz")
    print(f"  - ローパスフィルタ: {LOW_PASS_CUTOFF} Hz")
    print(f"  - マルチバンドノイズゲート: {NUM_BANDS}バンド")
    print(f"  - VAD（音声検出）: 有効")
    print(f"  - オーバーサブトラクション係数: {OVERSUBTRACTION_FACTOR}x")
    print("\n最初の数秒間は無音にしてノイズプロファイルを学習してください。")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

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
