"""
NIDAQ Beamforming Configuration
================================

NIDAQハードウェアとマイクの設定
"""

import numpy as np

# ========================================
# NIDAQ Hardware Configuration
# ========================================

# NIDAQ Device Name
# 確認方法: python -c "import nidaqmx; print(nidaqmx.system.System.local().devices)"
DEVICE_NAME = "Dev10"

# Microphone Channels
# 2つのマイクが接続されているアナログ入力チャンネル
MIC_CHANNELS = [
    f"{DEVICE_NAME}/ai0",  # 左マイク
    f"{DEVICE_NAME}/ai1"   # 右マイク
]

# ========================================
# Audio Configuration
# ========================================

# Sample Rate (Hz)
# 推奨: 20000 Hz (高品質) または 10000 Hz (低遅延)
SAMPLE_RATE = 20000

# Chunk Size (samples)
# 小さい値 = 低遅延、大きい値 = 高品質
# 推奨: 2048 (約100ms @ 20kHz)
CHUNK_SIZE = 2048

# Queue Size (chunks)
# バッファサイズ。遅延が大きい場合は減らす、音切れする場合は増やす
QUEUE_SIZE = 20

# ========================================
# Microphone Array Geometry
# ========================================

# Microphone Spacing (meters)
# 2つのマイク間の距離
# デフォルト: 0.76m (76cm)
MIC_SPACING = 0.76

# Microphone Positions (x, y, z) in meters
# 中心を(0, 0, 0)として、X軸上に配置
# 左マイク: -MIC_SPACING/2
# 右マイク: +MIC_SPACING/2
MIC_POSITIONS = np.array([
    [-MIC_SPACING / 2, 0.0, 0.0],  # 左マイク
    [ MIC_SPACING / 2, 0.0, 0.0]   # 右マイク
])

# ========================================
# Physical Constants
# ========================================

# Speed of Sound (m/s)
# 20°Cの空気中での音速
SPEED_OF_SOUND = 343.0

# ========================================
# Audio Processing Parameters
# ========================================

# Noise Gate Threshold
# この値以下のRMS音量は大幅に減衰される
NOISE_GATE_THRESHOLD = 0.01

# Gain Adjustment
# 出力音量の調整 (1.0 = そのまま、1.2 = 20%増幅)
GAIN = 1.2

# High-Pass Filter Cutoff (Hz)
# 低周波ノイズ除去用のハイパスフィルタ
# 人間の声は通常100Hz以上なので、80Hzでカット
HIGH_PASS_CUTOFF = 80

# ========================================
# Beamforming Parameters
# ========================================

# Default Target Angle (degrees)
# 0度 = 正面（マイクアレイに垂直）
# 90度 = 右側（エンドファイア）
# -90度 = 左側
DEFAULT_TARGET_ANGLE = 0.0

# ========================================
# Display Configuration
# ========================================

# Show Real-time Level Meter
SHOW_LEVEL_METER = True

# Level Meter Update Interval (chunks)
LEVEL_METER_UPDATE_INTERVAL = 5

# ========================================
# Advanced Settings
# ========================================

# Enable Echo Cancellation
# 出力音がマイクに戻るのを軽減（実験的機能）
ENABLE_ECHO_CANCELLATION = False

# Echo Buffer Size (samples)
ECHO_BUFFER_SIZE = CHUNK_SIZE * 2

# ========================================
# Validation
# ========================================

def validate_config():
    """設定の妥当性をチェック"""
    errors = []
    
    if SAMPLE_RATE <= 0:
        errors.append("SAMPLE_RATE must be positive")
    
    if CHUNK_SIZE <= 0:
        errors.append("CHUNK_SIZE must be positive")
    
    if MIC_SPACING <= 0:
        errors.append("MIC_SPACING must be positive")
    
    if len(MIC_CHANNELS) != 2:
        errors.append("Exactly 2 microphone channels required")
    
    if HIGH_PASS_CUTOFF >= SAMPLE_RATE / 2:
        errors.append(f"HIGH_PASS_CUTOFF ({HIGH_PASS_CUTOFF}) must be less than Nyquist frequency ({SAMPLE_RATE/2})")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# 設定を検証
if __name__ == "__main__":
    try:
        validate_config()
        print("✓ Configuration is valid")
        print()
        print("NIDAQ Configuration:")
        print(f"  Device: {DEVICE_NAME}")
        print(f"  Channels: {MIC_CHANNELS}")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Chunk Size: {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.1f} ms)")
        print(f"  Microphone Spacing: {MIC_SPACING*100:.1f} cm")
        print(f"  Default Target Angle: {DEFAULT_TARGET_ANGLE}°")
    except ValueError as e:
        print(f"✗ {e}")
