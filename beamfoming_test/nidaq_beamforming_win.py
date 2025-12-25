"""
NIDAQ Dual-Microphone Beamforming System (改良版)
=================================================

正しいDelay-and-Sumビームフォーミングを実装
時間領域での遅延補償による話者分離

必要なライブラリ:
pip install numpy scipy sounddevice nidaqmx

使用方法:
1. NIDAQに2つのマイクを接続（Dev10/ai0, Dev10/ai1）
2. nidaq_config.pyで設定を確認
3. このスクリプトを実行
4. 目的角度を入力
5. ビームフォーミングされた音声がスピーカーから出力
"""

import numpy as np
import sounddevice as sd
import nidaqmx
from scipy import signal
import queue
import sys
import time
import nidaq_config as config


class NIDAQBeamformer:
    """NIDAQを使用したビームフォーミングクラス（改良版）"""
    
    def __init__(self):
        """初期化"""
        # 設定を検証
        config.validate_config()
        
        # パラメータを設定
        self.sample_rate = config.SAMPLE_RATE
        self.chunk_size = config.CHUNK_SIZE
        self.mic_positions = config.MIC_POSITIONS
        self.num_mics = len(self.mic_positions)
        
        # 通信キュー
        self.audio_queue = queue.Queue(maxsize=config.QUEUE_SIZE)
        
        # 実行状態
        self.is_running = False
        
        # 現在の目的角度
        self.current_angle = config.DEFAULT_TARGET_ANGLE
        
        # 遅延サンプル数（後で計算）
        self.delay_samples = 0
        
        # ハイパスフィルタの設計
        self.setup_filters()
        
        # レベルメーター用
        self.chunk_counter = 0
        
        # 出力デバイスを自動選択
        self.output_device = self.select_output_device()
        
        print(f"\n=== NIDAQ Beamformer 初期化完了 ===")
        print(f"サンプリングレート: {self.sample_rate} Hz")
        print(f"チャンクサイズ: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f} ms)")
        print(f"マイク数: {self.num_mics}")
        print(f"マイク間距離: {config.MIC_SPACING*100:.1f} cm")
    
    def select_output_device(self):
        """出力デバイスを自動選択（Windows/Mac対応）"""
        import platform
        
        devices = sd.query_devices()
        os_type = platform.system()
        
        print(f"\nOS: {os_type}")
        
        # Windowsの場合
        if os_type == "Windows":
            for i, device in enumerate(devices):
                name_lower = device['name'].lower()
                if device['max_output_channels'] >= 1:
                    if any(keyword in name_lower for keyword in ['speaker', 'スピーカー', 'headphone', 'ヘッドフォン', 'default']):
                        print(f"出力デバイス: [{i}] {device['name']}")
                        return i
            
            try:
                default_output = sd.query_devices(kind='output')
                print(f"出力デバイス: [デフォルト] {default_output['name']}")
                return None
            except:
                print("出力デバイス: デフォルト")
                return None
        
        # Macの場合
        elif os_type == "Darwin":
            for i, device in enumerate(devices):
                name_lower = device['name'].lower()
                if ('macbook' in name_lower or 'built-in' in name_lower) and device['max_output_channels'] >= 1:
                    print(f"出力デバイス: [{i}] {device['name']}")
                    return i
            
            try:
                default_output = sd.query_devices(kind='output')
                print(f"出力デバイス: [デフォルト] {default_output['name']}")
                return None
            except:
                print("出力デバイス: デフォルト")
                return None
        
        else:
            print(f"出力デバイス: デフォルト（{os_type}）")
            return None
    
    def setup_filters(self):
        """フィルタの設計"""
        nyquist = self.sample_rate / 2
        
        if nyquist > config.HIGH_PASS_CUTOFF:
            normalized_cutoff = config.HIGH_PASS_CUTOFF / nyquist
            self.filter_b, self.filter_a = signal.butter(
                4, normalized_cutoff, btype="high", analog=False
            )
            self.filter_enabled = True
            print(f"ハイパスフィルタ: {config.HIGH_PASS_CUTOFF} Hz")
        else:
            self.filter_enabled = False
            print("ハイパスフィルタ: 無効")
    
    def calculate_delay(self, theta_deg):
        """
        指定角度に対する時間遅延を計算
        
        Parameters:
        -----------
        theta_deg : float
            目的角度（度）
            -90度 = 左側（左マイクが音源に近い）
            0度 = 正面（両マイク等距離）
            90度 = 右側（右マイクが音源に近い）
        
        Returns:
        --------
        delay_samples : int
            遅延サンプル数
        """
        self.current_angle = theta_deg
        theta_rad = np.deg2rad(theta_deg)
        
        # マイク間距離
        d = config.MIC_SPACING
        
        # 音源方向からの時間差
        # sin(theta) = 0 のとき（正面）: 時間差なし
        # sin(theta) = 1 のとき（右側）: 最大時間差 d/c
        # sin(theta) = -1 のとき（左側）: 最大時間差 -d/c
        time_delay = (d * np.sin(theta_rad)) / config.SPEED_OF_SOUND
        
        # サンプル数に変換
        delay_samples = int(abs(time_delay) * self.sample_rate)
        
        print(f"\n遅延計算: {theta_deg}度")
        print(f"  時間遅延: {time_delay*1000:.3f} ms")
        print(f"  遅延サンプル数: {delay_samples}")
        
        return delay_samples
    
    def apply_beamforming(self, multichannel_chunk):
        """
        Delay-and-Sumビームフォーミングを適用（時間領域）
        
        Parameters:
        -----------
        multichannel_chunk : ndarray
            マルチチャンネル音声データ
            shape: (chunk_size, num_mics)
        
        Returns:
        --------
        ndarray
            ビームフォーミング後の音声
            shape: (chunk_size,)
        """
        # 左右チャンネルを分離
        left_channel = multichannel_chunk[:, 0]
        right_channel = multichannel_chunk[:, 1]
        
        if self.current_angle < -10:  # 左側を強調
            # 左マイクが音源に近い
            # 右マイクの信号を遅延させて左マイクに合わせる
            delayed_right = np.pad(right_channel, (self.delay_samples, 0), mode='constant')[:-self.delay_samples or None]
            # 加算して強調、減算して抑制
            output = left_channel + delayed_right - 0.5 * right_channel
            
        elif self.current_angle > 10:  # 右側を強調
            # 右マイクが音源に近い
            # 左マイクの信号を遅延させて右マイクに合わせる
            delayed_left = np.pad(left_channel, (self.delay_samples, 0), mode='constant')[:-self.delay_samples or None]
            # 加算して強調、減算して抑制
            output = right_channel + delayed_left - 0.5 * left_channel
            
        else:  # 正面（0度付近）
            # 両チャンネルを均等に混合
            output = (left_channel + right_channel) / 2.0
        
        return output.astype(np.float32)
    
    def enhance_audio_quality(self, audio_data):
        """音質向上処理"""
        # 1. ハイパスフィルタ（低周波ノイズ除去）
        if self.filter_enabled:
            filtered = signal.filtfilt(self.filter_b, self.filter_a, audio_data)
        else:
            filtered = audio_data
        
        # 2. ノイズゲート（小さなノイズを除去）
        rms = np.sqrt(np.mean(filtered**2))
        if rms < config.NOISE_GATE_THRESHOLD:
            filtered = filtered * 0.1
        
        # 3. ゲイン調整
        filtered = filtered * config.GAIN
        
        # 4. クリッピング防止
        filtered = np.clip(filtered, -1.0, 1.0)
        
        return filtered
    
    def audio_output_callback(self, outdata, frames, time_info, status):
        """Sounddeviceの出力コールバック"""
        if status:
            print(f"出力Status: {status}", file=sys.stderr)
        
        try:
            data = self.audio_queue.get(block=False)
            processed_data = self.enhance_audio_quality(data)
            
            if len(processed_data) < frames:
                outdata[:len(processed_data)] = processed_data.reshape(-1, 1)
                outdata[len(processed_data):] = 0
            else:
                outdata[:] = processed_data.reshape(-1, 1)
        
        except queue.Empty:
            outdata[:] = 0
            if self.is_running:
                print("Buffer underflow", file=sys.stderr)
    
    def show_level_meter(self, data):
        """簡易レベルメーター表示"""
        if not config.SHOW_LEVEL_METER:
            return
        
        self.chunk_counter += 1
        if self.chunk_counter % config.LEVEL_METER_UPDATE_INTERVAL != 0:
            return
        
        rms = np.sqrt(np.mean(data**2))
        
        if rms > 1e-10:
            db = 20 * np.log10(rms)
        else:
            db = -100
        
        bar_length = int(max(0, min(50, (db + 60) / 60 * 50)))
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"\rLevel: [{bar}] {db:+6.1f} dB", end='', flush=True)
    
    def run(self, target_angle=None, duration=None):
        """ビームフォーミングを実行"""
        # 目的角度を設定
        if target_angle is None:
            print("\n目的角度を入力してください（度）:")
            print("  -90度 = 左側の話者")
            print("  0度 = 正面（両方）")
            print("  90度 = 右側の話者")
            target_angle = float(input("> "))
        
        # 遅延を計算
        self.delay_samples = self.calculate_delay(target_angle)
        
        # 実行時間を設定
        if duration is None:
            print("\n実行時間を入力してください（秒）:")
            print("（Enterキーのみで手動停止モード）")
            duration_input = input("> ")
            duration = float(duration_input) if duration_input.strip() else None
        
        print(f"\n=== NIDAQ Beamforming 開始 ===")
        print(f"目的角度: {target_angle}度")
        if duration:
            print(f"実行時間: {duration}秒")
        else:
            print("Ctrl+Cで停止してください")
        print()
        
        self.is_running = True
        start_time = time.time()
        
        try:
            with sd.OutputStream(
                device=self.output_device,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self.audio_output_callback
            ):
                with nidaqmx.Task() as task:
                    for channel in config.MIC_CHANNELS:
                        task.ai_channels.add_ai_voltage_chan(channel)
                    
                    task.timing.cfg_samp_clk_timing(
                        self.sample_rate,
                        samps_per_chan=self.chunk_size * 10
                    )
                    
                    print("🎤 録音中... 🔊 再生中...\n")
                    
                    while self.is_running:
                        if duration and (time.time() - start_time) >= duration:
                            break
                        
                        # NIDAQからデータを読み取り
                        data = task.read(number_of_samples_per_channel=self.chunk_size)
                        np_data = np.array(data, dtype=np.float32).T
                        
                        # ビームフォーミングを適用
                        beamformed = self.apply_beamforming(np_data)
                        
                        # レベルメーター表示
                        self.show_level_meter(beamformed)
                        
                        # キューに追加
                        try:
                            self.audio_queue.put(beamformed, block=True, timeout=1)
                        except queue.Full:
                            print("\nQueue full", file=sys.stderr)
        
        except KeyboardInterrupt:
            print("\n\n停止しました")
        except Exception as e:
            print(f"\n\nエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
        
        print("\n処理完了")


def list_nidaq_devices():
    """利用可能なNIDAQデバイスをリスト表示"""
    try:
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        devices = system.devices
        
        print("\n=== 利用可能なNIDAQデバイス ===")
        if devices:
            for device in devices:
                print(f"  - {device.name}")
                print(f"    製品タイプ: {device.product_type}")
                try:
                    print(f"    アナログ入力: {len(device.ai_physical_chans)} チャンネル")
                except:
                    pass
        else:
            print("  NIDAQデバイスが見つかりませんでした")
        print()
        
        return len(devices) > 0
    
    except ImportError:
        print("\n❌ nidaqmxライブラリがインストールされていません")
        return False
    except Exception as e:
        print(f"\n❌ NIDAQデバイスの検出に失敗: {e}")
        return False


def main():
    """メイン関数"""
    print("=" * 60)
    print("NIDAQ Dual-Microphone Beamforming (改良版)")
    print("=" * 60)
    
    if not list_nidaq_devices():
        print("\n設定を確認してください")
        return
    
    try:
        beamformer = NIDAQBeamformer()
    except Exception as e:
        print(f"\n❌ 初期化エラー: {e}")
        return
    
    beamformer.run()


if __name__ == "__main__":
    main()
