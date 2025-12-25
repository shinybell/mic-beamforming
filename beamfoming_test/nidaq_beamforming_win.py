"""
NIDAQ Dual-Microphone Beamforming System
=========================================

NIDAQハードウェアの2つのマイクを使用したリアルタイムビームフォーミング

必要なライブラリ:
pip install numpy scipy sounddevice nidaqmx

使用方法:
1. NIDAQに2つのマイクを接続（Dev10/ai0, Dev10/ai1）
2. nidaq_config.pyで設定を確認
3. このスクリプトを実行
4. 目的角度を入力
5. ビームフォーミングされた音声がスピーカーから出力（Windows/Macに対応）

特徴:
- リアルタイムストリーミング処理（超低遅延）
- 入力直後に処理して即座に出力
- Windows/Mac両対応したスピーカー選択
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
    """NIDAQを使用したビームフォーミングクラス"""
    
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
        
        # 周波数ビンを事前計算
        self.freqs = np.fft.rfftfreq(self.chunk_size, d=1.0/self.sample_rate)
        self.num_bins = len(self.freqs)
        
        # ステアリングベクトル（重み）
        self.steering_vector = np.ones((self.num_bins, self.num_mics), dtype=np.complex64)
        
        # 現在の目的角度
        self.current_angle = config.DEFAULT_TARGET_ANGLE
        
        # ハイパスフィルタの設計
        self.setup_filters()
        
        
        # エコーキャンセレーション用バッファ
        if config.ENABLE_ECHO_CANCELLATION:
            self.echo_buffer = np.zeros(config.ECHO_BUFFER_SIZE)
        
        # レベルメーター用
        self.chunk_counter = 0
        
        # 出力デバイスを自動選択
        self.output_device = self.select_output_device()
        
        print(f"\n=== NIDAQ Beamformer 初期化完了 ===")
        print(f"サンプリングレート: {self.sample_rate} Hz")
        print(f"チャンクサイズ: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f} ms)")
        print(f"マイク数: {self.num_mics}")
        print(f"マイク間距離: {config.MIC_SPACING*100:.1f} cm")
        print(f"周波数ビン数: {self.num_bins}")
    
    def select_output_device(self):
        """
        出力デバイスを自動選択（Windows/Mac対応）
        
        Returns:
        --------
        int or None
            出力デバイスID（Noneの場合はデフォルト）
        """
        import platform
        
        devices = sd.query_devices()
        os_type = platform.system()
        
        print(f"\nOS: {os_type}")
        
        # Windowsの場合
        if os_type == "Windows":
            # Windowsスピーカーを探す
            for i, device in enumerate(devices):
                name_lower = device['name'].lower()
                if device['max_output_channels'] >= 1:
                    # スピーカー、ヘッドフォン、またはデフォルトデバイスを優先
                    if any(keyword in name_lower for keyword in ['speaker', 'スピーカー', 'headphone', 'ヘッドフォン', 'default']):
                        print(f"出力デバイス: [{i}] {device['name']}")
                        return i
            
            # 見つからない場合はデフォルト
            try:
                default_output = sd.query_devices(kind='output')
                print(f"出力デバイス: [デフォルト] {default_output['name']}")
                return None
            except:
                print("出力デバイス: デフォルト")
                return None
        
        # Macの場合
        elif os_type == "Darwin":
            # MacBookスピーカーを探す
            for i, device in enumerate(devices):
                name_lower = device['name'].lower()
                if ('macbook' in name_lower or 'built-in' in name_lower) and device['max_output_channels'] >= 1:
                    print(f"出力デバイス: [{i}] {device['name']}")
                    return i
            
            # 見つからない場合はデフォルト
            try:
                default_output = sd.query_devices(kind='output')
                print(f"出力デバイス: [デフォルト] {default_output['name']}")
                return None
            except:
                print("出力デバイス: デフォルト")
                return None
        
        # その他のOS
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
    
    def update_steering_vector(self, theta_deg):
        """
        指定角度に対するステアリングベクトルを更新
        
        Parameters:
        -----------
        theta_deg : float
            目的角度（度）
            0度 = 正面（マイクアレイに垂直）
            90度 = 右側（エンドファイア）
            -90度 = 左側
        """
        self.current_angle = theta_deg
        theta_rad = np.deg2rad(theta_deg)
        
        # 方向ベクトル（音源方向）
        # 0度 = 正面（Y軸正方向）、90度 = 右（X軸正方向）
        ux = np.sin(theta_rad)
        uy = np.cos(theta_rad)
        uz = 0.0
        
        # 各マイクの時間遅延を計算
        # マイクが音源に近いほど、信号が早く到達する
        delays = np.dot(self.mic_positions, np.array([ux, uy, uz])) / config.SPEED_OF_SOUND
        
        # 各周波数に対する位相シフトを計算
        omega = 2 * np.pi * self.freqs
        
        # ステアリングベクトル = exp(j * omega * tau)
        # 信号を位相整列させるための補償
        self.steering_vector = np.exp(1j * np.outer(omega, delays))
        
        # 正規化（Delay-and-Sum）
        self.steering_vector /= self.num_mics
        
        print(f"\nステアリングベクトル更新: {theta_deg}度")
    
    def apply_beamforming(self, multichannel_chunk):
        """
        ビームフォーミングを適用
        
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
        # 1. FFT（周波数領域に変換）
        spectrum = np.fft.rfft(multichannel_chunk, axis=0)
        
        # 2. ステアリングベクトルを適用
        # spectrum: (num_bins, num_mics)
        # steering_vector: (num_bins, num_mics)
        beamformed_spectrum = np.sum(spectrum * self.steering_vector, axis=1)
        
        # 3. IFFT（時間領域に戻す）
        beamformed_chunk = np.fft.irfft(beamformed_spectrum, n=self.chunk_size)
        
        return beamformed_chunk.astype(np.float32)
    
    def enhance_audio_quality(self, audio_data):
        """
        音質向上処理
        
        Parameters:
        -----------
        audio_data : ndarray
            入力音声データ
        
        Returns:
        --------
        ndarray
            処理済み音声データ
        """
        # 1. ハイパスフィルタ（低周波ノイズ除去）
        if self.filter_enabled:
            filtered = signal.filtfilt(self.filter_b, self.filter_a, audio_data)
        else:
            filtered = audio_data
        
        # 2. ノイズゲート（小さなノイズを除去）
        rms = np.sqrt(np.mean(filtered**2))
        if rms < config.NOISE_GATE_THRESHOLD:
            filtered = filtered * 0.1  # ノイズを大幅に減衰
        
        # 3. ゲイン調整
        filtered = filtered * config.GAIN
        
        # 4. クリッピング防止
        filtered = np.clip(filtered, -1.0, 1.0)
        
        # 5. エコーキャンセレーション（オプション）
        if config.ENABLE_ECHO_CANCELLATION:
            if len(self.echo_buffer) >= len(filtered):
                echo_reduction = self.echo_buffer[:len(filtered)] * 0.1
                filtered = filtered - echo_reduction
            
            # エコーバッファを更新
            self.echo_buffer = np.roll(self.echo_buffer, -len(filtered))
            self.echo_buffer[-len(filtered):] = filtered
        
        return filtered
    
    def audio_output_callback(self, outdata, frames, time_info, status):
        """
        Sounddeviceの出力コールバック
        キューからデータを取得してスピーカーに出力
        """
        if status:
            print(f"出力Status: {status}", file=sys.stderr)
        
        try:
            # キューからデータを取得
            data = self.audio_queue.get(block=False)
            
            # 音質向上処理を適用
            processed_data = self.enhance_audio_quality(data)
            
            # データサイズチェック
            if len(processed_data) < frames:
                outdata[:len(processed_data)] = processed_data.reshape(-1, 1)
                outdata[len(processed_data):] = 0
                print("Buffer underrun (partial)", file=sys.stderr)
            else:
                outdata[:] = processed_data.reshape(-1, 1)
        
        except queue.Empty:
            # キューが空の場合は無音を出力
            outdata[:] = 0
            if self.is_running:
                print("Buffer underflow: Outputting silence", file=sys.stderr)
    
    def show_level_meter(self, data):
        """簡易レベルメーター表示"""
        if not config.SHOW_LEVEL_METER:
            return
        
        self.chunk_counter += 1
        if self.chunk_counter % config.LEVEL_METER_UPDATE_INTERVAL != 0:
            return
        
        # RMS計算
        rms = np.sqrt(np.mean(data**2))
        
        # dB変換
        if rms > 1e-10:
            db = 20 * np.log10(rms)
        else:
            db = -100
        
        # バー表示
        bar_length = int(max(0, min(50, (db + 60) / 60 * 50)))
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"\rLevel: [{bar}] {db:+6.1f} dB", end='', flush=True)
    
    def run(self, target_angle=None, duration=None):
        """
        ビームフォーミングを実行
        
        Parameters:
        -----------
        target_angle : float or None
            目的角度（度）。Noneの場合はユーザーに入力を求める
        duration : float or None
            実行時間（秒）。Noneの場合は手動停止まで継続
        """
        # 目的角度を設定
        if target_angle is None:
            print("\n目的角度を入力してください（度）:")
            print("  0度 = 正面（マイクアレイに垂直）")
            print("  90度 = 右側（エンドファイア）")
            print("  -90度 = 左側")
            target_angle = float(input("> "))
        
        # ステアリングベクトルを更新
        self.update_steering_vector(target_angle)
        
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
            # 出力ストリームを開始（Windows/Mac自動選択）
            with sd.OutputStream(
                device=self.output_device,  # 自動選択されたデバイス
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self.audio_output_callback
            ):
                # NIDAQタスクを開始
                with nidaqmx.Task() as task:
                    # マイクチャンネルを追加
                    for channel in config.MIC_CHANNELS:
                        task.ai_channels.add_ai_voltage_chan(channel)
                    
                    # タイミング設定
                    task.timing.cfg_samp_clk_timing(
                        self.sample_rate,
                        samps_per_chan=self.chunk_size * 10
                    )
                    
                    print("🎤 録音中... 🔊 再生中...\n")
                    print("ℹ️  リアルタイムストリーミング処理:")
                    print("   入力 → ビームフォーミング → 出力 (遅延: ~100ms)\n")
                    
                    while self.is_running:
                        # 時間チェック
                        if duration and (time.time() - start_time) >= duration:
                            break
                        
                        # ===== ストリーミング処理ループ =====
                        # 1. NIDAQからリアルタイムでデータを読み取り
                        # 戻り値: list of lists [[ch0_samples], [ch1_samples]]
                        data = task.read(number_of_samples_per_channel=self.chunk_size)
                        
                        # 2. numpy配列に変換して転置
                        # shape: (num_channels, chunk_size) -> (chunk_size, num_channels)
                        np_data = np.array(data, dtype=np.float32).T
                        
                        # 3. ビームフォーミングを即座に適用（周波数領域処理）
                        beamformed = self.apply_beamforming(np_data)
                        
                        # 4. レベルメーター表示
                        self.show_level_meter(beamformed)
                        
                        # 5. 処理済み音声を即座にキューに追加
                        # → 出力コールバックが自動的に取得してスピーカーから再生
                        try:
                            self.audio_queue.put(beamformed, block=True, timeout=1)
                        except queue.Full:
                            print("\nQueue full: Dropping data", file=sys.stderr)
        
        except KeyboardInterrupt:
            print("\n\n停止しました")
        except Exception as e:
            print(f"\n\nエラーが発生しました: {e}")
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
        print("インストール: pip install nidaqmx")
        return False
    except Exception as e:
        print(f"\n❌ NIDAQデバイスの検出に失敗: {e}")
        return False


def main():
    """メイン関数"""
    print("=" * 60)
    print("NIDAQ Dual-Microphone Beamforming")
    print("=" * 60)
    
    # NIDAQデバイスを確認
    if not list_nidaq_devices():
        print("\n設定を確認してください:")
        print("1. NIDAQハードウェアが接続されているか")
        print("2. NI-DAQmxドライバがインストールされているか")
        print("3. nidaq_config.pyのDEVICE_NAMEが正しいか")
        return
    
    # ビームフォーマーを初期化
    try:
        beamformer = NIDAQBeamformer()
    except Exception as e:
        print(f"\n❌ 初期化エラー: {e}")
        return
    
    # ビームフォーミングを実行
    beamformer.run()


if __name__ == "__main__":
    main()
