"""
AirPods エンドファイア型ビームフォーミング
==========================================

一直線上の配置で話者Aの音声のみをリアルタイム出力

配置:
A ←---→ 左AirPods ←--50cm--→ 右AirPods ←---→ B

必要なライブラリ:
pip install numpy scipy sounddevice

使用方法:
1. AirPodsを50cm離して一直線上に配置
2. 話者AとBが両端に位置
3. このスクリプトを実行
4. 話者Aの音声のみがMacBookスピーカーから出力される
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.fft import fft, ifft
import queue
import threading
import time


class EndfireBeamformer:
    """エンドファイア型ビームフォーミングクラス"""
    
    def __init__(self, 
                 sample_rate=48000,
                 block_size=2048,  # 低遅延のため小さめに設定
                 mic_distance=0.50,  # 50cm
                 sound_speed=343.0,
                 target_direction='left'):  # 'left' = A側, 'right' = B側
        """
        Parameters:
        -----------
        sample_rate : int
            サンプリングレート（Hz）
        block_size : int
            処理ブロックサイズ（小さいほど低遅延）
        mic_distance : float
            左右マイク間の距離（メートル）
        sound_speed : float
            音速（m/s）
        target_direction : str
            目的方向 ('left' = A側, 'right' = B側)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mic_distance = mic_distance
        self.sound_speed = sound_speed
        self.target_direction = target_direction
        
        # 入出力キュー
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # 処理状態
        self.is_running = False
        self.processing_thread = None
        
        # エンドファイア用パラメータ
        self.max_delay_samples = int(self.mic_distance / self.sound_speed * self.sample_rate)
        
        print(f"\n=== エンドファイア型ビームフォーミング ===")
        print(f"マイク間距離: {self.mic_distance*100:.1f} cm")
        print(f"最大遅延: {self.max_delay_samples} サンプル ({self.max_delay_samples/self.sample_rate*1000:.2f} ms)")
        print(f"目的方向: {'A側（左）' if target_direction == 'left' else 'B側（右）'}")
        print(f"ブロックサイズ: {block_size} サンプル ({block_size/sample_rate*1000:.1f} ms)")
    
    def list_audio_devices(self):
        """利用可能なオーディオデバイスをリスト表示"""
        print("\n=== 利用可能なオーディオデバイス ===")
        devices = sd.query_devices()
        
        print("\n【入力デバイス】")
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 2:
                print(f"[{i}] {device['name']}")
                print(f"    入力チャンネル数: {device['max_input_channels']}")
                print(f"    サンプリングレート: {device['default_samplerate']} Hz")
                print()
        
        print("【出力デバイス】")
        for i, device in enumerate(devices):
            if device['max_output_channels'] >= 1:
                print(f"[{i}] {device['name']}")
                print(f"    出力チャンネル数: {device['max_output_channels']}")
                print()
        
        return devices
    
    def select_device(self, device_type='input'):
        """デバイスを選択"""
        devices = sd.query_devices()
        
        if device_type == 'input':
            # AirPodsを自動検出
            airpods_indices = []
            for i, device in enumerate(devices):
                if 'airpods' in device['name'].lower() and device['max_input_channels'] >= 2:
                    airpods_indices.append(i)
            
            if len(airpods_indices) == 1:
                device_id = airpods_indices[0]
                print(f"\nAirPodsを検出しました: {devices[device_id]['name']}")
                return device_id
            elif len(airpods_indices) > 1:
                print("\n複数のAirPodsデバイスが見つかりました:")
                for idx in airpods_indices:
                    print(f"[{idx}] {devices[idx]['name']}")
                device_id = int(input("使用するデバイス番号を入力してください: "))
                return device_id
            else:
                print("\nAirPodsが見つかりませんでした。")
                self.list_audio_devices()
                device_id = int(input("入力デバイス番号を入力してください: "))
                return device_id
        else:
            # 出力デバイス（MacBookスピーカー）
            macbook_indices = []
            for i, device in enumerate(devices):
                if ('macbook' in device['name'].lower() or 
                    'built-in' in device['name'].lower()) and device['max_output_channels'] >= 1:
                    macbook_indices.append(i)
            
            if macbook_indices:
                device_id = macbook_indices[0]
                print(f"\n出力デバイス: {devices[device_id]['name']}")
                return device_id
            else:
                print("\nMacBookスピーカーが見つかりませんでした。")
                self.list_audio_devices()
                device_id = int(input("出力デバイス番号を入力してください: "))
                return device_id
    
    def endfire_beamforming_time_domain(self, left_channel, right_channel):
        """
        時間領域でのエンドファイア型ビームフォーミング（低遅延版）
        
        Parameters:
        -----------
        left_channel : ndarray
            左チャンネルの音声データ
        right_channel : ndarray
            右チャンネルの音声データ
        
        Returns:
        --------
        output : ndarray
            ビームフォーミング後の音声
        """
        # 遅延サンプル数を計算
        delay_samples = self.max_delay_samples
        
        if self.target_direction == 'left':
            # A側（左）を強調: 左マイクをそのまま、右マイクを遅延させて減算
            # 左から来る音は同相、右から来る音は逆相になる
            delayed_right = np.pad(right_channel, (delay_samples, 0), mode='constant')[:-delay_samples]
            output = left_channel - 0.5 * delayed_right
        else:
            # B側（右）を強調: 右マイクをそのまま、左マイクを遅延させて減算
            delayed_left = np.pad(left_channel, (delay_samples, 0), mode='constant')[:-delay_samples]
            output = right_channel - 0.5 * delayed_left
        
        return output
    
    def endfire_beamforming_frequency_domain(self, left_channel, right_channel):
        """
        周波数領域でのエンドファイア型ビームフォーミング（高品質版）
        
        Parameters:
        -----------
        left_channel : ndarray
            左チャンネルの音声データ
        right_channel : ndarray
            右チャンネルの音声データ
        
        Returns:
        --------
        output : ndarray
            ビームフォーミング後の音声
        """
        # FFTで周波数領域に変換
        left_fft = fft(left_channel)
        right_fft = fft(right_channel)
        
        # 周波数ビンを計算
        freqs = np.fft.fftfreq(len(left_channel), 1/self.sample_rate)
        
        # 出力信号を初期化
        output_fft = np.zeros_like(left_fft, dtype=complex)
        
        # 各周波数ビンに対してエンドファイアビームフォーミング
        for i, freq in enumerate(freqs[:len(freqs)//2 + 1]):
            if freq == 0:
                # DC成分
                if self.target_direction == 'left':
                    output_fft[i] = left_fft[i]
                else:
                    output_fft[i] = right_fft[i]
                continue
            
            # 時間遅延に対応する位相遅延
            time_delay = self.mic_distance / self.sound_speed
            phase_delay = 2 * np.pi * abs(freq) * time_delay
            
            if self.target_direction == 'left':
                # A側（左）を強調
                # 左マイク + 右マイク × 位相遅延 × 減衰係数
                weight_left = 1.0
                weight_right = -0.5 * np.exp(-1j * phase_delay)
                output_fft[i] = weight_left * left_fft[i] + weight_right * right_fft[i]
            else:
                # B側（右）を強調
                weight_right = 1.0
                weight_left = -0.5 * np.exp(-1j * phase_delay)
                output_fft[i] = weight_right * right_fft[i] + weight_left * left_fft[i]
            
            # 負の周波数成分も対称に設定
            if i > 0 and i < len(freqs)//2:
                output_fft[-i] = output_fft[i].conj()
        
        # 時間領域に戻す
        output = np.real(ifft(output_fft))
        
        return output
    
    def apply_noise_reduction(self, audio_data):
        """簡易ノイズリダクション"""
        # ハイパスフィルタ（低周波ノイズ除去）
        sos = signal.butter(4, 100, 'hp', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, audio_data)
        
        # ローパスフィルタ（高周波ノイズ除去）
        sos = signal.butter(4, 8000, 'lp', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, filtered)
        
        return filtered
    
    def processing_loop(self, method='frequency'):
        """音声処理ループ（別スレッドで実行）"""
        print("\n処理スレッド開始...")
        
        while self.is_running:
            try:
                # 入力キューからデータを取得
                audio_data = self.input_queue.get(timeout=0.1)
                
                # 左右チャンネルを分離
                left_channel = audio_data[:, 0]
                right_channel = audio_data[:, 1]
                
                # ビームフォーミング
                if method == 'time':
                    output = self.endfire_beamforming_time_domain(left_channel, right_channel)
                else:
                    output = self.endfire_beamforming_frequency_domain(left_channel, right_channel)
                
                # ノイズリダクション
                output = self.apply_noise_reduction(output)
                
                # 正規化（クリッピング防止）
                max_val = np.max(np.abs(output))
                if max_val > 0.8:
                    output = output * 0.8 / max_val
                
                # 出力キューに追加
                try:
                    self.output_queue.put_nowait(output)
                except queue.Full:
                    # キューが満杯の場合は古いデータを破棄
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(output)
                    except:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"処理エラー: {e}")
                continue
        
        print("処理スレッド終了")
    
    def audio_input_callback(self, indata, frames, time_info, status):
        """入力ストリームのコールバック"""
        if status:
            print(f"入力Status: {status}")
        
        try:
            self.input_queue.put_nowait(indata.copy())
        except queue.Full:
            # キューが満杯の場合は警告（データ損失）
            pass
    
    def audio_output_callback(self, outdata, frames, time_info, status):
        """出力ストリームのコールバック"""
        if status:
            print(f"出力Status: {status}")
        
        try:
            data = self.output_queue.get_nowait()
            outdata[:, 0] = data
        except queue.Empty:
            # データがない場合は無音
            outdata.fill(0)
    
    def run_realtime(self, method='frequency', duration=None):
        """
        リアルタイム処理を実行
        
        Parameters:
        -----------
        method : str
            処理方法 ('time' = 時間領域, 'frequency' = 周波数領域)
        duration : float or None
            実行時間（秒）。Noneの場合は手動停止まで継続
        """
        print(f"\n=== リアルタイム話者分離開始 ===")
        print(f"処理方法: {'時間領域（低遅延）' if method == 'time' else '周波数領域（高品質）'}")
        print(f"目的話者: {'A（左側）' if self.target_direction == 'left' else 'B（右側）'}")
        
        if duration:
            print(f"実行時間: {duration}秒")
        else:
            print("Ctrl+Cで停止してください")
        
        # デバイスを選択
        input_device = self.select_device('input')
        output_device = self.select_device('output')
        
        self.is_running = True
        
        # 処理スレッドを開始
        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            args=(method,),
            daemon=True
        )
        self.processing_thread.start()
        
        start_time = time.time()
        
        try:
            # 入出力ストリームを同時に開始
            with sd.InputStream(device=input_device,
                              channels=2,
                              samplerate=self.sample_rate,
                              blocksize=self.block_size,
                              callback=self.audio_input_callback), \
                 sd.OutputStream(device=output_device,
                               channels=1,
                               samplerate=self.sample_rate,
                               blocksize=self.block_size,
                               callback=self.audio_output_callback):
                
                print("\n🎤 録音中... 🔊 再生中...")
                print("話者Aの音声がMacBookスピーカーから出力されます\n")
                
                while self.is_running:
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    # ステータス表示
                    elapsed = time.time() - start_time
                    queue_status = f"入力キュー: {self.input_queue.qsize()}/10, 出力キュー: {self.output_queue.qsize()}/10"
                    print(f"\r経過時間: {elapsed:.1f}秒 | {queue_status}", end='', flush=True)
                    
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n停止しました")
        except Exception as e:
            print(f"\n\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
        
        print("\n処理完了")


def main():
    """メイン関数"""
    print("=" * 60)
    print("AirPods エンドファイア型ビームフォーミング")
    print("話者Aの音声をリアルタイム出力")
    print("=" * 60)
    print()
    print("配置:")
    print("  A ←---→ 左AirPods ←--50cm--→ 右AirPods ←---→ B")
    print()
    
    # ビームフォーマーを初期化
    beamformer = EndfireBeamformer(
        sample_rate=48000,
        block_size=2048,  # 低遅延
        mic_distance=0.50,  # 50cm
        target_direction='left'  # A側を出力
    )
    
    # 利用可能なデバイスを表示
    beamformer.list_audio_devices()
    
    # 実行時間を設定
    print("\n実行時間を入力してください（秒）:")
    print("（Enterキーのみで手動停止モード）")
    duration_input = input("> ")
    duration = float(duration_input) if duration_input.strip() else None
    
    # 処理方法を選択
    print("\n処理方法を選択してください:")
    print("1: 周波数領域（高品質・推奨）")
    print("2: 時間領域（超低遅延）")
    method_choice = input("> ").strip()
    method = 'time' if method_choice == '2' else 'frequency'
    
    # リアルタイム処理を実行
    beamformer.run_realtime(method=method, duration=duration)
    
    print("\n処理が完了しました！")


if __name__ == "__main__":
    main()
