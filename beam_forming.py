"""
AirPods Dual-Microphone Beamforming for Speaker Separation
===========================================================

このスクリプトは、AirPodsの左右のマイクから入力された音声を使用して、
2人の話者を空間的に分離します。

必要なライブラリ:
pip install numpy scipy sounddevice soundfile matplotlib

使用方法:
1. AirPodsをMacBookに接続
2. システム環境設定でAirPodsを入力デバイスとして選択
3. このスクリプトを実行
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from datetime import datetime
import queue
import threading
import time


class AirPodsBeamformer:
    """AirPodsの左右マイクを使用したビームフォーミングクラス"""
    
    def __init__(self, 
                 sample_rate=48000,
                 block_size=4096,
                 mic_distance=0.15,  # AirPods間の距離（メートル）
                 sound_speed=343.0):  # 音速（m/s）
        """
        Parameters:
        -----------
        sample_rate : int
            サンプリングレート（Hz）
        block_size : int
            処理ブロックサイズ
        mic_distance : float
            左右マイク間の距離（メートル）
        sound_speed : float
            音速（m/s）
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mic_distance = mic_distance
        self.sound_speed = sound_speed
        
        # 音声データキュー
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # 分離された音声を保存
        self.separated_audio = {
            'speaker1': [],  # 左側の話者
            'speaker2': []   # 右側の話者
        }
        
        # ビームフォーミングパラメータ
        self.angles = np.array([-45, 45])  # 2人の話者の想定角度（度）
        
    def list_audio_devices(self):
        """利用可能なオーディオデバイスをリスト表示"""
        print("\n=== 利用可能なオーディオデバイス ===")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 2:
                print(f"[{i}] {device['name']}")
                print(f"    入力チャンネル数: {device['max_input_channels']}")
                print(f"    サンプリングレート: {device['default_samplerate']} Hz")
                print()
        return devices
    
    def select_airpods_device(self):
        """AirPodsデバイスを自動検出または手動選択"""
        devices = sd.query_devices()
        
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
            device_id = int(input("使用するデバイス番号を入力してください: "))
            return device_id
    
    def calculate_steering_vector(self, angle_deg, frequency):
        """
        指定された角度に対するステアリングベクトルを計算
        
        Parameters:
        -----------
        angle_deg : float
            ビーム方向の角度（度）
        frequency : float
            周波数（Hz）
        
        Returns:
        --------
        steering_vector : ndarray
            ステアリングベクトル
        """
        angle_rad = np.deg2rad(angle_deg)
        
        # 時間遅延を計算
        time_delay = (self.mic_distance * np.sin(angle_rad)) / self.sound_speed
        
        # 位相遅延を計算
        phase_delay = 2 * np.pi * frequency * time_delay
        
        # ステアリングベクトル [左マイク, 右マイク]
        steering_vector = np.array([
            np.exp(-1j * phase_delay / 2),  # 左マイク
            np.exp(1j * phase_delay / 2)    # 右マイク
        ])
        
        return steering_vector
    
    def delay_and_sum_beamforming(self, left_channel, right_channel, angle_deg):
        """
        遅延和ビームフォーミングを実行
        
        Parameters:
        -----------
        left_channel : ndarray
            左チャンネルの音声データ
        right_channel : ndarray
            右チャンネルの音声データ
        angle_deg : float
            ビーム方向の角度（度）
        
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
        
        # 各周波数ビンに対してビームフォーミング
        for i, freq in enumerate(freqs[:len(freqs)//2]):
            if freq == 0:
                continue
            
            # ステアリングベクトルを計算
            steering_vec = self.calculate_steering_vector(angle_deg, abs(freq))
            
            # 入力信号ベクトル
            input_vec = np.array([left_fft[i], right_fft[i]])
            
            # ビームフォーミング（内積）
            output_fft[i] = np.dot(steering_vec.conj(), input_vec) / 2
            
            # 負の周波数成分も対称に設定
            if i > 0:
                output_fft[-i] = output_fft[i].conj()
        
        # 時間領域に戻す
        output = np.real(ifft(output_fft))
        
        return output
    
    def mvdr_beamforming(self, left_channel, right_channel, angle_deg):
        """
        MVDR (Minimum Variance Distortionless Response) ビームフォーミング
        
        Parameters:
        -----------
        left_channel : ndarray
            左チャンネルの音声データ
        right_channel : ndarray
            右チャンネルの音声データ
        angle_deg : float
            目的方向の角度（度）
        
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
        
        # 各周波数ビンに対してMVDRビームフォーミング
        for i, freq in enumerate(freqs[:len(freqs)//2]):
            if freq == 0:
                continue
            
            # ステアリングベクトルを計算
            steering_vec = self.calculate_steering_vector(angle_deg, abs(freq))
            
            # 入力信号ベクトル
            input_vec = np.array([left_fft[i], right_fft[i]])
            
            # 共分散行列を推定（簡易版）
            R = np.outer(input_vec, input_vec.conj()) + 1e-6 * np.eye(2)
            
            # MVDRウェイト計算
            try:
                R_inv = np.linalg.inv(R)
                numerator = np.dot(R_inv, steering_vec)
                denominator = np.dot(steering_vec.conj(), np.dot(R_inv, steering_vec))
                mvdr_weight = numerator / (denominator + 1e-10)
                
                # ビームフォーミング
                output_fft[i] = np.dot(mvdr_weight.conj(), input_vec)
            except:
                # 逆行列計算が失敗した場合は遅延和にフォールバック
                output_fft[i] = np.dot(steering_vec.conj(), input_vec) / 2
            
            # 負の周波数成分も対称に設定
            if i > 0:
                output_fft[-i] = output_fft[i].conj()
        
        # 時間領域に戻す
        output = np.real(ifft(output_fft))
        
        return output
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        オーディオストリームのコールバック関数
        """
        if status:
            print(f"Status: {status}")
        
        # キューにデータを追加
        self.audio_queue.put(indata.copy())
    
    def process_audio_stream(self, duration=None, method='delay_sum'):
        """
        リアルタイム音声処理
        
        Parameters:
        -----------
        duration : float or None
            録音時間（秒）。Noneの場合は手動停止まで継続
        method : str
            ビームフォーミング手法 ('delay_sum' or 'mvdr')
        """
        print(f"\n=== リアルタイム話者分離開始 ===")
        print(f"ビームフォーミング手法: {method}")
        print(f"話者1の方向: {self.angles[0]}度（左側）")
        print(f"話者2の方向: {self.angles[1]}度（右側）")
        
        if duration:
            print(f"録音時間: {duration}秒")
        else:
            print("Ctrl+Cで停止してください")
        
        # デバイスを選択
        device_id = self.select_airpods_device()
        
        # ビームフォーミング関数を選択
        if method == 'mvdr':
            beamform_func = self.mvdr_beamforming
        else:
            beamform_func = self.delay_and_sum_beamforming
        
        self.is_recording = True
        start_time = time.time()
        
        try:
            # 音声ストリームを開始
            with sd.InputStream(device=device_id,
                              channels=2,
                              samplerate=self.sample_rate,
                              blocksize=self.block_size,
                              callback=self.audio_callback):
                
                print("\n録音中...")
                
                while self.is_recording:
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    try:
                        # キューからデータを取得
                        audio_data = self.audio_queue.get(timeout=0.1)
                        
                        # 左右チャンネルを分離
                        left_channel = audio_data[:, 0]
                        right_channel = audio_data[:, 1]
                        
                        # 各話者方向にビームフォーミング
                        speaker1_audio = beamform_func(left_channel, right_channel, self.angles[0])
                        speaker2_audio = beamform_func(left_channel, right_channel, self.angles[1])
                        
                        # 分離された音声を保存
                        self.separated_audio['speaker1'].append(speaker1_audio)
                        self.separated_audio['speaker2'].append(speaker2_audio)
                        
                        # 進捗表示
                        elapsed = time.time() - start_time
                        print(f"\r経過時間: {elapsed:.1f}秒", end='', flush=True)
                        
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
        
        except KeyboardInterrupt:
            print("\n\n録音を停止しました")
        finally:
            self.is_recording = False
        
        print("\n処理完了")
    
    def save_separated_audio(self, output_dir='.'):
        """
        分離された音声をファイルに保存
        
        Parameters:
        -----------
        output_dir : str
            出力ディレクトリ
        """
        if not self.separated_audio['speaker1']:
            print("保存する音声データがありません")
            return
        
        # タイムスタンプを生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 各話者の音声を結合
        speaker1_full = np.concatenate(self.separated_audio['speaker1'])
        speaker2_full = np.concatenate(self.separated_audio['speaker2'])
        
        # 正規化
        speaker1_full = speaker1_full / (np.max(np.abs(speaker1_full)) + 1e-10)
        speaker2_full = speaker2_full / (np.max(np.abs(speaker2_full)) + 1e-10)
        
        # ファイルに保存
        filename1 = f"{output_dir}/speaker1_left_{timestamp}.wav"
        filename2 = f"{output_dir}/speaker2_right_{timestamp}.wav"
        
        sf.write(filename1, speaker1_full, self.sample_rate)
        sf.write(filename2, speaker2_full, self.sample_rate)
        
        print(f"\n=== 音声ファイルを保存しました ===")
        print(f"話者1（左側）: {filename1}")
        print(f"話者2（右側）: {filename2}")
        print(f"サンプリングレート: {self.sample_rate} Hz")
        print(f"録音時間: {len(speaker1_full)/self.sample_rate:.2f}秒")
    
    def visualize_separation(self):
        """分離結果を可視化"""
        if not self.separated_audio['speaker1']:
            print("可視化する音声データがありません")
            return
        
        # 音声を結合
        speaker1_full = np.concatenate(self.separated_audio['speaker1'])
        speaker2_full = np.concatenate(self.separated_audio['speaker2'])
        
        # 時間軸を作成
        time_axis = np.arange(len(speaker1_full)) / self.sample_rate
        
        # プロット
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 話者1の波形
        axes[0].plot(time_axis, speaker1_full, linewidth=0.5)
        axes[0].set_title('話者1（左側）の分離音声')
        axes[0].set_ylabel('振幅')
        axes[0].grid(True, alpha=0.3)
        
        # 話者2の波形
        axes[1].plot(time_axis, speaker2_full, linewidth=0.5)
        axes[1].set_title('話者2（右側）の分離音声')
        axes[1].set_ylabel('振幅')
        axes[1].grid(True, alpha=0.3)
        
        # スペクトログラム（話者1）
        f, t, Sxx = signal.spectrogram(speaker1_full, self.sample_rate, nperseg=1024)
        axes[2].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        axes[2].set_title('話者1のスペクトログラム')
        axes[2].set_ylabel('周波数 (Hz)')
        axes[2].set_xlabel('時間 (秒)')
        axes[2].set_ylim([0, 8000])
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"separation_result_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"\n可視化結果を保存しました: {filename}")
        
        plt.show()


def main():
    """メイン関数"""
    print("=" * 60)
    print("AirPods Dual-Microphone Beamforming")
    print("話者分離システム")
    print("=" * 60)
    
    # ビームフォーマーを初期化
    beamformer = AirPodsBeamformer(
        sample_rate=48000,
        block_size=4096,
        mic_distance=0.15  # AirPods間の距離を調整可能
    )
    
    # 利用可能なデバイスを表示
    beamformer.list_audio_devices()
    
    # 録音時間を設定
    print("\n録音時間を入力してください（秒）:")
    print("（Enterキーのみで手動停止モード）")
    duration_input = input("> ")
    duration = float(duration_input) if duration_input.strip() else None
    
    # ビームフォーミング手法を選択
    print("\nビームフォーミング手法を選択してください:")
    print("1: Delay-and-Sum（遅延和）- 高速・安定")
    print("2: MVDR - 高精度・計算量大")
    method_choice = input("> ").strip()
    method = 'mvdr' if method_choice == '2' else 'delay_sum'
    
    # 音声処理を実行
    beamformer.process_audio_stream(duration=duration, method=method)
    
    # 結果を保存
    beamformer.save_separated_audio()
    
    # 可視化
    print("\n結果を可視化しますか？ (y/n)")
    if input("> ").strip().lower() == 'y':
        beamformer.visualize_separation()
    
    print("\n処理が完了しました！")


if __name__ == "__main__":
    main()