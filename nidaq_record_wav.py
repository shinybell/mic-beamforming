"""
NIDAQ Dual-Microphone WAV Recorder
===================================

NIDAQハードウェアの2つのマイクから音声を録音してWAVファイルに保存

必要なライブラリ:
pip install nidaqmx numpy scipy

使用方法:
1. NIDAQに2つのマイクを接続
2. デバイス名とチャンネルを確認・設定
3. python nidaq_record_wav.py
4. 録音が終了したらCtrl+Cで停止
"""

import nidaqmx
import numpy as np
from scipy.io import wavfile
import datetime
import sys
import signal


class NIDAQRecorder:
    """NIDAQを使用した2マイク録音クラス"""

    def __init__(self, device_name="Dev10", sample_rate=44100, record_time=10):
        """
        初期化

        Parameters:
        -----------
        device_name : str
            NIDAQデバイス名（例: "Dev10"）
        sample_rate : int
            サンプリングレート（Hz）
        record_time : float
            録音時間（秒）、Noneの場合は手動停止まで録音
        """
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.record_time = record_time

        # チャンネル設定
        self.channels = [
            f"{device_name}/ai0",  # マイク1
            f"{device_name}/ai1",  # マイク2
        ]

        self.is_recording = True
        self.recorded_data = []

    def setup_signal_handler(self):
        """Ctrl+C でキャプチャを停止するためのシグナルハンドラ"""

        def signal_handler(sig, frame):
            print("\n録音を停止しています...")
            self.is_recording = False

        signal.signal(signal.SIGINT, signal_handler)

    def record(self):
        """録音を実行"""
        print(f"デバイス: {self.device_name}")
        print(f"チャンネル: {self.channels}")
        print(f"サンプリングレート: {self.sample_rate} Hz")

        if self.record_time:
            print(f"録音時間: {self.record_time} 秒")
            samples_to_read = int(self.sample_rate * self.record_time)
        else:
            print("録音時間: 手動停止まで（Ctrl+C で停止）")
            samples_to_read = nidaqmx.constants.READ_ALL_AVAILABLE

        print("\n録音を開始します...")
        self.setup_signal_handler()

        try:
            with nidaqmx.Task() as task:
                # 2つのマイクチャンネルを追加
                for channel in self.channels:
                    task.ai_channels.add_ai_voltage_chan(
                        channel, min_val=-10.0, max_val=10.0
                    )

                # サンプリングレートを設定
                task.timing.cfg_samp_clk_timing(
                    rate=self.sample_rate,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                )

                # 録音開始
                task.start()

                if self.record_time:
                    # 指定時間録音
                    data = task.read(
                        number_of_samples_per_channel=samples_to_read,
                        timeout=self.record_time + 5,
                    )
                    self.recorded_data = np.array(data).T
                    print(f"録音完了: {len(self.recorded_data)} サンプル")
                else:
                    # 手動停止まで録音
                    chunk_size = self.sample_rate  # 1秒ごとに読み込み
                    while self.is_recording:
                        data = task.read(
                            number_of_samples_per_channel=chunk_size, timeout=2.0
                        )
                        chunk = np.array(data).T
                        self.recorded_data.append(chunk)
                        print(f"録音中... {len(self.recorded_data)} 秒", end="\r")

                    # データを結合
                    self.recorded_data = np.vstack(self.recorded_data)
                    print(f"\n録音完了: {len(self.recorded_data)} サンプル")

                task.stop()

        except nidaqmx.errors.DaqError as e:
            print(f"\nNIDAQエラー: {e}")
            print("\nデバイス名を確認してください:")
            print(
                'python -c "import nidaqmx; print(nidaqmx.system.System.local().devices)"'
            )
            sys.exit(1)
        except Exception as e:
            print(f"\nエラー: {e}")
            sys.exit(1)

    def save_wav(self, filename_prefix=None):
        """録音データを2つのモノラルWAVファイルに保存"""
        if len(self.recorded_data) == 0:
            print("録音データがありません")
            return

        # ファイル名が指定されていない場合、タイムスタンプを使用
        if filename_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"nidaq_recording_{timestamp}"
        else:
            # .wavを削除
            if filename_prefix.endswith(".wav"):
                filename_prefix = filename_prefix[:-4]

        duration = len(self.recorded_data) / self.sample_rate

        # 各チャンネルを個別に保存
        for ch_idx in range(self.recorded_data.shape[1]):
            # チャンネルデータを取得
            channel_data = self.recorded_data[:, ch_idx]

            # データを-1.0～1.0に正規化
            max_val = np.abs(channel_data).max()
            if max_val > 0:
                normalized_data = channel_data / max_val
            else:
                normalized_data = channel_data

            # 16bitに変換
            audio_data = (normalized_data * 32767).astype(np.int16)

            # ファイル名を作成（ch1, ch2）
            filename = f"{filename_prefix}_ch{ch_idx + 1}.wav"

            # WAVファイルに保存（モノラル）
            wavfile.write(filename, self.sample_rate, audio_data)

            print(f"\nチャンネル{ch_idx + 1} 保存完了:")
            print(f"  ファイル名: {filename}")
            print(f"  サンプリングレート: {self.sample_rate} Hz")
            print(f"  チャンネル数: 1 (モノラル)")
            print(f"  録音時間: {duration:.2f} 秒")
            print(f"  サンプル数: {len(audio_data)}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("NIDAQ Dual-Microphone WAV Recorder")
    print("=" * 60)
    print()

    # 設定
    device_name = "Dev7"  # 必要に応じて変更
    sample_rate = 10000  # サンプリングレート（Hz）
    record_time = None  # None: 手動停止まで、数値: 指定秒数

    # 録音時間を入力
    user_input = input("録音時間を秒数で入力（Enter=手動停止まで録音）: ").strip()
    if user_input:
        try:
            record_time = float(user_input)
        except ValueError:
            print("無効な入力です。手動停止モードで録音します。")
            record_time = None

    # 出力ファイル名のプレフィックスを入力
    filename = input(
        "出力ファイル名のプレフィックス（Enter=自動生成、_ch1.wavと_ch2.wavが追加されます）: "
    ).strip()
    if not filename:
        filename = None

    print()

    # 録音実行
    recorder = NIDAQRecorder(
        device_name=device_name, sample_rate=sample_rate, record_time=record_time
    )

    recorder.record()
    recorder.save_wav(filename)


if __name__ == "__main__":
    main()
