"""
WAV to Mel Spectrogram Converter
=================================

WAVファイルをメルスペクトログラム画像に変換するスクリプト

必要なライブラリ:
pip install librosa matplotlib numpy

使用方法:
python wav_to_melspec.py input.wav [output.png]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def create_mel_spectrogram(
    wav_path,
    output_path=None,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmax=8000,
    figsize=(12, 6),
):
    """
    WAVファイルからメルスペクトログラムを生成して画像として保存

    Args:
        wav_path: 入力WAVファイルのパス
        output_path: 出力画像のパス（Noneの場合は自動生成）
        n_fft: FFTウィンドウサイズ
        hop_length: ホップ長
        n_mels: メルフィルタバンクの数
        fmax: 最大周波数（Hz）
        figsize: 図のサイズ（幅, 高さ）

    Returns:
        output_path: 保存された画像のパス
    """
    # 出力パスが指定されていない場合、自動生成
    if output_path is None:
        base_name = os.path.splitext(wav_path)[0]
        output_path = f"{base_name}_melspec.png"

    print(f"Processing: {wav_path}")

    try:
        # WAVファイルを読み込み
        y, sr = librosa.load(wav_path, sr=None)
        duration = len(y) / sr

        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Samples: {len(y)}")

        # メルスペクトログラムを計算
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax
        )

        # dBスケールに変換
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 可視化
        plt.figure(figsize=figsize)
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmax=fmax,
            cmap="viridis",
        )

        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram: {os.path.basename(wav_path)}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()

        # 保存
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("Usage: python wav_to_melspec.py <input.wav> [output.png]")
        print("")
        print("Arguments:")
        print("  input.wav   : Input WAV file path")
        print("  output.png  : Output image path (optional)")
        print("")
        print("Examples:")
        print("  python wav_to_melspec.py audio.wav")
        print("  python wav_to_melspec.py audio.wav spectrogram.png")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # ファイルの存在確認
    if not os.path.exists(wav_path):
        print(f"Error: File not found: {wav_path}")
        sys.exit(1)

    # WAVファイルかチェック
    if not wav_path.lower().endswith((".wav", ".wave")):
        print(f"Warning: File may not be a WAV file: {wav_path}")

    print("=" * 60)
    print("WAV to Mel Spectrogram Converter")
    print("=" * 60)
    print()

    # メルスペクトログラムを生成
    result = create_mel_spectrogram(wav_path, output_path)

    if result:
        print()
        print("=" * 60)
        print("✓ Conversion complete!")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("✗ Conversion failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
