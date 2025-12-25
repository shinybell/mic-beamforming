"""
AirPods デバイステストスクリプト
オーディオデバイスの検出と基本的な動作確認
"""

import sounddevice as sd
import numpy as np

def test_audio_devices():
    """利用可能なオーディオデバイスをテスト"""
    print("=" * 60)
    print("オーディオデバイステスト")
    print("=" * 60)
    print()
    
    # すべてのデバイスを表示
    print("=== 利用可能なオーディオデバイス ===\n")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        print(f"[{i}] {device['name']}")
        print(f"    最大入力チャンネル: {device['max_input_channels']}")
        print(f"    最大出力チャンネル: {device['max_output_channels']}")
        print(f"    デフォルトサンプリングレート: {device['default_samplerate']} Hz")
        print()
    
    # 2チャンネル以上の入力デバイスを検出
    print("=== ステレオ入力可能なデバイス ===\n")
    stereo_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] >= 2:
            stereo_devices.append(i)
            print(f"[{i}] {device['name']}")
            print(f"    入力チャンネル数: {device['max_input_channels']}")
            print()
    
    # AirPodsを検出
    print("=== AirPodsデバイス ===\n")
    airpods_devices = []
    for i, device in enumerate(devices):
        if 'airpods' in device['name'].lower():
            airpods_devices.append(i)
            print(f"[{i}] {device['name']}")
            print(f"    入力チャンネル数: {device['max_input_channels']}")
            print(f"    出力チャンネル数: {device['max_output_channels']}")
            print()
    
    if not airpods_devices:
        print("AirPodsが見つかりませんでした。")
        print("AirPodsを接続してから再度実行してください。")
    
    # デフォルトデバイスを表示
    print("=== デフォルトデバイス ===\n")
    try:
        default_input = sd.query_devices(kind='input')
        print(f"入力: {default_input['name']}")
        print(f"  チャンネル数: {default_input['max_input_channels']}")
        print()
    except Exception as e:
        print(f"デフォルト入力デバイスの取得に失敗: {e}\n")
    
    try:
        default_output = sd.query_devices(kind='output')
        print(f"出力: {default_output['name']}")
        print(f"  チャンネル数: {default_output['max_output_channels']}")
        print()
    except Exception as e:
        print(f"デフォルト出力デバイスの取得に失敗: {e}\n")
    
    # 簡易録音テスト
    if stereo_devices:
        print("=== 録音テスト ===\n")
        print("ステレオ入力デバイスで簡易録音テストを実行しますか？")
        print("（3秒間録音して、左右チャンネルの音量を確認します）")
        response = input("実行する場合は 'y' を入力: ").strip().lower()
        
        if response == 'y':
            # デバイスを選択
            if len(stereo_devices) == 1:
                device_id = stereo_devices[0]
            else:
                print("\nテストするデバイスを選択してください:")
                for idx in stereo_devices:
                    print(f"[{idx}] {devices[idx]['name']}")
                device_id = int(input("デバイス番号: "))
            
            print(f"\nデバイス [{device_id}] {devices[device_id]['name']} で録音中...")
            print("何か音を出してください（3秒間）...")
            
            # 録音
            duration = 3  # 秒
            sample_rate = int(devices[device_id]['default_samplerate'])
            
            try:
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=2,
                    device=device_id,
                    dtype='float32'
                )
                sd.wait()
                
                # 左右チャンネルの音量を計算
                left_channel = recording[:, 0]
                right_channel = recording[:, 1]
                
                left_rms = np.sqrt(np.mean(left_channel**2))
                right_rms = np.sqrt(np.mean(right_channel**2))
                
                print("\n=== 録音結果 ===")
                print(f"左チャンネルRMS: {left_rms:.6f}")
                print(f"右チャンネルRMS: {right_rms:.6f}")
                
                if left_rms > 0.001 and right_rms > 0.001:
                    print("\n✓ 両チャンネルで音声が検出されました！")
                    print("  ビームフォーミングシステムを使用できます。")
                elif left_rms > 0.001 or right_rms > 0.001:
                    print("\n⚠ 片方のチャンネルのみで音声が検出されました。")
                    print("  デバイスの設定を確認してください。")
                else:
                    print("\n✗ 音声が検出されませんでした。")
                    print("  マイクの設定や音量を確認してください。")
                
            except Exception as e:
                print(f"\n録音テストに失敗しました: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_audio_devices()
    except KeyboardInterrupt:
        print("\n\nテストを中断しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
