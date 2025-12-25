"""
エンドファイア型ビームフォーミング - 簡易テストスクリプト
配置確認と基本動作テスト用
"""

import numpy as np
import sounddevice as sd
import time


def test_endfire_setup():
    """エンドファイア配置のテスト"""
    print("=" * 60)
    print("エンドファイア型ビームフォーミング - セットアップテスト")
    print("=" * 60)
    print()
    
    # 配置図を表示
    print("【配置図】")
    print()
    print("    A ←--1m--→ 左● ←--50cm--→ ●右 ←--1m--→ B")
    print()
    print("  話者A: 出力対象")
    print("  話者B: 抑制対象")
    print()
    print("=" * 60)
    print()
    
    # デバイス一覧
    print("【ステップ1】オーディオデバイスの確認")
    print()
    devices = sd.query_devices()
    
    print("入力デバイス（2チャンネル以上）:")
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] >= 2:
            input_devices.append(i)
            marker = " ← AirPods?" if 'airpods' in device['name'].lower() else ""
            print(f"  [{i}] {device['name']}{marker}")
            print(f"      チャンネル数: {device['max_input_channels']}")
    print()
    
    print("出力デバイス:")
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] >= 1:
            output_devices.append(i)
            marker = " ← MacBook?" if 'macbook' in device['name'].lower() or 'built-in' in device['name'].lower() else ""
            print(f"  [{i}] {device['name']}{marker}")
    print()
    
    if not input_devices:
        print("❌ ステレオ入力デバイスが見つかりません")
        return False
    
    # デバイス選択
    print("【ステップ2】デバイスの選択")
    print()
    
    if len(input_devices) == 1:
        input_device = input_devices[0]
        print(f"入力デバイス: [{input_device}] {devices[input_device]['name']}")
    else:
        input_device = int(input("入力デバイス番号を選択: "))
    
    if len(output_devices) == 1:
        output_device = output_devices[0]
        print(f"出力デバイス: [{output_device}] {devices[output_device]['name']}")
    else:
        output_device = int(input("出力デバイス番号を選択: "))
    
    print()
    
    # ステレオ録音テスト
    print("【ステップ3】ステレオ録音テスト")
    print()
    print("3秒間録音します。")
    print("話者Aの位置（左側）から音を出してください。")
    input("準備ができたらEnterキーを押してください...")
    
    sample_rate = 48000
    duration = 3
    
    print("\n録音中（3秒）...")
    recording_a = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,
        device=input_device,
        dtype='float32'
    )
    sd.wait()
    
    # 左右チャンネルの音量を計算
    left_rms_a = np.sqrt(np.mean(recording_a[:, 0]**2))
    right_rms_a = np.sqrt(np.mean(recording_a[:, 1]**2))
    
    print(f"\n【話者A位置からの録音結果】")
    print(f"  左チャンネル: {left_rms_a:.6f}")
    print(f"  右チャンネル: {right_rms_a:.6f}")
    print(f"  左/右 比率: {left_rms_a/right_rms_a:.2f}" if right_rms_a > 0 else "  右チャンネル無音")
    
    # 期待値: 左チャンネルの方が大きい（Aに近い）
    if left_rms_a > right_rms_a * 1.2:
        print("  ✓ 正しく配置されています（左チャンネルが大きい）")
        test_a_ok = True
    elif left_rms_a > right_rms_a * 0.8:
        print("  ⚠ 左右の差が小さいです（配置を調整してください）")
        test_a_ok = False
    else:
        print("  ✗ 配置が逆かもしれません（左チャンネルが小さい）")
        test_a_ok = False
    
    print()
    
    # 話者B位置のテスト
    print("次に、話者Bの位置（右側）から音を出してください。")
    input("準備ができたらEnterキーを押してください...")
    
    print("\n録音中（3秒）...")
    recording_b = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,
        device=input_device,
        dtype='float32'
    )
    sd.wait()
    
    left_rms_b = np.sqrt(np.mean(recording_b[:, 0]**2))
    right_rms_b = np.sqrt(np.mean(recording_b[:, 1]**2))
    
    print(f"\n【話者B位置からの録音結果】")
    print(f"  左チャンネル: {left_rms_b:.6f}")
    print(f"  右チャンネル: {right_rms_b:.6f}")
    print(f"  右/左 比率: {right_rms_b/left_rms_b:.2f}" if left_rms_b > 0 else "  左チャンネル無音")
    
    # 期待値: 右チャンネルの方が大きい（Bに近い）
    if right_rms_b > left_rms_b * 1.2:
        print("  ✓ 正しく配置されています（右チャンネルが大きい）")
        test_b_ok = True
    elif right_rms_b > left_rms_b * 0.8:
        print("  ⚠ 左右の差が小さいです（配置を調整してください）")
        test_b_ok = False
    else:
        print("  ✗ 配置が逆かもしれません（右チャンネルが小さい）")
        test_b_ok = False
    
    print()
    print("=" * 60)
    
    # 簡易ビームフォーミングテスト
    print("\n【ステップ4】簡易ビームフォーミングテスト")
    print()
    
    # 遅延サンプル数を計算
    mic_distance = 0.50  # 50cm
    sound_speed = 343.0
    delay_samples = int(mic_distance / sound_speed * sample_rate)
    
    print(f"マイク間距離: {mic_distance*100:.0f}cm")
    print(f"遅延サンプル数: {delay_samples} ({delay_samples/sample_rate*1000:.2f}ms)")
    print()
    
    # 話者A位置の音声に対してビームフォーミング
    left_a = recording_a[:, 0]
    right_a = recording_a[:, 1]
    
    # 遅延を適用
    delayed_right_a = np.pad(right_a, (delay_samples, 0), mode='constant')[:-delay_samples]
    
    # エンドファイアビームフォーミング（A側を強調）
    output_a = left_a - 0.5 * delayed_right_a
    output_rms_a = np.sqrt(np.mean(output_a**2))
    
    print(f"【話者A位置の音声】")
    print(f"  ビームフォーミング前（左のみ）: {left_rms_a:.6f}")
    print(f"  ビームフォーミング後: {output_rms_a:.6f}")
    print(f"  変化: {output_rms_a/left_rms_a:.2f}x" if left_rms_a > 0 else "  計算不可")
    
    # 話者B位置の音声に対してビームフォーミング
    left_b = recording_b[:, 0]
    right_b = recording_b[:, 1]
    
    delayed_right_b = np.pad(right_b, (delay_samples, 0), mode='constant')[:-delay_samples]
    output_b = left_b - 0.5 * delayed_right_b
    output_rms_b = np.sqrt(np.mean(output_b**2))
    
    print(f"\n【話者B位置の音声】")
    print(f"  ビームフォーミング前（左のみ）: {left_rms_b:.6f}")
    print(f"  ビームフォーミング後: {output_rms_b:.6f}")
    print(f"  変化: {output_rms_b/left_rms_b:.2f}x" if left_rms_b > 0 else "  計算不可")
    
    # 分離性能を計算
    if output_rms_a > 0 and output_rms_b > 0:
        separation_db = 20 * np.log10(output_rms_a / output_rms_b)
        print(f"\n【分離性能】")
        print(f"  A/B比率: {separation_db:.1f} dB")
        
        if separation_db > 10:
            print("  ✓ 良好な分離性能です！")
        elif separation_db > 5:
            print("  ⚠ 分離性能は普通です（配置を調整すると改善するかも）")
        else:
            print("  ✗ 分離性能が低いです（配置を見直してください）")
    
    print()
    print("=" * 60)
    print("\n【総合評価】")
    
    if test_a_ok and test_b_ok:
        print("✓ セットアップは正しく完了しています！")
        print("  endfire_beamforming.py を実行できます。")
        return True
    else:
        print("⚠ セットアップに問題があります。")
        print("\n【改善案】")
        if not test_a_ok:
            print("  - 話者Aの位置を左AirPodsに近づける")
        if not test_b_ok:
            print("  - 話者Bの位置を右AirPodsに近づける")
        print("  - AirPodsが一直線上に並んでいるか確認")
        print("  - マイク間距離が正確に50cmか確認")
        return False


if __name__ == "__main__":
    try:
        success = test_endfire_setup()
        print()
        if success:
            print("次のステップ: python endfire_beamforming.py")
        else:
            print("配置を調整してから、もう一度このテストを実行してください。")
    except KeyboardInterrupt:
        print("\n\nテストを中断しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
