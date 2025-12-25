"""
オーディオデバイス詳細チェックツール
AirPodsが認識されない問題を診断
"""

import sounddevice as sd


def check_all_devices():
    """すべてのオーディオデバイスを詳細表示"""
    print("=" * 70)
    print("オーディオデバイス詳細チェック")
    print("=" * 70)
    print()
    
    devices = sd.query_devices()
    
    print(f"検出されたデバイス数: {len(devices)}")
    print()
    
    for i, device in enumerate(devices):
        print(f"【デバイス {i}】")
        print(f"  名前: {device['name']}")
        print(f"  入力チャンネル数: {device['max_input_channels']}")
        print(f"  出力チャンネル数: {device['max_output_channels']}")
        print(f"  デフォルトサンプリングレート: {device['default_samplerate']} Hz")
        print(f"  ホストAPI: {device['hostapi']}")
        
        # AirPods関連のキーワードをチェック
        name_lower = device['name'].lower()
        keywords = ['airpods', 'air pods', 'bluetooth', 'wireless']
        matches = [kw for kw in keywords if kw in name_lower]
        if matches:
            print(f"  ⭐ キーワード検出: {', '.join(matches)}")
        
        # 入力可能なデバイスをハイライト
        if device['max_input_channels'] >= 2:
            print(f"  ✓ ステレオ入力可能")
        elif device['max_input_channels'] == 1:
            print(f"  ⚠ モノラル入力のみ")
        
        print()
    
    # デフォルトデバイスを表示
    print("=" * 70)
    print("デフォルトデバイス")
    print("=" * 70)
    print()
    
    try:
        default_input = sd.query_devices(kind='input')
        print(f"デフォルト入力: [{default_input['index']}] {default_input['name']}")
        print(f"  入力チャンネル数: {default_input['max_input_channels']}")
    except Exception as e:
        print(f"デフォルト入力デバイスなし: {e}")
    
    print()
    
    try:
        default_output = sd.query_devices(kind='output')
        print(f"デフォルト出力: [{default_output['index']}] {default_output['name']}")
        print(f"  出力チャンネル数: {default_output['max_output_channels']}")
    except Exception as e:
        print(f"デフォルト出力デバイスなし: {e}")
    
    print()
    print("=" * 70)
    print("推奨事項")
    print("=" * 70)
    print()
    
    # AirPodsを探す
    airpods_devices = []
    for i, device in enumerate(devices):
        name_lower = device['name'].lower()
        if 'airpods' in name_lower or 'air pods' in name_lower:
            airpods_devices.append((i, device))
    
    if airpods_devices:
        print("✓ AirPods関連デバイスが見つかりました:")
        for idx, dev in airpods_devices:
            print(f"  [{idx}] {dev['name']}")
            print(f"      入力: {dev['max_input_channels']}ch, 出力: {dev['max_output_channels']}ch")
            
            if dev['max_input_channels'] >= 2:
                print(f"      → このデバイスを使用できます！")
            elif dev['max_input_channels'] == 1:
                print(f"      → モノラル入力のみです（ステレオ録音不可）")
            else:
                print(f"      → 入力デバイスとして使用できません")
        print()
        print("【使用方法】")
        print("プログラム実行時に、上記のデバイス番号を入力してください。")
    else:
        print("❌ AirPods関連デバイスが見つかりませんでした。")
        print()
        print("【確認事項】")
        print("1. AirPodsがMacBookとBluetooth接続されているか確認")
        print("2. システム環境設定 > サウンド > 入力 でAirPodsが表示されるか確認")
        print("3. AirPodsを一度切断して再接続")
        print("4. Macを再起動")
        print()
        print("【代替案】")
        print("ステレオ入力可能なデバイス:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 2:
                print(f"  [{i}] {device['name']}")


if __name__ == "__main__":
    try:
        check_all_devices()
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
