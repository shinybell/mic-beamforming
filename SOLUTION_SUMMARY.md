# 🔍 診断結果と解決策

## 問題の原因

**AirPods Proはモノラル（1チャンネル）入力のみをサポート**

診断ツールの実行結果:
```
た　のAirPods Pro
  入力チャンネル数: 1  ← ステレオ録音不可
```

ビームフォーミングには**2チャンネル（左右独立）**が必要ですが、AirPodsでは不可能です。

## 📊 利用可能なデバイス

現在のMacBookで検出されたステレオ入力可能デバイス:

| デバイス番号 | 名前 | 用途 |
|------------|------|------|
| 0 | BlackHole 2ch | 仮想オーディオ（テスト用） |
| 6 | Microsoft Teams Audio | 仮想オーディオ（テスト用） |

## ✅ 解決策

### 1. テスト実行（今すぐ可能）

BlackHoleを使用してシステムをテスト:

```bash
source venv/bin/activate
python beam_forming.py

# 入力:
# 録音時間: 10
# 手法: 1
# デバイス番号: 0
```

### 2. 実用化（外部デバイス購入）

**推奨ハードウェア:**

**予算 〜1万円:**
- Zoom H1n (約8,000円)
- Behringer UMC202HD + マイク2本

**予算 1〜3万円:**
- Blue Yeti (約15,000円) ← 推奨
- Zoom H2n (約18,000円)

**予算 3万円〜:**
- Zoom H4n Pro (約30,000円)

### 3. 2つのマイクで集約デバイス作成

手順:
1. 2つのUSBマイクを用意
2. Audio MIDI設定 > 集約デバイスを作成
3. 2つのマイクを選択
4. プログラムで集約デバイスを選択

## 🚀 次のステップ

### すぐにテストしたい場合
```bash
python check_audio_devices.py  # デバイス確認
python beam_forming.py          # BlackHole (0) でテスト
```

### 実用化したい場合
1. 外部ステレオマイクを購入
2. 接続後 `python check_audio_devices.py` で確認
3. `python beam_forming.py` で実行

## 📚 詳細ドキュメント

- **`AIRPODS_LIMITATION.md`**: 詳細な説明と解決策
- **`check_audio_devices.py`**: デバイス診断ツール
- **`COMPARISON.md`**: システムの使い分けガイド
