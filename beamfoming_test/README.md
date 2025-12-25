# NIDAQ Beamforming System

NIDAQハードウェアの2つのマイクを使用したリアルタイムビームフォーミングシステムです。

## ✨ 特徴

- **Windows/Mac両対応** - スピーカーを自動選択
- **リアルタイムストリーミング処理** - 入力直後に処理して即座に出力
- **超低遅延** - 約100msの遅延で音声を出力
- **高品質ビームフォーミング** - 周波数領域Delay-and-Sum方式

## 📦 必要なファイル

このディレクトリには以下のファイルが含まれています:

- `nidaq_beamforming.py` - メインプログラム
- `nidaq_config.py` - 設定ファイル
- `requirements.txt` - 依存関係
- `NIDAQ_GUIDE.md` - 詳細ガイド
- `NIDAQ_SUMMARY.md` - クイックリファレンス
- `setup.sh` - セットアップスクリプト（自動生成）

## 🚀 クイックスタート

### 1. セットアップ

```bash
cd /Users/takakurakanta/tasks/mic-beamforming/beamfoming_test

# 仮想環境を作成
python3 -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. 設定確認

```bash
# 設定ファイルを検証
python nidaq_config.py

# NIDAQデバイスを確認
python -c "import nidaqmx; print([d.name for d in nidaqmx.system.System.local().devices])"
```

### 3. 実行

```bash
python nidaq_beamforming.py
```

## 🎤 ハードウェア要件

- **NIDAQデバイス** (例: Dev10)
- **マイク × 2**
- **NI-DAQmxドライバ**

## ⚙️ 設定

### デバイス名の変更

`nidaq_config.py` の15行目:
```python
DEVICE_NAME = "Dev10"  # 実際のデバイス名に変更
```

### マイク間距離の変更

`nidaq_config.py` の22行目:
```python
MIC_SPACING = 0.76  # メートル単位
```

## 📚 詳細ドキュメント

- **NIDAQ_GUIDE.md**: 完全なセットアップガイド
- **NIDAQ_SUMMARY.md**: クイックリファレンス

## 🔧 トラブルシューティング

### NIDAQデバイスが見つからない

1. NIDAQハードウェアがUSBで接続されているか確認
2. NI-DAQmxドライバがインストールされているか確認
3. デバイス名が正しいか確認

詳細は `NIDAQ_GUIDE.md` を参照してください。

## 📊 システム仕様

- サンプリングレート: 20,000 Hz
- チャンクサイズ: 2,048 samples (~102ms)
- マイク間距離: 76cm (デフォルト)
- 処理方式: 周波数領域Delay-and-Sum

## 🎯 使用例

```bash
# 正面の話者を聞く
python nidaq_beamforming.py
# 目的角度: 0

# 右側の話者を聞く
python nidaq_beamforming.py
# 目的角度: 90
```

---

詳細は `NIDAQ_GUIDE.md` をご覧ください。
