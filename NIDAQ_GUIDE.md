# NIDAQ Beamforming セットアップガイド

## 📋 概要

NIDAQハードウェアの2つのマイクを使用したリアルタイムビームフォーミングシステムです。

## 🔧 必要なハードウェア

- **NIDAQ デバイス** (例: NI USB-6211, NI USB-6001など)
- **マイク × 2** (ダイナミックマイクまたはコンデンサーマイク)
- **オーディオインターフェース** (マイクに電源が必要な場合)

## 📦 ソフトウェア要件

### 1. NI-DAQmxドライバのインストール

National Instruments公式サイトからダウンロード:
https://www.ni.com/ja-jp/support/downloads/drivers/download.ni-daqmx.html

### 2. Pythonライブラリのインストール

```bash
cd /Users/takakurakanta/tasks
source venv/bin/activate
pip install -r requirements.txt
```

## 🎤 ハードウェアセットアップ

### マイク接続

```
NIDAQ Device (Dev10)
├── ai0 (アナログ入力0) ← 左マイク
└── ai1 (アナログ入力1) ← 右マイク
```

### マイク配置

```
    左マイク          右マイク
       ●  ←--76cm--→  ●
    (ai0)           (ai1)
```

**重要:**
- 2つのマイクを一直線上に配置
- 間隔: 76cm (デフォルト、変更可能)
- 同じ高さに設置
- 障害物がないように

## ⚙️ 設定

### デバイス名の確認

```bash
python -c "import nidaqmx; print([d.name for d in nidaqmx.system.System.local().devices])"
```

出力例: `['Dev10']`

### 設定ファイルの編集

`nidaq_config.py`を開いて、必要に応じて以下を変更:

```python
# デバイス名（上記で確認した名前）
DEVICE_NAME = "Dev10"

# マイク間距離（メートル）
MIC_SPACING = 0.76  # 76cm

# サンプリングレート
SAMPLE_RATE = 20000  # 20kHz

# チャンクサイズ（遅延調整）
CHUNK_SIZE = 2048  # 約100ms
```

## 🚀 使用方法

### 基本的な実行

```bash
source venv/bin/activate
python nidaq_beamforming.py
```

### 実行フロー

1. **デバイス確認**
   ```
   === 利用可能なNIDAQデバイス ===
     - Dev10
       製品タイプ: USB-6211
       アナログ入力: 16 チャンネル
   ```

2. **目的角度の入力**
   ```
   目的角度を入力してください（度）:
     0度 = 正面（マイクアレイに垂直）
     90度 = 右側（エンドファイア）
     -90度 = 左側
   > 0
   ```

3. **実行時間の入力**
   ```
   実行時間を入力してください（秒）:
   （Enterキーのみで手動停止モード）
   > [Enter]
   ```

4. **録音・再生開始**
   ```
   🎤 録音中... 🔊 再生中...
   
   Level: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] -12.3 dB
   ```

5. **停止**: Ctrl+C

## 📐 角度の設定

### 角度の意味

```
        0度（正面）
          ↑
          |
-90度 ←--●--●--→ 90度
    (左)      (右)
          |
          ↓
       180度（背面）
```

### 使用例

**例1: 正面の話者を聞く**
```
目的角度: 0度
配置: 話者がマイクアレイの正面（垂直方向）
```

**例2: 右側の話者を聞く（エンドファイア）**
```
目的角度: 90度
配置: 話者がマイクアレイの右端の延長線上
```

**例3: 左側の話者を聞く**
```
目的角度: -90度
配置: 話者がマイクアレイの左端の延長線上
```

## 🎛️ パラメータ調整

### マイク間距離の変更

実際の配置に合わせて `nidaq_config.py` を編集:

```python
MIC_SPACING = 0.50  # 50cmの場合
MIC_SPACING = 1.00  # 1mの場合
```

**推奨距離:**
- 近距離（30-50cm）: 高周波に強い、狭い指向性
- 中距離（50-80cm）: バランス型（推奨）
- 遠距離（80-150cm）: 低周波に強い、広い指向性

### サンプリングレートの変更

```python
SAMPLE_RATE = 10000  # 低遅延（約200ms）
SAMPLE_RATE = 20000  # 高品質（約100ms）- 推奨
SAMPLE_RATE = 44100  # 最高品質（約46ms）
```

### 遅延の調整

```python
CHUNK_SIZE = 1024  # 低遅延（約50ms）
CHUNK_SIZE = 2048  # バランス（約100ms）- 推奨
CHUNK_SIZE = 4096  # 高品質（約200ms）
```

### 音質調整

```python
# ノイズゲート（小さな音を除去）
NOISE_GATE_THRESHOLD = 0.01  # 小さくするとノイズ除去が強くなる

# ゲイン（音量調整）
GAIN = 1.2  # 大きくすると音量が上がる

# ハイパスフィルタ（低音ノイズ除去）
HIGH_PASS_CUTOFF = 80  # 大きくすると低音が減る
```

## 🔧 トラブルシューティング

### ❌ NIDAQデバイスが見つからない

**確認事項:**
1. NIDAQハードウェアがUSBで接続されているか
2. NI-DAQmxドライバがインストールされているか
3. デバイスマネージャーで認識されているか

**解決策:**
```bash
# デバイスを確認
python -c "import nidaqmx; print(nidaqmx.system.System.local().devices)"

# ドライバを再インストール
# https://www.ni.com/ja-jp/support/downloads/drivers/download.ni-daqmx.html
```

### ❌ チャンネルエラー

```
Error: Channel 'Dev10/ai0' not found
```

**解決策:**
1. NI MAX (Measurement & Automation Explorer) を開く
2. デバイスを確認
3. `nidaq_config.py` のチャンネル名を修正

### ❌ 音が聞こえない

**確認事項:**
1. マイクが正しく接続されているか
2. マイクに電源が供給されているか（コンデンサーマイクの場合）
3. 音量が適切か

**テスト:**
```bash
# 単一チャンネルのテスト
python -c "
import nidaqmx
import numpy as np
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan('Dev10/ai0')
    task.timing.cfg_samp_clk_timing(20000, samps_per_chan=1000)
    data = task.read(100)
    print(f'RMS: {np.sqrt(np.mean(np.array(data)**2)):.6f}')
"
```

期待値: RMS > 0.001 (音がある場合)

### ❌ 音が歪む

**原因と解決策:**

1. **入力レベルが高すぎる**
   - マイクとの距離を離す
   - `GAIN` を小さくする（例: 0.8）

2. **クリッピング**
   - 入力電圧範囲を確認
   - NIDAQの設定で入力範囲を調整

### ❌ 遅延が大きい

**解決策:**
```python
# nidaq_config.py
CHUNK_SIZE = 1024  # 2048 → 1024
QUEUE_SIZE = 10    # 20 → 10
```

### ❌ 音が途切れる

**解決策:**
```python
# nidaq_config.py
CHUNK_SIZE = 4096  # 2048 → 4096
QUEUE_SIZE = 30    # 20 → 30
```

## 📊 性能指標

### 処理遅延

| 設定 | チャンクサイズ | サンプリングレート | 遅延 |
|-----|--------------|------------------|------|
| 低遅延 | 1024 | 20000 Hz | 約50ms |
| バランス | 2048 | 20000 Hz | 約100ms |
| 高品質 | 4096 | 20000 Hz | 約200ms |

### 分離性能

**理想的な条件:**
- 目的方向: 0dB（そのまま）
- 90度オフ: -10dB〜-15dB
- 180度オフ: -15dB〜-20dB

## 💡 使用例

### 例1: 会議録音

```
配置:
    発表者
      🗣️
      |
      | 0度
      |
    ●---●
   (マイク)
```

```bash
python nidaq_beamforming.py
# 目的角度: 0
# 実行時間: 3600 (1時間)
```

### 例2: インタビュー

```
配置:
質問者 ←→ ●---● ←→ ゲスト
🗣️    -90°     90°    🗣️
```

質問者の声を聞く:
```bash
python nidaq_beamforming.py
# 目的角度: -90
```

ゲストの声を聞く:
```bash
python nidaq_beamforming.py
# 目的角度: 90
```

## 📚 参考情報

### ファイル構成

```
/Users/takakurakanta/tasks/
├── nidaq_beamforming.py  # メインプログラム
├── nidaq_config.py       # 設定ファイル
├── requirements.txt      # 依存関係
└── NIDAQ_GUIDE.md        # このガイド
```

### 関連ドキュメント

- **COMPARISON.md**: 他のビームフォーミング方式との比較
- **TECHNICAL.md**: 技術的な詳細
- **AIRPODS_LIMITATION.md**: AirPodsの制限について

### 参考リンク

- NI-DAQmx Python API: https://nidaqmx-python.readthedocs.io/
- National Instruments: https://www.ni.com/

## 🎓 次のステップ

1. **基本動作確認**: 10秒間のテスト実行
2. **角度調整**: 複数の角度で試す
3. **パラメータ最適化**: 環境に合わせて調整
4. **実用テスト**: 実際のシーンで長時間テスト

---

**準備完了！** まずは `python nidaq_beamforming.py` を実行してテストしてください。
