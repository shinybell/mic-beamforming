# リアルタイム音声処理システム

NI-DAQmxからの電圧データをリアルタイムで処理し、音声として復元・出力するシステムです。

元のリポジトリ: [mic-beamforming](https://github.com/shinybell/mic-beamforming)

## 機能

- **NI-DAQmx入力**: NI-DAQmxデバイスから電圧データを読み取り、音声として復元
- **音質向上処理**:
  - ハイパスフィルタ（低周波ノイズ除去）
  - ノイズゲート
  - ゲイン調整
  - エコーキャンセレーション
  - クリッピング防止

## インストール

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

**注意**: NI-DAQmxを使用する場合は、NI-DAQmxドライバを事前にインストールする必要があります。

## 使用方法

```bash
python main.py
```

デフォルトで`Dev1/ai0`からデータを読み取ります。デバイス名を変更する場合は、コード内の`"Dev1/ai0"`を編集してください。

## ライセンス

（ライセンス情報を追加してください）

