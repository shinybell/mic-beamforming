#!/bin/bash

# AirPods Beamforming セットアップスクリプト

echo "=== AirPods Beamforming セットアップ ==="
echo ""

# 仮想環境の作成
if [ ! -d "venv" ]; then
    echo "仮想環境を作成中..."
    python3 -m venv venv
fi

# 仮想環境をアクティベート
echo "仮想環境をアクティベート中..."
source venv/bin/activate

# 依存関係をインストール
echo "依存関係をインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "使用方法:"
echo "1. 仮想環境をアクティベート: source venv/bin/activate"
echo "2. スクリプトを実行: python beam_forming.py"
echo "3. 終了時: deactivate"
echo ""
