#!/bin/bash
# キーワードマイニングツール起動スクリプト

echo "🔍 キーワードマイニングツールを起動しています..."

# 作業ディレクトリを移動
cd "$(dirname "$0")"

# 依存ライブラリの確認
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 必要なライブラリをインストールしています..."
    pip install -r requirements.txt
fi

# Streamlitアプリを起動
echo "🌐 ブラウザで http://localhost:8501 を開いてください"
python3 -m streamlit run app/main.py --server.port 8501
