# Keyword Mining Tool

Search Consoleのキーワードデータを分析し、ワードクラウド・共起ネットワーク・カテゴリ分析を行うツールです。

## 機能

- **ワードクラウド**: 単語の出現頻度を視覚化（高解像度PNG出力対応）
- **CTR × 掲載順位マップ**: キーワードのパフォーマンスを4象限で分類
- **単語パフォーマンス**: 単語別のCTR・順位をスコア化
- **カテゴリ分析**: 単語を意味カテゴリ別に自動分類
- **共起ネットワーク**: 同時に検索される単語の関係性を可視化
- **HTMLレポート出力**: 分析結果をレポートとしてエクスポート

## 業種別カテゴリ

### 医療機関（773語）
診療科、施設、病名・疾患、症状、体の部位、治療・施術、検査・診断、薬・医薬品など

### 一般企業（401語）
業種・業界、商品・サービス、企業・組織、アクション、対象・ターゲット、品質・特徴など

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
streamlit run app/main.py
```

ブラウザで http://localhost:8501 にアクセス

## データ入力方法

### 1. CSVアップロード
以下のフォーマットのCSVファイルをアップロード：
```
query,clicks,impressions,ctr,position
キーワード1,100,1000,10.0,5.2
キーワード2,50,800,6.25,8.1
```

### 2. Search Console連携
Google Cloud Consoleで認証情報（OAuth 2.0クライアントID）を作成し、JSONを貼り付けて認証

## 必要なライブラリ

- streamlit
- pandas
- numpy
- wordcloud
- matplotlib
- japanize-matplotlib
- networkx
- plotly
- janome（日本語形態素解析）
- google-auth, google-auth-oauthlib, google-api-python-client（Search Console連携用）

## ライセンス

MIT License
# keyword-analysis
