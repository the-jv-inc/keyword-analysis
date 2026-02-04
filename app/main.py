"""
キーワードマイニングツール
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import pickle
import base64

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import japanize_matplotlib
import networkx as nx
import plotly.graph_objects as go

from janome.tokenizer import Tokenizer

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ページ設定
st.set_page_config(page_title="Keyword Mining", page_icon="K", layout="wide")

# Google Trends風CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');

    .main .block-container {
        max-width: 1000px;
        padding: 2rem 1.5rem;
    }

    #MainMenu, footer, header { visibility: hidden; }

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    h1 {
        font-size: 1.75rem !important;
        font-weight: 400 !important;
        color: #202124 !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #202124 !important;
        margin-bottom: 0.25rem !important;
    }

    .desc-text {
        font-size: 0.85rem;
        color: #5f6368;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .stMetric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e8eaed;
    }

    .stMetric label {
        font-size: 0.7rem !important;
        color: #5f6368 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #202124 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e8eaed;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #5f6368;
        padding: 0.75rem 1.25rem;
        border-bottom: 3px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #1a73e8 !important;
        border-bottom: 3px solid #1a73e8 !important;
    }

    /* Google風ボタン */
    .stButton > button {
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        border: 1px solid #dadce0 !important;
        background: white !important;
        color: #1a73e8 !important;
        transition: background 0.2s, box-shadow 0.2s !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #f8f9fa !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: #1a73e8 !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: #1557b0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }

    /* ダウンロードボタン */
    .stDownloadButton > button {
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        border-radius: 4px !important;
        padding: 10px 16px !important;
        border: 1px solid #dadce0 !important;
        background: white !important;
        color: #3c4043 !important;
        transition: all 0.2s !important;
    }

    .stDownloadButton > button:hover {
        background: #f1f3f4 !important;
        border-color: #c6c6c6 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #e8eaed;
        margin: 2rem 0;
    }

    .stDataFrame {
        border: 1px solid #e8eaed;
        border-radius: 8px;
    }

    /* アップロードエリア */
    .upload-area {
        border: 2px dashed #dadce0;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        margin-bottom: 1rem;
    }

    .upload-area:hover {
        border-color: #1a73e8;
        background: #f8f9fa;
    }

    /* ファイルアップローダー非表示 */
    [data-testid="stFileUploader"] {
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section > input + div {
        display: none !important;
    }

    [data-testid="stFileUploader"] section > button {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 定数
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(APP_DIR, 'token.pickle')
CREDENTIALS_PATH = os.path.join(APP_DIR, 'credentials.json')

# OAuth設定（環境変数から読み込み）
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
REDIRECT_URI = os.environ.get('REDIRECT_URI', 'https://keyword-analysis-rrzzyfm8ktruqrca4k7nfv.streamlit.app')
COLORS = {
    'blue': '#1a73e8',
    'red': '#ea4335',
    'yellow': '#fbbc04',
    'green': '#34a853',
    'gray': '#5f6368',
    'light_gray': '#e8eaed',
    'bg': '#f8f9fa',
    'text': '#202124',
}

# セッション状態
defaults = {
    'credentials': None, 'keyword_data': None, 'authenticated': False,
    'sites': [], 'analysis_results': None, 'filter_keyword': '', 'display_count': 100,
    'industry': '医療機関',  # 業種選択
    'oauth_tokens': None  # OAuthトークン保存用
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def apply_theme(fig, height=400):
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Roboto, sans-serif', color=COLORS['text'], size=12),
        height=height,
        margin=dict(l=60, r=40, t=60, b=80),
        hoverlabel=dict(bgcolor='white', font_size=12, bordercolor=COLORS['light_gray']),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=COLORS['light_gray'], gridwidth=1,
        showline=True, linecolor=COLORS['light_gray'], linewidth=1,
        tickfont=dict(size=11, color=COLORS['gray']),
        title_font=dict(size=12, color=COLORS['gray'])
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=COLORS['light_gray'], gridwidth=1,
        showline=True, linecolor=COLORS['light_gray'], linewidth=1,
        tickfont=dict(size=11, color=COLORS['gray']),
        title_font=dict(size=12, color=COLORS['gray'])
    )
    return fig

# 認証機能（Web OAuth対応）
def get_google_auth_url():
    """Google OAuth認証URLを生成"""
    if not GOOGLE_CLIENT_ID:
        return None

    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': ' '.join(SCOPES),
        'response_type': 'code',
        'access_type': 'offline',
        'prompt': 'consent'
    }

    from urllib.parse import urlencode
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return auth_url

def exchange_code_for_tokens(code):
    """認証コードをトークンに交換"""
    import requests

    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'code': code,
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }

    try:
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            tokens = response.json()
            # Credentialsオブジェクトを作成
            creds = Credentials(
                token=tokens.get('access_token'),
                refresh_token=tokens.get('refresh_token'),
                token_uri='https://oauth2.googleapis.com/token',
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                scopes=SCOPES
            )
            return creds
    except Exception as e:
        st.error(f"トークン取得エラー: {e}")
    return None

def save_credentials_json(content):
    try:
        with open(CREDENTIALS_PATH, 'w') as f:
            json.dump(json.loads(content), f)
        return True
    except:
        return False

def load_saved_credentials():
    # セッションステートからトークンを読み込み
    if 'oauth_tokens' in st.session_state and st.session_state.oauth_tokens:
        try:
            tokens = st.session_state.oauth_tokens
            creds = Credentials(
                token=tokens.get('access_token'),
                refresh_token=tokens.get('refresh_token'),
                token_uri='https://oauth2.googleapis.com/token',
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                scopes=SCOPES
            )
            if creds and creds.valid:
                return creds
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                st.session_state.oauth_tokens = {
                    'access_token': creds.token,
                    'refresh_token': creds.refresh_token
                }
                return creds
        except:
            pass
    return None

def authenticate():
    if not os.path.exists(CREDENTIALS_PATH):
        return None
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=8080, prompt='consent')
        with open(TOKEN_PATH, 'wb') as f:
            pickle.dump(creds, f)
        return creds
    except:
        return None

def get_service(creds):
    try:
        return build('searchconsole', 'v1', credentials=creds)
    except:
        return None

def get_sites(service):
    try:
        return [s['siteUrl'] for s in service.sites().list().execute().get('siteEntry', [])]
    except:
        return []

def get_data(service, url, start, end):
    try:
        resp = service.searchanalytics().query(siteUrl=url, body={
            'startDate': start, 'endDate': end, 'dimensions': ['query'], 'rowLimit': 5000
        }).execute()
        return pd.DataFrame([{
            'query': r['keys'][0], 'clicks': r['clicks'], 'impressions': r['impressions'],
            'ctr': round(r['ctr']*100, 2), 'position': round(r['position'], 1)
        } for r in resp.get('rows', [])])
    except:
        return pd.DataFrame()

def logout():
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)
    for key in ['credentials', 'authenticated', 'sites', 'oauth_tokens']:
        st.session_state[key] = defaults.get(key, None)

# ===================================
# 業種別カテゴリ分類付き専門用語辞書
# ===================================

# 共通カテゴリ（全業種で使用）
COMMON_CATEGORIES = {
    '評価・品質': {
        'おすすめ', 'オススメ', '人気', '有名', '評判', '口コミ', 'レビュー',
        '上手', '上手い', 'うまい', '信頼', '安心', '丁寧', '親切',
        '最新', '先進', '最先端', '高度', '専門', '特化', '実績',
        'ランキング', '比較', 'ベスト', 'トップ', 'No1', 'ナンバーワン',
        '良い', '悪い', 'いい', 'ダメ', '最高', '最悪', '満足', '不満',
    },
    '価格・費用': {
        '料金', '費用', '価格', '値段', '相場', 'コスト', '見積もり', '見積り',
        '安い', '格安', '激安', '低価格', 'リーズナブル', 'お得', '割引', 'セール',
        '高い', '高額', '高級', 'プレミアム',
        '無料', 'タダ', '0円', 'キャンペーン', 'クーポン', 'ポイント',
        '分割', 'ローン', 'クレジット', '月額', '年額', 'サブスク',
    },
    '時間・アクセス': {
        '予約', '当日', '即日', '今日', '明日', '土曜', '日曜', '祝日',
        '夜間', '深夜', '早朝', '24時間', '年中無休', '営業時間', '定休日',
        '待ち時間', '待たない', 'すぐ', '短時間', '即対応', 'スピード',
        '近く', '近い', '駅近', '駅前', '徒歩', 'アクセス', '駐車場',
        '通いやすい', '行きやすい', '便利',
    },
    '地域': {
        '東京', '大阪', '名古屋', '福岡', '札幌', '仙台', '広島', '横浜', '神戸', '京都',
        '埼玉', '千葉', '神奈川', '愛知', '兵庫', '北海道', '沖縄',
        '新宿', '渋谷', '池袋', '銀座', '品川', '上野', '秋葉原',
        '梅田', '難波', 'なんば', '心斎橋', '天王寺',
        '駅前', '駅近', '近く', '周辺', '市内', '県内', '地域', 'エリア',
    },
}

# ===== 医療機関向けカテゴリ =====
MEDICAL_CATEGORIES = {
    '診療科': {
        # 内科系
        '内科', '消化器内科', '循環器内科', '呼吸器内科', '神経内科', '腎臓内科',
        '内分泌内科', '糖尿病内科', '血液内科', 'リウマチ科', 'アレルギー科',
        '感染症科', '心療内科', '総合内科', '老年内科', '肝臓内科', '膠原病科',
        # 外科系
        '外科', '消化器外科', '心臓血管外科', '呼吸器外科', '脳神経外科', '乳腺外科',
        '整形外科', '形成外科', '美容外科', '小児外科', '移植外科', '血管外科',
        # 専門科
        '眼科', '耳鼻咽喉科', '耳鼻科', '皮膚科', '泌尿器科', '産婦人科', '産科', '婦人科',
        '小児科', '精神科', 'メンタルクリニック', '神経科', '放射線科', '麻酔科',
        'ペインクリニック', 'リハビリテーション科', '救急科', '緩和ケア科', '腫瘍科',
        # 歯科系
        '歯科', '矯正歯科', '小児歯科', '口腔外科', '審美歯科', 'インプラント科', '歯科口腔外科',
    },

    '施設': {
        'クリニック', 'ホスピタル', '総合病院', '大学病院', '専門外来', '診療所',
        '医院', '病院', 'センター', '健診センター', 'ドッククリニック',
        '接骨院', '整骨院', '鍼灸院', '治療院', '薬局', 'ドラッグストア',
        '介護施設', '老人ホーム', 'デイサービス', 'リハビリ施設',
    },

    '病名・疾患': {
        # 内科系疾患
        '高血圧', '糖尿病', '脂質異常症', '動脈硬化', '心筋梗塞', '脳卒中', '脳梗塞',
        '狭心症', '不整脈', '心不全', '肺炎', '喘息', '気管支炎', 'COPD',
        '胃潰瘍', '十二指腸潰瘍', '逆流性食道炎', '胃炎', '腸炎', '過敏性腸症候群',
        '肝炎', '肝硬変', '脂肪肝', '膵炎', '胆石', '腎不全', '腎炎',
        'インフルエンザ', 'コロナ', 'ピロリ', 'ノロウイルス', '帯状疱疹',
        'メタボリック', 'メタボリックシンドローム', '痛風', '貧血', '甲状腺',
        # がん
        'がん', '癌', '腫瘍', '悪性腫瘍', '良性腫瘍', '乳がん', '胃がん', '大腸がん',
        '肺がん', '肝臓がん', '膵臓がん', '前立腺がん', '子宮がん', '卵巣がん',
        # 整形外科系
        '骨折', '脱臼', '捻挫', 'ヘルニア', '椎間板ヘルニア', '脊柱管狭窄症',
        '腰痛', '肩こり', '関節痛', '五十肩', '四十肩', '腱鞘炎', '関節リウマチ',
        '変形性膝関節症', '変形性股関節症', 'ぎっくり腰', '坐骨神経痛', '頸椎症',
        # 眼科系
        '白内障', '緑内障', '網膜剥離', '加齢黄斑変性', '糖尿病網膜症',
        'ドライアイ', '結膜炎', '眼瞼下垂', '斜視', '弱視', '近視', '遠視', '乱視', '老眼',
        '飛蚊症', '眼精疲労', 'ものもらい',
        # 耳鼻科系
        '中耳炎', '外耳炎', '難聴', '突発性難聴', 'メニエール病', '耳鳴り',
        '副鼻腔炎', '蓄膿症', 'アレルギー性鼻炎', '花粉症', '鼻炎', '扁桃炎',
        '咽頭炎', '喉頭炎', '声帯ポリープ', 'いびき', '睡眠時無呼吸症候群',
        # 皮膚科系
        'アトピー', 'アトピー性皮膚炎', '湿疹', '蕁麻疹', '乾癬', '水虫', '白癬',
        'ニキビ', '吹き出物', 'シミ', 'シワ', 'たるみ', 'ほくろ', 'イボ', 'あざ',
        '脱毛症', '円形脱毛症', 'AGA', '薄毛', 'ヘルペス', '帯状疱疹',
        # 泌尿器科系
        '膀胱炎', '前立腺肥大', '前立腺炎', '尿路結石', '腎結石', '尿失禁', '頻尿',
        'ED', '勃起不全', '性病', '性感染症',
        # 婦人科系
        '子宮筋腫', '子宮内膜症', '卵巣嚢腫', '月経困難症', '生理不順', '更年期障害',
        '不妊症', '妊娠', '出産', 'つわり', '乳腺炎',
        # 精神科系
        'うつ病', '鬱', '不眠症', '睡眠障害', 'パニック障害', '適応障害',
        '統合失調症', '双極性障害', 'ADHD', '発達障害', '自律神経失調症', '認知症',
        # 歯科系
        '虫歯', '歯周病', '歯肉炎', '歯槽膿漏', '知覚過敏', '顎関節症', '口内炎',
        '親知らず', '歯並び', '不正咬合', '出っ歯', '受け口',
        # 小児科系
        '発熱', '風邪', '手足口病', 'RSウイルス', 'おたふく', 'はしか', '水疱瘡',
    },

    '症状': {
        '痛み', '痛い', '腫れ', '腫れる', 'かゆみ', 'かゆい', 'しびれ', 'だるい',
        '頭痛', '腹痛', '胸痛', '背中痛', '首痛', '歯痛', '関節痛', '筋肉痛',
        '吐き気', '嘔吐', '下痢', '便秘', '血便', '血尿', '頻尿', '残尿感',
        '咳', '痰', '鼻水', '鼻づまり', 'くしゃみ', '喉の痛み', '声がれ',
        '発熱', '微熱', '悪寒', '倦怠感', '疲労', 'めまい', 'ふらつき', '立ちくらみ',
        '動悸', '息切れ', '胸やけ', 'むくみ', '浮腫', '冷え', '冷え性',
        'かすみ目', '充血', '目やに', '涙目', '眼痛',
        '耳鳴り', '耳垂れ', '耳痛', '聞こえにくい',
        '出血', 'あざ', '発疹', '湿疹', 'じんましん', '水ぶくれ', 'ただれ',
        '抜け毛', 'フケ', 'べたつき', '乾燥', 'ひび割れ',
        '不眠', '眠れない', '食欲不振', '体重減少', '体重増加',
        '物忘れ', '集中力低下', 'イライラ', '不安', '憂鬱',
    },

    '体の部位': {
        '頭', '顔', '額', 'おでこ', '目', '眼', '鼻', '耳', '口', '唇', '舌', '歯', '歯茎',
        '顎', 'あご', '頬', 'ほほ', '首', '喉', 'のど', '肩', '腕', '肘', '手首', '手', '指',
        '胸', '乳房', 'おっぱい', '背中', '腰', 'お腹', '腹部', 'おへそ', 'お尻', '股',
        '太もも', '膝', 'ひざ', 'すね', 'ふくらはぎ', '足首', '足', '足裏', 'かかと', 'つま先',
        '心臓', '肺', '肝臓', '腎臓', '胃', '腸', '大腸', '小腸', '膵臓', '脾臓', '胆嚢',
        '膀胱', '子宮', '卵巣', '前立腺', '甲状腺', '副腎',
        '脳', '神経', '血管', '動脈', '静脈', 'リンパ', '骨', '関節', '筋肉', '腱', '靭帯',
        '皮膚', '毛', '髪', '爪', 'まぶた', 'まつげ', '眉毛',
    },

    '治療・施術': {
        # 一般治療
        '治療', '手術', 'オペ', '処置', '施術', 'ケア', '療法', 'セラピー',
        '投薬', '点滴', '注射', '予防接種', 'ワクチン', '輸血', '透析',
        'リハビリ', 'リハビリテーション', '理学療法', '作業療法', '言語療法',
        # 歯科治療
        'インプラント', 'ホワイトニング', 'セラミック', 'マウスピース', 'クリーニング',
        'ブリッジ', 'スケーリング', 'ルートプレーニング', 'リテーナー', 'アライナー',
        'ブラケット', 'ワイヤー', 'オールセラミック', 'ジルコニア', 'ラミネート',
        'ベニア', 'クラウン', 'インレー', 'オンレー', 'メタルボンド', '入れ歯',
        'デンチャー', 'インビザライン', 'クリアアライナー', '抜歯', '根管治療', '神経治療',
        # 眼科治療
        'レーシック', 'ICL', 'オルソケラトロジー', 'フェイキック', '眼内レンズ',
        '白内障手術', '緑内障手術', '硝子体手術', 'レーザー治療',
        # 美容治療
        'ボトックス', 'ヒアルロン', 'ヒアルロン酸', 'プラセンタ', 'ピーリング', 'レーザー',
        'ダーマペン', 'フォトフェイシャル', 'エレクトロポレーション', 'イオン導入',
        'リフトアップ', 'サーマクール', 'ハイフ', 'ウルセラ', 'スレッドリフト',
        'フェイスリフト', '脂肪吸引', '豊胸', 'シリコン', 'プロテーゼ',
        '二重整形', '埋没法', '切開法', '隆鼻術', '小顔整形', 'エラ削り',
        'ケミカルピーリング', 'トレチノイン', 'ハイドロキノン', '脱毛', 'レーザー脱毛',
        # 整形外科治療
        'ブロック注射', 'トリガーポイント', 'AKA', 'PRP', '人工関節', '骨接合術',
        'カイロプラクティック', 'マッサージ', 'ストレッチ', '牽引', 'テーピング',
        # 内視鏡系
        '内視鏡', '胃カメラ', '大腸カメラ', 'カプセル内視鏡', '腹腔鏡', '胸腔鏡',
        # その他
        '漢方', '鍼灸', '鍼', '灸', '指圧', '整体', 'オステオパシー',
        '放射線治療', '化学療法', '抗がん剤', '免疫療法', 'ホルモン療法',
    },

    '検査・診断': {
        '検査', '診断', '診察', '問診', '触診', '聴診', '視診',
        'CT', 'MRI', 'レントゲン', 'X線', 'エコー', '超音波', 'PET',
        '血液検査', '尿検査', '便検査', '心電図', '脳波', '筋電図',
        '内視鏡検査', '胃カメラ検査', '大腸カメラ検査', 'マンモグラフィ',
        '健診', '検診', '人間ドック', 'ドック', 'スクリーニング', '精密検査',
        'がん検診', '乳がん検診', '子宮がん検診', '肺がん検診', '大腸がん検診',
        '眼底検査', '視力検査', '眼圧検査', '聴力検査',
        'アレルギー検査', '遺伝子検査', 'PCR', '抗体検査', '抗原検査',
        'セカンドオピニオン', '紹介状', '診断書',
    },

    '薬・医薬品': {
        '薬', '医薬品', '処方薬', '処方箋', '市販薬', 'OTC',
        '抗生物質', '抗菌薬', '解熱剤', '鎮痛剤', '痛み止め', '睡眠薬', '安定剤',
        '胃薬', '整腸剤', '下剤', '便秘薬', '下痢止め', '吐き気止め',
        '目薬', '点眼薬', '軟膏', 'クリーム', '湿布', 'パップ',
        'ジェネリック', '後発医薬品', '先発医薬品', 'サプリ', 'サプリメント',
        'ビタミン', 'ミネラル', 'プロテイン', 'アミノ酸',
        'インスリン', 'ステロイド', '抗ヒスタミン', '降圧剤', '利尿剤',
    },

    # === 形容詞系（どんな状態・条件か） ===
    '評価・品質': {
        'おすすめ', 'オススメ', '人気', '有名', '評判', '口コミ', 'レビュー',
        '名医', '専門医', '認定医', '指導医', 'ベテラン', '実績', '症例数',
        '上手', '上手い', 'うまい', '腕がいい', '信頼', '安心', '丁寧',
        '最新', '先進', '最先端', '高度', '専門', '特化',
        'ランキング', '比較', 'ベスト', 'トップ', 'No1', 'ナンバーワン',
    },

    '価格・費用': {
        '料金', '費用', '価格', '値段', '相場', 'コスト',
        '安い', '格安', '激安', '低価格', 'リーズナブル', 'お得', '割引',
        '高い', '高額', '高級',
        '無料', 'タダ', '0円', '保険適用', '保険診療', '自費', '自由診療',
        '分割', 'ローン', 'クレジット', '医療費控除', '助成金', '補助金',
    },

    '時間・アクセス': {
        '予約', '当日', '即日', '今日', '明日', '土曜', '日曜', '祝日',
        '夜間', '深夜', '早朝', '24時間', '年中無休', '休診日',
        '待ち時間', '待たない', 'すぐ', '短時間', '日帰り',
        '近く', '近い', '駅近', '駅前', '徒歩', 'アクセス', '駐車場',
        '通いやすい', '行きやすい',
    },

    '対象・条件': {
        '初診', '再診', '初めて', '初回', 'カウンセリング', '相談',
        '男性', '女性', '子供', '子ども', '小児', '赤ちゃん', '乳児', '幼児',
        '高齢者', 'お年寄り', 'シニア', '妊婦', '妊娠中', '授乳中',
        '痛くない', '無痛', '麻酔', '局所麻酔', '全身麻酔', '鎮静',
        '日帰り', '入院', '通院', '在宅', '往診', '訪問診療',
        'オンライン診療', '遠隔診療', 'リモート', 'テレビ電話',
    },
}

# ===== 一般企業向けカテゴリ =====
GENERAL_CATEGORIES = {
    '業種・業界': {
        'IT', 'システム', 'ソフトウェア', 'アプリ', 'Web', 'ウェブ', 'インターネット',
        '製造', 'メーカー', '工場', '生産', '加工', '組立',
        '建設', '建築', '土木', 'リフォーム', 'リノベーション', '設計', '施工',
        '不動産', '賃貸', '売買', '仲介', '管理', 'マンション', 'アパート', '戸建て',
        '小売', '販売', 'ショップ', '店舗', 'ECサイト', 'ネットショップ', '通販',
        '飲食', 'レストラン', 'カフェ', '居酒屋', 'バー', 'ファストフード',
        '金融', '銀行', '証券', '保険', 'ローン', '融資', '投資',
        '教育', '学習', '塾', 'スクール', '研修', 'セミナー', '講座',
        '人材', '派遣', '紹介', '転職', '求人', '採用', 'キャリア',
        '広告', 'マーケティング', 'PR', 'プロモーション', 'ブランディング',
        'コンサル', 'コンサルティング', '顧問', 'アドバイザー',
        '物流', '配送', '運送', '倉庫', '輸送', 'デリバリー',
        '旅行', '観光', 'ツアー', 'ホテル', '宿泊', '航空', 'トラベル',
        '美容', 'エステ', 'サロン', 'ネイル', 'ヘア', 'スパ', 'マッサージ',
        'フィットネス', 'ジム', 'スポーツ', 'ヨガ', 'ピラティス', 'トレーニング',
        '介護', '福祉', 'ヘルスケア', 'シニア', 'デイサービス',
        '法律', '弁護士', '司法書士', '行政書士', '税理士', '会計士',
    },

    '商品・サービス': {
        '商品', '製品', 'サービス', 'プラン', 'コース', 'パッケージ', 'オプション',
        'ソリューション', 'ツール', 'システム', 'アプリ', 'ソフト', 'プラットフォーム',
        '機能', '特徴', 'メリット', 'デメリット', '違い', '比較',
        '導入', '利用', '活用', '運用', 'サポート', '保守', 'メンテナンス',
        'カスタマイズ', 'オーダーメイド', 'オリジナル', '限定', '新商品', '新サービス',
        '定番', 'ロングセラー', 'ヒット', '話題', 'トレンド',
    },

    '企業・組織': {
        '会社', '企業', '法人', '株式会社', '有限会社', '合同会社', 'LLC',
        '本社', '支社', '支店', '営業所', '事務所', 'オフィス',
        '大手', '中小', 'ベンチャー', 'スタートアップ', '老舗', '新興',
        '上場', '非上場', '外資', '国内', 'グローバル', '地元', '地域密着',
        '代表', '社長', 'CEO', '経営者', '創業者', 'ファウンダー',
    },

    'アクション': {
        '購入', '買う', '申込', '申し込み', '契約', '登録', '加入', '入会',
        '問い合わせ', '相談', '見積もり', '見積り', '資料請求', 'お問い合わせ',
        '予約', '注文', 'オーダー', '発注', '依頼', 'お願い',
        '検討', '比較', '選び方', '選ぶ', '探す', '探し方', '見つける',
        'ダウンロード', 'インストール', '登録', 'ログイン', 'サインアップ',
        '解約', 'キャンセル', '退会', '返品', '返金', 'クーリングオフ',
        '変更', '更新', '切り替え', '乗り換え', 'アップグレード',
    },

    '対象・ターゲット': {
        '個人', '法人', 'BtoB', 'BtoC', '企業向け', '個人向け',
        '初心者', '初めて', '入門', 'ビギナー', '未経験',
        '経験者', '上級者', 'プロ', '専門家', 'エキスパート',
        '男性', '女性', '学生', '社会人', '主婦', 'シニア', '若者',
        '中小企業', '大企業', 'スタートアップ', 'フリーランス', '個人事業主',
    },

    '品質・特徴': {
        '高品質', '低品質', '品質', 'クオリティ', '信頼性', '安定性', '耐久性',
        '実績', '経験', 'ノウハウ', '技術', 'スキル', '専門性',
        '対応', '丁寧', '迅速', 'スピーディ', '柔軟', 'フレキシブル',
        '安全', 'セキュリティ', '保証', '保障', 'アフターサービス', 'サポート',
        '簡単', 'シンプル', '使いやすい', '便利', '手軽', 'お手軽',
        '効果', '効率', '成果', '結果', 'パフォーマンス', 'ROI',
    },
}

# ===== 業種設定 =====
INDUSTRY_CONFIGS = {
    '医療機関': {
        'categories': {**MEDICAL_CATEGORIES, **COMMON_CATEGORIES},
        'label': '医療機関（病院・クリニック・歯科等）',
    },
    '一般企業': {
        'categories': {**GENERAL_CATEGORIES, **COMMON_CATEGORIES},
        'label': '一般企業（IT・小売・サービス等）',
    },
}

def get_industry_terms(industry='医療機関'):
    """業種に応じた用語セットを取得"""
    config = INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS['医療機関'])
    terms = set()
    for category_terms in config['categories'].values():
        terms.update(category_terms)
    return terms

def get_term_to_category(industry='医療機関'):
    """業種に応じた用語→カテゴリ逆引き辞書を取得"""
    config = INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS['医療機関'])
    term_to_cat = {}
    for category, terms in config['categories'].items():
        for term in terms:
            if term not in term_to_cat:
                term_to_cat[term] = []
            term_to_cat[term].append(category)
    return term_to_cat

# デフォルト（後方互換性のため）
MEDICAL_TERMS = get_industry_terms('医療機関')
TERM_TO_CATEGORY = get_term_to_category('医療機関')

@st.cache_resource
def get_tokenizer():
    return Tokenizer()

def tokenize(query, tokenizer, industry='医療機関'):
    """形態素解析（専門用語を保護、長い用語を優先）"""
    import re
    query_str = str(query)
    found_terms = []

    # 業種に応じた用語セットを取得
    industry_terms = get_industry_terms(industry)

    # 長い用語から優先的にマッチ（部分一致を防ぐ）
    sorted_terms = sorted(industry_terms, key=len, reverse=True)

    for term in sorted_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(query_str):
            for match in pattern.finditer(query_str):
                found_terms.append(match.group())
            query_str = pattern.sub(' ', query_str)

    # 通常の形態素解析
    tokens = [t.surface for t in tokenizer.tokenize(query_str)
              if t.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞', '副詞']
              and len(t.surface) > 1 and not t.surface.isdigit()]

    # 専門用語を追加
    tokens.extend(found_terms)

    return tokens

def calc_score(row):
    pos_score = max(0, (20 - row['position']) / 20) * 50
    ctr_score = min(row['ctr'] / 10, 1) * 30
    click_score = min(row['clicks'] / 100, 1) * 20
    return round(pos_score + ctr_score + click_score, 1)

def classify(row, avg_ctr, med_pos):
    high_ctr = row['ctr'] >= avg_ctr
    good_pos = row['position'] <= med_pos
    if good_pos and high_ctr: return 'star'
    if not good_pos and high_ctr: return 'potential'
    if good_pos and not high_ctr: return 'improve'
    return 'stable'

def get_word_category(word, industry='医療機関'):
    """単語のカテゴリを取得"""
    term_to_cat = get_term_to_category(industry)
    return term_to_cat.get(word, ['その他'])

def analyze(df, tokenizer, filter_kw='', industry='医療機関'):
    if filter_kw:
        df = df[df['query'].str.contains(filter_kw, case=False, na=False)]
    if df.empty:
        return None

    df = df.copy()
    df['score'] = df.apply(calc_score, axis=1)
    avg_ctr, med_pos = df['ctr'].mean(), df['position'].median()
    df['category'] = df.apply(lambda x: classify(x, avg_ctr, med_pos), axis=1)

    results = {
        'word_freq': Counter(), 'cooccurrence': Counter(),
        'word_position': defaultdict(lambda: {'前方': 0, '後方': 0, '単体': 0}),
        'word_stats': defaultdict(lambda: {'ctr_sum': 0, 'pos_sum': 0, 'count': 0}),
        'word_categories': defaultdict(Counter),  # カテゴリ別の単語出現
        'category_freq': Counter(),  # カテゴリ別の総出現回数
        'total_imp': df['impressions'].sum(),
        'total_clicks': df['clicks'].sum(),
        'count': len(df),
        'avg_ctr': avg_ctr,
        'avg_pos': df['position'].mean(),
        'med_pos': med_pos,
        'df': df
    }

    for _, row in df.iterrows():
        tokens = tokenize(row['query'], tokenizer, industry)
        for i, t in enumerate(tokens):
            results['word_freq'][t] += row['impressions']
            results['word_stats'][t]['ctr_sum'] += row['ctr']
            results['word_stats'][t]['pos_sum'] += row['position']
            results['word_stats'][t]['count'] += 1

            # カテゴリ分類を追加
            categories = get_word_category(t, industry)
            for cat in categories:
                results['word_categories'][cat][t] += row['impressions']
                results['category_freq'][cat] += row['impressions']

            if len(tokens) == 1:
                results['word_position'][t]['単体'] += row['impressions']
            elif i == 0:
                results['word_position'][t]['前方'] += row['impressions']
            elif i == len(tokens) - 1:
                results['word_position'][t]['後方'] += row['impressions']

        for pair in combinations(tokens, 2):
            results['cooccurrence'][tuple(sorted(pair))] += row['impressions']

    return results

def get_color(word, wp):
    pos = wp.get(word, {'前方': 0, '後方': 0, '単体': 0})
    total = sum(pos.values())
    if total == 0: return COLORS['gray']
    if pos['単体'] / total > 0.5: return COLORS['red']
    if pos['前方'] > pos['後方']: return COLORS['blue']
    if pos['後方'] > pos['前方']: return COLORS['green']
    return '#9334e6'

def create_wordcloud(wf, wp):
    """高解像度ワードクラウドを生成（図と画像バイナリを返す）"""
    font_path = None
    try:
        import japanize_matplotlib
        font_path = os.path.join(os.path.dirname(japanize_matplotlib.__file__), 'fonts', 'ipaexg.ttf')
    except:
        pass
    if not font_path or not os.path.exists(font_path):
        for fp in ['/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc', '/Library/Fonts/ヒラギノ角ゴ ProN W3.otf',
                   '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
            if os.path.exists(fp):
                font_path = fp
                break
    if not wf:
        return None, None
    try:
        # 高解像度設定
        wc = WordCloud(
            width=1600, height=640, background_color='white', font_path=font_path,
            max_words=80, color_func=lambda word, **kw: get_color(word, wp),
            prefer_horizontal=0.9, scale=2
        ).generate_from_frequencies(wf)

        # 高DPIでmatplotlib図を作成
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)

        # PNG画像データを取得（ダウンロード用）
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight', pad_inches=0.1,
                    facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_bytes = img_buffer.getvalue()

        return fig, img_bytes
    except:
        return None, None

def create_scatter(r, limit=100):
    df = r['df'].nlargest(limit, 'impressions').copy()
    if df.empty:
        return None

    cat_colors = {'star': COLORS['yellow'], 'potential': COLORS['green'], 'stable': COLORS['blue'], 'improve': COLORS['red']}
    cat_labels = {'star': 'スター', 'potential': 'ポテンシャル', 'stable': '安定', 'improve': '要改善'}

    fig = go.Figure()
    fig.add_hline(y=r['avg_ctr'], line_dash="dot", line_color="#dadce0", line_width=1)
    fig.add_vline(x=r['med_pos'], line_dash="dot", line_color="#dadce0", line_width=1)

    for cat in ['star', 'potential', 'stable', 'improve']:
        sub = df[df['category'] == cat]
        if not sub.empty:
            sizes = sub['impressions'].apply(lambda x: min(28, max(8, np.log(x+1)*2.8)))
            fig.add_trace(go.Scatter(
                x=sub['position'], y=sub['ctr'], mode='markers', name=cat_labels[cat],
                marker=dict(size=sizes, color=cat_colors[cat], opacity=0.8, line=dict(width=1, color='white')),
                text=sub['query'], hovertemplate="<b>%{text}</b><br>順位: %{x:.1f}<br>CTR: %{y:.1f}%<extra></extra>"
            ))

    y_max = df['ctr'].max() if not df.empty else 10
    x_max = df['position'].max() if not df.empty else 100

    fig.update_layout(
        xaxis_title='掲載順位（左が上位）', yaxis_title='CTR（%）',
        xaxis=dict(autorange='reversed', range=[x_max * 1.1, 0]),
        yaxis=dict(range=[0, y_max * 1.25]),
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center', font=dict(size=11), bgcolor='rgba(255,255,255,0)'),
        title=dict(text=f'<span style="font-size:11px;color:#5f6368">平均CTR: {r["avg_ctr"]:.2f}%　｜　中央順位: {r["med_pos"]:.1f}</span>', x=0.5, y=0.98, xanchor='center')
    )
    return apply_theme(fig, 480)

def create_word_chart(r, limit=12):
    ws, wf = r['word_stats'], r['word_freq']
    if not ws:
        return None

    data = []
    for w, f in wf.most_common(limit):
        s = ws.get(w, {})
        c = s.get('count', 1)
        data.append({'word': w, 'volume': f, 'ctr': s.get('ctr_sum', 0)/c if c else 0, 'pos': s.get('pos_sum', 0)/c if c else 0})

    df = pd.DataFrame(data)
    ctr_norm = (df['ctr'] - df['ctr'].min()) / (df['ctr'].max() - df['ctr'].min() + 0.01)
    pos_norm = (df['pos'].max() - df['pos']) / (df['pos'].max() - df['pos'].min() + 0.01)
    df['score'] = ctr_norm * 50 + pos_norm * 50

    fig = go.Figure(go.Bar(
        y=df['word'], x=df['volume'], orientation='h',
        marker=dict(color=df['score'], colorscale=[[0, COLORS['red']], [0.5, COLORS['yellow']], [1, COLORS['green']]], line=dict(width=0)),
        text=df.apply(lambda x: f"CTR {x['ctr']:.1f}%　順位 {x['pos']:.0f}", axis=1),
        textposition='inside', textfont=dict(size=11, color='white'),
        hovertemplate="<b>%{y}</b><br>表示回数: %{x:,}<extra></extra>"
    ))
    fig.update_layout(yaxis=dict(categoryorder='total ascending'), xaxis_title='表示回数', bargap=0.2)
    return apply_theme(fig, 360)

def create_network(r, n=35):
    cooc, wf = r['cooccurrence'], r['word_freq']
    if not cooc:
        return None

    pairs = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:n]
    G = nx.Graph()
    for (w1, w2), c in pairs:
        G.add_edge(w1, w2, weight=c)

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    max_w = max(d['weight'] for _, _, d in G.edges(data=True))
    max_f = max(wf.values()) if wf else 1

    edge_traces = [go.Scatter(
        x=[pos[e[0]][0], pos[e[1]][0], None], y=[pos[e[0]][1], pos[e[1]][1], None],
        mode='lines', line=dict(width=max(0.5, e[2]['weight']/max_w*2), color='rgba(95,99,104,0.25)'), hoverinfo='none'
    ) for e in G.edges(data=True)]

    node_sizes = [14 + min(16, wf.get(n, 0)/max_f*16) for n in G.nodes()]
    palette = [COLORS['blue'], COLORS['green'], COLORS['yellow'], COLORS['red']] * 10

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode='markers+text', text=list(G.nodes()), textposition='top center',
        textfont=dict(size=10, color=COLORS['text']),
        hovertext=[f"{n}: {wf.get(n,0):,}" for n in G.nodes()], hoverinfo='text',
        marker=dict(size=node_sizes, color=palette[:len(G.nodes())], line=dict(width=2, color='white'))
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10))
    return apply_theme(fig, 400)

def create_category_chart(r):
    """カテゴリ別の単語分布を表示"""
    cat_freq = r.get('category_freq', {})
    if not cat_freq:
        return None

    # カテゴリを出現順にソート
    sorted_cats = sorted(cat_freq.items(), key=lambda x: x[1], reverse=True)[:12]
    if not sorted_cats:
        return None

    categories = [c[0] for c in sorted_cats]
    values = [c[1] for c in sorted_cats]

    # カテゴリごとの色
    cat_colors = {
        '診療科': COLORS['blue'], '施設': '#4285f4', '病名・疾患': COLORS['red'],
        '症状': '#ea4335', '体の部位': '#fbbc04', '治療・施術': COLORS['green'],
        '検査・診断': '#34a853', '薬・医薬品': '#9334e6', '評価・品質': '#ff6d01',
        '価格・費用': '#46bdc6', '時間・アクセス': '#7baaf7', '対象・条件': '#ee675c',
        '地域': '#fcc934', 'その他': COLORS['gray']
    }
    colors = [cat_colors.get(c, COLORS['gray']) for c in categories]

    fig = go.Figure(go.Bar(
        y=categories, x=values, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:,}' for v in values],
        textposition='auto', textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>出現回数: %{x:,}<extra></extra>"
    ))
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        xaxis_title='出現回数（表示回数ベース）',
        bargap=0.25
    )
    return apply_theme(fig, 380)

def create_category_detail_table(r, category):
    """特定カテゴリの単語詳細テーブルを作成"""
    word_cats = r.get('word_categories', {})
    if category not in word_cats:
        return None

    words = word_cats[category].most_common(15)
    return pd.DataFrame([{'単語': w, '出現回数': f'{c:,}'} for w, c in words])

# HTMLレポート生成（PDF代替）
def generate_html_report(r):
    cat_labels = {'star': 'スター', 'potential': 'ポテンシャル', 'stable': '安定', 'improve': '要改善'}
    cat_counts = r['df']['category'].value_counts()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>キーワード分析レポート</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; color: #202124; }}
h1 {{ font-size: 1.5rem; font-weight: 400; border-bottom: 2px solid #1a73e8; padding-bottom: 0.5rem; }}
h2 {{ font-size: 1.1rem; font-weight: 500; color: #1a73e8; margin-top: 2rem; }}
.summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }}
.card {{ background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; }}
.card-value {{ font-size: 1.5rem; font-weight: 500; color: #202124; }}
.card-label {{ font-size: 0.75rem; color: #5f6368; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #e8eaed; }}
th {{ background: #f8f9fa; font-weight: 500; }}
.date {{ color: #5f6368; font-size: 0.85rem; }}
</style></head><body>
<h1>キーワード分析レポート</h1>
<p class="date">作成日: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>

<h2>サマリー</h2>
<div class="summary">
<div class="card"><div class="card-value">{r['count']:,}</div><div class="card-label">クエリ数</div></div>
<div class="card"><div class="card-value">{r['total_imp']:,}</div><div class="card-label">表示回数</div></div>
<div class="card"><div class="card-value">{r['total_clicks']:,}</div><div class="card-label">クリック数</div></div>
<div class="card"><div class="card-value">{r['avg_ctr']:.2f}%</div><div class="card-label">平均CTR</div></div>
<div class="card"><div class="card-value">{r['avg_pos']:.1f}</div><div class="card-label">平均順位</div></div>
</div>

<h2>カテゴリ分布</h2>
<table><tr><th>カテゴリ</th><th>件数</th><th>割合</th></tr>"""

    for cat, label in cat_labels.items():
        count = cat_counts.get(cat, 0)
        pct = count / r['count'] * 100 if r['count'] > 0 else 0
        html += f"<tr><td>{label}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>"

    html += """</table><h2>単語ランキング Top20</h2><table><tr><th>#</th><th>単語</th><th>出現回数</th></tr>"""
    for i, (word, freq) in enumerate(r['word_freq'].most_common(20), 1):
        html += f"<tr><td>{i}</td><td>{word}</td><td>{freq:,}</td></tr>"

    html += """</table><h2>共起ペア Top20</h2><table><tr><th>#</th><th>単語ペア</th><th>共起回数</th></tr>"""
    for i, ((w1, w2), freq) in enumerate(r['cooccurrence'].most_common(20), 1):
        html += f"<tr><td>{i}</td><td>{w1} + {w2}</td><td>{freq:,}</td></tr>"

    html += """</table><h2>効率スコア Top20</h2><table><tr><th>#</th><th>クエリ</th><th>CTR</th><th>順位</th><th>スコア</th></tr>"""
    for i, (_, row) in enumerate(r['df'].nlargest(20, 'score').iterrows(), 1):
        html += f"<tr><td>{i}</td><td>{row['query']}</td><td>{row['ctr']:.1f}%</td><td>{row['position']:.1f}</td><td>{row['score']:.1f}</td></tr>"

    html += "</table></body></html>"
    return html.encode('utf-8')

DESC = {
    'scatter': 'キーワードを掲載順位とCTRでマッピング。円の大きさは表示回数を表します。<br><span style="color:#1a73e8;font-size:0.8rem;">💡 各データポイントにマウスを重ねると、キーワード詳細が表示されます</span>',
    'word': '検索クエリに含まれる単語の出現頻度。色はパフォーマンススコア（緑=良好、赤=改善余地）。',
    'cloud': '単語の出現頻度を視覚化。色は出現位置（青=前方、緑=後方、赤=単体）。',
    'network': '同時に検索される単語の関係性。線が太いほど共起頻度が高い。<br><span style="color:#1a73e8;font-size:0.8rem;">💡 ノードにマウスを重ねると表示回数が確認できます</span>',
    'cooc': '同じクエリ内で一緒に出現する単語ペアのランキング。',
    'score': '順位・CTR・クリック数から算出したスコアの高いクエリ。',
    'category': '検索キーワードを意味カテゴリ別に分類。主語（診療科・施設・病名）、形容詞（評価・価格・条件）などで分類されます。',
}

DEMO_DATA = """query,clicks,impressions,ctr,position
歯科 インプラント 費用,150,2500,6.0,5.2
インプラント おすすめ,120,2000,6.0,4.8
インプラント 痛くない,98,1800,5.4,6.1
近くの歯科,85,1500,5.7,5.5
歯科 おすすめ,72,1200,6.0,7.2
ホワイトニング 歯科,68,1100,6.2,4.5
歯科 ランキング,62,1050,5.9,6.8
親知らず 抜歯,58,980,5.9,5.9
矯正歯科 費用,54,920,5.9,8.1
歯科 予約,51,850,6.0,6.5
小児歯科 おすすめ,49,820,6.0,7.8
歯科 日曜,47,780,6.0,5.2
審美歯科 セラミック,45,750,6.0,4.1
歯科 名医,42,700,6.0,6.3
虫歯 治療,40,680,5.9,7.5
歯周病 治療,38,650,5.8,8.2
歯科 クリーニング,36,620,5.8,7.1
歯科 近く,34,600,5.7,5.8
歯科 口コミ,32,580,5.5,6.9
歯科 評判,30,550,5.5,7.5"""

def main():
    st.title("Keyword Mining")

    # OAuthコールバック処理（URLにcodeパラメータがある場合）
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.authenticated:
        code = query_params['code']
        with st.spinner("Googleアカウントに接続中..."):
            creds = exchange_code_for_tokens(code)
            if creds:
                st.session_state.credentials = creds
                st.session_state.authenticated = True
                st.session_state.oauth_tokens = {
                    'access_token': creds.token,
                    'refresh_token': creds.refresh_token
                }
                # URLパラメータをクリア
                st.query_params.clear()
                st.rerun()
            else:
                st.error("認証に失敗しました。もう一度お試しください。")
                st.query_params.clear()

    if not st.session_state.authenticated:
        creds = load_saved_credentials()
        if creds:
            st.session_state.credentials = creds
            st.session_state.authenticated = True

    # データがない場合：userlocal風の入力画面
    if st.session_state.keyword_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <p style="color: #5f6368; font-size: 0.95rem;">Search Consoleのキーワードデータを分析し、ワードクラウド・共起ネットワーク・ランキングを生成します</p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["CSVアップロード", "Search Console連携"])

        with tab1:
            st.markdown("""
            <div style="border: 2px dashed #dadce0; border-radius: 12px; padding: 2.5rem; text-align: center; background: #fafafa; margin: 1rem 0;">
                <p style="font-size: 1.1rem; color: #202124; margin-bottom: 0.5rem;">CSVファイルをドラッグ＆ドロップ</p>
                <p style="font-size: 0.85rem; color: #5f6368;">または下のボタンからファイルを選択</p>
            </div>
            """, unsafe_allow_html=True)

            file = st.file_uploader("CSVファイルを選択", type=['csv'], label_visibility="collapsed")

            if file:
                try:
                    df = pd.read_csv(file)
                    st.session_state.keyword_data = df
                    st.session_state.analysis_results = None
                    st.rerun()
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

            st.markdown("---")
            st.markdown("**CSVフォーマット**")
            st.code("query,clicks,impressions,ctr,position", language=None)

            col1, col2 = st.columns(2)
            sample = "query,clicks,impressions,ctr,position\n歯科 インプラント,150,2500,6.0,5.2\nホワイトニング 費用,120,2000,6.0,4.8"
            col1.download_button("サンプルCSVをダウンロード", sample, "sample.csv", use_container_width=True)
            if col2.button("デモデータで試す", type="primary", use_container_width=True):
                st.session_state.keyword_data = pd.read_csv(StringIO(DEMO_DATA))
                st.session_state.analysis_results = None
                st.rerun()

        with tab2:
            if st.session_state.authenticated:
                st.success("Google アカウント接続済み")
                if st.button("ログアウト"):
                    logout()
                    st.rerun()

                service = get_service(st.session_state.credentials)
                if service:
                    if not st.session_state.sites:
                        st.session_state.sites = get_sites(service)

                    if st.session_state.sites:
                        site = st.selectbox("サイトを選択", st.session_state.sites)
                        c1, c2 = st.columns(2)
                        start = c1.date_input("開始日", datetime.now().date() - timedelta(days=28))
                        end = c2.date_input("終了日", datetime.now().date() - timedelta(days=3))

                        if st.button("データを取得", type="primary", use_container_width=True):
                            with st.spinner("取得中..."):
                                df = get_data(service, site, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                                if not df.empty:
                                    st.session_state.keyword_data = df
                                    st.session_state.analysis_results = None
                                    st.rerun()
                                else:
                                    st.warning("データがありません")
            else:
                # Googleでログインボタン
                if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem 0;">
                        <p style="color: #5f6368; margin-bottom: 1.5rem;">Search Consoleのデータに直接アクセスするには、Googleアカウントでログインしてください</p>
                    </div>
                    """, unsafe_allow_html=True)

                    auth_url = get_google_auth_url()
                    if auth_url:
                        # Googleログインボタン（リンクとして表示）
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <a href="{auth_url}" target="_self" style="
                                display: inline-flex;
                                align-items: center;
                                gap: 12px;
                                background: white;
                                border: 1px solid #dadce0;
                                border-radius: 4px;
                                padding: 12px 24px;
                                text-decoration: none;
                                color: #3c4043;
                                font-family: 'Roboto', sans-serif;
                                font-weight: 500;
                                font-size: 14px;
                                transition: background 0.2s, box-shadow 0.2s;
                            " onmouseover="this.style.background='#f8f9fa'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';"
                               onmouseout="this.style.background='white'; this.style.boxShadow='none';">
                                <svg width="18" height="18" viewBox="0 0 24 24">
                                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                                </svg>
                                Googleでログイン
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("OAuth設定が見つかりません。環境変数 GOOGLE_CLIENT_ID と GOOGLE_CLIENT_SECRET を設定してください。")

    # データがある場合：分析結果表示
    else:
        df = st.session_state.keyword_data

        # 業種選択（上部に配置）
        industry_col1, industry_col2 = st.columns([1, 3])
        with industry_col1:
            industry_options = list(INDUSTRY_CONFIGS.keys())
            current_idx = industry_options.index(st.session_state.industry) if st.session_state.industry in industry_options else 0
            selected_industry = st.selectbox(
                "業種カテゴリ",
                industry_options,
                index=current_idx,
                format_func=lambda x: INDUSTRY_CONFIGS[x]['label'],
                key="industry_select"
            )
            if selected_industry != st.session_state.industry:
                st.session_state.industry = selected_industry
                st.session_state.analysis_results = None
                st.rerun()

        # 上部コントロール
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        filter_kw = col1.text_input("絞り込み", value=st.session_state.filter_keyword, placeholder="キーワードを入力", label_visibility="collapsed")
        if col2.button("適用", use_container_width=True):
            st.session_state.filter_keyword = filter_kw
            st.session_state.analysis_results = None
        if col3.button("クリア", use_container_width=True):
            st.session_state.filter_keyword = ''
            st.session_state.analysis_results = None
            st.rerun()
        if col4.button("新規データ", use_container_width=True):
            st.session_state.keyword_data = None
            st.session_state.analysis_results = None
            st.rerun()

        tokenizer = get_tokenizer()
        if st.session_state.analysis_results is None:
            with st.spinner("分析中..."):
                st.session_state.analysis_results = analyze(df, tokenizer, st.session_state.filter_keyword, st.session_state.industry)

        r = st.session_state.analysis_results
        if r is None:
            st.warning("該当データなし")
            return

        # KPI
        st.divider()
        cols = st.columns(5)
        metrics = [
            ("クエリ数", f"{r['count']:,}", "分析対象のキーワード数"),
            ("表示回数", f"{r['total_imp']:,}", "検索結果での総表示回数"),
            ("クリック数", f"{r['total_clicks']:,}", "総クリック数"),
            ("平均CTR", f"{r['avg_ctr']:.2f}%", "クリック率の平均"),
            ("平均順位", f"{r['avg_pos']:.1f}", "掲載順位の平均"),
        ]
        for col, (label, value, tip) in zip(cols, metrics):
            col.metric(label, value, help=tip)

        st.divider()

        # ワードクラウド（最初に表示）
        st.subheader("ワードクラウド")
        st.markdown(f'<p class="desc-text">{DESC["cloud"]}</p>', unsafe_allow_html=True)
        wc_fig, wc_img_bytes = create_wordcloud(dict(r['word_freq']), r['word_position'])
        if wc_fig:
            st.pyplot(wc_fig)
            if wc_img_bytes:
                st.download_button(
                    "画像をダウンロード",
                    wc_img_bytes,
                    f"wordcloud_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=False
                )

        st.divider()

        # CTR × 掲載順位
        st.subheader("CTR × 掲載順位")
        st.markdown(f'<p class="desc-text">{DESC["scatter"]}</p>', unsafe_allow_html=True)
        display_options = [50, 100, 200, 500]
        idx = display_options.index(st.session_state.display_count) if st.session_state.display_count in display_options else 1
        display_count = st.selectbox("表示件数", display_options, index=idx, key="scatter_limit")
        st.session_state.display_count = display_count
        fig = create_scatter(r, limit=display_count)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # 単語パフォーマンス
        st.subheader("単語パフォーマンス")
        st.markdown(f'<p class="desc-text">{DESC["word"]}</p>', unsafe_allow_html=True)
        fig = create_word_chart(r)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # カテゴリ分析（新規追加）
        st.subheader("単語カテゴリ分析")
        st.markdown('<p class="desc-text">検索キーワードに含まれる単語を意味カテゴリ別に分類。ユーザーの検索意図を把握できます。</p>', unsafe_allow_html=True)

        cat_col1, cat_col2 = st.columns([3, 2])

        with cat_col1:
            cat_fig = create_category_chart(r)
            if cat_fig:
                st.plotly_chart(cat_fig, use_container_width=True)

        with cat_col2:
            # カテゴリ選択で詳細表示
            cat_freq = r.get('category_freq', {})
            if cat_freq:
                sorted_cats = sorted(cat_freq.keys(), key=lambda x: cat_freq[x], reverse=True)
                selected_cat = st.selectbox("カテゴリを選択して詳細を表示", sorted_cats, key="cat_select")
                if selected_cat:
                    cat_detail = create_category_detail_table(r, selected_cat)
                    if cat_detail is not None and not cat_detail.empty:
                        st.dataframe(cat_detail, use_container_width=True, hide_index=True)

        st.divider()

        # 共起ネットワーク
        st.subheader("共起ネットワーク")
        st.markdown(f'<p class="desc-text">{DESC["network"]}</p>', unsafe_allow_html=True)
        fig = create_network(r)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ランキング
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("共起ランキング")
            st.markdown(f'<p class="desc-text">{DESC["cooc"]}</p>', unsafe_allow_html=True)
            cooc_df = pd.DataFrame([{'単語ペア': f"{w1} + {w2}", '回数': f"{c:,}"} for (w1, w2), c in r['cooccurrence'].most_common(10)])
            if not cooc_df.empty:
                st.dataframe(cooc_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("効率スコアランキング")
            st.markdown(f'<p class="desc-text">{DESC["score"]}</p>', unsafe_allow_html=True)
            score_df = r['df'].nlargest(10, 'score')[['query', 'ctr', 'position', 'score']].copy()
            score_df.columns = ['クエリ', 'CTR', '順位', 'スコア']
            score_df['CTR'] = score_df['CTR'].apply(lambda x: f"{x:.1f}%")
            score_df['順位'] = score_df['順位'].apply(lambda x: f"{x:.1f}")
            st.dataframe(score_df, use_container_width=True, hide_index=True)

        st.divider()

        # エクスポート
        st.subheader("エクスポート")
        cols = st.columns(4)
        cols[0].download_button("HTMLレポート", generate_html_report(r), f"report_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html", use_container_width=True)
        cols[1].download_button("全データ CSV", r['df'].to_csv(index=False).encode('utf-8-sig'), "keyword_data.csv", use_container_width=True)
        word_data = [{'単語': w, '出現': f} for w, f in r['word_freq'].most_common()]
        cols[2].download_button("単語 CSV", pd.DataFrame(word_data).to_csv(index=False).encode('utf-8-sig'), "word_data.csv", use_container_width=True)
        cooc_data = [{'単語1': w1, '単語2': w2, '共起': c} for (w1, w2), c in r['cooccurrence'].most_common()]
        cols[3].download_button("共起 CSV", pd.DataFrame(cooc_data).to_csv(index=False).encode('utf-8-sig'), "cooccurrence_data.csv", use_container_width=True)

if __name__ == "__main__":
    main()
