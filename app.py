#@title オルカン／TOPIX推移と近似線（強化版）
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from matplotlib import font_manager
import urllib.request
import os

# 日本語フォント設定
font_path = "/tmp/NotoSansJP.ttf"
if not os.path.exists(font_path):
    urllib.request.urlretrieve(
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
        font_path
    )
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

st.set_page_config(page_title="市場指標チャート（強化版）", layout="wide")
st.title("📈 日米主要指標・回帰分析（強化版）")

TICKER_MAP = {
    "全世界株式 (オルカン)": "2559.T",
    "TOPIX (国内株式)":    "1306.T",
    "ドル円 (為替)":       "JPY=X"
}

st.sidebar.header("🎛️ 操作パネル")
銘柄選択 = st.sidebar.selectbox("銘柄選択", list(TICKER_MAP.keys()))
表示期間_日 = st.sidebar.slider("表示期間（日）", min_value=30, max_value=1000, value=180, step=10)
実行 = st.sidebar.button("📊 チャートを表示", type="primary")

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def run_analysis(label, days):
    ticker_symbol = TICKER_MAP[label]
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days + 150)

    with st.spinner("データ取得中..."):
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

    if data.empty:
        st.error("データの取得に失敗しました。")
        return

    df_ser = data['Close'].copy()
    if isinstance(df_ser, pd.DataFrame):
        df_ser = df_ser.iloc[:, 0]
    df_ser = df_ser.dropna()
    df_ser = df_ser[df_ser > 0]

    if len(df_ser) > 20:
        ma20      = df_ser.rolling(window=20, min_periods=5).mean()
        threshold = ma20.shift(1) * 0.5
        df_ser    = df_ser[df_ser >= threshold]
        df_ser    = df_ser.dropna()

    df_all  = df_ser.copy()
    df_plot = df_ser.tail(days).copy()

    x = np.arange(len(df_plot))
    y = df_plot.values.flatten()

    slope, intercept = np.polyfit(x, y, 1)
    regression_line  = slope * x + intercept
    residuals        = y - regression_line
    sigma            = residuals.std()

    current_val    = float(y[-1])
    theory_val     = float(regression_line[-1])
    diff_pct       = (current_val - theory_val) / theory_val * 100
    all_deviations = (y - regression_line) / regression_line * 100
    percentile     = float(np.sum(all_deviations < diff_pct) / len(all_deviations) * 100)

    ma25 = df_all.rolling(25).mean().reindex(df_plot.index)
    ma75 = df_all.rolling(75).mean().reindex(df_plot.index)
    rsi  = calc_rsi(df_all, 14).reindex(df_plot.index)

    sigma_level = (current_val - theory_val) / sigma if sigma > 0 else 0
    rsi_now     = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50

    if   sigma_level >  2 and rsi_now > 65:
        judgment, j_color = "⚠️ 強い買われ過ぎ", "red"
    elif sigma_level >  1 and rsi_now > 60:
        judgment, j_color = "🔴 買われ過ぎ気味",  "red"
    elif sigma_level < -2 and rsi_now < 35:
        judgment, j_color = "💡 強い売られ過ぎ", "blue"
    elif sigma_level < -1 and rsi_now < 40:
        judgment, j_color = "🔵 売られ過ぎ気味",  "blue"
    else:
        judgment, j_color = "⬜ 中立圏",          "gray"

    st.subheader(f"【{label}】{days}日間・回帰分析")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("現在値",  f"{current_val:,.1f}円")
    col2.metric("回帰値",  f"{theory_val:,.1f}円")
    col3.metric("乖離率",  f"{diff_pct:+.2f}%")
    col4.metric("σ水準",   f"{sigma_level:+.2f}σ")
    col5.metric("RSI(14)", f"{rsi_now:.1f}")

    st.markdown(
        f"**パーセンタイル:** {percentile:.0f}%　｜　"
        f"**データ日:** {df_plot.index[-1].strftime('%m/%d')} ※前営業日終値　｜　"
        f"**判定:** :{j_color}[{judgment}]"
    )

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_plot.index, y,               label='実勢',   color='tab:blue', linewidth=2)
    ax1.plot(df_plot.index, regression_line, label='回帰線', color='tab:red',  linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.fill_between(df_plot.index, regression_line - sigma,   regression_line + sigma,   alpha=0.12, color='orange', label='±1σ')
    ax1.fill_between(df_plot.index, regression_line - 2*sigma, regression_line + 2*sigma, alpha=0.07, color='orange', label='±2σ')
    if ma25.notna().sum() > 0:
        ax1.plot(df_plot.index, ma25, label='MA25', color='green',  linewidth=1.2, alpha=0.8)
    if ma75.notna().sum() > 0:
        ax1.plot(df_plot.index, ma75, label='MA75', color='purple', linewidth=1.2, alpha=0.8)
    ax1.set_title(f"【{label}】{days}日間・回帰分析（強化版）", fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    bar_colors = ['crimson' if v > 0 else 'steelblue' for v in all_deviations]
    ax2.bar(df_plot.index, all_deviations, color=bar_colors, alpha=0.6, width=1)
    ax2.axhline(0, color='black', linewidth=0.8)
    for lvl, ls in [(sigma/theory_val*100, ':'), (2*sigma/theory_val*100, '--')]:
        ax2.axhline( lvl, color='orange', linewidth=0.9, linestyle=ls, alpha=0.8)
        ax2.axhline(-lvl, color='orange', linewidth=0.9, linestyle=ls, alpha=0.8)
    ax2.set_ylabel('乖離率(%)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_plot.index, rsi, color='purple', linewidth=1.5)
    ax3.axhline(70, color='red',  linewidth=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(30, color='blue', linewidth=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(50, color='gray', linewidth=0.6, linestyle=':',  alpha=0.5)
    ax3.fill_between(df_plot.index, rsi, 70, where=(rsi >= 70), alpha=0.2, color='red')
    ax3.fill_between(df_plot.index, rsi, 30, where=(rsi <= 30), alpha=0.2, color='blue')
    ax3.set_ylim(0, 100)
    ax3.set_yticks([30, 50, 70])
    ax3.set_ylabel('RSI(14)', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

if 実行:
    run_analysis(銘柄選択, 表示期間_日)
else:
    st.info("👈 左のパネルで銘柄・期間を選んで「チャートを表示」を押してください。")
