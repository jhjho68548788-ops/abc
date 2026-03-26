"""
포스코 홀딩스 주가 데이터 대시보드 (Streamlit)

- 데이터 소스: yfinance (Yahoo Finance)
- 통화: 종목 데이터의 currency(없으면 KRW 가정)
- 스케일: 가격/거래량을 사용자가 보기 편한 단위로 선택
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="포스코 홀딩스 주가 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRICE_UNITS = {
    "원": 1.0,
    "천원": 1_000.0,
    "만원": 10_000.0,
    "십만원": 100_000.0,
}

VOLUME_UNITS = {
    "주": 1.0,
    "천 주": 1_000.0,
    "백만 주": 1_000_000.0,
    "십억 주": 1_000_000_000.0,
}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_history(
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=auto_adjust, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        # 스트림릿/플롯에 tz-aware DateTime이 섞이면 축이 지저분해질 수 있어 제거
        df.index = df.index.tz_localize(None)
    df = df.reset_index()
    # yfinance는 컬럼명이 보통 (Open, High, Low, Close, Volume) 형태
    df = df.rename(columns={"index": "Date", "Datetime": "Date"})
    if "Date" not in df.columns:
        # 혹시라도 인덱스 이름이 다를 때 대비
        df = df.rename(columns={df.columns[0]: "Date"})
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_currency(ticker: str) -> str:
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info or {}
        currency = info.get("currency")
        return str(currency) if currency else "KRW"
    except Exception:
        return "KRW"


def _annualization_factor(interval: str) -> float:
    # 간격이 1d/1wk/1mo일 때만 정확도 기대치가 높습니다.
    # intraday(예: 60m)일 경우엔 여기 값을 보수적으로 252로 둡니다.
    if interval.endswith("mo"):
        return 12.0
    if interval.endswith("wk"):
        return 52.0
    if interval.endswith("d"):
        return 252.0
    return 252.0


def _format_scaled_price(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{decimals}f}"


def main() -> None:
    # 모든 텍스트를 볼드체로 보이도록(가독성 향상) 앱 전역 스타일을 적용합니다.
    st.markdown(
        """
        <style>
        [data-testid="stApp"] * {
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📈 포스코 홀딩스 주가 데이터 대시보드")
    st.caption("yfinance에서 불러온 OHLC/거래량 데이터를 통화와 스케일까지 고려해 시각화합니다.")

    with st.sidebar:
        st.header("설정")

        ticker = st.text_input("티커 (Yahoo Finance)", value="005490.KS").strip()
        period = st.selectbox(
            "조회 기간",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=4,
            help="기간/간격 조합에 따라 yfinance가 제공하는 데이터가 달라질 수 있습니다.",
        )
        interval = st.selectbox(
            "데이터 간격",
            options=["1d", "1wk", "1mo", "60m", "30m", "15m", "5m"],
            index=0,
        )
        auto_adjust = st.checkbox("자동 조정 가격 사용", value=True)

        chart_type = st.radio(
            "차트 타입",
            options=["캔들스틱", "종가 선"],
            horizontal=True,
            index=0,
        )

        col_ma = st.columns(2)
        with col_ma[0]:
            show_ma = st.checkbox("이동평균선", value=True)
        with col_ma[1]:
            ma_fast = st.selectbox("단기 MA", options=[5, 10, 20], index=0, disabled=not show_ma)

        ma_slow = None
        if show_ma:
            ma_slow = st.selectbox("장기 MA", options=[20, 60, 120], index=0)

        price_unit_label = st.selectbox("가격 표시 단위", options=list(PRICE_UNITS.keys()), index=1)
        price_scale = float(PRICE_UNITS[price_unit_label])

        volume_unit_label = st.selectbox("거래량 표시 단위", options=list(VOLUME_UNITS.keys()), index=1)
        volume_scale = float(VOLUME_UNITS[volume_unit_label])

    if not ticker:
        st.warning("티커가 비어 있습니다.")
        return

    with st.spinner("데이터를 불러오는 중입니다..."):
        currency = load_currency(ticker)
        try:
            df = load_history(ticker=ticker, period=period, interval=interval, auto_adjust=auto_adjust)
        except Exception as e:
            st.error("데이터 로딩에 실패했습니다.")
            st.code(str(e))
            return

    if df.empty:
        st.warning("해당 기간/간격으로 데이터를 가져오지 못했습니다. (티커/기간/간격 확인)")
        return

    # 기본 숫자 컬럼 정리
    needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼이 없습니다: {missing}")
        return

    # 스케일된 표시용 컬럼
    for col in ["Open", "High", "Low", "Close"]:
        df[col + "_S"] = df[col] / price_scale
    df["Volume_S"] = df["Volume"] / volume_scale

    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()

    latest = float(df["Close"].iloc[-1])
    first = float(df["Close"].iloc[0])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else float("nan")

    period_return = (latest / first - 1.0) if first else float("nan")
    daily_return_pct = ((latest - prev) / prev * 100.0) if prev and not math.isnan(prev) else float("nan")
    vol_rolling = float(df["Volume"].tail(20).mean()) if len(df) >= 2 else float("nan")

    # 연환산 변동성(추정): 표준편차 * sqrt(연간 관측치)
    rets = df["Return"].dropna()
    ann_factor = _annualization_factor(interval)
    ann_vol = float(rets.std(ddof=1) * math.sqrt(ann_factor)) if len(rets) >= 2 else float("nan")

    # MA
    if show_ma:
        if ma_slow is None:
            ma_slow = ma_fast * 4
        if ma_slow <= ma_fast:
            ma_slow = ma_fast + 1
        df["MA_fast"] = df["Close"].rolling(int(ma_fast)).mean() / price_scale
        df["MA_slow"] = df["Close"].rolling(int(ma_slow)).mean() / price_scale

    latest_scaled = latest / price_scale
    high_scaled = float(df["High"].max()) / price_scale
    low_scaled = float(df["Low"].min()) / price_scale

    st.subheader("핵심 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최신 종가", f"{_format_scaled_price(latest_scaled, 2)} {price_unit_label}")
    c2.metric("기간 누적 수익률", f"{period_return:.1%}" if not math.isnan(period_return) else "-")
    c3.metric("전일(또는 직전 구간) 변동률", f"{daily_return_pct:.2f}%" if not math.isnan(daily_return_pct) else "-")
    c4.metric("연환산 변동성(추정)", f"{ann_vol:.1%}" if not math.isnan(ann_vol) else "-")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("구간 고가", f"{_format_scaled_price(high_scaled, 2)} {price_unit_label}")
    c6.metric("구간 저가", f"{_format_scaled_price(low_scaled, 2)} {price_unit_label}")
    c7.metric("최근 20개 평균 거래량", f"{_format_scaled_price(vol_rolling / volume_scale, 2)} {volume_unit_label}")
    c8.metric("통화", currency)

    st.divider()

    # 차트
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    if chart_type == "캔들스틱":
        hover = (
            "날짜: %{x}<br>"
            f"시가: %{{open:,.2f}} {price_unit_label} ({currency})<br>"
            f"고가: %{{high:,.2f}} {price_unit_label} ({currency})<br>"
            f"저가: %{{low:,.2f}} {price_unit_label} ({currency})<br>"
            f"종가: %{{close:,.2f}} {price_unit_label} ({currency})"
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open_S"],
                high=df["High_S"],
                low=df["Low_S"],
                close=df["Close_S"],
                name="OHLC",
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
    else:
        hover = (
            "날짜: %{x}<br>"
            f"종가: %{{y:,.2f}} {price_unit_label} ({currency})"
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close_S"],
                mode="lines",
                name="종가",
                line={"width": 2},
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )

    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA_fast"],
                mode="lines",
                name=f"MA {ma_fast}",
                line={"width": 1.6, "dash": "solid"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA_slow"],
                mode="lines",
                name=f"MA {ma_slow}",
                line={"width": 1.6, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

    # 거래량(아래)
    vol_hover = (
        "날짜: %{x}<br>"
        f"거래량: %{{y:,.2f}} {volume_unit_label}"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume_S"],
            name="거래량",
            marker={"color": "rgba(0,0,0,0.25)"},
            hovertemplate=vol_hover,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text=f"가격 ({price_unit_label})", row=1, tickformat=",.2f")
    fig.update_yaxes(title_text=f"거래량 ({volume_unit_label})", row=2, tickformat=",.2f")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=780,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("데이터 미리보기")
    dfd = pd.DataFrame(
        {
            "Date": df["Date"],
            f"시가 ({price_unit_label})": df["Open"] / price_scale,
            f"고가 ({price_unit_label})": df["High"] / price_scale,
            f"저가 ({price_unit_label})": df["Low"] / price_scale,
            f"종가 ({price_unit_label})": df["Close"] / price_scale,
            f"거래량 ({volume_unit_label})": df["Volume"] / volume_scale,
        }
    )

    # Date는 너무 길면 보기 어려우니 날짜 포맷 통일
    if pd.api.types.is_datetime64_any_dtype(dfd["Date"]):
        fmt = "%Y-%m-%d %H:%M" if interval not in ["1d", "1wk", "1mo"] else "%Y-%m-%d"
        dfd["Date"] = pd.to_datetime(dfd["Date"]).dt.strftime(fmt)

    st.dataframe(dfd.tail(60), use_container_width=True, hide_index=True, height=420)


if __name__ == "__main__":
    main()
