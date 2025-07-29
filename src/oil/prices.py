import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_data
def fetch_prices(
    symbol,
    period,
    interval="1d",
    progress=False,
    auto_adjust=True,
    multi_level_index=False,
):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=progress,
        auto_adjust=auto_adjust,
        multi_level_index=multi_level_index,
    )
    return df


def show_price_analysis():
    st.subheader("📉 Oil Price Analysis")

    ticker = st.selectbox("Choose crude type:", ["Brent", "WTI"])
    symbol = "BZ=F" if ticker == "Brent" else "CL=F"
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)

    df = fetch_prices(
        symbol=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )

    # Data validation: check for empty or missing data
    if df.empty or df.isna().all().all():
        st.warning(
            "No data retrieved. Try a different period or check your connection."
        )
        return

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in df or df["Volume"].isna().all():
        df["Volume"] = 0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(128, 128, 128, 0.5)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(text=f"{ticker} Crude Oil Price", font=dict(size=22), x=0.5),
        xaxis=dict(
            title="Date",
            showgrid=True,
            tickangle=0,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Price (USD)",
            showgrid=True,
        ),
        yaxis2=dict(
            title="Volume",
            showgrid=False,
        ),
        template="plotly_white",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
