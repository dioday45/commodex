from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LinearRegression


def get_contract_symbols(
    oil_type: str, start_month: int = 0, count: int = 12
) -> pd.Series:
    month_codes = "FGHJKMNQUVXZ"
    today = datetime.today()
    symbols = []
    prefix_map = {
        "WTI": "CL",
        "Brent": "BZ",
        "Gasoline": "RB",
    }

    prefix = prefix_map.get(oil_type, "CL")

    for i in range(start_month, start_month + count):
        month = (today.month - 1 + i) % 12
        year = today.year + ((today.month - 1 + i) // 12)
        code = f"{prefix}{month_codes[month]}{str(year)[-2:]}.NYM"
        symbols.append(code)

    return pd.Series(symbols)


@st.cache_data
def fetch_forward_curve_from_contracts(symbols: pd.Series) -> pd.Series:
    prices = {}
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )
            if not df.empty:
                prices[sym] = df["Close"].iloc[-1]
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
    return pd.Series(prices)


def show_futures_analysis():
    st.subheader("🔁 Oil Futures Curve (Built from Real Contracts)")

    st.markdown(
        "This section displays the **futures price curve** for selected oil-related commodities. "
        "Each point represents the market price for a specific delivery month. "
        "The shape of the curve provides insight into market structure—such as contango or backwardation—"
        "which informs traders about storage costs, supply expectations, and future price pressures."
    )

    oil_type = st.selectbox("Choose benchmark:", ["WTI", "Brent", "Gasoline"])

    years = st.slider("Select how many years to display", 1, 3, 1)

    symbols = get_contract_symbols(oil_type, count=12 * years)
    curve = fetch_forward_curve_from_contracts(symbols).dropna()
    symbols = symbols[symbols.index.isin(curve.index)]
    curve = curve.astype(float)
    if curve.empty:
        st.warning("Could not fetch futures prices.")
        return
    if curve.isnull().any():
        st.warning("Some contracts returned missing prices and were dropped.")

    # Build labels like 'Jul 2024'
    today = datetime.today().replace(day=1)
    labels = [
        (today + relativedelta(months=i)).strftime("%b %Y") for i in range(len(curve))
    ]
    curve.index = labels

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve.values,
            mode="lines+markers",
            name=f"{oil_type} Futures",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve.values,
            mode="markers+text",
            text=[f"{v:.2f}" for v in curve.values],
            textposition="top center",
            name="Futures Price",
            hovertext=symbols.values,
            hoverinfo="text",
        )
    )

    x_numeric = np.arange(len(curve))
    y_values = curve.values
    b, m = polyfit(x_numeric, y_values, 1)
    trend = m * x_numeric + b

    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=trend,
            mode="lines",
            name="Trendline",
            line=dict(dash="dash", color="gray"),
        )
    )

    fig.update_layout(
        title=f"{oil_type} Futures Curve (Real Contracts)",
        xaxis_title="Delivery Month",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    spread = curve.iloc[-1] - curve.iloc[0]
    structure = "Contango" if spread > 0 else "Backwardation"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Back-to-Front Spread",
        f"{spread:.2f} USD",
        delta=f"{(spread / curve.iloc[0]) * 100:.2f}%",
    )
    col2.metric("Curve Std Dev", f"{curve.std():.2f} USD")
    model = LinearRegression().fit(x_numeric.reshape(-1, 1), y_values)
    r2 = model.score(x_numeric.reshape(-1, 1), y_values)
    col3.metric("Curve Linear Fit (R²)", f"{r2:.2f}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown(
            "- **Back-to-Front Spread**: Difference between last and first contract prices. "
            "Positive spread suggests contango; negative implies backwardation.\n"
            "- **Curve Std Dev**: Measures the variability of prices along the curve.\n"
            "- **Curve Linear Fit (R²)**: Indicates how closely the curve resembles a straight line. "
            "A higher R² implies a more uniform term structure."
        )

    st.info(f"Market structure: **{structure}**")

    st.subheader("📊 Calendar Spreads")
    st.markdown(
        "The **calendar spread** shows the price difference between each consecutive contract on the curve. "
        "It highlights changes in the curve's shape, and is often used to assess short-term market pressures "
        "and arbitrage opportunities (e.g. rolling futures positions)."
    )

    calendar_spreads = curve.diff().dropna()
    spread_labels = [
        f"{curve.index[i-1]} → {curve.index[i]}" for i in range(1, len(curve))
    ]

    fig_spread = go.Figure()
    fig_spread.add_trace(
        go.Scatter(
            x=spread_labels,
            y=calendar_spreads.values,
            mode="lines+markers+text",
            text=[f"{s:.2f}" for s in calendar_spreads.values],
            textposition="top center",
            line=dict(color="blue"),
            name="Calendar Spread",
        )
    )

    fig_spread.update_layout(
        title="Calendar Spreads Between Consecutive Contracts",
        xaxis_title="Contract Pair",
        yaxis_title="Price Difference (USD)",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(fig_spread, use_container_width=True)

    st.caption(
        "📘 **Disclaimer**: The data used in this dashboard is sourced from Yahoo Finance via the `yfinance` library. "
        "While convenient for educational and prototyping purposes, this data may be subject to delays, adjustments, "
        "or inaccuracies. It is not intended for real-time trading decisions or professional use without further validation."
    )
