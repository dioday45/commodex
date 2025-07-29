import streamlit as st
import numpy as np
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet


def show_supply_demand_analysis():
    st.subheader("U.S. Oil Supply & Demand Analysis")
    st.markdown(
        "This section provides a comprehensive overview of U.S. oil supply and demand fundamentals, including production, refinery activity, and end-user consumption trends. "
        "Data is sourced from the U.S. Energy Information Administration (EIA), a reliable and authoritative provider of energy statistics. "
        "Understanding these supply-demand dynamics is critical for analyzing oil market balance, price trends, and macroeconomic implications."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "U.S. Crude Production",
            "U.S. Crude Consumption",
            "Imports/Exports",
            "Forecasting",
        ]
    )

    with tab1:
        show_production_section()
    with tab2:
        show_consumption_section()
    with tab3:
        show_import_export_section()
    with tab4:
        show_forecasting_section()


#### Production
@st.cache_data
def fetch_production_data():
    url = "https://api.eia.gov/v2/petroleum/sum/snd/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "monthly",
        "data[0]": "value",
        "facets[duoarea][]": ["R10", "R20", "R30", "R40", "R50"],
        "facets[process][]": ["FPF"],
        "facets[product][]": ["EPC0"],
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data["response"]["data"])
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Pivot PADD-level production
    df_top_pivot = df.pivot_table(
        index="Date", columns="duoarea", values="value", aggfunc="sum"
    )
    df_top_pivot = df_top_pivot.rename(
        columns={
            "R10": "PADD 1",
            "R20": "PADD 2",
            "R30": "PADD 3",
            "R40": "PADD 4",
            "R50": "PADD 5",
        }
    )

    # Compute U.S. total from sum of PADDs
    df_top_pivot["US Total"] = df_top_pivot[
        ["PADD 1", "PADD 2", "PADD 3", "PADD 4", "PADD 5"]
    ].sum(axis=1)
    return df_top_pivot


def show_production_section():
    st.subheader("U.S. Field Crude Oil Production by PADD (Monthly)")
    st.markdown(
        "This chart shows monthly crude oil production across the five main Petroleum Administration for Defense Districts (PADDs). "
        "It highlights how production is geographically distributed and how regional trends contribute to national supply. "
        "The U.S. total is calculated as the sum of all five PADDs."
    )
    df_top_pivot = fetch_production_data()

    # Plot total and stacked PADDs
    fig = go.Figure()
    for col in ["PADD 1", "PADD 2", "PADD 3", "PADD 4", "PADD 5"]:
        fig.add_trace(
            go.Scatter(
                x=df_top_pivot.index,
                y=df_top_pivot[col],
                mode="lines",
                stackgroup="one",
                name=col,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_top_pivot.index,
            y=df_top_pivot["US Total"],
            mode="lines",
            name="US Total",
            line=dict(width=2, color="black"),
        )
    )

    fig.update_layout(
        title="U.S. Crude Oil Production by PADD",
        xaxis_title="Date",
        yaxis_title="MMBL/D",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Metrics")
    st.markdown(
        "These metrics provide a snapshot of the latest U.S. crude oil production level, along with short-term and long-term trend indicators.\n\n"
    )
    latest_value = df_top_pivot["US Total"].iloc[-1]
    prev_month = df_top_pivot["US Total"].iloc[-2]
    last_year = df_top_pivot["US Total"].iloc[-13] if len(df_top_pivot) > 12 else None

    # Compute additional metrics
    df_top_pivot["Growth YoY %"] = df_top_pivot["US Total"].pct_change(12) * 100
    df_top_pivot["Volatility 3M"] = df_top_pivot["US Total"].rolling(3).std()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "YoY Growth",
        f"{df_top_pivot['Growth YoY %'].iloc[-1]:.2f}%",
        f"{df_top_pivot['Growth YoY %'].iloc[-1] - df_top_pivot['Growth YoY %'].iloc[-2]:+.2f}% MoM",
    )
    col2.metric(
        "Production Volatility (3M STD)",
        f"{df_top_pivot['Volatility 3M'].iloc[-1]:.0f} MMBL/D",
    )
    col3.metric(
        "Latest U.S. Production",
        f"{latest_value:,.0f} MMBL/D",
        f"{latest_value - prev_month:+.0f} MoM",
    )

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **YoY Growth** shows the annualized growth rate in national crude production.\n"
            "- **Production Volatility (3M STD)** highlights how stable or volatile production levels have been in recent months.\n"
            "- **Latest U.S. Production** gives the most recent national value and its month-over-month change."
        )


@st.cache_data
def fetch_consumption_data():
    url = "https://api.eia.gov/v2/petroleum/sum/snd/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "monthly",
        "data[0]": "value",
        "facets[duoarea][]": ["R10", "R20", "R30", "R40", "R50"],
        "facets[product][]": ["EPC0"],
        "facets[process][]": ["YIR"],
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data["response"]["data"])
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df_pivot = df.pivot_table(
        index="Date", columns="duoarea", values="value", aggfunc="sum"
    )
    df_pivot = df_pivot.rename(
        columns={
            "R10": "PADD 1",
            "R20": "PADD 2",
            "R30": "PADD 3",
            "R40": "PADD 4",
            "R50": "PADD 5",
        }
    )
    df_pivot["US Total"] = df_pivot[
        ["PADD 1", "PADD 2", "PADD 3", "PADD 4", "PADD 5"]
    ].sum(axis=1)
    return df_pivot


def show_consumption_section():
    st.subheader("U.S. Crude Oil Consumption by PADD (Monthly)")
    st.markdown(
        "This chart shows monthly crude oil refinery input (proxy for consumption) across the five main PADDs."
    )

    df_pivot = fetch_consumption_data()

    fig = go.Figure()
    for col in ["PADD 1", "PADD 2", "PADD 3", "PADD 4", "PADD 5"]:
        fig.add_trace(
            go.Scatter(
                x=df_pivot.index,
                y=df_pivot[col],
                mode="lines",
                stackgroup="one",
                name=col,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_pivot.index,
            y=df_pivot["US Total"],
            mode="lines",
            name="US Total",
            line=dict(width=2, color="black"),
        )
    )

    fig.update_layout(
        title="U.S. Crude Oil Consumption by PADD",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    latest_value = df_pivot["US Total"].iloc[-1]
    prev_month = df_pivot["US Total"].iloc[-2]
    df_pivot["Growth YoY %"] = df_pivot["US Total"].pct_change(12) * 100
    df_pivot["Volatility 3M"] = df_pivot["US Total"].rolling(3).std()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "YoY Growth",
        f"{df_pivot['Growth YoY %'].iloc[-1]:.2f}%",
        f"{df_pivot['Growth YoY %'].iloc[-1] - df_pivot['Growth YoY %'].iloc[-2]:+.2f}% MoM",
    )
    col2.metric(
        "Consumption Volatility (3M STD)",
        f"{df_pivot['Volatility 3M'].iloc[-1]:.0f} MBBL/D",
    )
    col3.metric(
        "Latest U.S. Consumption",
        f"{latest_value:,.0f} MBBL/D",
        f"{latest_value - prev_month:+.0f} MoM",
    )


@st.cache_data()
def fetch_import_export_data():
    url = "https://api.eia.gov/v2/petroleum/move/wkly/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "weekly",
        "data[0]": "value",
        "facets[duoarea][]": ["NUS-Z00"],
        "facets[product][]": ["EPC0"],
        "facets[process][]": ["EEX", "IMX"],
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data["response"]["data"])


def show_import_export_section():
    st.subheader("U.S. Crude Oil Imports and Exports")
    st.markdown(
        "This section visualizes weekly U.S. **crude oil** imports and exports. "
        "It helps track cross-border flows and assess the U.S. net trade position in crude oil. "
        "All data is sourced from the U.S. EIA and reported in Million Barrels per Day (MBBL/D)."
    )

    df = fetch_import_export_data()

    # Clean and prepare
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["Date", "process", "value"]]
    df_imports = df[df["process"] == "IMX"]
    df_exports = df[df["process"] == "EEX"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_imports["Date"],
            y=df_imports["value"],
            name="Imports",
            line=dict(color="royalblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_exports["Date"],
            y=df_exports["value"],
            name="Exports",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="U.S. Crude Oil Imports, Exports, and Net Position (Weekly, MBBL/D)",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def show_forecasting_section():
    st.subheader("U.S. Oil Supply & Demand Forecasting Module")
    st.markdown(
        "This module provides a 12-month forecast of U.S. oil supply and demand, using historical data to project future trends. "
        "It includes confidence intervals to assess forecast uncertainty."
    )

    production = fetch_production_data()["US Total"]
    consumption = fetch_consumption_data()["US Total"]

    # Prepare data for Prophet
    prod_df = production.reset_index()
    prod_df.columns = ["ds", "y"]

    cons_df = consumption.reset_index()
    cons_df.columns = ["ds", "y"]

    # Fit Prophet models
    prod_model = Prophet()
    prod_model.fit(prod_df)

    cons_model = Prophet()
    cons_model.fit(cons_df)

    # Make future dataframe (3 months)
    future_prod = prod_model.make_future_dataframe(periods=12, freq="MS")
    forecast_prod = prod_model.predict(future_prod)

    future_cons = cons_model.make_future_dataframe(periods=12, freq="MS")
    forecast_cons = cons_model.predict(future_cons)

    # Plot using Plotly
    fig = go.Figure()

    # Production actual and forecast
    fig.add_trace(
        go.Scatter(
            x=prod_df["ds"], y=prod_df["y"], name="Production Actual", mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_prod["ds"],
            y=forecast_prod["yhat"],
            name="Production Forecast",
            mode="lines",
            line=dict(dash="dash"),
        )
    )

    # Consumption actual and forecast
    fig.add_trace(
        go.Scatter(
            x=cons_df["ds"], y=cons_df["y"], name="Consumption Actual", mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_cons["ds"],
            y=forecast_cons["yhat"],
            name="Consumption Forecast",
            mode="lines",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="3-Month Forecast of U.S. Oil Production and Consumption",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast Model Performance Metrics ---
    st.subheader("Forecast Model Performance Metrics")

    # Compute metrics for last 12 months of historical data (in-sample)
    def compute_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, rmse, mape

    # For production
    n_hist = len(prod_df)
    y_true_prod = prod_df["y"].values[-12:]
    y_pred_prod = forecast_prod["yhat"].values[:n_hist][-12:]
    prod_mae, prod_rmse, prod_mape = compute_metrics(y_true_prod, y_pred_prod)

    # For consumption
    n_hist_cons = len(cons_df)
    y_true_cons = cons_df["y"].values[-12:]
    y_pred_cons = forecast_cons["yhat"].values[:n_hist_cons][-12:]
    cons_mae, cons_rmse, cons_mape = compute_metrics(y_true_cons, y_pred_cons)

    st.markdown(
        "The table below summarizes the in-sample forecast errors for production and consumption models."
    )

    # Build DataFrame for metrics
    metrics_data = {
        "Production": [prod_mae, prod_rmse, prod_mape],
        "Consumption": [cons_mae, cons_rmse, cons_mape],
    }
    metrics_index = ["MAE (MBBL/D)", "RMSE (MBBL/D)", "MAPE (%)"]
    metrics_df = pd.DataFrame(metrics_data, index=metrics_index)

    # Format: two decimals, percent for MAPE
    def format_metric(val, is_percent=False):
        if is_percent:
            return f"{val:.2f}%"
        else:
            return f"{val:.2f}"

    metrics_df_display = metrics_df.copy()
    for idx in metrics_df_display.index:
        if "MAPE" in idx:
            metrics_df_display.loc[idx] = metrics_df_display.loc[idx].apply(
                lambda v: format_metric(v, is_percent=True)
            )
        else:
            metrics_df_display.loc[idx] = metrics_df_display.loc[idx].apply(
                lambda v: format_metric(v, is_percent=False)
            )

    st.dataframe(metrics_df_display, use_container_width=True)
