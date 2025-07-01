import streamlit as st
import numpy as np
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def show_supply_demand_analysis():
    st.subheader("U.S. Oil Supply & Demand Analysis")
    st.markdown(
        "This section provides a comprehensive overview of U.S. oil supply and demand fundamentals, including production, refinery activity, and end-user consumption trends. "
        "Data is sourced from the U.S. Energy Information Administration (EIA), a reliable and authoritative provider of energy statistics. "
        "Understanding these supply-demand dynamics is critical for analyzing oil market balance, price trends, and macroeconomic implications."
    )

    tab1, tab2, tab3 = st.tabs(["Supply", "Demand", "Imports/Exports"])

    with tab1:
        show_production_section()
    with tab2:
        show_product_supplied_section()
    with tab3:
        show_import_export_section()


#### Production
def show_production_section():
    st.subheader("U.S. Crude Oil Production by PADD (Monthly)")
    st.markdown(
        "This chart shows monthly crude oil production across the five main Petroleum Administration for Defense Districts (PADDs). "
        "It highlights how production is geographically distributed and how regional trends contribute to national supply. "
        "The U.S. total is calculated as the sum of all five PADDs."
    )
    url = "https://api.eia.gov/v2/petroleum/crd/crpdn/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "monthly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data["response"]["data"])
    # Filter only crude oil field production
    df = df[(df["product"] == "EPC0") & (df["process"] == "FPF")]
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Separate state-level and PADD-level rows
    df_states = df[df["duoarea"].str.startswith("S")].copy()
    valid_padds = ["R10", "R20", "R30", "R40", "R50"]
    df_padds = df[df["duoarea"].isin(valid_padds)].copy()

    # Pivot PADD-level production
    df_top_pivot = df_padds.pivot_table(
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

    st.subheader("Top 5 Producing States (Monthly)")
    st.markdown(
        "This chart presents the top five U.S. oil-producing states based on cumulative output over the selected time range. "
        "It allows you to visualize which states are the primary contributors to U.S. crude production and how their output has evolved over time."
    )
    # Compute top 5 states by cumulative output
    top_states = (
        df_states.groupby("duoarea")["value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    df_top_states = df_states[df_states["duoarea"].isin(top_states)]
    df_states_pivot = df_top_states.pivot_table(
        index="Date", columns="duoarea", values="value", aggfunc="sum"
    )
    # Map readable state names
    state_name_map = (
        df_top_states[["duoarea", "area-name"]]
        .drop_duplicates()
        .set_index("duoarea")["area-name"]
        .to_dict()
    )
    df_states_pivot = df_states_pivot.rename(columns=state_name_map)

    fig_states = go.Figure()
    for col in df_states_pivot.columns:
        fig_states.add_trace(
            go.Scatter(
                x=df_states_pivot.index,
                y=df_states_pivot[col],
                mode="lines",
                stackgroup="one",
                name=col,
            )
        )
    fig_states.update_layout(
        title="Top 5 Producing States",
        xaxis_title="Date",
        yaxis_title="MMBL/D",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_states, use_container_width=True)

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


# --- Placeholder for refinery inputs section ---
def show_product_supplied_section():
    st.subheader("U.S. Product Supplied")
    st.markdown(
        "Product supplied represents the volume of petroleum products delivered to the U.S. domestic market, and is commonly used as a proxy for end-user demand. "
        "This section helps track consumption patterns and highlight trends across key petroleum products such as gasoline, distillate, and jet fuel. "
        "All figures are expressed in **Million Barrels per Day (MBBL/D)**."
    )

    url = "https://api.eia.gov/v2/petroleum/cons/wpsup/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["response"]["data"])
    # Keep relevant columns
    df = df[["period", "product-name", "value"]]

    # Convert date and value
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Rename columns for clarity
    df = df.rename(columns={"product-name": "Product", "value": "Volume"})

    # Divide Volume by 1000 to convert to MBBL/D

    # Sort by date
    df = df.sort_values("Date")

    # Define a consistent color map by product
    product_list = df["Product"].unique()
    color_sequence = px.colors.qualitative.Plotly
    color_map = {
        prod: color_sequence[i % len(color_sequence)]
        for i, prod in enumerate(sorted(product_list))
    }

    st.markdown(
        "#### Product Supplied by Product Type\n"
        "This line chart shows the weekly volume of petroleum products supplied to the U.S. market, broken down by product type. "
        "It helps identify short-term fluctuations, seasonal patterns, and major demand disruptions such as the COVID-19 pandemic or extreme weather events."
    )

    # Line plot of product supplied with product as hue
    fig = go.Figure()
    for product in df["Product"].unique():
        df_product = df[df["Product"] == product]
        fig.add_trace(
            go.Scatter(
                x=df_product["Date"],
                y=df_product["Volume"],
                mode="lines",
                name=product,
                line=dict(color=color_map[product]),
            )
        )

    fig.update_layout(
        title="U.S. Weekly Product Supplied by Product Type (MBBL/D)",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pivot the data to have products as columns
    df_wide = df.pivot(index="Date", columns="Product", values="Volume")
    # Drop rows with missing data
    df_wide = df_wide.dropna()

    col1, col2 = st.columns(2)

    col1.markdown(
        "##### Weekly Demand Composition\n"
        "This stacked area chart shows how each product contributes to total U.S. petroleum demand over time. "
        "A stable or rising share suggests growing reliance on that product. "
        "Notice how gasoline demand dominates during summer months, while distillates are more evenly distributed."
    )

    # 1. Stacked Area Chart (Demand Composition)
    df_pct = df_wide.div(df_wide.sum(axis=1), axis=0) * 100
    fig_stack = go.Figure()
    for col in df_pct.columns:
        fig_stack.add_trace(
            go.Scatter(
                x=df_pct.index,
                y=df_pct[col],
                mode="lines",
                stackgroup="one",
                name=col,
                line=dict(color=color_map[col]),
            )
        )
    fig_stack.update_layout(
        title="Share of Products in Total Weekly Demand",
        xaxis_title="Date",
        yaxis_title="Percentage",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", x=0.05, y=-0.25),
    )
    col1.plotly_chart(fig_stack, use_container_width=True)

    col2.markdown(
        "##### Correlation Across Product Demand\n"
        "This heatmap shows how weekly demand for various petroleum products correlates over time. "
        "Strong correlations may indicate shared economic drivers (e.g., gasoline and jet fuel rising together with mobility). "
        "Negative or low correlations can signal different seasonality or use cases."
    )

    # 4. Correlation Heatmap of Products
    corr = df_wide.corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            showscale=False,
        )
    )
    fig_corr.update_layout(
        title="Correlation Matrix of Weekly Product Volumes", height=600
    )
    col2.plotly_chart(fig_corr, use_container_width=True)

    # --- Simplified Weekly Demand Metrics ---
    st.markdown(
        "#### Year-over-Year Demand Change\n"
        "This section compares the current weekly product demand with the same week one year ago. "
        "It highlights structural demand changes and helps differentiate between seasonal noise and longer-term shifts. "
        "Products with strong positive YoY change may signal recovery or growing economic activity, while negative values may reflect efficiency gains or fuel switching."
    )

    df_yoy = df.copy()
    df_yoy["YoY Change (%)"] = df_yoy.groupby("Product")["Volume"].transform(
        lambda x: x.pct_change(52) * 100
    )
    latest_date = df_yoy["Date"].max()
    df_latest = df_yoy[df_yoy["Date"] == latest_date]

    num_cols = 6
    products = df_latest["Product"].unique()
    cols = st.columns(num_cols)
    for idx, product in enumerate(products):

        if pd.notnull(
            df_latest[df_latest["Product"] == product]["YoY Change (%)"].values[0]
        ):
            with cols[idx]:
                st.metric(
                    label=f"{product}",
                    value=f"{df_latest[df_latest['Product'] == product]['Volume'].values[0]:,.2f} MBBL/D",
                    delta=f"{df_latest[df_latest['Product'] == product]['YoY Change (%)'].values[0]:+.1f}%",
                    delta_color="normal",
                )


def show_import_export_section():
    st.subheader("U.S. Crude Oil Imports and Exports")
    st.markdown(
        "This section visualizes weekly U.S. **crude oil** imports and exports. "
        "It helps track cross-border flows and assess the U.S. net trade position in crude oil. "
        "All data is sourced from the U.S. EIA and reported in Million Barrels per Day (MBBL/D)."
    )

    url = "https://api.eia.gov/v2/petroleum/move/wkly/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "facets[product][]": "EP00",
        "facets[product][]": "EPC0",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["response"]["data"])

    # Clean and prepare
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(
        columns={"product-name": "Product", "process-name": "Flow", "value": "Volume"}
    )
    # Only keep national-level commercial crude oil imports and exports
    df = df[
        (df["duoarea"] == "NUS-Z00")
        & (df["product"] == "EPC0")
        & (df["process"].isin(["IMX", "EEX"]))
    ]

    # Pivot to Imports and Exports by product
    df_grouped = df.groupby(["Date", "Flow"])["Volume"].sum().unstack()

    # Plot net imports (Imports - Exports)
    df_grouped["Net Imports"] = (
        df_grouped["Imports Excluding SPR"] - df_grouped["Exports"]
    )
    df_grouped = df_grouped.sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_grouped.index,
            y=df_grouped["Imports Excluding SPR"],
            name="Imports",
            line=dict(color="royalblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_grouped.index,
            y=df_grouped["Exports"],
            name="Exports",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_grouped.index,
            y=df_grouped["Net Imports"],
            name="Net Imports",
            line=dict(color="green", dash="dot"),
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

    st.subheader("Summary Metrics")
    st.markdown(
        "These metrics summarize the most recent weekly trade flows for U.S. crude oil. "
        "Net imports reflect the difference between imports and exports, indicating the U.S. trade position in crude oil."
    )

    latest = df_grouped.iloc[-1]
    prev = df_grouped.iloc[-2]

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Imports", f"{latest['Imports Excluding SPR']:.0f} MBBL/D")
    col2.metric("Latest Exports", f"{latest['Exports']:.0f} MBBL/D")
    col3.metric(
        "Net Imports",
        f"{latest['Net Imports']:.0f} MBBL/D",
        delta=f"{latest['Net Imports'] - prev['Net Imports']:+.0f} WoW",
    )


def show_forecasting_section():

    st.subheader("U.S. Oil Supply & Demand Forecast")
    st.markdown(
        "This section compares historical and projected U.S. oil supply and demand, using weekly data from the EIA. "
        "Both production (supply) and product supplied (demand) are shown, with 3-month forecasts and confidence intervals."
    )

    # --- Fetch product supplied (demand) data ---
    url = "https://api.eia.gov/v2/petroleum/cons/wpsup/data/"
    params = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["response"]["data"])
    df["Date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"product-name": "Product", "value": "Volume"})
    df["Volume"] = df["Volume"] / 1000  # convert to MBBL/D
    # Aggregate total demand
    df_total = df.groupby("Date")["Volume"].sum().sort_index()
    df_total = df_total.asfreq("ME").interpolate()

    # --- Fetch production data ---
    url_prod = "https://api.eia.gov/v2/petroleum/crd/crpdn/data/"
    params_prod = {
        "api_key": st.secrets["EIA_KEY"],
        "frequency": "monthly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response_prod = requests.get(url_prod, params=params_prod)
    data_prod = response_prod.json()
    df_prod = pd.DataFrame(data_prod["response"]["data"])
    df_prod["Date"] = pd.to_datetime(df_prod["period"])
    df_prod["value"] = pd.to_numeric(df_prod["value"], errors="coerce")
    # Only keep U.S. PADDs
    df_prod = df_prod[df_prod["duoarea"].isin(["R10", "R20", "R30", "R40", "R50"])]
    df_prod = df_prod.groupby("Date")["value"].sum().sort_index()
    df_prod = df_prod.asfreq("W-FRI").interpolate()
    df_prod = df_prod / 1000  # to MMBL/D

    # --- Plot historical production vs product supplied ---
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(x=df_total.index, y=df_total, name="Product Supplied")
    )
    fig_hist.add_trace(go.Scatter(x=df_prod.index, y=df_prod, name="Production"))

    fig_hist.update_layout(
        title="U.S. Oil Supply vs Demand (Historical, MBBL/D)",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Forecast both series for 12 weeks ---
    forecast_horizon = 12
    fit_demand = ExponentialSmoothing(
        df_total, seasonal="add", seasonal_periods=52
    ).fit(optimized=True)
    forecast_demand = fit_demand.forecast(forecast_horizon)
    se_demand = np.std(fit_demand.resid)
    ci_upper_d = forecast_demand + 1.96 * se_demand
    ci_lower_d = forecast_demand - 1.96 * se_demand

    fit_prod = ExponentialSmoothing(df_prod, seasonal="add", seasonal_periods=52).fit(
        optimized=True
    )
    forecast_prod = fit_prod.forecast(forecast_horizon)
    se_prod = np.std(fit_prod.resid)
    ci_upper_p = forecast_prod + 1.96 * se_prod
    ci_lower_p = forecast_prod - 1.96 * se_prod

    # --- Plot forecast with confidence intervals ---
    fig_fc = go.Figure()

    fig_fc.add_trace(
        go.Scatter(
            x=forecast_demand.index,
            y=ci_upper_d,
            line=dict(width=0),
            name="Demand CI",
            showlegend=False,
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_demand.index,
            y=ci_lower_d,
            fill="tonexty",
            fillcolor="rgba(99,110,250,0.2)",
            line=dict(width=0),
            name="Demand Forecast CI",
        )
    )

    fig_fc.add_trace(
        go.Scatter(
            x=forecast_prod.index,
            y=ci_upper_p,
            line=dict(width=0),
            name="Production CI",
            showlegend=False,
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_prod.index,
            y=ci_lower_p,
            fill="tonexty",
            fillcolor="rgba(239,85,59,0.2)",
            line=dict(width=0),
            name="Production Forecast CI",
        )
    )

    # Add forecast lines (center) for both
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_demand.index,
            y=forecast_demand,
            name="Demand Forecast",
            line=dict(color="rgb(99,110,250)", dash="dash"),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_prod.index,
            y=forecast_prod,
            name="Production Forecast",
            line=dict(color="rgb(239,85,59)", dash="dash"),
        )
    )

    fig_fc.update_layout(
        title="3-Month Forecast: U.S. Oil Supply & Demand (Confidence Intervals Only)",
        xaxis_title="Date",
        yaxis_title="MBBL/D",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_fc, use_container_width=True)
