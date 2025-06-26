import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


def show_supply_demand_analysis():
    st.subheader("U.S. Oil Supply & Demand Analysis")
    st.markdown(
        "This section highlights key supply-demand fundamentals including U.S. crude production, inventories, and other key drivers."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Production", "Refinery Inputs", "Import/Export", "Forecasting"]
    )

    with tab1:
        show_production_section()
    with tab2:
        show_refinery_inputs_section()
    with tab3:
        st.warning("Imports/Exports section is under construction.")
    with tab4:
        st.warning("Forecasting module is under construction.")


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
def show_refinery_inputs_section():
    st.subheader("🏭 U.S. Refinery Inputs")
    st.markdown(
        "Refinery inputs represent the volume of crude oil processed by refineries. "
        "They serve as a direct proxy for domestic crude oil demand and are closely tied to transportation and industrial activity."
    )
    st.info("Refinery input data will be integrated here using EIA API.")
