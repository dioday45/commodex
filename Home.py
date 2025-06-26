import streamlit as st

st.set_page_config(
    page_title="Commodex – Commodity Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Commodex – Commodity Market Dashboard")

st.markdown(
    """
    ### Project Overview

    **Commodex** is a multi-asset commodity dashboard designed for analysts, traders, and researchers seeking actionable insights into key physical markets.
    The platform consolidates fundamental and market data from trusted public sources to support decision-making in trading and macroeconomic analysis.

    The dashboard is organized into three major sectors:
    """
)

tab1, tab2, tab3 = st.tabs(["U.S. Crude Oil", "LNG", "Agricultural Commodities"])

with tab1:
    st.markdown(
        """
        ### U.S. Crude Oil Supply & Demand

        This section focuses on U.S. domestic oil fundamentals, leveraging high-frequency EIA data to monitor:

        - Monthly **crude production** across PADDs and top oil-producing states
        - **Refinery demand** as a proxy for domestic crude consumption
        - **Import/export flows** to assess net trade positioning
        - **Forward curves** to analyze market structure (contango/backwardation)
        - A planned **forecasting module** to anticipate inventory shifts

        Ideal for macro-driven oil traders and fundamental analysts.
        """
    )

with tab2:
    st.markdown(
        """
        ### Global LNG Market (Coming Soon)

        This module will offer insights into global LNG flows, prices, and capacity utilization.

        - Monitor global liquefaction/export capacity
        - Track imports by region (Europe, Asia, U.S.)
        - Compare regional LNG spot prices and spreads
        - Assess seasonal trends and shipping dynamics

        Targeted at participants in the natural gas and global energy transition markets.
        """
    )

with tab3:
    st.markdown(
        """
        ### Agricultural Commodities (Coming Soon)

        The agriculture section will explore fundamentals across grains and oilseeds:

        - Global and regional **production data**
        - Trade balances and export trends
        - Weather-driven **supply shocks**
        - Seasonal demand patterns

        Designed to support grain traders, ag economists, and policy researchers.
        """
    )

st.markdown("---")
st.markdown(
    "*Disclaimer: The content and data presented on this dashboard are for educational and informational purposes only and should not be construed as financial advice or relied upon for trading decisions.*"
)
