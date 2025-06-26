# pages/1_Oil.py

import streamlit as st

st.set_page_config(page_title="Oil Dashboard", layout="wide")
st.title("🛢️ Oil Dashboard")

subpage = st.radio(
    "Navigate:",
    ["Futures Curve", "Supply & Demand Analysis", "Spot Price Analysis"],
    horizontal=True,
)

if subpage == "Spot Price Analysis":
    from src.oil.prices import show_price_analysis

    show_price_analysis()

elif subpage == "Futures Curve":
    from src.oil.futures import show_futures_analysis

    show_futures_analysis()

elif subpage == "Supply & Demand Analysis":
    from src.oil.supply_demand import show_supply_demand_analysis

    show_supply_demand_analysis()
