import streamlit as st
import pandas as pd
import numpy as np
import os
from modules.rules import get_trend, interpret_rsi, interpret_delta, interpret_vega

# --- CONFIG ---
st.set_page_config(page_title="Market Intelligence Console", layout="wide")
st.title("Market Intelligence Dashboard")
st.markdown("*A tabular decision-support console for NIFTY options.*")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Read the verbatim file
    file_path = os.path.join("data", "dummy_option_data.xlsx")
    df = pd.read_excel(file_path)
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Please place 'dummy_option_data.xlsx' in the 'data' folder.")
    st.stop()

# --- PHASE 2: TIME-BASED UPDATES ---
st.sidebar.header("Simulation Controls")
timestamps = df['timestamp'].unique()
selected_time = st.sidebar.selectbox("Select Intraday Time", timestamps)

# Filter data for the selected timestamp
current_data = df[df['timestamp'] == selected_time]
nifty_spot = current_data['nifty_price'].iloc[0]
atm_strike = current_data['atm_strike'].iloc[0]

# --- 4.1 HEADER (MARKET CONTEXT) ---
st.header("1. Market Context")
col1, col2, col3, col4 = st.columns(4)
col1.metric("NIFTY Spot", f"{nifty_spot:.2f}")
col2.metric("Fixed ATM", f"{atm_strike}")
col3.metric("Time", str(selected_time).split("T")[-1] if "T" in str(selected_time) else str(selected_time))
col4.metric("Day Change", "+0.45%") # Mocked for dashboard completeness

st.divider()

# --- 4.2 UNDERLYING (NIFTY STATE) ---
st.header("2. Underlying (NIFTY State)")
# Mocking technicals as they require historical series not present in single intraday snapshots
mock_rsi = 65.4
rsi_interp, rsi_bias = interpret_rsi(mock_rsi)

nifty_state_data = {
    "Parameter": ["RSI", "Price vs 20 EMA", "Price vs 50 EMA", "Momentum (ROC)"],
    "Value": [f"{mock_rsi}", "Above", "Above", "+1.2%"],
    "Trend": ["⬆️", "⬆️", "⬆️", "⬆️"],
    "Interpretation": [rsi_interp, "Bullish Trend", "Macro Bullish", "Accelerating"],
    "Action Bias": [rsi_bias, "Bullish", "Bullish", "Bullish"]
}
st.dataframe(pd.DataFrame(nifty_state_data), use_container_width=True, hide_index=True)

st.divider()

# --- 4.3 ATM CALL & PUT ANALYSIS ---
st.header("3. ATM Analysis")
atm_data = current_data[current_data['strike'] == atm_strike]

if not atm_data.empty:
    atm_call = atm_data.iloc[0] # Assuming first match is correct for simplicity
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ATM Call")
        call_table = {
            "Parameter": ["Premium", "Delta", "Theta"],
            "Value": [f"{atm_call['call_premium']:.2f}", f"{atm_call['delta']:.2f}", f"{atm_call['theta']:.2f}"],
            "Trend": ["⬆️", "⬆️", "⬇️"],
            "Interpretation": ["Premium Expanding", "Gaining Sensitivity", "Decay Accelerating"],
            "Action Bias": ["Bullish", "Bullish", "Neutral"]
        }
        st.dataframe(pd.DataFrame(call_table), use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("ATM Put")
        put_table = {
            "Parameter": ["Premium", "Delta", "Theta"],
            "Value": [f"{atm_call['put_premium']:.2f}", f"{-atm_call['delta']:.2f}", f"{atm_call['theta']:.2f}"], # Simplified put delta
            "Trend": ["⬇️", "⬇️", "⬇️"],
            "Interpretation": ["Premium Collapsing", "Losing Sensitivity", "Decay Accelerating"],
            "Action Bias": ["Bearish", "Bearish", "Neutral"]
        }
        st.dataframe(pd.DataFrame(put_table), use_container_width=True, hide_index=True)

st.divider()

# --- 4.4 CUMULATIVE GREEKS ---
st.header("4. Cumulative Greeks (±10 Strikes)")
st.markdown("Aggregated risk exposure across the chain.")

agg_delta = current_data['delta'].sum()
agg_gamma = current_data['gamma'].sum()
agg_vega = current_data['vega'].sum()

greeks_table = {
    "Greek": ["Delta", "Gamma", "Vega"],
    "Absolute Value": [f"{agg_delta:.2f}", f"{agg_gamma:.4f}", f"{agg_vega:.2f}"],
    "Change Since Open": ["+12.4", "-0.002", "+1.5"],
    "Interpretation": ["Net Long Build", "Pinning Risk Low", "Vol Expanding"]
}
st.dataframe(pd.DataFrame(greeks_table), use_container_width=True, hide_index=True)

st.divider()

# --- 4.7 FINAL REGIME BOX ---
st.header("5. Auto Interpretation & Regime Box")
st.info("**Market Structure:** Bullish Trend | **Volatility:** Expanding | **Positioning:** Call Heavy")
st.success("**Suggested Playbook:** Buy on Dips / Long Call Spreads. Avoid naked short puts due to rising Vega.")