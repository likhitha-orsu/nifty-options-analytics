import os
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dhanhq import DhanContext, dhanhq
from modules.rules import get_trend, interpret_rsi, interpret_delta, interpret_vega

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")

# --- AUTHENTICATION STATE ---
if "dhan_authenticated" not in st.session_state:
    st.session_state["dhan_authenticated"] = False
    st.session_state["client_id"] = ""
    st.session_state["access_token"] = ""

if not st.session_state["dhan_authenticated"]:
    st.title(" Connect to Dhan API")
    with st.form("dhan_login_form"):
        input_client_id = st.text_input("Dhan Client ID", value=st.session_state["client_id"], help="Enter your Dhan Client ID")
        input_token = st.text_input("Access Token", value=st.session_state["access_token"], type="password", help="Enter your 0-day or valid Access Token")
        submit_btn = st.form_submit_button("Launch Dashboard")
        
        if submit_btn:
            if input_client_id.strip() == "" or input_token.strip() == "":
                st.error("Both Client ID and Access Token are required.")
            else:
                st.session_state["client_id"] = input_client_id.strip()
                st.session_state["access_token"] = input_token.strip()
                st.session_state["dhan_authenticated"] = True
                st.rerun()
    st.stop() 

CLIENT_ID    = st.session_state["client_id"]
ACCESS_TOKEN = st.session_state["access_token"]
BASE_URL     = "https://api.dhan.co/v2"

# --- API HELPER FUNCTIONS ---
NIFTY_SCRIP   = 13          
NIFTY_SEG     = "IDX_I"     
NIFTY_SEC_ID  = "13"        
NSE_EQ_SEG    = "NSE_EQ"

def _headers() -> dict:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
    }

def _post(path: str, payload: dict):
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def _get(path: str):
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def fetch_nifty_ltp() -> float:
    resp = _post("/marketfeed/ltp", {"IDX_I": [NIFTY_SCRIP]})
    return float(resp["data"]["IDX_I"][str(NIFTY_SCRIP)]["last_price"])

@st.cache_data(ttl=180)
def fetch_expiry_list() -> list:
    resp = _post("/optionchain/expirylist", {
        "UnderlyingScrip": NIFTY_SCRIP,
        "UnderlyingSeg": NIFTY_SEG,
    })
    return resp.get("data", [])

@st.cache_data(ttl=180)
def fetch_option_chain(expiry: str) -> pd.DataFrame:
    resp = _post("/optionchain", {
        "UnderlyingScrip": NIFTY_SCRIP,
        "UnderlyingSeg": NIFTY_SEG,
        "Expiry": expiry,
    })
    rows = []
    chain_data = resp.get("data", {})
    oc = chain_data.get("oc", {})          
    for strike_str, v in oc.items():
        strike = float(strike_str)
        ce = v.get("ce", {})
        pe = v.get("pe", {})
        rows.append({
            "strike":        strike,
            "call_premium":  ce.get("last_price", 0.0),
            "call_oi":       ce.get("oi", 0),
            "call_volume":   ce.get("volume", 0),
            "call_iv":       ce.get("implied_volatility", 0.0),
            "put_premium":   pe.get("last_price", 0.0),
            "put_oi":        pe.get("oi", 0),
            "put_volume":    pe.get("volume", 0),
            "put_iv":        pe.get("implied_volatility", 0.0),
            "delta":         ce.get("greeks", {}).get("delta", 0.0),
            "gamma":         ce.get("greeks", {}).get("gamma", 0.0),
            "theta":         ce.get("greeks", {}).get("theta", 0.0),
            "vega":          ce.get("greeks", {}).get("vega", 0.0),
            "put_delta":     pe.get("greeks", {}).get("delta", 0.0),
            "put_theta":     pe.get("greeks", {}).get("theta", 0.0),
        })
    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

@st.cache_data(ttl=300)
def fetch_intraday_history() -> pd.DataFrame:
    today = datetime.date.today().strftime("%Y-%m-%d")
    resp = _post("/charts/intraday", {
        "securityId": NIFTY_SEC_ID,
        "exchangeSegment": "IDX_I",
        "instrument": "INDEX",
        "interval": "5",
        "oi": False,
        "fromDate": f"{today} 09:15:00",
        "toDate":   f"{today} 15:30:00",
    })
    data = resp.get("data", {})
    closes = data.get("c", [])
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame({"close": closes})

# --- INDICATOR FUNCTIONS ---
def calc_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1: return 50.0
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)

def calc_ema(series: pd.Series, span: int) -> float:
    if len(series) < span: return float(series.iloc[-1])
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def calc_roc(series: pd.Series, period: int = 10) -> float:
    if len(series) < period + 1: return 0.0
    roc = ((series.iloc[-1] - series.iloc[-1 - period]) / series.iloc[-1 - period]) * 100
    return round(float(roc), 2)

def find_atm(spot: float, strikes: pd.Series) -> float:
    return float(strikes.iloc[(strikes - spot).abs().argsort().iloc[0]])

def day_change_pct(hist_df: pd.DataFrame) -> str:
    if hist_df.empty or len(hist_df) < 2: return "N/A"
    first = hist_df["close"].iloc[0]
    last  = hist_df["close"].iloc[-1]
    pct   = ((last - first) / first) * 100
    sign  = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"

def auto_regime(spot, ema20, ema50, rsi, agg_vega, call_oi, put_oi):
    bullish = spot > ema20 > ema50
    bearish = spot < ema20 < ema50
    structure = "Bullish Trend" if bullish else ("Bearish Trend" if bearish else "Sideways / Choppy")
    vol_regime = "Expanding" if agg_vega > 0 else "Contracting"
    pcr = put_oi / call_oi if call_oi > 0 else 1.0
    positioning = "Put Heavy (Bearish hedge)" if pcr > 1.2 else ("Call Heavy (Bullish bets)" if pcr < 0.8 else "Balanced")
    
    if bullish and rsi < 70: playbook = "Buy on Dips / Long Call Spreads. Avoid naked short puts."
    elif bullish and rsi >= 70: playbook = "Overbought — consider Bull Put Spreads."
    elif bearish and rsi > 30: playbook = "Sell rallies / Long Put Spreads."
    elif bearish and rsi <= 30: playbook = "Oversold — consider Bear Call Spreads."
    else: playbook = "Range-bound — Iron Condors or short straddles."
    
    info_line  = f"**Market Structure:** {structure} | **Volatility:** {vol_regime} | **Positioning:** {positioning}"
    return info_line, playbook

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", ["NIFTY Dashboard", "Trade Decision Engine"])

st.sidebar.divider()
st.sidebar.header("Live Controls")
if st.sidebar.button("🚪 Disconnect API Keys"):
    st.session_state["dhan_authenticated"] = False
    st.session_state["client_id"] = ""
    st.session_state["access_token"] = ""
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if page == "NIFTY Dashboard":
    st.title("Market Intelligence Dashboard")
    try:
        expiries = fetch_expiry_list()
        if not expiries:
            st.error("Could not fetch expiry list from Dhan.")
            st.stop()
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

    selected_expiry = st.sidebar.selectbox("Select Expiry", expiries)
    strike_range    = st.sidebar.slider("Strikes around ATM (±N)", min_value=5, max_value=20, value=10)

    nifty_spot = fetch_nifty_ltp()
    chain_df = fetch_option_chain(selected_expiry)
    hist_df = fetch_intraday_history()

    atm_strike  = find_atm(nifty_spot, chain_df["strike"])
    day_chg     = day_change_pct(hist_df)
    closes      = hist_df["close"] if not hist_df.empty else pd.Series([nifty_spot])
    rsi_val     = calc_rsi(closes)
    ema20       = calc_ema(closes, 20)
    ema50       = calc_ema(closes, 50)
    roc_val     = calc_roc(closes)
    now_str     = datetime.datetime.now().strftime("%H:%M:%S")

    all_strikes  = sorted(chain_df["strike"].unique())
    atm_idx      = all_strikes.index(atm_strike)
    low_idx      = max(0, atm_idx - strike_range)
    high_idx     = min(len(all_strikes) - 1, atm_idx + strike_range)
    nearby_strikes = all_strikes[low_idx:high_idx + 1]
    filtered_df  = chain_df[chain_df["strike"].isin(nearby_strikes)]

    atm_row = chain_df[chain_df["strike"] == atm_strike]
    if atm_row.empty:
        st.error("ATM strike not found in option chain.")
        st.stop()
    atm = atm_row.iloc[0]

    st.header("1. Market Context")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NIFTY Spot",  f"{nifty_spot:.2f}")
    col2.metric("ATM Strike",  f"{atm_strike:.0f}")
    col3.metric("Time",        now_str)
    col4.metric("Day Change",  day_chg)
    st.divider()

    st.header("2. Underlying (NIFTY State)")
    rsi_interp, rsi_bias = interpret_rsi(rsi_val)
    vs_ema20 = "Above" if nifty_spot > ema20 else "Below"
    vs_ema50 = "Above" if nifty_spot > ema50 else "Below"
    
    nifty_state_data = {
        "Parameter": ["RSI (14)", "Price vs 20 EMA", "Price vs 50 EMA", "Momentum (ROC 10)"],
        "Value": [f"{rsi_val}", vs_ema20, vs_ema50, f"{roc_val:+.2f}%"],
        "Trend": ["⬆️" if rsi_val > 50 else "⬇️", "⬆️" if nifty_spot > ema20 else "⬇️", "⬆️" if nifty_spot > ema50 else "⬇️", "⬆️" if roc_val > 0 else "⬇️"],
        "Interpretation": [rsi_interp, f"Price {'above' if nifty_spot > ema20 else 'below'} 20 EMA", f"Price {'above' if nifty_spot > ema50 else 'below'} 50 EMA", "Accelerating" if roc_val > 0 else "Decelerating"],
        "Action Bias": [rsi_bias, "Bullish" if nifty_spot > ema20 else "Bearish", "Bullish" if nifty_spot > ema50 else "Bearish", "Bullish" if roc_val > 0 else "Bearish"],
    }
    st.dataframe(pd.DataFrame(nifty_state_data), width="stretch", hide_index=True)
    st.divider()

    st.header("3. ATM Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"ATM Call ({atm_strike:.0f} CE)")
        call_delta_interp, call_delta_bias = interpret_delta(float(atm["delta"]))
        call_table = {
            "Parameter": ["Premium", "Delta", "Theta", "IV"],
            "Value": [f"{atm['call_premium']:.2f}", f"{atm['delta']:.4f}", f"{atm['theta']:.4f}", f"{atm['call_iv']:.2f}%"],
            "Trend": ["⬆️", "⬆️", "⬇️", "➡️"],
            "Interpretation": ["Market price", call_delta_interp, "Time decay", "Implied volatility"],
            "Action Bias": [call_delta_bias, call_delta_bias, "Neutral", "Monitor"],
        }
        st.dataframe(pd.DataFrame(call_table), width="stretch", hide_index=True)

    with col2:
        st.subheader(f"ATM Put ({atm_strike:.0f} PE)")
        put_delta_interp, put_delta_bias = interpret_delta(float(atm["put_delta"]))
        put_table = {
            "Parameter": ["Premium", "Delta", "Theta", "IV"],
            "Value": [f"{atm['put_premium']:.2f}", f"{atm['put_delta']:.4f}", f"{atm['put_theta']:.4f}", f"{atm['put_iv']:.2f}%"],
            "Trend": ["⬇️", "⬇️", "⬇️", "➡️"],
            "Interpretation": ["Market price", put_delta_interp, "Time decay", "Implied volatility"],
            "Action Bias": [put_delta_bias, put_delta_bias, "Neutral", "Monitor"],
        }
        st.dataframe(pd.DataFrame(put_table), width="stretch", hide_index=True)

    st.divider()
    st.header("4. Auto Interpretation & Regime Box")
    info_line, playbook = auto_regime(nifty_spot, ema20, ema50, rsi_val, filtered_df["vega"].sum(), filtered_df["call_oi"].sum(), filtered_df["put_oi"].sum())
    st.info(info_line)
    st.success(f"**Suggested Playbook:** {playbook}")



elif page == "Trade Decision Engine":
    st.title("Options vs Futures Decision Engine")
   
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("User Strategy Inputs")
        stock_name = st.selectbox("Stock Name", ["Reliance", "Polycab", "IDEA"])
        trade_type = st.radio("Trade Type", ["Long", "Short"], horizontal=True)
        stop_loss = st.number_input("Stop Loss (Price)", value=1280.0, step=1.0)
        momentum = st.radio("Momentum Expectation", ["Explosive", "Orderly"], horizontal=True)
        conviction = st.radio("Trade Conviction", ["High", "Medium"], horizontal=True)
        intended_lots = st.number_input("Intended Lots", value=100, step=10)
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    with col2:
        st.subheader("Simulated Market Data")
        mock_spot = {"Reliance": 1302.50, "Polycab": 5210.00, "IDEA": 14.90}[stock_name]
        mock_future = {"Reliance": 1305.00, "Polycab": 5230.00, "IDEA": 15.00}[stock_name]
        
        spot_price = st.number_input("Spot Price", value=mock_spot, step=0.1)
        future_price = st.number_input("Future Price", value=mock_future, step=0.1)
        option_premium = st.number_input("Option Premium", value=45.5, step=0.1)
        best_bid = st.number_input("Best Bid", value=45.0, step=0.1)
        best_ask = st.number_input("Best Ask", value=46.0, step=0.1)
        open_interest = st.number_input("Open Interest (Lots)", value=15000, step=100)

    st.divider()

    if run_analysis:
        st.header("Decision Scorecard")
        
        # --- RULE 1 & 2 ---
        vote_momentum = "Options" if momentum == "Explosive" else "Futures"
        vote_conviction = "Futures" if conviction == "High" else "Options"

        # --- RULE 3: Liquidity ---
        mid_price = (best_ask + best_bid) / 2
        spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else 100
        oi_ratio = (intended_lots / open_interest) * 100 if open_interest > 0 else 100
        
        liquidity_pass = (spread_pct <= 3) and (oi_ratio < 2)
        vote_liquidity = "Options" if liquidity_pass else "Futures"
        liq_detail = f"Spread: {spread_pct:.1f}% | OI Ratio: {oi_ratio:.1f}%"

        # --- RULE 4: Cost (OPR) ---
        strike_multiples = {"Reliance": 10, "Polycab": 50, "IDEA": 1}
        mult = strike_multiples.get(stock_name, 10)
        atm_strike = round(spot_price / mult) * mult
        
        if trade_type == "Long":
            intrinsic = max(spot_price - atm_strike, 0)
        else:
            intrinsic = max(atm_strike - spot_price, 0)
            
        extrinsic = max(option_premium - intrinsic, 0)
        future_cost = abs(future_price - spot_price)
        
        opr = extrinsic / future_cost if future_cost > 0 else 0
        vote_cost = "Futures" if opr > 4 else "Options"
        cost_detail = f"ATM: {atm_strike} | OPR: {opr:.2f}"

        # --- RULE 5: Stop Loss ---
        sl_pct = (abs(spot_price - stop_loss) / spot_price) * 100 if spot_price > 0 else 0
        vote_sl = "Options" if sl_pct > 4 else "Futures"
        sl_detail = f"SL Distance: {sl_pct:.2f}%"

        # --- Tally ---
        votes = [vote_momentum, vote_conviction, vote_liquidity, vote_cost, vote_sl]
        options_count = votes.count("Options")
        futures_count = votes.count("Futures")
        winner = "OPTIONS" if options_count >= 3 else "FUTURES"
        
        st.markdown(f"<h2 style='text-align: center; color: {'#4CAF50' if winner=='OPTIONS' else '#2196F3'};'>🏆 Recommended Instrument: {winner} ({max(options_count, futures_count)} to {min(options_count, futures_count)})</h2>", unsafe_allow_html=True)
        st.write("")

        results_data = {
            "Factor": ["1. Momentum", "2. Conviction", "3. Liquidity Test", "4. Cost Valuation (OPR)", "5. Risk Profile (SL%)"],
            "User Input / Logic": [momentum, conviction, liq_detail, cost_detail, sl_detail],
            "Recommendation": [vote_momentum, vote_conviction, vote_liquidity, vote_cost, vote_sl]
        }
        
        st.dataframe(pd.DataFrame(results_data), width="stretch", hide_index=True)