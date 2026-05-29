import os
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dhanhq import DhanContext, dhanhq
from modules.rules import get_trend, interpret_rsi, interpret_delta, interpret_vega

st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")
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
                # Store in session state
                st.session_state["client_id"] = input_client_id.strip()
                st.session_state["access_token"] = input_token.strip()
                st.session_state["dhan_authenticated"] = True
                st.rerun()
    st.stop() 

CLIENT_ID    = st.session_state["client_id"]
ACCESS_TOKEN = st.session_state["access_token"]
BASE_URL     = "https://api.dhan.co/v2"

# NIFTY index scrip details on Dhan
NIFTY_SCRIP   = 13          # UnderlyingScrip for NIFTY
NIFTY_SEG     = "IDX_I"     # Segment for index
NIFTY_SEC_ID  = "13"        # securityId string for LTP call
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

@st.cache_data(ttl=60)   # refresh every 60 seconds
def fetch_nifty_ltp() -> float:
    """Fetch live NIFTY spot price."""
    resp = _post("/marketfeed/ltp", {"IDX_I": [NIFTY_SCRIP]})
    return float(resp["data"]["IDX_I"][str(NIFTY_SCRIP)]["last_price"])

@st.cache_data(ttl=180)  # refresh every 3 minutes (Dhan rate limit)
def fetch_expiry_list() -> list:
    """Fetch available expiry dates for NIFTY options."""
    resp = _post("/optionchain/expirylist", {
        "UnderlyingScrip": NIFTY_SCRIP,
        "UnderlyingSeg": NIFTY_SEG,
    })
    return resp.get("data", [])

@st.cache_data(ttl=180)
def fetch_option_chain(expiry: str) -> pd.DataFrame:
    """
    Fetch NIFTY option chain for a given expiry and return a
    flat DataFrame with one row per strike containing both
    call and put data plus Greeks.
    """
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
            # Greeks — Dhan returns these inside each leg
            "delta":         ce.get("greeks", {}).get("delta", 0.0),
            "gamma":         ce.get("greeks", {}).get("gamma", 0.0),
            "theta":         ce.get("greeks", {}).get("theta", 0.0),
            "vega":          ce.get("greeks", {}).get("vega", 0.0),
            "put_delta":     pe.get("greeks", {}).get("delta", 0.0),
            "put_theta":     pe.get("greeks", {}).get("theta", 0.0),
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df

@st.cache_data(ttl=300)
def fetch_intraday_history() -> pd.DataFrame:
    """Fetch today's 5-min NIFTY candles for RSI / EMA calculation."""
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
    df = pd.DataFrame({"close": closes})
    return df

def calc_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)

def calc_ema(series: pd.Series, span: int) -> float:
    if len(series) < span:
        return float(series.iloc[-1])
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def calc_roc(series: pd.Series, period: int = 10) -> float:
    if len(series) < period + 1:
        return 0.0
    roc = ((series.iloc[-1] - series.iloc[-1 - period]) / series.iloc[-1 - period]) * 100
    return round(float(roc), 2)

def find_atm(spot: float, strikes: pd.Series) -> float:
    return float(strikes.iloc[(strikes - spot).abs().argsort().iloc[0]])

def day_change_pct(hist_df: pd.DataFrame) -> str:
    if hist_df.empty or len(hist_df) < 2:
        return "N/A"
    first = hist_df["close"].iloc[0]
    last  = hist_df["close"].iloc[-1]
    pct   = ((last - first) / first) * 100
    sign  = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"

def auto_regime(spot: float, ema20: float, ema50: float,
                rsi: float, agg_vega: float,
                call_oi: float, put_oi: float) -> tuple[str, str]:
    """Return (structure_label, playbook_text)."""
    bullish = spot > ema20 > ema50
    bearish = spot < ema20 < ema50
    structure = "Bullish Trend" if bullish else ("Bearish Trend" if bearish else "Sideways / Choppy")

    vol_regime = "Expanding" if agg_vega > 0 else "Contracting"
    pcr = put_oi / call_oi if call_oi > 0 else 1.0
    positioning = "Put Heavy (Bearish hedge)" if pcr > 1.2 else (
                  "Call Heavy (Bullish bets)"  if pcr < 0.8 else "Balanced")

    if bullish and rsi < 70:
        playbook = "Buy on Dips / Long Call Spreads. Avoid naked short puts due to rising Vega."
    elif bullish and rsi >= 70:
        playbook = "Overbought — consider Bull Put Spreads or partial profit booking on calls."
    elif bearish and rsi > 30:
        playbook = "Sell rallies / Long Put Spreads. Avoid naked short calls."
    elif bearish and rsi <= 30:
        playbook = "Oversold — consider Bear Call Spreads or watch for reversal signals."
    else:
        playbook = "Range-bound — Iron Condors or short straddles near ATM if IV is elevated."

    info_line  = f"**Market Structure:** {structure} | **Volatility:** {vol_regime} | **Positioning:** {positioning}"
    return info_line, playbook

st.title("Market Intelligence Dashboard")
st.markdown("*A tabular decision-support console for NIFTY options — powered by live Dhan data.*")

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

try:
    expiries = fetch_expiry_list()
    if not expiries:
        st.error("Could not fetch expiry list from Dhan. Verify your credentials in the console.")
        st.stop()
except Exception as e:
    st.error(f"Authentication failed or invalid token: {e}")
    if st.button("Re-enter Credentials"):
        st.session_state["dhan_authenticated"] = False
        st.rerun()
    st.stop()

selected_expiry = st.sidebar.selectbox("Select Expiry", expiries)
strike_range    = st.sidebar.slider("Strikes around ATM (±N)", min_value=5, max_value=20, value=10)

try:
    nifty_spot = fetch_nifty_ltp()
except Exception as e:
    st.error(f"Failed to fetch NIFTY spot: {e}")
    st.stop()

try:
    chain_df = fetch_option_chain(selected_expiry)
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")
    st.stop()

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
ema20_bias = "Bullish" if nifty_spot > ema20 else "Bearish"
ema50_bias = "Bullish" if nifty_spot > ema50 else "Bearish"
roc_bias   = "Bullish" if roc_val > 0 else "Bearish"
roc_trend  = "⬆️" if roc_val > 0 else "⬇️"

nifty_state_data = {
    "Parameter":     ["RSI (14)", "Price vs 20 EMA", "Price vs 50 EMA", "Momentum (ROC 10)"],
    "Value":         [f"{rsi_val}", vs_ema20, vs_ema50, f"{roc_val:+.2f}%"],
    "Trend":         ["⬆️" if rsi_val > 50 else "⬇️", "⬆️" if nifty_spot > ema20 else "⬇️",
                      "⬆️" if nifty_spot > ema50 else "⬇️", roc_trend],
    "Interpretation": [rsi_interp,
                       f"Price {'above' if nifty_spot > ema20 else 'below'} 20 EMA ({ema20:.2f})",
                       f"Price {'above' if nifty_spot > ema50 else 'below'} 50 EMA ({ema50:.2f})",
                       "Accelerating" if roc_val > 0 else "Decelerating"],
    "Action Bias":   [rsi_bias, ema20_bias, ema50_bias, roc_bias],
}
st.dataframe(pd.DataFrame(nifty_state_data), width="stretch", hide_index=True)
st.divider()

st.header("3. ATM Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"ATM Call  ({atm_strike:.0f} CE)")
    call_delta_interp, call_delta_bias = interpret_delta(float(atm["delta"]))
    call_table = {
        "Parameter":     ["Premium", "Delta", "Theta", "IV"],
        "Value":         [f"{atm['call_premium']:.2f}", f"{atm['delta']:.4f}",
                          f"{atm['theta']:.4f}",        f"{atm['call_iv']:.2f}%"],
        "Trend":         ["⬆️", "⬆️", "⬇️", "➡️"],
        "Interpretation": ["Market price of call",  call_delta_interp,
                           "Time decay per day",    "Implied volatility"],
        "Action Bias":   [call_delta_bias, call_delta_bias, "Neutral", "Monitor"],
    }
    st.dataframe(pd.DataFrame(call_table), width="stretch", hide_index=True)

with col2:
    st.subheader(f"ATM Put  ({atm_strike:.0f} PE)")
    put_delta_interp, put_delta_bias = interpret_delta(float(atm["put_delta"]))
    put_table = {
        "Parameter":     ["Premium", "Delta", "Theta", "IV"],
        "Value":         [f"{atm['put_premium']:.2f}", f"{atm['put_delta']:.4f}",
                          f"{atm['put_theta']:.4f}",   f"{atm['put_iv']:.2f}%"],
        "Trend":         ["⬇️", "⬇️", "⬇️", "➡️"],
        "Interpretation": ["Market price of put",  put_delta_interp,
                           "Time decay per day",   "Implied volatility"],
        "Action Bias":   [put_delta_bias, put_delta_bias, "Neutral", "Monitor"],
    }
    st.dataframe(pd.DataFrame(put_table), width="stretch", hide_index=True)

st.divider()

st.header(f"4. Cumulative Greeks (±{strike_range} Strikes around ATM)")
st.markdown("Aggregated risk exposure across the filtered chain.")

agg_delta = filtered_df["delta"].sum()
agg_gamma = filtered_df["gamma"].sum()
agg_vega  = filtered_df["vega"].sum()
agg_theta = filtered_df["theta"].sum()
vega_interp, vega_bias = interpret_vega(float(agg_vega))

greeks_table = {
    "Greek":          ["Delta", "Gamma", "Vega", "Theta"],
    "Aggregated Value": [f"{agg_delta:.4f}", f"{agg_gamma:.6f}",
                         f"{agg_vega:.4f}",  f"{agg_theta:.4f}"],
    "Interpretation": [
        "Net Long build" if agg_delta > 0 else "Net Short build",
        "High pinning risk" if agg_gamma > 0.05 else "Low pinning risk",
        vega_interp,
        "Chain decaying fast" if agg_theta < -50 else "Moderate decay",
    ],
    "Bias": [
        "Bullish" if agg_delta > 0 else "Bearish",
        "Caution" if agg_gamma > 0.05 else "Neutral",
        vega_bias,
        "Neutral",
    ],
}
st.dataframe(pd.DataFrame(greeks_table), width="stretch", hide_index=True)

total_call_oi = filtered_df["call_oi"].sum()
total_put_oi  = filtered_df["put_oi"].sum()
pcr           = total_put_oi / total_call_oi if total_call_oi > 0 else 0
c1, c2, c3 = st.columns(3)
c1.metric("Total Call OI", f"{total_call_oi:,.0f}")
c2.metric("Total Put OI",  f"{total_put_oi:,.0f}")
c3.metric("PCR (Put/Call OI)", f"{pcr:.2f}")
st.divider()

st.header("5. Option Chain (Filtered Strikes)")
display_chain = filtered_df[[
    "strike", "call_premium", "call_oi", "call_iv",
    "delta", "gamma", "vega", "theta",
    "put_premium", "put_oi", "put_iv"
]].copy()
display_chain.columns = [
    "Strike", "Call LTP", "Call OI", "Call IV%",
    "Delta", "Gamma", "Vega", "Theta",
    "Put LTP", "Put OI", "Put IV%"
]

def highlight_atm(row):
    color = "background-color: #fffacd" if row["Strike"] == atm_strike else ""
    return [color] * len(row)

st.dataframe(
    display_chain.style.apply(highlight_atm, axis=1).format(precision=2),
    width="stretch", hide_index=True
)
st.divider()
st.header("6. Auto Interpretation & Regime Box")
info_line, playbook = auto_regime(
    nifty_spot, ema20, ema50, rsi_val,
    agg_vega, total_call_oi, total_put_oi
)
st.info(info_line)
st.success(f"**Suggested Playbook:** {playbook}")
st.caption(f"Data refreshed at {now_str} | Expiry: {selected_expiry} | Cache TTL: 60s spot / 180s chain")