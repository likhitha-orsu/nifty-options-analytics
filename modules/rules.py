def get_trend(current, previous):
    if current > previous * 1.001: return "Rising", "⬆️"
    elif current < previous * 0.999: return "Falling", "⬇️"
    return "Stable", "➡️"

def interpret_rsi(rsi_val):
    if rsi_val > 70: return "Overbought", "Bearish"
    elif rsi_val < 30: return "Oversold", "Bullish"
    return "Neutral", "Neutral"

def interpret_delta(delta_trend, price_trend):
    if delta_trend == "Rising" and price_trend == "Rising":
        return "Long Buildup", "Bullish"
    elif delta_trend == "Falling" and price_trend == "Falling":
        return "Short Buildup", "Bearish"
    return "Consolidation", "Neutral"

def interpret_vega(vega_trend):
    if vega_trend == "Rising": return "Vol Expansion", "Neutral/Hedge"
    elif vega_trend == "Falling": return "Vol Contraction", "Yield Generation"
    return "Stable Vol", "Neutral"