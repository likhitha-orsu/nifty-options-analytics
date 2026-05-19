def get_trend(current, previous):
    if current > previous * 1.001:
        return "Rising", "⬆️"
    elif current < previous * 0.999:
        return "Falling", "⬇️"
    return "Stable", "➡️"


def interpret_rsi(rsi_val: float) -> tuple[str, str]:
    """Takes raw RSI float. Returns (interpretation, bias)."""
    if rsi_val > 70:
        return "Overbought", "Bearish"
    elif rsi_val < 30:
        return "Oversold", "Bullish"
    return "Neutral", "Neutral"


def interpret_delta(delta_val: float) -> tuple[str, str]:
    """
    Takes raw delta float (call delta: 0 to 1, put delta: -1 to 0).
    Returns (interpretation, bias).
    """
    if delta_val >= 0.55:
        return "Deep ITM / Strong Directional", "Bullish"
    elif delta_val >= 0.45:
        return "Near ATM / Balanced Sensitivity", "Neutral"
    elif delta_val >= 0.2:
        return "OTM / Low Sensitivity", "Neutral"
    elif delta_val <= -0.55:
        return "Deep ITM Put / Strong Bearish", "Bearish"
    elif delta_val <= -0.45:
        return "Near ATM Put / Balanced", "Neutral"
    elif delta_val <= -0.2:
        return "OTM Put / Low Sensitivity", "Neutral"
    return "Near Zero Delta", "Neutral"


def interpret_vega(vega_val: float) -> tuple[str, str]:
    """
    Takes raw aggregated vega float.
    Returns (interpretation, bias).
    """
    if vega_val > 50:
        return "Vol Expansion — IV rising across chain", "Neutral/Hedge"
    elif vega_val < -50:
        return "Vol Contraction — IV falling", "Yield Generation"
    return "Stable Vol", "Neutral"