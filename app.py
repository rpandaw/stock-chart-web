import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT = 15
USE_HA_RSI = True


# ============================================================
# YAHOO SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120 Safari/537.36"
})


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=60)
def download_history(symbol, interval="15m", days=3):

    end_ts = int(time.time())

    start_ts = int(
        (datetime.utcnow() - timedelta(days=days)).timestamp()
    )

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + symbol.replace(".", "-")
    )

    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true"
    }

    try:

        r = SESSION.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        r.raise_for_status()

        js = r.json()

        result = js.get("chart", {}).get("result")

        if not result:
            return None

        result = result[0]

        timestamps = result.get("timestamp")

        quote = result.get(
            "indicators", {}
        ).get(
            "quote", [{}]
        )[0]

        if not timestamps:
            return None

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume")
        })

        adj = result.get(
            "indicators", {}
        ).get("adjclose", [])

        if adj and "adjclose" in adj[0]:
            df["AdjClose"] = adj[0]["adjclose"]
        else:
            df["AdjClose"] = df["Close"]

        df = df.dropna(
            subset=["Close", "AdjClose", "Volume"]
        )

        df = (
            df.drop_duplicates("Date")
              .sort_values("Date")
              .reset_index(drop=True)
        )

        if len(df) < 40:
            return None

        return df

    except Exception:
        return None


# ============================================================
# RSI
# ============================================================

def calc_rsi(df, period=14, use_heikin_ashi=False):

    if use_heikin_ashi:

        source = (
            df["Open"]
            + df["High"]
            + df["Low"]
            + df["Close"]
        ) / 4.0

    else:

        source = df["Close"]

    delta = source.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    c = df["AdjClose"]

    df["EMA10"] = c.ewm(
        span=10,
        adjust=False
    ).mean()

    df["EMA20"] = c.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = c.ewm(
        span=50,
        adjust=False
    ).mean()

    df["RSI14"] = calc_rsi(
        df,
        period=14,
        use_heikin_ashi=USE_HA_RSI
    )

    return df


# ============================================================
# CREATE CHART
# ============================================================

def create_chart(df, symbol, interval):

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1]
        }
    )

    x = df.index

    # --------------------------------------------------------
    # CANDLESTICKS
    # --------------------------------------------------------

    for i, row in df.iterrows():

        candle_color = (
            "green"
            if row["Close"] >= row["Open"]
            else "red"
        )

        ax1.vlines(
            i,
            row["Low"],
            row["High"],
            color=candle_color,
            linewidth=1
        )

        ax1.add_patch(
            Rectangle(
                (
                    i - 0.3,
                    min(row["Open"], row["Close"])
                ),
                0.6,
                abs(row["Close"] - row["Open"]),
                facecolor=candle_color,
                edgecolor=candle_color
            )
        )

    # --------------------------------------------------------
    # EMA20
    # --------------------------------------------------------

    ax1.plot(
        x,
        df["EMA20"],
        label="EMA20",
        linewidth=1.5
    )

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    ax1.set_title(
        "{} - {}".format(symbol, interval),
        fontsize=16,
        fontweight="bold"
    )

    ax1.set_ylabel("Price")

    ax1.legend(
        loc="upper left"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    ax2.plot(
        x,
        df["RSI14"],
        label="RSI14",
        linewidth=1.5
    )

    ax2.fill_between(
        x,
        60,
        df["RSI14"],
        where=(df["RSI14"] > 60),
        color="green",
        alpha=0.25
    )

    ax2.fill_between(
        x,
        df["RSI14"],
        40,
        where=(df["RSI14"] < 40),
        color="red",
        alpha=0.25
    )

    ax2.axhline(
        60,
        linestyle="--",
        linewidth=1
    )

    ax2.axhline(
        40,
        linestyle="--",
        linewidth=1
    )

    ax2.axhline(
        50,
        linestyle=":",
        linewidth=0.8
    )

    ax2.set_ylim(
        0,
        100
    )

    ax2.set_ylabel("RSI")

    ax2.legend(
        loc="upper left"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    step = 8

    ticks = df.index[::step]

    ax2.set_xticks(ticks)

    ax2.set_xticklabels(
        df.loc[
            ticks,
            "Date"
        ].dt.strftime("%m-%d %H:%M"),
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    return fig


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="US Stock Chart",
    layout="wide"
)

st.title("US Stock Chart")


# ============================================================
# USER INPUT
# ============================================================

col1, col2, col3, col4 = st.columns(
    [3, 2, 2, 1]
)


with col1:

    symbol = st.text_input(
        "Stock Symbol",
        "MRVL"
    ).strip().upper()


with col2:

    interval = st.selectbox(
        "Interval",
        options=[
            "1m",
            "5m",
            "15m",
            "30m",
            "60m",
            "1h",
            "1d"
        ],
        index=2,
        format_func=lambda x: {
            "1m": "1 Minute",
            "5m": "5 Minutes",
            "15m": "15 Minutes",
            "30m": "30 Minutes",
            "60m": "60 Minutes",
            "1h": "1 Hour",
            "1d": "Daily"
        }[x]
    )


with col3:

    days = st.selectbox(
        "Days",
        options=[
            1,
            2,
            3,
            5,
            7,
            10,
            15,
            30,
            60
        ],
        index=2
    )


with col4:

    st.write("")

    chart_button = st.button(
        "Chart",
        use_container_width=True
    )


# ============================================================
# DISPLAY CHART
# ============================================================

if chart_button:

    if not symbol:

        st.error(
            "Please enter a stock symbol."
        )

    else:

        with st.spinner(
            "Downloading {} data...".format(
                symbol
            )
        ):

            df = download_history(
                symbol,
                interval=interval,
                days=days
            )

        if df is None:

            st.error(
                "Could not download data for {}. "
                "Check the symbol, interval and "
                "available Yahoo Finance history.".format(
                    symbol
                )
            )

        else:

            df = add_indicators(df)

            fig = create_chart(
                df,
                symbol,
                interval
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)
