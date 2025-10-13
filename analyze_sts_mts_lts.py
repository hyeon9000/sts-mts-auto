import argparse, os, datetime, numpy as np, pandas as pd, yfinance as yf
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

BUY_STS = float(os.getenv("STS_BUY", 88))
BUY_MTS = float(os.getenv("MTS_BUY", 83))
BUY_LTS = float(os.getenv("LTS_BUY", 83))
SELL_STS = float(os.getenv("STS_SELL", 87))
SELL_LTS = float(os.getenv("LTS_SELL", 82))

# ---- NEW: 항상 1차원 Series 보장 ----
def as_series(x, index=None):
    """
    x가 (n,1) DataFrame/ndarray 여도 1차원 Series로 바꿔준다.
    index가 주어지면 해당 index로 맞춰준다.
    """
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # 단일 컬럼만 취함
        return x.iloc[:, 0]
    arr = np.asarray(x).ravel()
    if index is not None:
        return pd.Series(arr, index=index)
    return pd.Series(arr)

def ichimoku(df, conv=9, base=26, spanb=52, disp=26):
    high = as_series(df["High"])
    low  = as_series(df["Low"])
    convl = (high.rolling(conv).max() + low.rolling(conv).min())/2
    basel = (high.rolling(base).max() + low.rolling(base).min())/2
    spana = ((convl + basel)/2).shift(disp)
    spanb = ((high.rolling(spanb).max() + low.rolling(spanb).min())/2).shift(disp)
    return convl, basel, spana, spanb

def obv(df):
    close = as_series(df["Close"])
    vol   = as_series(df["Volume"])
    sign = np.sign(close.diff()).fillna(0)
    return (sign * vol).cumsum()

def compute(df):
    # 모든 핵심 컬럼을 1차원 Series로 보정
    close = as_series(df["Close"], index=df.index)
    open_ = as_series(df["Open"],  index=df.index)
    high  = as_series(df["High"],  index=df.index)
    low   = as_series(df["Low"],   index=df.index)
    vol   = as_series(df["Volume"],index=df.index)

    out = pd.DataFrame(index=df.index)
    out["Close"] = close
    out["Open"]  = open_
    out["Volume"]= vol

    # TA 인디케이터 입력도 1차원 Series 보장
    sma5   = SMAIndicator(close, 5).sma_indicator()
    sma20  = SMAIndicator(close, 20).sma_indicator()
    sma60  = SMAIndicator(close, 60).sma_indicator()
    sma120 = SMAIndicator(close,120).sma_indicator()

    rsi  = RSIIndicator(close,14).rsi()
    macd = MACD(close, 26, 12, 9)
    bb   = BollingerBands(close, 20, 2)

    # Ichimoku/OBV
    conv, base, spana, spanb = ichimoku(pd.DataFrame({"High":high,"Low":low}))
    obv20 = obv(pd.DataFrame({"Close":close,"Volume":vol}))
    vol20 = vol.rolling(20).mean()

    out = out.assign(
        SMA5=sma5, SMA20=sma20, SMA60=sma60, SMA120=sma120,
        RSI14=rsi,
        MACD=macd.macd(), MACD_SIG=macd.macd_signal(), MACD_HIST=macd.macd_diff(),
        BB_M=bb.bollinger_mavg(), BB_U=bb.bollinger_hband(), BB_L=bb.bollinger_lband()
    )
    out["BB_STD"]  = (out["BB_U"] - out["BB_M"]) / 2
    out["ICH_CONV"]= conv
    out["ICH_BASE"]= base
    out["ICH_SPANA"]= spana
    out["ICH_SPANB"]= spanb
    out["OBV"]     = obv20
    out["VOL20"]   = vol20

    # ---- scoring ----
    def sts(i):
        s=0.0; p=out["Close"].iat[i]
        s+=15 if p>out["SMA20"].iat[i] else 0
        r=out["RSI14"].iat[i]
        if not np.isnan(r): s+=max(0,min(20,(r-30)*(20/40)))
        mh=out["MACD_HIST"].iat[i]; s+=15 if (not np.isnan(mh) and mh>0) else 0
        std=out["BB_STD"].iat[i]; m=out["BB_M"].iat[i]
        if not np.isnan(std) and std!=0:
            z=(p-m)/std; s+=max(0,min(15,(z+2)*(15/4)))
        v=out["Volume"].iat[i]; v20=out["VOL20"].iat[i]
        if not np.isnan(v20) and v20>0 and p>=out["Open"].iat[i] and (v/v20)>=1.2: s+=10
        s+=10 if out["SMA5"].iat[i]>out["SMA20"].iat[i] else 0
        if i>=1 and not np.isnan(out["RSI14"].iat[i-1]) and not np.isnan(r):
            s+=5 if r>=out["RSI14"].iat[i-1] else 0
        return min(100.0,s)

    def mts(i):
        s=0.0; p=out["Close"].iat[i]
        s+=15 if p>out["SMA60"].iat[i] else 0
        s+=10 if out["SMA20"].iat[i]>out["SMA60"].iat[i] else 0
        if i>=5 and not np.isnan(out["SMA60"].iat[i-5]): s+=10 if out["SMA60"].iat[i]>out["SMA60"].iat[i-5] else 0
        s+=10 if p>out["SMA120"].iat[i] else 0
        s+=10 if out["SMA60"].iat[i]>out["SMA120"].iat[i] else 0
        cloud=max(out["ICH_SPANA"].iat[i],out["ICH_SPANB"].iat[i])
        s+=15 if p>cloud else 0
        s+=5 if out["ICH_CONV"].iat[i]>out["ICH_BASE"].iat[i] else 0
        if i>=5 and not np.isnan(out["OBV"].iat[i-5]): s+=10 if out["OBV"].iat[i]>out["OBV"].iat[i-5] else 0
        look=out["Close"].rolling(252).max().iat[i]
        if not np.isnan(look) and look>0:
            ratio=out["Close"].iat[i]/look
            s+=max(0,min(10,(ratio-0.8)*(10/0.2)))
        return min(100.0,s)

    def lts(i):
        s=0.0; p=out["Close"].iat[i]
        s+=20 if p>out["SMA120"].iat[i] else 0
        cloud=max(out["ICH_SPANA"].iat[i],out["ICH_SPANB"].iat[i])
        s+=20 if p>cloud else 0
        look=out["Close"].rolling(252).max().iat[i]
        if not np.isnan(look) and look>0:
            ratio=p/look; s+=max(0,min(20,(ratio-0.75)*(20/0.25)))
        if i>=10 and not np.isnan(out["SMA120"].iat[i-10]): s+=20 if out["SMA120"].iat[i]>out["SMA120"].iat[i-10] else 0
        if i>=20 and not np.isnan(out["OBV"].iat[i-20]): s+=20 if out["OBV"].iat[i]>out["OBV"].iat[i-20] else 0
        return min(100.0,s)

    STS=[]; MTS=[]; LTS=[]
    for i in range(len(out)):
        STS.append(sts(i)); MTS.append(mts(i)); LTS.append(lts(i))
    out["STS"]=STS; out["MTS"]=MTS; out["LTS"]=LTS
    return out

def run(market):
    tickers = open(f"tickers_{market}.txt").read().strip().splitlines()
    rows=[]
    for t in tickers:
        try:
            df = yf.download(t, period="420d", interval="1d",
                             auto_adjust=True, progress=False)
            # 일부 케이스에서 멀티컬럼/2D → 단일 컬럼 보정
            for col in ["Open","High","Low","Close","Volume"]:
                if col in df.columns and isinstance(df[col], pd.DataFrame):
                    df[col] = df[col].iloc[:,0]

            if df.empty or len(df) < 130:
                continue

            out  = compute(df)
            last = out.iloc[-1]
            close = float(last["Close"])
            STS = float(last["STS"]); MTS = float(last["MTS"]); LTS = float(last["LTS"])

            rows.append({
                "ticker": t, "close": round(close,3),
                "STS": round(STS,2), "MTS": round(MTS,2), "LTS": round(LTS,2),
                "BUY_CANDIDATE": (STS>=BUY_STS and MTS>=BUY_MTS and LTS>=BUY_LTS),
                "SELL_SIGNAL":  (STS<=SELL_STS and LTS<=SELL_LTS),
            })
        except Exception as e:
            print("ERR", t, e)

    if not rows:
        return

    rep_dir = os.path.join("reports", market)
    os.makedirs(rep_dir, exist_ok=True)

    rep = (pd.DataFrame(rows)
             .sort_values(["BUY_CANDIDATE","STS","MTS","LTS","close"],
                          ascending=[False,False,False,False,False]))
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    rep.to_csv(os.path.join(rep_dir, "latest.csv"), index=False, encoding="utf-8")
    rep.to_csv(os.path.join(rep_dir, f"{today}_summary.csv"), index=False, encoding="utf-8")

    buy  = rep[rep["BUY_CANDIDATE"]==True].copy()
    sell = rep[rep["SELL_SIGNAL"]==True].copy()

    lines = [
        f"[{market.upper()} Summary — {today}]",
        f"총 종목: {len(rep)}",
        f"BUY 후보군(≥{BUY_STS}/{BUY_MTS}/{BUY_LTS}): {len(buy)}",
        f"SELL 후보군(STS≤{SELL_STS} & LTS≤{SELL_LTS}): {len(sell)}",
        ""
    ]
    if not buy.empty:
        lines.append("[BUY Top 10]")
        top = buy.sort_values(["STS","MTS","LTS","close"], ascending=False).head(10)
        for i, r in enumerate(top.itertuples(), 1):
            lines.append(f"{i}) {r.ticker:>8s}  Close {r.close}  STS {r.STS}  MTS {r.MTS}  LTS {r.LTS}")
        lines.append("")

    if not sell.empty:
        lines.append("[SELL Candidates]")
        top = sell.sort_values(["LTS","STS"], ascending=True).head(20)
        for r in top.itertuples():
            lines.append(f"- {r.ticker:>8s}  Close {r.close}  STS {r.STS}  LTS {r.LTS}")
        lines.append("")

    with open(os.path.join(rep_dir, f"{today}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["kr","us"], required=True)
    a = ap.parse_args()
    run(a.market)
