# analyze_sts_mts_lts.py
import argparse, os, datetime, numpy as np, pandas as pd, yfinance as yf

# ----- TA -----
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ====== 파라미터(환경변수로 덮어쓰기 가능) ======
BUY_STS  = float(os.getenv("STS_BUY", 88))
BUY_MTS  = float(os.getenv("MTS_BUY", 83))
BUY_LTS  = float(os.getenv("LTS_BUY", 83))
SELL_STS = float(os.getenv("STS_SELL", 87))
SELL_LTS = float(os.getenv("LTS_SELL", 82))

MIN_ROWS = 130        # 최소 데이터 길이(기본 6개월+)
DL_PERIOD = "420d"    # 다운로드 기간
DL_INTERVAL = "1d"    # 주기

# ====== 유틸 ======
def as_series(x, index=None):
    """(n,1) DataFrame/ndarray도 1D Series로 강제."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    arr = np.asarray(x).ravel()
    return pd.Series(arr, index=index) if index is not None else pd.Series(arr)

def read_tickers(file_path):
    """
    tickers_*.txt 읽기
    - 허용 포맷: 'TICKER' 또는 'TICKER,이름'
    - 공백/주석(#) 무시
    """
    items = []
    with open(file_path, encoding="utf-8") as f:
        for raw in f.read().splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                t, name = [p.strip() for p in s.split(",", 1)]
                items.append((t, name))
            else:
                items.append((s, ""))  # 이름 없음
    return items

# ====== 핵심 계산 ======
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    다중 지표 기반으로 STS/MTS/LTS 산출
    - STS: 단기 모멘텀·과열·패턴(1~3주)
    - MTS: 중기 트렌드·구름·자금유입(1~3개월)
    - LTS: 장기 추세·상승장 속도(6~12개월)
    """
    # 데이터 정형화
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    close = as_series(df["Close"], df.index)
    high  = as_series(df["High"], df.index)
    low   = as_series(df["Low"], df.index)
    open_ = as_series(df["Open"], df.index)
    vol   = as_series(df["Volume"], df.index)

    out = pd.DataFrame(index=df.index)
    out["Open"], out["High"], out["Low"], out["Close"], out["Volume"] = open_, high, low, close, vol

    # ----- 이동평균/지표들 -----
    sma5   = SMAIndicator(close, 5).sma_indicator()
    sma20  = SMAIndicator(close, 20).sma_indicator()
    sma60  = SMAIndicator(close, 60).sma_indicator()
    sma120 = SMAIndicator(close, 120).sma_indicator()
    ema20  = EMAIndicator(close, 20).ema_indicator()
    ema60  = EMAIndicator(close, 60).ema_indicator()

    rsi14 = RSIIndicator(close, 14).rsi()
    macd  = MACD(close, 26, 12, 9)
    macd_line, macd_sig, macd_hist = macd.macd(), macd.macd_signal(), macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    bb_m, bb_u, bb_l = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
    bb_std = (bb_u - bb_m) / 2

    # Ichimoku (ta 라이브러리) — 선행스팬 A/B, 기준선/전환선
    ichi = IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52, visual=True)
    ich_conv = ichi.ichimoku_conversion_line()
    ich_base = ichi.ichimoku_base_line()
    ich_a    = ichi.ichimoku_a()
    ich_b    = ichi.ichimoku_b()

    # OBV 간이버전: 등락 방향 * 거래량 누적
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * vol).cumsum()
    vol20 = vol.rolling(20).mean()

    # 변동성/추세강도
    atr14 = AverageTrueRange(high, low, close, window=14).average_true_range()
    adx14 = ADXIndicator(high, low, close, window=14).adx()

    # Stochastic (추가 참조만)
    stoch_k = StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()

    # 결과 프레임
    out = out.assign(
        SMA5=sma5, SMA20=sma20, SMA60=sma60, SMA120=sma120,
        EMA20=ema20, EMA60=ema60,
        RSI14=rsi14,
        MACD=macd_line, MACD_SIG=macd_sig, MACD_HIST=macd_hist,
        BB_M=bb_m, BB_U=bb_u, BB_L=bb_l, BB_STD=bb_std,
        ICH_CONV=ich_conv, ICH_BASE=ich_base, ICH_A=ich_a, ICH_B=ich_b,
        OBV=obv, VOL20=vol20, ATR14=atr14, ADX14=adx14, STOCH_K=stoch_k
    )

    # ----- 스코어 함수 -----
    def sts(i):
        s = 0.0; p = out["Close"].iat[i]
        # 가격 위치/단기 추세
        s += 15 if p > out["SMA20"].iat[i] else 0
        s += 10 if out["SMA5"].iat[i] > out["SMA20"].iat[i] else 0
        # 모멘텀/과매수·과매도 해소
        r = out["RSI14"].iat[i]
        if not np.isnan(r):
            s += max(0, min(20, (r - 30) * (20 / 40)))  # 30~70 구간에서 선형 가점
        # MACD histogram 양전환
        mh = out["MACD_HIST"].iat[i]
        s += 15 if (not np.isnan(mh) and mh > 0) else 0
        # 볼린저 z-score
        std = out["BB_STD"].iat[i]; m = out["BB_M"].iat[i]
        if not np.isnan(std) and std != 0:
            z = (p - m) / std
            s += max(0, min(15, (z + 2) * (15 / 4)))   # -2~+2 범위 매핑
        # 거래량 동반 양봉
        v = out["Volume"].iat[i]; v20 = out["VOL20"].iat[i]
        if not np.isnan(v20) and v20 > 0 and p >= out["Open"].iat[i] and (v / v20) >= 1.2:
            s += 10
        # 단기 개선
        if i >= 1 and not np.isnan(out["RSI14"].iat[i-1]) and not np.isnan(r):
            s += 5 if r >= out["RSI14"].iat[i-1] else 0
        # 변동성 과도 확대 감점 (너무 과열)
        if not np.isnan(out["ATR14"].iat[i]) and not np.isnan(p) and p > 0:
            atrp = (out["ATR14"].iat[i] / p) * 100
            if atrp > 6:  # 변동성 과다 시 약간 감점
                s -= min(5, (atrp - 6))
        return max(0.0, min(100.0, s))

    def mts(i):
        s = 0.0; p = out["Close"].iat[i]
        s += 15 if p > out["SMA60"].iat[i] else 0
        s += 10 if out["SMA20"].iat[i] > out["SMA60"].iat[i] else 0
        s += 10 if p > out["SMA120"].iat[i] else 0
        s += 10 if out["SMA60"].iat[i] > out["SMA120"].iat[i] else 0
        # 구름 상단 돌파
        cloud = max(out["ICH_A"].iat[i], out["ICH_B"].iat[i])
        s += 15 if p > cloud else 0
        # 전환선>기준선
        s += 5 if out["ICH_CONV"].iat[i] > out["ICH_BASE"].iat[i] else 0
        # OBV 개선
        if i >= 5 and not np.isnan(out["OBV"].iat[i-5]):
            s += 10 if out["OBV"].iat[i] > out["OBV"].iat[i-5] else 0
        # ADX로 추세 강도 반영(20~40 구간 가점)
        adx = out["ADX14"].iat[i]
        if not np.isnan(adx):
            s += max(0, min(10, (adx - 20) * (10 / 20)))
        # 1년 고점 대비 위치
        look = out["Close"].rolling(252).max().iat[i]
        if not np.isnan(look) and look > 0:
            ratio = p / look
            s += max(0, min(10, (ratio - 0.8) * (10 / 0.2)))
        return max(0.0, min(100.0, s))

    def lts(i):
        s = 0.0; p = out["Close"].iat[i]
        s += 20 if p > out["SMA120"].iat[i] else 0
        cloud = max(out["ICH_A"].iat[i], out["ICH_B"].iat[i])
        s += 20 if p > cloud else 0
        look = out["Close"].rolling(252).max().iat[i]
        if not np.isnan(look) and look > 0:
            ratio = p / look
            s += max(0, min(20, (ratio - 0.75) * (20 / 0.25)))
        if i >= 10 and not np.isnan(out["SMA120"].iat[i-10]):
            s += 20 if out["SMA120"].iat[i] > out["SMA120"].iat[i-10] else 0
        if i >= 20 and not np.isnan(out["OBV"].iat[i-20]):
            s += 20 if out["OBV"].iat[i] > out["OBV"].iat[i-20] else 0
        # 매우 과열(스토캐스틱 90↑) 약간 감점
        k = out["STOCH_K"].iat[i]
        if not np.isnan(k) and k > 90:
            s -= min(5, (k - 90) / 2)
        return max(0.0, min(100.0, s))

    STS=[]; MTS=[]; LTS=[]
    for i in range(len(out)):
        STS.append(sts(i)); MTS.append(mts(i)); LTS.append(lts(i))
    out["STS"], out["MTS"], out["LTS"] = STS, MTS, LTS
    return out

# ====== 실행 ======
def run(market):
    tickers_with_names = read_tickers(f"tickers_{market}.txt")

    rows = []
    for t, name in tickers_with_names:
        try:
            df = yf.download(t, period=DL_PERIOD, interval=DL_INTERVAL,
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < MIN_ROWS:
                print(f"SKIP {t} - not enough data")
                continue
            out  = compute(df)
            last = out.iloc[-1]

            close = float(last["Close"])
            STS   = float(last["STS"]); MTS=float(last["MTS"]); LTS=float(last["LTS"])
            rows.append({
                "ticker": t,
                "name": name,
                "close": round(close, 3),
                "STS": round(STS, 2),
                "MTS": round(MTS, 2),
                "LTS": round(LTS, 2),
                "BUY_CANDIDATE": (STS>=BUY_STS and MTS>=BUY_MTS and LTS>=BUY_LTS),
                "SELL_SIGNAL":  (STS<=SELL_STS and LTS<=SELL_LTS),
            })
        except Exception as e:
            print("ERR", t, e)

    if not rows:
        print("No rows produced")
        return

    rep_dir = os.path.join("reports", market)
    os.makedirs(rep_dir, exist_ok=True)

    rep = (pd.DataFrame(rows)
           .sort_values(["BUY_CANDIDATE","STS","MTS","LTS","close"],
                        ascending=[False, False, False, False, False]))
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # CSV 저장
    rep.to_csv(os.path.join(rep_dir, "latest.csv"), index=False, encoding="utf-8")
    rep.to_csv(os.path.join(rep_dir, f"{today}_summary.csv"), index=False, encoding="utf-8")

    # 통계
    buy  = rep[rep["BUY_CANDIDATE"]==True].copy()
    sell = rep[rep["SELL_SIGNAL"]==True].copy()
    avg_sts = float(rep["STS"].mean()); avg_mts = float(rep["MTS"].mean()); avg_lts = float(rep["LTS"].mean())

    # 텍스트 요약
    lines = [
        f"[{market.upper()} Summary — {today}]",
        f"총 종목: {len(rep)}",
        f"BUY 후보군(≥{BUY_STS}/{BUY_MTS}/{BUY_LTS}): {len(buy)}",
        f"SELL 후보군(STS≤{SELL_STS} & LTS≤{SELL_LTS}): {len(sell)}",
        f"평균 점수 — STS {avg_sts:.2f} / MTS {avg_mts:.2f} / LTS {avg_lts:.2f}",
        ""
    ]
    if not buy.empty:
        lines.append("[BUY Top 10]")
        top = buy.sort_values(["STS","MTS","LTS","close"], ascending=False).head(10)
        for i, r in enumerate(top.itertuples(), 1):
            nm = (r.name if isinstance(r.name, str) and r.name else r.ticker)
            lines.append(f"{i}) {nm:<20s} ({r.ticker})  Close {r.close}  STS {r.STS}  MTS {r.MTS}  LTS {r.LTS}")
        lines.append("")
    if not sell.empty:
        lines.append("[SELL Candidates]")
        top = sell.sort_values(["LTS","STS"], ascending=True).head(20)
        for r in top.itertuples():
            nm = (r.name if isinstance(r.name, str) and r.name else r.ticker)
            lines.append(f"-  {nm:<20s} ({r.ticker})  Close {r.close}  STS {r.STS}  LTS {r.LTS}")
        lines.append("")

    # 저장
    with open(os.path.join(rep_dir, f"{today}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(rep_dir, "latest_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["kr","us"], required=True)
    args = ap.parse_args()
    run(args.market)
