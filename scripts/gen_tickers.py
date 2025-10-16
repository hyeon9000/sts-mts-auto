# scripts/gen_tickers.py
import os, sys, math, time, re, json
import pandas as pd
import numpy as np

# ---- KR: pykrx로 KOSPI200 / KOSDAQ150 ----
from datetime import datetime, timedelta
from pykrx import stock as krx

# ---- US: 나스닥 공식 TXT + yfinance로 시총 ----
import requests
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KR_FILE = os.path.join(ROOT, "tickers_kr.txt")
US_FILE = os.path.join(ROOT, "tickers_us.txt")

def sanitize_name(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = s.replace(",", " ")  # CSV 한 줄 포맷 유지
    return s

# ---------------- KR (KOSPI200 + KOSDAQ150) ----------------
def build_kr():
    # pykrx는 "YYYYMMDD" 형식 날짜를 요구
    d = datetime.today().strftime("%Y%m%d")

    # 코스피200 / 코스닥150 구성종목
    k200 = krx.get_index_portfolio_deposit_file("코스피 200", d)
    kq150 = krx.get_index_portfolio_deposit_file("코스닥 150", d)

    rows = []

    for code in k200:
        name = krx.get_market_ticker_name(code)
        rows.append((f"{code}.KS", sanitize_name(name)))

    for code in kq150:
        name = krx.get_market_ticker_name(code)
        rows.append((f"{code}.KQ", sanitize_name(name)))

    # 중복 제거 및 정렬
    df = pd.DataFrame(rows, columns=["ticker","name"]).drop_duplicates("ticker")
    df = df.sort_values("ticker").reset_index(drop=True)

    # 저장
    with open(KR_FILE, "w", encoding="utf-8") as f:
        for t,n in df.itertuples(index=False):
            f.write(f"{t},{n}\n")

    print(f"[KR] Saved {len(df)} tickers -> {KR_FILE}")

# ---------------- US (NASDAQ200 + NYSE100 by MarketCap) ----------------
def fetch_symbol_tables():
    # 나스닥 공식 테이블 (TXT)
    # 1) NASDAQ 상장
    nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    # 2) 기타 거래소(주로 NYSE/AMEX/ARCA)
    other_url  = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    na = requests.get(nasdaq_url, timeout=30).text
    ot = requests.get(other_url, timeout=30).text

    def parse_txt(txt, sep="|"):
        lines = [x for x in txt.strip().splitlines() if not x.startswith("File Creation Time")]
        header = lines[0].split(sep)
        data = [l.split(sep) for l in lines[1:] if "Symbol" not in l and "File Creation Time" not in l]
        return pd.DataFrame(data, columns=header)

    nasdaq = parse_txt(na)
    other  = parse_txt(ot)

    return nasdaq, other

def pick_us_by_marketcap():
    nasdaq, other = fetch_symbol_tables()

    # 정리
    # 나스닥: Test Issue != Y, ETF/Right/Warrant 제외
    def clean(df, sym_col="Symbol", name_col="Security Name"):
        df = df.copy()
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"]!="Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"]!="Y"]
        # 이름/심볼
        df["ticker"] = df[sym_col].str.strip()
        df["name"]   = df[name_col].astype(str).map(sanitize_name)
        # 심볼 필터 (우선 보통주 중심)
        df = df[~df["ticker"].str.contains(r"[\.\$]")]  # 우선 복합/권리표기 제거
        df = df[df["ticker"].str.len()<=5]              # 너무 긴 심볼 제거(선별)
        return df[["ticker","name"]].drop_duplicates("ticker")

    nasdaq = clean(nasdaq, "Symbol", "Security Name")
    other  = clean(other,  "ACT Symbol", "Security Name")

    # other에는 Exchange(예: N=NYSE, A=AMEX, P=ARCA)가 있다면 NYSE만 우선 선별
    if "Exchange" in other.columns:
        nyse = other[other["Exchange"]=="N"][["ticker","name"]].copy()
    else:
        nyse = other.copy()

    # 시총 취득(yfinance) — 대량 질의는 배치로
    def enrich_marketcap(df):
        caps = []
        syms = df["ticker"].tolist()
        B = 50
        for i in range(0, len(syms), B):
            chunk = syms[i:i+B]
            tk = yf.Tickers(" ".join(chunk))
            for t in chunk:
                try:
                    info = tk.tickers[t].fast_info
                    mc = getattr(info, "market_cap", None)
                    if mc is None:
                        # fallback
                        info2 = tk.tickers[t].info
                        mc = info2.get("marketCap")
                except Exception:
                    mc = None
                caps.append(mc if isinstance(mc,(int,float)) else np.nan)
            time.sleep(0.2)
        df = df.copy()
        df["marketCap"] = caps
        df = df.dropna(subset=["marketCap"])
        df = df.sort_values("marketCap", ascending=False).reset_index(drop=True)
        return df

    nasdaq_top = enrich_marketcap(nasdaq).head(200)
    nyse_top   = enrich_marketcap(nyse).head(100)

    # 저장
    with open(US_FILE, "w", encoding="utf-8") as f:
        for t,n in pd.concat([nasdaq_top, nyse_top]).itertuples(index=False):
            f.write(f"{t},{n}\n")

    print(f"[US] Saved NASDAQ{len(nasdaq_top)} + NYSE{len(nyse_top)} -> {US_FILE}")

def main():
    os.makedirs(os.path.join(ROOT, "scripts"), exist_ok=True)
    build_kr()
    pick_us_by_marketcap()

if __name__ == "__main__":
    main()
