# scripts/gen_tickers.py
import os, sys, time, math, csv, io, threading
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import requests
import pandas as pd
import yfinance as yf

# ---- 옵션(원하면 숫자만 변경) ----
KR_KOSPI200_INDEX = "1028"   # KOSPI 200
KR_KOSDAQ150_INDEX = "2001"  # KOSDAQ 150
US_NASDAQ_TOP = 200
US_NYSE_TOP = 100

WRITE_KR = "tickers_kr.txt"
WRITE_US = "tickers_us.txt"

FALLBACK_DIR = os.path.join("scripts", "fallback")
FALLBACK_KR_KOSPI200 = os.path.join(FALLBACK_DIR, "kr_kospi200.csv")      # header: ticker,name  (예: 005930,삼성전자)
FALLBACK_KR_KOSDAQ150 = os.path.join(FALLBACK_DIR, "kr_kosdaq150.csv")    # header: ticker,name  (예: 035720,카카오)
FALLBACK_US_NASDAQ = os.path.join(FALLBACK_DIR, "nasdaqlisted.csv")       # NASDAQ Trader 포맷(.txt와 흡사, '|' 구분)
FALLBACK_US_OTHER  = os.path.join(FALLBACK_DIR, "otherlisted.csv")        # NYSE/AMEX 포맷

# ---------- 유틸 ----------
def log(msg: str):
    print(f"[gen] {msg}", flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def http_get(url: str, timeout=30, retries=3, sleep=2) -> Optional[str]:
    last = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            last = f"HTTP {resp.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(sleep * (i+1))
    log(f"GET 실패: {url} / last={last}")
    return None

# ---------- KR: pykrx ----------
def kr_fetch_by_pykrx() -> Optional[pd.DataFrame]:
    """
    pykrx로 KOSPI200 + KOSDAQ150 가져오기. 실패 시 None.
    """
    try:
        from pykrx import stock
    except Exception as e:
        log(f"pykrx import 실패: {e}")
        return None

    def get_index_members(idx_code: str, when: str) -> List[str]:
        try:
            return stock.get_index_portfolio_deposit_file(idx_code, when)
        except Exception:
            return []

    # 오늘/어제/그제 순차 시도 (휴장/지연 대응)
    for d in range(0, 7):
        day = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
        kospi200 = get_index_members(KR_KOSPI200_INDEX, day)
        kosdaq150 = get_index_members(KR_KOSDAQ150_INDEX, day)
        if kospi200 and kosdaq150:
            log(f"pykrx 성공 (기준일 {day}) / KOSPI200 {len(kospi200)}, KOSDAQ150 {len(kosdaq150)}")
            rows = []
            for t in kospi200:
                name = stock.get_market_ticker_name(t)
                rows.append((f"{t}.KS", name))
            for t in kosdaq150:
                name = stock.get_market_ticker_name(t)
                # yfinance 코스닥 접미사는 .KQ
                rows.append((f"{t}.KQ", name))
            df = pd.DataFrame(rows, columns=["ticker", "name"])
            return df
    return None

def kr_fetch_from_fallback() -> Optional[pd.DataFrame]:
    ok = True
    if not os.path.exists(FALLBACK_KR_KOSPI200):
        log(f"fallback 없음: {FALLBACK_KR_KOSPI200}"); ok = False
    if not os.path.exists(FALLBACK_KR_KOSDAQ150):
        log(f"fallback 없음: {FALLBACK_KR_KOSDAQ150}"); ok = False
    if not ok:
        return None

    def read_simple_csv(p):
        # header: ticker,name  (ticker는 숫자코드)
        df = pd.read_csv(p)
        assert {"ticker","name"}.issubset(df.columns)
        return df

    k1 = read_simple_csv(FALLBACK_KR_KOSPI200).copy()
    k1["ticker"] = k1["ticker"].astype(str).str.zfill(6) + ".KS"
    k2 = read_simple_csv(FALLBACK_KR_KOSDAQ150).copy()
    k2["ticker"] = k2["ticker"].astype(str).str.zfill(6) + ".KQ"
    df = pd.concat([k1, k2], ignore_index=True)[["ticker","name"]]
    log(f"fallback KR 로드 성공: {len(df)} rows")
    return df

def build_kr_list() -> List[Tuple[str, str]]:
    df = kr_fetch_by_pykrx()
    if df is None:
        df = kr_fetch_from_fallback()
    if df is None:
        raise RuntimeError("KR 리스트 생성 실패 (pykrx & fallback 모두 실패)")
    # 중복 제거 & 정렬
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    rows = list(df.itertuples(index=False, name=None))
    return rows

# ---------- US: 심볼 테이블 + 시총 ----------
NASDAQ_URLS = [
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
]
OTHER_URLS = [
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]

def parse_pipe_table(text: str) -> pd.DataFrame:
    """
    NASDAQ Trader 형식('|' 구분) 파싱.
    맨 마지막 'File Creation Time' 라인은 제외.
    """
    lines = [ln for ln in text.strip().splitlines() if "File Creation Time" not in ln]
    raw = "\n".join(lines)
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str)
    return df

def us_fetch_symbol_tables() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    na = None
    for u in NASDAQ_URLS:
        t = http_get(u, timeout=30, retries=3, sleep=3)
        if t: 
            na = parse_pipe_table(t); break
    if na is None:
        # fallback 파일
        if os.path.exists(FALLBACK_US_NASDAQ):
            na = pd.read_csv(FALLBACK_US_NASDAQ, sep="|", dtype=str)
            log("fallback nasdaqlisted.csv 사용")
        else:
            log("NASDAQ 표 로드 실패")
            return None

    ot = None
    for u in OTHER_URLS:
        t = http_get(u, timeout=30, retries=3, sleep=3)
        if t:
            ot = parse_pipe_table(t); break
    if ot is None:
        if os.path.exists(FALLBACK_US_OTHER):
            ot = pd.read_csv(FALLBACK_US_OTHER, sep="|", dtype=str)
            log("fallback otherlisted.csv 사용")
        else:
            log("OTHER 표 로드 실패")
            return None

    return na, ot

def yfin_market_caps(tickers: List[str], workers: int = 8, retries: int = 2) -> Dict[str, Optional[float]]:
    """
    yfinance로 시총 가져오기. fast_info.market_cap 우선, 실패 시 info['marketCap'] 재시도.
    멀티쓰레드로 속도 보강.
    """
    caps: Dict[str, Optional[float]] = {}
    lock = threading.Lock()

    def work(batch: List[str]):
        for t in batch:
            cap = None
            for r in range(retries+1):
                try:
                    tk = yf.Ticker(t)
                    # fast_info가 가장 빠름
                    cap = getattr(tk, "fast_info", None)
                    cap = getattr(cap, "market_cap", None)
                    if cap is None:
                        cap = tk.info.get("marketCap")
                    if cap: 
                        break
                except Exception:
                    pass
                time.sleep(0.5*(r+1))
            with lock:
                caps[t] = float(cap) if cap is not None else None

    if not tickers:
        return caps

    # 분할
    n = max(1, math.ceil(len(tickers)/workers))
    threads = []
    for i in range(0, len(tickers), n):
        th = threading.Thread(target=work, args=(tickers[i:i+n],))
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    return caps

def us_pick_by_marketcap() -> List[Tuple[str, str]]:
    res = us_fetch_symbol_tables()
    if res is None:
        raise RuntimeError("US 심볼 테이블 로드 실패")
    na, ot = res

    # 필터링: 테스트/ETF/권리증 제외
    def clean(df: pd.DataFrame, symcol: str = "Symbol", namecol: str = "Security Name") -> pd.DataFrame:
        df = df.copy()
        cols = {c.lower(): c for c in df.columns}
        def get(col):
            return cols.get(col.lower(), col)
        sym = get("Symbol"); name = get("Security Name")
        etf = get("ETF"); test = get("Test Issue")
        # 없는 컬럼은 기본값 처리
        if etf not in df.columns: df[etf] = "N"
        if test not in df.columns: df[test] = "N"
        df = df[(df[test] == "N") & (df[etf] == "N")]
        return df[[sym, name]].rename(columns={sym:"symbol", name:"name"}).dropna()

    na2 = clean(na)
    ot2 = clean(ot)

    # yfinance 티커로 그대로 사용 (NASDAQ/NYSE/AMEX 공통)
    nas_symbols = na2["symbol"].tolist()
    ny_symbols = ot2["symbol"].tolist()

    log(f"NASDAQ 후보 {len(nas_symbols)}, NYSE/AMEX 후보 {len(ny_symbols)} → 시총 수집 중...")
    caps_na = yfin_market_caps(nas_symbols, workers=12)
    caps_ny = yfin_market_caps(ny_symbols, workers=12)

    def topn(df: pd.DataFrame, caps: Dict[str, Optional[float]], n: int) -> List[Tuple[str,str]]:
        tmp = df.copy()
        tmp["cap"] = tmp["symbol"].map(caps)
        tmp = tmp.dropna(subset=["cap"]).sort_values("cap", ascending=False).head(n)
        return list(tmp[["symbol","name"]].itertuples(index=False, name=None))

    top_na = topn(na2, caps_na, US_NASDAQ_TOP)
    top_ny = topn(ot2, caps_ny, US_NYSE_TOP)

    rows = top_na + top_ny
    log(f"US 선정 결과: NASDAQ {len(top_na)} + NYSE/AMEX {len(top_ny)} = {len(rows)}")
    return rows

# ---------- 메인 ----------
def write_txt(path: str, rows: List[Tuple[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for t, name in rows:
            f.write(f"{t},{name}\n")
    log(f"Saved {len(rows)} -> {path}")

def main():
    ensure_dir(FALLBACK_DIR)

    # KR
    try:
        kr_rows = build_kr_list()
        write_txt(WRITE_KR, kr_rows)
    except Exception as e:
        log(f"KR 생성 실패: {e}")

    # US
    try:
        us_rows = us_pick_by_marketcap()
        write_txt(WRITE_US, us_rows)
    except Exception as e:
        log(f"US 생성 실패: {e}")

if __name__ == "__main__":
    main()
