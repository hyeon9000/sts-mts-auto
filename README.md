# sts-mts-auto

자동으로 한국/미국 주식의 STS/MTS/LTS 점수를 계산해 매일 리포트를 생성합니다.
iOS에서 ZIP 업로드 후에는 PC 없이도 매일 자동 실행됩니다.

## 사용법
1) Public 레포에 ZIP의 모든 파일을 업로드
2) `tickers_kr.txt`, `tickers_us.txt`에 감시 티커 편집
3) Actions 탭 Enable 후 `Run workflow`
4) 결과는 `reports/<market>/latest.csv`, `<date>_summary.txt`

## 기본 컷오프
- BUY: STS≥88 / MTS≥83 / LTS≥83
- SELL: STS≤87 / LTS≤82
