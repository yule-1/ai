# PER/PBR 파이프라인 코드 보관

요청하신 "데이터 수집 → 계산 → 차트 생성" 과정을 재현 가능한 코드로 별도 저장합니다.

## 스크립트
- `scripts/per_pbr_openkrx_opendart_pipeline.py`

## 기능
- OpenDART + OpenKRX 기반 연말 PER/PBR 계산
- 2025 잠정 + 2026/2027 예상치 생성
- 2027 보수/기준/낙관 시나리오 계산
- 연도별 PER/PBR 막대차트 + 2016~2024 평균선
- 결과물 CSV/PNG/리포트 저장

## 예시 실행
```bash
cd /Users/nyang/.openclaw/workspace/ai
python3 scripts/per_pbr_openkrx_opendart_pipeline.py \
  --date 20260222 \
  --ticker 005930 \
  --name 삼성전자 \
  --dart-key "$DART_KEY" \
  --krx-key "$KRX_KEY"

python3 scripts/per_pbr_openkrx_opendart_pipeline.py \
  --date 20260222 \
  --ticker 000660 \
  --name SK하이닉스 \
  --dart-key "$DART_KEY" \
  --krx-key "$KRX_KEY"
```

## 저장 규칙
- `ai/YYYYMMDD/data`
- `ai/YYYYMMDD/charts`
- `ai/YYYYMMDD/report`

원하면 다음 단계로, 이 스크립트를 `scripts/auto_chart_pipeline.py`와 연결해서
한 번의 명령으로 Notion 동기화까지 자동 처리하도록 확장할 수 있습니다.
