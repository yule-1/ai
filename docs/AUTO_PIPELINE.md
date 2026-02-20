# Auto Pipeline (A + C)

자동으로 아래를 처리합니다.

- **A**: `data/*.csv` → `interactive/*.html` (Plotly 인터랙티브)
- **GitHub Pages**: 인터랙티브 링크 공개
- **C**: Notion DB 인덱스 동기화 + 인터랙티브 차트 임베드 페이지 생성

## 파일
- 스크립트: `scripts/auto_chart_pipeline.py`

## 실행
```bash
cd /Users/nyang/.openclaw/workspace/ai
python3 scripts/auto_chart_pipeline.py --date 20260220 --push --notion-sync
```

## 옵션
- `--date YYYYMMDD` : 날짜 폴더 지정 (기본: 오늘)
- `--push` : git commit/push 자동
- `--notion-sync` : Notion DB/페이지 동기화 수행

## Notion 동기화 환경변수
`--notion-sync` 사용 시 필수:

```bash
export NOTION_TOKEN='...'
export NOTION_PARENT_PAGE_ID='30d0c893d12080a18f2bf0e5779264d1'
```

## 기대 폴더 구조
```text
ai/
  YYYYMMDD/
    data/*.csv
    charts/*.png
    interactive/*.html
```

## 출력
- 인터랙티브 차트 생성 개수
- GitHub Pages URL
- (Notion 사용 시) DB ID / 생성 row 수 / 임베드 페이지 URL
