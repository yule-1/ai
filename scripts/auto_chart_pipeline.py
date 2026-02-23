#!/usr/bin/env python3
"""
Auto pipeline for A+C workflow:
A) Build Plotly interactive HTML from CSV files
C) Sync Notion DB entries + make an embed page of interactive charts

Usage:
  python3 scripts/auto_chart_pipeline.py --date 20260220 --push --notion-sync

Required env for Notion sync:
  NOTION_TOKEN
  NOTION_PARENT_PAGE_ID   (32-hex or dashed UUID)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def ensure_plotly() -> None:
    try:
        import plotly.express as _  # noqa
    except Exception:
        run(["python3", "-m", "pip", "install", "--user", "plotly", "-q"])


def build_interactive_html(data_dir: Path, out_dir: Path) -> List[Path]:
    import plotly.express as px

    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    for csv in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
            if df.empty or len(df.columns) < 2:
                continue

            date_col = df.columns[0]
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            if parsed.notna().mean() < 0.6:
                continue

            d = df.copy()
            d[date_col] = parsed

            num_cols = [c for c in d.columns[1:] if pd.api.types.is_numeric_dtype(d[c])]
            if not num_cols:
                for c in d.columns[1:]:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                num_cols = [c for c in d.columns[1:] if pd.api.types.is_numeric_dtype(d[c])]
            if not num_cols:
                continue

            m = d[[date_col] + num_cols].melt(id_vars=[date_col], var_name="series", value_name="value").dropna()
            title = csv.stem.replace("_", " ").title()

            fig = px.line(m, x=date_col, y="value", color="series", template="plotly_dark", title=title)
            fig.update_layout(legend_title_text="", hovermode="x unified")

            html_path = out_dir / f"{csv.stem}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
            created.append(html_path)
        except Exception:
            continue

    return created


def normalize_uuid(raw: str) -> str:
    s = raw.strip().replace("-", "")
    if len(s) != 32:
        raise ValueError("Invalid 32-hex UUID")
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"


@dataclass
class NotionConfig:
    token: str
    parent_page_id: str


class NotionClient:
    def __init__(self, cfg: NotionConfig):
        import requests

        self.requests = requests
        self.base = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {cfg.token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        self.parent = normalize_uuid(cfg.parent_page_id)

    def _post(self, path: str, payload: dict):
        r = self.requests.post(f"{self.base}{path}", headers=self.headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, payload: dict):
        r = self.requests.patch(f"{self.base}{path}", headers=self.headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str):
        r = self.requests.get(f"{self.base}{path}", headers=self.headers, timeout=20)
        r.raise_for_status()
        return r.json()

    def ensure_parent_access(self) -> None:
        self._get(f"/pages/{self.parent}")

    def ensure_index_db(self) -> str:
        res = self._post(
            "/search",
            {
                "query": "AI Charts Data Index",
                "filter": {"property": "object", "value": "database"},
            },
        )

        for db in res.get("results", []):
            if db.get("parent", {}).get("page_id") == self.parent:
                return db["id"]

        created = self._post(
            "/databases",
            {
                "parent": {"type": "page_id", "page_id": self.parent},
                "title": [{"type": "text", "text": {"content": "AI Charts Data Index"}}],
                "properties": {
                    "Name": {"title": {}},
                    "Date": {"date": {}},
                    "Type": {
                        "select": {
                            "options": [
                                {"name": "interactive", "color": "blue"},
                                {"name": "data", "color": "green"},
                                {"name": "chart", "color": "purple"},
                            ]
                        }
                    },
                    "URL": {"url": {}},
                    "Path": {"rich_text": {}},
                },
            },
        )
        return created["id"]

    def get_existing_names(self, db_id: str) -> set:
        out = set()
        q = self._post(f"/databases/{db_id}/query", {"page_size": 100})
        for row in q.get("results", []):
            t = row.get("properties", {}).get("Name", {}).get("title", [])
            if t:
                out.add(t[0].get("plain_text", ""))
        return out

    def add_index_rows(self, db_id: str, date_iso: str, rows: List[Tuple[str, str, str]]) -> int:
        existing = self.get_existing_names(db_id)
        created = 0
        for typ, name, url in rows:
            if name in existing:
                continue
            self._post(
                "/pages",
                {
                    "parent": {"database_id": db_id},
                    "properties": {
                        "Name": {"title": [{"type": "text", "text": {"content": name}}]},
                        "Date": {"date": {"start": date_iso}},
                        "Type": {"select": {"name": typ}},
                        "URL": {"url": url},
                        "Path": {"rich_text": [{"type": "text", "text": {"content": name}}]},
                    },
                },
            )
            created += 1
        return created

    def create_embed_page(self, title: str, rows: List[Tuple[str, str, str]]) -> str:
        page = self._post(
            "/pages",
            {
                "parent": {"page_id": self.parent},
                "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
            },
        )
        pid = page["id"]

        def pretty(name: str) -> str:
            stem = Path(name).stem
            label = stem
            # common KR labels
            replaces = {
                "samsung": "삼성전자",
                "skhynix": "SK하이닉스",
                "yearly": "연도별",
                "yearend": "연말",
                "with": "포함",
                "and": "및",
                "per": "PER",
                "pbr": "PBR",
                "mcap": "시가총액",
                "historical": "과거",
                "actual": "실제치",
                "adjusted": "수정",
                "close": "종가",
                "full": "전체",
                "timeseries": "시계열",
                "yfinance": "yfinance",
                "interactive": "인터랙티브",
                "scenarios": "시나리오",
                "estimate": "추정",
                "estimates": "추정치",
                "provisional": "잠정치",
                "yesterday": "어제 기준",
                "price": "주가",
            }
            for en, ko in replaces.items():
                label = label.replace(en, ko)
            return label.replace("_", " ")

        blocks = [
            {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": "Interactive + Charts + Data links"}}]}},
        ]

        sections = [
            ("interactive", "Interactive Charts"),
            ("chart", "Chart Images (PNG)"),
            ("data", "Data (CSV)"),
        ]

        for typ, heading in sections:
            items = [(n, u) for t, n, u in rows if t == typ]
            if not items:
                continue
            blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": heading}}]}})
            for name, url in items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": pretty(name), "link": {"url": url}}}]
                    },
                })
                blocks.append({"object": "block", "type": "bookmark", "bookmark": {"url": url}})

        for i in range(0, len(blocks), 90):
            self._patch(f"/blocks/{pid}/children", {"children": blocks[i : i + 90]})
        return page.get("url", "")


def ensure_github_pages(repo_dir: Path) -> str:
    # idempotent: create if missing
    run(["gh", "api", f"repos/yule-1/ai/pages", "-X", "POST", "-F", "source[branch]=main", "-F", "source[path]=/"], cwd=repo_dir, check=False)
    return "https://yule-1.github.io/ai/"


def git_commit_push(repo_dir: Path, message: str) -> None:
    run(["git", "add", "-A"], cwd=repo_dir)
    diff = run(["git", "status", "--porcelain"], cwd=repo_dir)
    if not diff.stdout.strip():
        return
    run(["git", "commit", "-m", message], cwd=repo_dir)
    run(["git", "push"], cwd=repo_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="Folder date YYYYMMDD")
    p.add_argument("--repo", default="/Users/nyang/.openclaw/workspace/ai")
    p.add_argument("--push", action="store_true")
    p.add_argument("--notion-sync", action="store_true")
    args = p.parse_args()

    repo = Path(args.repo)
    root = repo / args.date
    data_dir = root / "data"
    charts_dir = root / "charts"
    interactive_dir = root / "interactive"

    if not data_dir.exists():
        raise SystemExit(f"Missing data dir: {data_dir}")

    ensure_plotly()
    created = build_interactive_html(data_dir, interactive_dir)

    pages_url = ensure_github_pages(repo)

    if args.push:
        git_commit_push(repo, f"Auto pipeline update for {args.date}: interactive charts + sync artifacts")

    notion_result = ""
    if args.notion_sync:
        token = os.getenv("NOTION_TOKEN", "").strip()
        parent = os.getenv("NOTION_PARENT_PAGE_ID", "").strip()
        if not token or not parent:
            raise SystemExit("NOTION_TOKEN / NOTION_PARENT_PAGE_ID env vars are required for --notion-sync")

        nc = NotionClient(NotionConfig(token=token, parent_page_id=parent))
        nc.ensure_parent_access()
        db_id = nc.ensure_index_db()
        date_iso = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}"

        rows: List[Tuple[str, str, str]] = []
        for f in sorted(interactive_dir.glob("*.html")):
            rows.append(("interactive", f.name, f"{pages_url}{args.date}/interactive/{f.name}"))
        for f in sorted(data_dir.glob("*.csv")):
            rows.append(("data", f.name, f"https://raw.githubusercontent.com/yule-1/ai/main/{args.date}/data/{f.name}"))
        for f in sorted(charts_dir.glob("*.png")):
            rows.append(("chart", f.name, f"https://raw.githubusercontent.com/yule-1/ai/main/{args.date}/charts/{f.name}"))

        created_rows = nc.add_index_rows(db_id, date_iso, rows)
        embed_url = nc.create_embed_page(f"{date_iso} Interactive + Charts + Data", rows)
        notion_result = f"db_id={db_id}, created_rows={created_rows}, embed_page={embed_url}"

    print(f"date={args.date}")
    print(f"interactive_created={len(created)}")
    for f in created:
        print(f" - {f.relative_to(repo)}")
    print(f"github_pages={pages_url}")
    if notion_result:
        print(f"notion={notion_result}")


if __name__ == "__main__":
    main()
