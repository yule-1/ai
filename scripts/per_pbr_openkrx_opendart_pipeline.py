#!/usr/bin/env python3
"""
Build annual year-end PER/PBR dataset + estimates + charts for KRX stocks
using OpenDART + OpenKRX + (optional) FnGuide consensus table.

Example:
  python3 scripts/per_pbr_openkrx_opendart_pipeline.py \
    --date 20260222 --ticker 005930 --name 삼성전자 --dart-key $DART --krx-key $KRX

Outputs under ai/YYYYMMDD:
- data/*
- charts/*
- report/*
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams


def setup_font() -> None:
    fp = Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
    if fp.exists():
        rcParams['font.family'] = fm.FontProperties(fname=str(fp)).get_name()
    rcParams['axes.unicode_minus'] = False


def get_corp_code(dart_key: str, ticker: str) -> str:
    content = requests.get(f'https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={dart_key}', timeout=30).content
    zf = zipfile.ZipFile(io.BytesIO(content))
    root = ET.fromstring(zf.read(zf.namelist()[0]))
    for li in root.findall('list'):
        if li.findtext('stock_code') == ticker:
            return li.findtext('corp_code')
    raise RuntimeError(f'corp_code not found for {ticker}')


def krx_close(krx_key: str, ticker: str, yyyymmdd: str):
    arr = requests.get(
        'https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd',
        headers={'AUTH_KEY': krx_key},
        params={'basDd': yyyymmdd},
        timeout=30,
    ).json().get('OutBlock_1', [])
    for x in arr:
        if x.get('ISU_CD') == ticker:
            return float(str(x.get('TDD_CLSPRC', '0')).replace(',', ''))
    return None


def year_end_close(krx_key: str, ticker: str, year: int):
    d = dt.date(year, 12, 31)
    for i in range(15):
        ds = (d - dt.timedelta(days=i)).strftime('%Y%m%d')
        p = krx_close(krx_key, ticker, ds)
        if p is not None:
            return ds, p
    return None, None


def dart_fin_rows(dart_key: str, corp_code: str, year: int):
    for fs in ('CFS', 'OFS'):
        j = requests.get(
            'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json',
            params={
                'crtfc_key': dart_key,
                'corp_code': corp_code,
                'bsns_year': str(year),
                'reprt_code': '11011',
                'fs_div': fs,
            },
            timeout=30,
        ).json()
        if j.get('status') == '000':
            return j.get('list', [])
    return []


def extract_eps_equity(rows):
    eps = None
    eq = None
    for r in rows:
        an = (r.get('account_nm') or '')
        aid = (r.get('account_id') or '').lower()
        v = r.get('thstrm_amount')
        if v is None:
            continue
        try:
            val = float(str(v).replace(',', ''))
        except Exception:
            continue
        if eps is None and ('기본주당이익' in an or aid.endswith('basicearningslosspershare')):
            eps = val
        if eq is None and ('지배기업 소유주지분' in an or aid.endswith('equityattributabletoownersofparent')):
            eq = val
    return eps, eq


def extract_shares(dart_key: str, corp_code: str, year: int):
    j = requests.get(
        'https://opendart.fss.or.kr/api/stockTotqySttus.json',
        params={
            'crtfc_key': dart_key,
            'corp_code': corp_code,
            'bsns_year': str(year),
            'reprt_code': '11011',
        },
        timeout=30,
    ).json()
    if j.get('status') != '000':
        return None
    s = 0.0
    for r in j.get('list', []):
        n = r.get('istc_totqy')
        if not n:
            continue
        try:
            s += float(str(n).replace(',', ''))
        except Exception:
            pass
    return s if s > 0 else None


def fetch_fnguide_consensus(ticker: str):
    ua = {'User-Agent': 'Mozilla/5.0'}
    main_url = f'https://comp.fnguide.com/SVO2/asp/SVD_Main.asp?pGB=1&gicode=A{ticker}&cID=&MenuYn=Y&ReportGB=&NewMenuID=11&stkGb=701'
    inv_url = f'https://comp.fnguide.com/SVO2/asp/SVD_Invest.asp?pGB=1&gicode=A{ticker}&cID=&MenuYn=Y&ReportGB=D&NewMenuID=105&stkGb=701'

    main_tables = pd.read_html(StringIO(requests.get(main_url, headers=ua, timeout=30).text))
    inv_tables = pd.read_html(StringIO(requests.get(inv_url, headers=ua, timeout=30).text))

    cons_target = cons_eps = cons_per = analysts = None
    for t in main_tables:
        txt = ' '.join(map(str, t.columns)) + ' ' + ' '.join(map(str, t.head(2).values.flatten()))
        if '목표주가' in txt and '추정기관수' in txt:
            row = t.iloc[-1]
            for c in t.columns:
                cn, v = str(c), str(row[c])
                try:
                    if '목표주가' in cn:
                        cons_target = float(v.replace(',', ''))
                    if cn.strip() == 'EPS' or ' EPS' in cn:
                        cons_eps = float(v.replace(',', ''))
                    if cn.strip() == 'PER' or ' PER' in cn:
                        cons_per = float(v.replace(',', ''))
                    if '추정기관수' in cn:
                        analysts = int(float(v))
                except Exception:
                    pass
            if cons_target is not None:
                break

    eps25 = bps25 = per25 = pbr25 = None
    for t in inv_tables:
        txt = ' '.join(map(str, t.columns)) + ' ' + ' '.join(map(str, t.iloc[:, 0].astype(str).tolist()))
        if '2025/12' in txt and 'PER' in txt and 'PBR' in txt:
            cols = [str(c[-1] if isinstance(c, tuple) else c) for c in t.columns]
            yidx = max(range(len(cols)), key=lambda i: 1 if '2025/12' in cols[i] else 0)
            for _, r in t.iterrows():
                k = str(r.iloc[0])
                try:
                    vv = float(str(r.iloc[yidx]).replace(',', ''))
                except Exception:
                    continue
                if k.startswith('EPS'):
                    eps25 = vv
                if k.startswith('BPS'):
                    bps25 = vv
                if k.startswith('PER'):
                    per25 = vv
                if k.startswith('PBR'):
                    pbr25 = vv
            break

    return {
        'cons_target': cons_target,
        'cons_eps': cons_eps,
        'cons_per': cons_per,
        'analysts': analysts,
        'eps25': eps25,
        'bps25': bps25,
        'per25': per25,
        'pbr25': pbr25,
    }


def main():
    setup_font()

    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYYMMDD')
    ap.add_argument('--ticker', required=True, help='KRX short code, e.g. 005930')
    ap.add_argument('--name', required=True)
    ap.add_argument('--dart-key', required=True)
    ap.add_argument('--krx-key', required=True)
    ap.add_argument('--start-year', type=int, default=1990)
    ap.add_argument('--end-year', type=int, default=2025)
    ap.add_argument('--repo-root', default='/Users/nyang/.openclaw/workspace/ai')
    args = ap.parse_args()

    out = Path(args.repo_root) / args.date
    (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'charts').mkdir(parents=True, exist_ok=True)
    (out / 'report').mkdir(parents=True, exist_ok=True)

    corp = get_corp_code(args.dart_key, args.ticker)

    rows = []
    for y in range(args.start_year, args.end_year + 1):
        d, p = year_end_close(args.krx_key, args.ticker, y)
        fin_rows = dart_fin_rows(args.dart_key, corp, y)
        eps, eq = extract_eps_equity(fin_rows)
        sh = extract_shares(args.dart_key, corp, y)
        bps = (eq / sh) if (eq and sh) else None
        per = (p / eps) if (p and eps and eps > 0) else None
        pbr = (p / bps) if (p and bps and bps > 0) else None
        rows.append({'year': y, 'year_end_date': d, 'price_close': p, 'eps': eps, 'bps': bps, 'per': per, 'pbr': pbr})

    actual = pd.DataFrame(rows)

    cons = fetch_fnguide_consensus(args.ticker)
    _, price25 = year_end_close(args.krx_key, args.ticker, 2025)
    row25 = actual[actual['year'] == 2025].iloc[0]

    eps25 = cons['eps25'] if cons['eps25'] is not None else row25['eps']
    bps25 = cons['bps25'] if cons['bps25'] is not None else row25['bps']
    per25 = cons['per25'] if cons['per25'] is not None else row25['per']
    pbr25 = cons['pbr25'] if cons['pbr25'] is not None else row25['pbr']

    valid = actual.dropna(subset=['per', 'pbr', 'eps', 'bps'])
    if cons['cons_target'] is None or cons['cons_eps'] is None:
        rec = valid[valid['year'] >= 2016]
        base_per = float(rec['per'].median())
        cons_eps = float((eps25 or valid['eps'].iloc[-1]) * 1.12)
        cons_target = cons_eps * base_per
        cons_per = base_per
        analysts = 0
    else:
        cons_target, cons_eps, cons_per, analysts = cons['cons_target'], cons['cons_eps'], cons['cons_per'], cons['analysts']

    per26 = cons_target / cons_eps

    bps_series = actual.dropna(subset=['bps'])[['year', 'bps']]
    br = bps_series[bps_series['year'] >= 2016]
    if len(br) >= 2:
        y0, y1 = int(br.iloc[0]['year']), int(br.iloc[-1]['year'])
        g = ((br.iloc[-1]['bps'] / br.iloc[0]['bps']) ** (1 / (y1 - y0)) - 1) if y1 > y0 else 0.08
    else:
        g = 0.08
    bps26 = (bps25 if bps25 else br.iloc[-1]['bps']) * (1 + g)
    pbr26 = cons_target / bps26

    scenarios = []
    for name, eg, per_ass, bg in [
        ('보수(Bear)', 0.05, max(7.5, per26 - 1.5), max(0.05, g - 0.02)),
        ('기준(Base)', 0.12, per26, max(0.06, g)),
        ('낙관(Bull)', 0.20, per26 + 1.2, g + 0.02),
    ]:
        eps27 = cons_eps * (1 + eg)
        bps27 = bps26 * (1 + bg)
        price27 = eps27 * per_ass
        pbr27 = price27 / bps27
        scenarios.append({
            'scenario': name,
            'eps_2027_est': round(eps27, 2),
            'bps_2027_est': round(bps27, 2),
            'per_2027_est': round(per_ass, 2),
            'price_2027_est': round(price27, 0),
            'pbr_2027_est': round(pbr27, 2),
        })
    S = pd.DataFrame(scenarios)
    base_row = S[S['scenario'] == '기준(Base)'].iloc[0]

    upd = pd.DataFrame([
        {'year': 2025, 'type': '잠정/연말(실제치)', 'price': round(price25, 0) if price25 else None, 'eps': round(eps25, 2) if eps25 else None,
         'bps': round(bps25, 2) if bps25 else None, 'per': round(per25, 2) if per25 else None, 'pbr': round(pbr25, 2) if pbr25 else None,
         'note': 'FnGuide/원천데이터'},
        {'year': 2026, 'type': '예상치(컨센서스 연동)', 'price': round(cons_target, 0), 'eps': round(cons_eps, 2),
         'bps': round(bps26, 2), 'per': round(per26, 2), 'pbr': round(pbr26, 2), 'note': f'추정기관 {analysts}개'},
        {'year': 2027, 'type': '예상치(기준시나리오)', 'price': float(base_row['price_2027_est']), 'eps': float(base_row['eps_2027_est']),
         'bps': float(base_row['bps_2027_est']), 'per': float(base_row['per_2027_est']), 'pbr': float(base_row['pbr_2027_est']),
         'note': '3시나리오 중 기준(Base)'},
    ])

    plot_df = actual[actual['year'] <= 2024][['year', 'per', 'pbr']].copy()
    plot_df = pd.concat([plot_df, upd[['year', 'per', 'pbr']]], ignore_index=True).dropna().sort_values('year')

    avg = actual[(actual['year'] >= 2016) & (actual['year'] <= 2024)].dropna(subset=['per', 'pbr'])
    per_avg = float(avg['per'].mean()) if len(avg) else 0
    pbr_avg = float(avg['pbr'].mean()) if len(avg) else 0

    tag = 'skhynix' if args.ticker == '000660' else 'samsung'
    actual_path = out / 'data' / f'{tag}_yearend_per_pbr_1990_2025_opendart_openkrx.csv'
    upd_path = out / 'data' / f'{tag}_2025_provisional_2026_2027_estimates_per_pbr.csv'
    sc_path = out / 'data' / f'{tag}_2027_per_pbr_three_scenarios.csv'
    merged_path = out / 'data' / f'{tag}_yearend_per_pbr_1990_2027_with_estimates.csv'

    actual.to_csv(actual_path, index=False, encoding='utf-8-sig')
    upd.to_csv(upd_path, index=False, encoding='utf-8-sig')
    S.to_csv(sc_path, index=False, encoding='utf-8-sig')
    plot_df.to_csv(merged_path, index=False, encoding='utf-8-sig')

    BG, PANEL, GRID, TXT, SUB = '#0f1117', '#151823', '#2a3040', '#e8ecf3', '#9aa4b2'

    def style(ax, title, ylabel):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(axis='y', color=GRID, alpha=0.55, linewidth=0.8)
        ax.tick_params(colors=SUB, labelsize=9)
        ax.set_title(title, color=TXT, fontsize=14, pad=10)
        ax.set_ylabel(ylabel, color=SUB)

    colors = ['#7C83FD' if y <= 2024 else '#37B4FF' if y == 2025 else '#00C2A8' if y == 2026 else '#F6C90E' for y in plot_df['year']]

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG)
    ax.bar(plot_df['year'].astype(str), plot_df['per'], color=colors)
    ax.axhline(per_avg, color='#FF7A59', linestyle='--', linewidth=1.8, label=f'2016~2024 평균 PER: {per_avg:.2f}')
    style(ax, f'{args.name} 연도별 PER (2025 잠정/2026·2027 추정 포함)', 'PER (배)')
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / 'charts' / f'{tag}_yearly_per_bar_with_2025_2027_and_avgline.png', dpi=180, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG)
    ax.bar(plot_df['year'].astype(str), plot_df['pbr'], color=colors)
    ax.axhline(pbr_avg, color='#FF7A59', linestyle='--', linewidth=1.8, label=f'2016~2024 평균 PBR: {pbr_avg:.2f}')
    style(ax, f'{args.name} 연도별 PBR (2025 잠정/2026·2027 추정 포함)', 'PBR (배)')
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / 'charts' / f'{tag}_yearly_pbr_bar_with_2025_2027_and_avgline.png', dpi=180, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.bar(S['scenario'], S['per_2027_est'], color=['#FF7A59', '#7C83FD', '#00C2A8'])
    style(ax, f'{args.name} 2027E PER 시나리오', 'PER (배)')
    plt.tight_layout()
    plt.savefig(out / 'charts' / f'{tag}_2027_per_scenarios_bar.png', dpi=180, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.bar(S['scenario'], S['pbr_2027_est'], color=['#FF7A59', '#7C83FD', '#00C2A8'])
    style(ax, f'{args.name} 2027E PBR 시나리오', 'PBR (배)')
    plt.tight_layout()
    plt.savefig(out / 'charts' / f'{tag}_2027_pbr_scenarios_bar.png', dpi=180, facecolor=BG)
    plt.close(fig)

    with open(out / 'report' / f'{tag}_2025_2027_per_pbr_update_with_analyst_consensus_ko.md', 'w', encoding='utf-8') as f:
        f.write(f'# {args.name} 2025 잠정치 및 2026/2027 예상 PER·PBR\n\n')
        f.write(f'- 최근 컨센서스(추정기관 {analysts}개): 목표주가 {cons_target:,.0f}원, EPS {cons_eps:,.0f}원, PER {(cons_per if cons_per else per26):.2f}배\n')
        f.write(f'- 2016~2024 평균: PER {per_avg:.2f}, PBR {pbr_avg:.2f}\n\n')
        f.write('## 2025~2027 요약\n\n')
        f.write(upd.to_string(index=False))
        f.write('\n\n## 2027 시나리오\n\n')
        f.write(S.to_string(index=False))

    print(json.dumps({
        'ticker': args.ticker,
        'name': args.name,
        'consensus': {'target': cons_target, 'eps': cons_eps, 'per': cons_per, 'analysts': analysts},
        'avg_2016_2024': {'per': per_avg, 'pbr': pbr_avg},
        'outputs_root': str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
