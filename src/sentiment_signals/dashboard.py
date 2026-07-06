"""Static HTML dashboard: headlines next to price moves for a chosen ticker/date range."""
from __future__ import annotations

import html

import pandas as pd

_ROW_TEMPLATE = """
<tr class="{cls}">
  <td>{date}</td>
  <td class="ret">{ret}</td>
  <td class="close">{close}</td>
  <td class="sent">{sent}</td>
  <td class="headline">{headline}</td>
</tr>
"""

_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; background: #0b0d12; color: #e6e6e6; }}
h1 {{ font-size: 1.3rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #2a2e37; text-align: left; font-size: 0.9rem; }}
th {{ color: #9aa4b2; text-transform: uppercase; font-size: 0.75rem; }}
.pos {{ color: #4fd18b; }}
.neg {{ color: #ef6a6a; }}
.neutral-row {{ color: #cfd3da; }}
tr.pos-row {{ background: rgba(79, 209, 139, 0.07); }}
tr.neg-row {{ background: rgba(239, 106, 106, 0.07); }}
</style></head>
<body>
<h1>{title}</h1>
<p>{subtitle}</p>
<table>
<thead><tr><th>Date</th><th>Daily return</th><th>Close</th><th>Sentiment</th><th>Headline</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body></html>
"""


def build_dashboard_html(
    ticker: str,
    prices: pd.DataFrame,
    headlines: pd.DataFrame,
    start: str,
    end: str,
    score_col: str = "lexicon_score",
) -> str:
    """prices: DataFrame indexed by date with a 'Close' column (and daily return computed
    here). headlines: DataFrame with columns date, ticker, text, score_col; may have zero,
    one, or many rows per date."""
    window = prices.loc[start:end].copy()
    window["ret"] = window["Close"].pct_change()

    hl = headlines[headlines["ticker"] == ticker].copy()
    hl["date"] = pd.to_datetime(hl["date"])

    rows_html = []
    for date, row in window.iterrows():
        day_headlines = hl[hl["date"] == date]
        ret = row["ret"]
        ret_cls = "pos" if pd.notna(ret) and ret > 0 else ("neg" if pd.notna(ret) and ret < 0 else "")
        ret_str = f"{ret:+.2%}" if pd.notna(ret) else "-"
        close_str = f"{row['Close']:.2f}"

        if day_headlines.empty:
            row_cls = "neutral-row"
            rows_html.append(
                _ROW_TEMPLATE.format(
                    cls=row_cls, date=date.date(), ret=f'<span class="{ret_cls}">{ret_str}</span>',
                    close=close_str, sent="-", headline="<em>no headlines</em>",
                )
            )
            continue

        for _, hrow in day_headlines.iterrows():
            s = hrow[score_col]
            row_cls = "pos-row" if s > 0.1 else ("neg-row" if s < -0.1 else "neutral-row")
            rows_html.append(
                _ROW_TEMPLATE.format(
                    cls=row_cls, date=date.date(), ret=f'<span class="{ret_cls}">{ret_str}</span>',
                    close=close_str, sent=f"{s:+.2f}", headline=html.escape(hrow["text"]),
                )
            )

    return _PAGE_TEMPLATE.format(
        title=f"{ticker} headlines vs price moves",
        subtitle=f"{start} to {end} &mdash; synthetic headlines aligned to real {ticker} daily closes",
        rows="\n".join(rows_html),
    )
