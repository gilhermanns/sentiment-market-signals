import pandas as pd

from sentiment_signals.dashboard import build_dashboard_html


def test_dashboard_includes_headline_text_and_handles_no_headline_days():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"Close": [100.0, 102.0, 101.0]}, index=dates)
    headlines = pd.DataFrame(
        {
            "date": [dates[1]],
            "ticker": ["AAPL"],
            "text": ["Apple shares surged after record profit"],
            "lexicon_score": [0.8],
        }
    )
    html_out = build_dashboard_html(
        "AAPL", prices, headlines, str(dates[0].date()), str(dates[-1].date())
    )
    assert "Apple shares surged after record profit" in html_out
    assert "no headlines" in html_out
    assert "AAPL" in html_out


def test_dashboard_escapes_html_in_headline_text():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    prices = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)
    headlines = pd.DataFrame(
        {
            "date": [dates[0]],
            "ticker": ["AAPL"],
            "text": ["<script>alert(1)</script>"],
            "lexicon_score": [0.0],
        }
    )
    html_out = build_dashboard_html(
        "AAPL", prices, headlines, str(dates[0].date()), str(dates[-1].date())
    )
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out
