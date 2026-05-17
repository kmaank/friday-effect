import warnings
warnings.filterwarnings("ignore")

from itertools import combinations

import io
from flask import Flask, render_template, jsonify, request, send_file
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

INDEXES = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "Dow Jones":    "^DJI",
    "Russell 2000": "^RUT",
    "FTSE 100":     "^FTSE",
    "DAX":          "^GDAXI",
    "Nikkei 225":   "^N225",
    "Hang Seng":    "^HSI",
    "CAC 40":       "^FCHI",
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def build_df(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = raw["Close"].squeeze()
    close.index = pd.to_datetime(close.index)
    df = pd.DataFrame({"Close": close})
    df["Return"]  = df["Close"].pct_change() * 100
    df["DayName"] = df.index.day_name()
    df["Year"]    = df.index.year
    df = df.dropna()
    df = df[df["DayName"].isin(DAYS)]
    return df


def box_stats(arr):
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo  = float(np.min(arr[arr >= q1 - 1.5 * iqr])) if arr.size else float(q1)
    hi  = float(np.max(arr[arr <= q3 + 1.5 * iqr])) if arr.size else float(q3)
    return {"q1": float(q1), "median": float(med), "q3": float(q3),
            "whisker_low": lo, "whisker_high": hi}


@app.route("/")
def index():
    return render_template("index.html", indexes=INDEXES)


@app.route("/api/analysis")
def analysis():
    ticker     = request.args.get("ticker", "^GSPC")
    start_year = int(request.args.get("start", 2010))
    end_year   = int(request.args.get("end",   2025))
    start      = f"{start_year}-01-01"
    end        = f"{end_year}-12-31"

    try:
        df = build_df(ticker, start, end)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # ── Descriptive stats per day ──────────────────────────────────────────
    desc = {}
    for day in DAYS:
        arr = df.loc[df["DayName"] == day, "Return"].values
        if len(arr) == 0:
            continue
        bs = box_stats(arr)
        desc[day] = {
            "n":       int(len(arr)),
            "mean":    float(arr.mean()),
            "median":  float(np.median(arr)),
            "std":     float(arr.std()),
            "neg_pct": float((arr < 0).mean() * 100),
            **bs,
            # downsample outliers for violin (max 500 pts each)
            "sample":  arr[np.random.choice(len(arr), min(len(arr), 500), replace=False)].tolist(),
        }

    fri    = df.loc[df["DayName"] == "Friday",  "Return"].values
    nonfri = df.loc[df["DayName"] != "Friday",  "Return"].values

    # ── SET 1: Welch t-test ────────────────────────────────────────────────
    t_stat, p_two = stats.ttest_ind(fri, nonfri, equal_var=False)
    p_one_t = float(p_two / 2 if t_stat < 0 else 1 - p_two / 2)

    # ── SET 2: One-way ANOVA ───────────────────────────────────────────────
    groups  = [df.loc[df["DayName"] == d, "Return"].values for d in DAYS]
    f_stat, p_anova = stats.f_oneway(*groups)

    # ── SET 3: Proportion z-test ───────────────────────────────────────────
    n1, n2 = len(fri), len(nonfri)
    p1, p2 = (fri < 0).mean(), (nonfri < 0).mean()
    p_pool = ((fri < 0).sum() + (nonfri < 0).sum()) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_stat = float((p1 - p2) / se) if se > 0 else 0.0
    p_one_z = float(1 - stats.norm.cdf(z_stat))

    # ── Post-hoc pairwise t-tests (Bonferroni) ────────────────────────────
    pairwise = []
    for d1, d2 in combinations(DAYS, 2):
        g1 = df.loc[df["DayName"] == d1, "Return"].values
        g2 = df.loc[df["DayName"] == d2, "Return"].values
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        p_bonf = min(float(p) * 10, 1.0)
        pairwise.append({"pair": f"{d1} vs {d2}", "t": float(t),
                         "p_raw": float(p), "p_bonf": p_bonf,
                         "sig": p_bonf < 0.05})

    # ── Histogram (Friday vs non-Friday) ──────────────────────────────────
    bins = np.linspace(-5, 5, 55)
    centers = ((bins[:-1] + bins[1:]) / 2).tolist()
    fri_h,  _ = np.histogram(fri,    bins=bins, density=True)
    non_h,  _ = np.histogram(nonfri, bins=bins, density=True)

    # ── Yearly mean return by weekday (heatmap) ───────────────────────────
    years  = sorted(df["Year"].unique().tolist())
    heat   = {day: [] for day in DAYS}
    for yr in years:
        yr_df = df[df["Year"] == yr]
        for day in DAYS:
            vals = yr_df.loc[yr_df["DayName"] == day, "Return"].values
            heat[day].append(float(vals.mean()) if len(vals) else None)

    # ── Cumulative Friday vs non-Friday return ────────────────────────────
    # Both series share the FULL date axis. Each line only increments on its
    # own days (0-contribution on the other days) so the x-axes align.
    df_sorted = df.sort_index()
    fri_inc   = df_sorted["Return"].where(df_sorted["DayName"] == "Friday",  0.0)
    non_inc   = df_sorted["Return"].where(df_sorted["DayName"] != "Friday",  0.0)
    full_dates = [str(d.date()) for d in df_sorted.index]

    return jsonify({
        "status":      "ok",
        "ticker":      ticker,
        "date_range":  f"{df.index.min().date()} → {df.index.max().date()}",
        "n_days":      int(len(df)),
        "descriptive": desc,
        "tests": {
            "ttest":      {"statistic": float(t_stat),  "p_one": p_one_t,  "reject": p_one_t  < 0.05},
            "anova":      {"statistic": float(f_stat),  "p_val": float(p_anova), "reject": float(p_anova) < 0.05},
            "proportion": {"statistic": z_stat, "p_one": p_one_z, "reject": p_one_z < 0.05,
                           "p_fri_neg": float(p1*100), "p_non_neg": float(p2*100)},
        },
        "pairwise": pairwise,
        "histogram": {"bins": centers, "friday": fri_h.tolist(), "nonfriday": non_h.tolist()},
        "heatmap":   {"years": [int(y) for y in years], "days": DAYS, "data": heat},
        "cumulative": {
            "dates":     full_dates,
            "fri_vals":  fri_inc.cumsum().values.tolist(),
            "non_vals":  non_inc.cumsum().values.tolist(),
        },
    })


@app.route("/api/download")
def download():
    ticker     = request.args.get("ticker", "^GSPC")
    start_year = int(request.args.get("start", 2010))
    end_year   = int(request.args.get("end",   2025))
    fmt        = request.args.get("fmt", "csv")   # "csv" or "xlsx"

    # Friendly index name for the filename
    index_name = next((k for k, v in INDEXES.items() if v == ticker), ticker)
    safe_name  = index_name.replace(" ", "_").replace("/", "-")
    filename   = f"{safe_name}_{start_year}-{end_year}.{fmt}"

    try:
        df = build_df(ticker, f"{start_year}-01-01", f"{end_year}-12-31")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    export = pd.DataFrame({
        "Date":              df.index.date,
        "Close":             df["Close"].round(4),
        "Daily_Return_Pct":  df["Return"].round(6),
        "Day_of_Week":       df["DayName"],
        "Year":              df["Year"],
        "Is_Friday":         (df["DayName"] == "Friday").map({True: "Yes", False: "No"}),
    })

    buf = io.BytesIO()
    if fmt == "xlsx":
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export.to_excel(writer, index=False, sheet_name=f"{safe_name}")
            ws = writer.sheets[safe_name]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 18
        buf.seek(0)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        buf.write(export.to_csv(index=False).encode())
        buf.seek(0)
        mime = "text/csv"

    return send_file(buf, mimetype=mime,
                     as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
