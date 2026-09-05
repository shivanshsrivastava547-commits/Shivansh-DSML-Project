#!/usr/bin/env python3
"""Build the mini-project notebook, figures, and PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from PIL import Image as PILImage
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "2019_nC0v_20200121_20200126 - SUMMARY.csv"
FIG_DIR = ROOT / "figures"
NOTEBOOK_PATH = ROOT / "MiniProject_COVID19_nCoV_EDA.ipynb"
PDF_PATH = ROOT / "MiniProject_COVID19_nCoV_EDA_Report.pdf"
CLEANED_PATH = ROOT / "cleaned_ncov_summary.csv"

STUDENT = "Shivansh Srivastava"
REG_NO = "RA2411056030011"

NAVY = "#0F2C59"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
TEAL = "#0d9488"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_FILE)
    df = raw.copy()
    df.columns = [
        "province",
        "country",
        "last_update",
        "confirmed",
        "suspected",
        "recovered",
        "deaths",
    ]
    df["country"] = df["country"].astype(str).str.strip()
    df["province"] = df["province"].fillna("").astype(str).str.strip()
    name_fix = {
        "Mainland China": "Mainland China",
        "China": "Mainland China",
    }
    df["country"] = df["country"].replace(name_fix)
    for col in ["confirmed", "suspected", "recovered", "deaths"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["last_update"] = pd.to_datetime(df["last_update"], format="mixed")
    df["date"] = df["last_update"].dt.normalize()
    df["location"] = np.where(
        df["province"] != "", df["province"] + ", " + df["country"], df["country"]
    )
    df["is_china"] = df["country"].eq("Mainland China")
    df["active"] = (df["confirmed"] - df["deaths"] - df["recovered"]).clip(lower=0)
    df = df.sort_values(["date", "last_update", "country", "province"]).reset_index(drop=True)

    daily_rows = []
    for d, g in df.groupby("date"):
        last = g.sort_values("last_update").drop_duplicates(
            ["province", "country"], keep="last"
        )
        daily_rows.append(
            {
                "date": d,
                "confirmed": int(last["confirmed"].sum()),
                "deaths": int(last["deaths"].sum()),
                "recovered": int(last["recovered"].sum()),
                "suspected": int(last["suspected"].sum()),
                "active": int(last["active"].sum()),
                "locations": int(len(last)),
                "countries": int(last["country"].nunique()),
                "china_confirmed": int(last.loc[last["is_china"], "confirmed"].sum()),
                "row_confirmed": int(last.loc[~last["is_china"], "confirmed"].sum()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    daily["new_confirmed"] = daily["confirmed"].diff().fillna(daily["confirmed"]).astype(int)
    daily["new_deaths"] = daily["deaths"].diff().fillna(daily["deaths"]).astype(int)
    daily["recovery_rate"] = (daily["recovered"] / daily["confirmed"] * 100).round(2)
    daily["death_rate"] = (daily["deaths"] / daily["confirmed"] * 100).round(2)

    last_day = df["date"].max()
    latest = (
        df[df["date"] == last_day]
        .sort_values("last_update")
        .drop_duplicates(["province", "country"], keep="last")
        .copy()
    )
    latest["recovery_rate"] = np.where(
        latest["confirmed"] > 0,
        (latest["recovered"] / latest["confirmed"] * 100).round(2),
        0,
    )
    latest["death_rate"] = np.where(
        latest["confirmed"] > 0,
        (latest["deaths"] / latest["confirmed"] * 100).round(2),
        0,
    )
    latest = latest.sort_values("confirmed", ascending=False).reset_index(drop=True)

    by_country = (
        latest.groupby("country", as_index=False)[["confirmed", "deaths", "recovered", "active"]]
        .sum()
        .sort_values("confirmed", ascending=False)
        .reset_index(drop=True)
    )
    by_country["recovery_rate"] = np.where(
        by_country["confirmed"] > 0,
        (by_country["recovered"] / by_country["confirmed"] * 100).round(2),
        0,
    )
    by_country["death_rate"] = np.where(
        by_country["confirmed"] > 0,
        (by_country["deaths"] / by_country["confirmed"] * 100).round(2),
        0,
    )
    return df, daily, latest, by_country


def style_plots() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#cbd5e1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def savefig(name: str) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def make_figures(df: pd.DataFrame, daily: pd.DataFrame, latest: pd.DataFrame) -> dict[str, Path]:
    style_plots()
    figs: dict[str, Path] = {}
    top10 = latest.head(10)
    top5 = latest.head(5)["location"].tolist()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(daily["date"], daily["confirmed"], label="Confirmed", color=BLUE, lw=2.4)
    ax.plot(daily["date"], daily["recovered"], label="Recovered", color=GREEN, lw=2.4)
    ax.plot(daily["date"], daily["deaths"], label="Deaths", color=RED, lw=2.4)
    ax.set_title("Global: Cumulative 2019-nCoV Cases, 21–26 January 2020")
    ax.set_xlabel("Date")
    ax.set_ylabel("People")
    ax.legend()
    fig.autofmt_xdate()
    figs["c1"] = savefig("chart01_cumulative.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(daily["date"], daily["new_confirmed"], color=BLUE, width=0.7)
    ax.set_title("Global: Daily New Confirmed Cases")
    ax.set_xlabel("Date")
    ax.set_ylabel("New confirmed cases")
    fig.autofmt_xdate()
    figs["c2"] = savefig("chart02_daily_new.png")

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    sns.barplot(
        data=top10, y="location", x="confirmed", hue="location",
        palette="Blues_r", legend=False, ax=ax,
    )
    ax.set_title("Top 10 Locations by Confirmed Cases (26 January 2020)")
    ax.set_xlabel("Confirmed cases")
    ax.set_ylabel("")
    figs["c3"] = savefig("chart03_top10.png")

    rec = top10.sort_values("recovery_rate")
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    sns.barplot(
        data=rec, y="location", x="recovery_rate", hue="location",
        palette="Greens", legend=False, ax=ax,
    )
    ax.set_title("Recovery Rate (%) — Top 10 Worst-Hit Locations")
    ax.set_xlabel("Recovery rate (%)")
    ax.set_ylabel("")
    figs["c4"] = savefig("chart04_recovery.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for loc in top5:
        sub = (
            df[df["location"] == loc]
            .sort_values("last_update")
            .drop_duplicates("date", keep="last")
        )
        ax.plot(sub["date"], sub["confirmed"], label=loc, lw=2.2)
    ax.set_title("Confirmed Cases Over Time — Top 5 Locations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Confirmed cases")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    figs["c5"] = savefig("chart05_top5_trend.png")

    hubei = int(latest.loc[latest["province"] == "Hubei", "confirmed"].sum())
    other_china = int(latest.loc[latest["is_china"] & (latest["province"] != "Hubei"), "confirmed"].sum())
    rest = int(latest.loc[~latest["is_china"], "confirmed"].sum())
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.pie(
        [hubei, other_china, rest],
        labels=["Hubei", "Rest of Mainland China", "Rest of world"],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Blues_r", 3),
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Share of Confirmed Cases on 26 January 2020")
    figs["c6"] = savefig("chart06_share_pie.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(daily["date"], daily["recovery_rate"], color=GREEN, lw=2.4, label="Recovery rate")
    ax.plot(daily["date"], daily["death_rate"], color=RED, lw=2.4, label="Death rate")
    ax.set_title("Global Recovery Rate vs Death Rate Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    fig.autofmt_xdate()
    figs["c7"] = savefig("chart07_rates.png")

    sizeable = latest[latest["confirmed"] >= 1]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    sns.scatterplot(
        data=sizeable, x="confirmed", y="deaths", size="confirmed",
        sizes=(40, 700), hue="confirmed", palette="Reds", legend=False, ax=ax,
    )
    for _, r in sizeable.nlargest(6, "confirmed").iterrows():
        ax.annotate(
            r["location"].split(",")[0],
            (r["confirmed"], r["deaths"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("Deaths vs Confirmed Cases by Location (bubble = case count)")
    ax.set_xlabel("Confirmed cases")
    ax.set_ylabel("Deaths")
    figs["c8"] = savefig("chart08_scatter.png")

    corr_cols = ["confirmed", "deaths", "recovered", "new_confirmed", "recovery_rate", "death_rate"]
    corr = daily[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Global 2019-nCoV Metrics")
    figs["c9"] = savefig("chart09_heatmap.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    box = df.copy()
    box["day"] = box["date"].dt.strftime("%d Jan")
    sns.boxplot(
        data=box[box["confirmed"] > 0], x="day", y="confirmed",
        hue="day", palette="Blues", legend=False, ax=ax,
    )
    ax.set_title("Spread of Location-Level Confirmed Counts by Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Confirmed cases (per location)")
    figs["c10"] = savefig("chart10_boxplot.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(daily))
    w = 0.38
    ax.bar(x - w / 2, daily["china_confirmed"], width=w, color=BLUE, label="Mainland China")
    ax.bar(x + w / 2, daily["row_confirmed"], width=w, color=ORANGE, label="Rest of world")
    ax.set_xticks(x, [d.strftime("%d Jan") for d in daily["date"]])
    ax.set_title("Mainland China vs Rest of World — Confirmed Cases")
    ax.set_xlabel("Date")
    ax.set_ylabel("Confirmed cases")
    ax.legend()
    figs["c11"] = savefig("chart11_china_vs_world.png")

    intl = latest[~latest["is_china"]].sort_values("confirmed", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    sns.barplot(
        data=intl, y="location", x="confirmed", hue="location",
        palette="Oranges_r", legend=False, ax=ax,
    )
    ax.set_title("International Spread Outside Mainland China (26 January 2020)")
    ax.set_xlabel("Confirmed cases")
    ax.set_ylabel("")
    figs["c12"] = savefig("chart12_international.png")
    return figs


def notebook_source() -> list[tuple[str, str]]:
    """Return (cell_type, source) pairs for the student notebook."""
    return [
        (
            "markdown",
            f"""# Mini Project — Exploratory Data Analysis on the Early 2019-nCoV Outbreak

**Student:** {STUDENT}  
**Registration Number:** {REG_NO}

**Goal:** Uncover country-wise and province-wise trends, recovery rates, and the first international spillover of the 2019 novel coronavirus, using nothing but the tools from Module 1: NumPy, Pandas, and Matplotlib/Seaborn.

**Data source** (local CSV, no extra download needed):
- **Case data:** `2019_nC0v_20200121_20200126 - SUMMARY.csv` — aggregated Johns Hopkins CSSE daily situation reports covering **21–26 January 2020**.
- Columns: `Province/State`, `Country`, `Date last updated`, `Confirmed`, `Suspected`, `Recovered`, `Deaths`.

**Workflow:** Load → Inspect → Clean → Explore (groupby/pivot) → Visualise (12 charts) → Summarise findings.

> **Note on the data window:** this file is an *early-outbreak* snapshot. Vaccines did not exist yet, testing was limited, and almost every confirmed case sat inside Mainland China — especially Hubei province. The interesting story is how fast the counts grew in six days, and which other countries had already reported imported cases.""",
        ),
        ("markdown", "## 1. Setup"),
        (
            "code",
            """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 20)

DATA_FILE = "2019_nC0v_20200121_20200126 - SUMMARY.csv"
print("Ready.")""",
        ),
        (
            "markdown",
            """## 2. Load the Data

We read the summary CSV with `pd.read_csv()` — exactly the File I/O skill from Module 1. The file name has spaces, so we keep it in quotes.""",
        ),
        (
            "code",
            """raw = pd.read_csv(DATA_FILE)

print("Shape:", raw.shape)
print("Columns:", list(raw.columns))
raw.head()""",
        ),
        (
            "markdown",
            """## 3. Inspect Before Touching Anything

Always check shape, dtypes, and missing values first — this tells you exactly what cleaning is needed.""",
        ),
        ("code", "raw.dtypes"),
        ("code", "raw.isna().sum()"),
        (
            "code",
            """print("Number of unique country names:", raw["Country"].nunique())
sorted(raw["Country"].unique())""",
        ),
        (
            "code",
            """print("Number of unique Province/State values:", raw["Province/State"].nunique(dropna=True))
raw["Date last updated"].unique()[:12]""",
        ),
        (
            "markdown",
            """Several problems jump out immediately:

1. **Numeric columns have missing values** — empty cells in `Confirmed`, `Suspected`, `Recovered`, and `Deaths` mean "not reported", which we should treat as 0 for counts.
2. **Country names are messy**: `Singapore ` and `Malaysia ` have trailing spaces, so they would be counted as different countries if we group naively.
3. **`Province/State` is blank** for countries that report nationally (Thailand, Japan, France, …).
4. **Dates are stored as text** in mixed formats (`1/21/2020 10pm`, `1/22/20 12:00`, `1/24/2020  12am`).
5. Some early rows list **Brazil / Mexico / Colombia / Philippines with no confirmed cases** — they appear because they were being *watched*, not because they had an outbreak yet.

This is exactly the kind of real-world messiness we fix *before* doing any analysis, never after.""",
        ),
        ("markdown", "## 4. Clean the Data"),
        (
            "code",
            """df = raw.copy()

# Rename to clean, snake_case column names
df.columns = [
    "province", "country", "last_update",
    "confirmed", "suspected", "recovered", "deaths",
]

# Strip whitespace from names
df["country"] = df["country"].astype(str).str.strip()
df["province"] = df["province"].fillna("").astype(str).str.strip()

# Harmonise a couple of country aliases
df["country"] = df["country"].replace({"China": "Mainland China"})

# Fix data types — missing counts become 0
for col in ["confirmed", "suspected", "recovered", "deaths"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# Mixed timestamp formats (12am / 10pm / 23:00) parse with format="mixed"
df["last_update"] = pd.to_datetime(df["last_update"], format="mixed")
df["date"] = df["last_update"].dt.normalize()

# A readable location label, plus a China flag
df["location"] = np.where(df["province"] != "", df["province"] + ", " + df["country"], df["country"])
df["is_china"] = df["country"].eq("Mainland China")

# Derived column: active cases = confirmed - deaths - recovered
df["active"] = (df["confirmed"] - df["deaths"] - df["recovered"]).clip(lower=0)

df = df.sort_values(["date", "last_update", "country", "province"]).reset_index(drop=True)

print("Rows:", len(df))
print("Countries after cleaning:", df["country"].nunique())
print("Date range:", df["date"].min().date(), "→", df["date"].max().date())
df.dtypes""",
        ),
        ("code", 'df.isna().sum().sum()   # should be 0 — fully clean now'),
        (
            "code",
            """print("Clean country list:")
sorted(df["country"].unique())""",
        ),
        (
            "markdown",
            """## 5. Global-Level Summary (groupby)

The CSV is a stack of situation reports, so the same province appears many times. For a fair daily total we take the **last report of each location on each calendar day**, then `groupby` the date.""",
        ),
        (
            "code",
            """def last_snapshot(frame):
    \"\"\"Keep the latest report for every province/country pair.\"\"\"
    return (
        frame.sort_values("last_update")
             .drop_duplicates(["province", "country"], keep="last")
    )

daily_rows = []
for day, group in df.groupby("date"):
    last = last_snapshot(group)
    daily_rows.append({
        "date": day,
        "confirmed": int(last["confirmed"].sum()),
        "deaths": int(last["deaths"].sum()),
        "recovered": int(last["recovered"].sum()),
        "suspected": int(last["suspected"].sum()),
        "active": int(last["active"].sum()),
        "locations": int(len(last)),
        "countries": int(last["country"].nunique()),
        "china_confirmed": int(last.loc[last["is_china"], "confirmed"].sum()),
        "row_confirmed": int(last.loc[~last["is_china"], "confirmed"].sum()),
    })

daily = pd.DataFrame(daily_rows)
daily["new_confirmed"] = daily["confirmed"].diff().fillna(daily["confirmed"]).astype(int)
daily["recovery_rate"] = (daily["recovered"] / daily["confirmed"] * 100).round(2)
daily["death_rate"] = (daily["deaths"] / daily["confirmed"] * 100).round(2)
daily""",
        ),
        (
            "code",
            """latest_date = df["date"].max()
latest_row = daily.iloc[-1]

print(f"As of {latest_date.date()}:")
print(f"  Total confirmed : {latest_row['confirmed']:,}")
print(f"  Total recovered : {latest_row['recovered']:,}")
print(f"  Total deaths    : {latest_row['deaths']:,}")
print(f"  Suspected       : {latest_row['suspected']:,}")
print(f"  Recovery rate   : {latest_row['recovery_rate']}%")
print(f"  Death rate      : {latest_row['death_rate']}%")
print(f"  Countries       : {latest_row['countries']}")""",
        ),
        (
            "markdown",
            """## 6. Country / Province Snapshot (latest date)

**pivot_table**: confirmed cases by country and date, so we can see when each country's count moved.""",
        ),
        (
            "code",
            """latest = last_snapshot(df[df["date"] == latest_date]).copy()
latest["recovery_rate"] = np.where(
    latest["confirmed"] > 0,
    (latest["recovered"] / latest["confirmed"] * 100).round(2),
    0,
)
latest["death_rate"] = np.where(
    latest["confirmed"] > 0,
    (latest["deaths"] / latest["confirmed"] * 100).round(2),
    0,
)
latest = latest.sort_values("confirmed", ascending=False).reset_index(drop=True)

top10 = latest.head(10)
top10[["location", "confirmed", "deaths", "recovered", "recovery_rate", "death_rate"]]""",
        ),
        (
            "code",
            """by_country = (
    latest.groupby("country", as_index=False)[["confirmed", "deaths", "recovered"]]
          .sum()
          .sort_values("confirmed", ascending=False)
)
by_country""",
        ),
        (
            "code",
            """# Confirmed cases by country × date (locations outside Mainland China stay visible)
country_day = (
    df.sort_values("last_update")
      .drop_duplicates(["date", "province", "country"], keep="last")
)

pivot = country_day.pivot_table(
    values="confirmed",
    index="country",
    columns="date",
    aggfunc="sum",
    fill_value=0,
)
pivot.loc[by_country["country"].head(8)].rename(
    columns=lambda d: d.strftime("%Y-%m-%d")
)""",
        ),
        (
            "markdown",
            """## 7. Visualisations (12 charts)

We'll work through global trends, Hubei's dominance, and the first international cases — using both Matplotlib and Seaborn, matching the Module 1 handbook patterns.""",
        ),
        ("markdown", "### Chart 1 — Global cumulative trend"),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(daily["date"], daily["confirmed"], label="Confirmed", color="#2563eb", lw=2)
ax.plot(daily["date"], daily["recovered"], label="Recovered", color="#16a34a", lw=2)
ax.plot(daily["date"], daily["deaths"], label="Deaths", color="#dc2626", lw=2)
ax.set_title("Global: Cumulative 2019-nCoV Cases Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("People")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 2 — Daily new confirmed cases"),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(daily["date"], daily["new_confirmed"], color="#2563eb", width=0.7)
ax.set_title("Global: Daily New Confirmed Cases")
ax.set_xlabel("Date"); ax.set_ylabel("New cases")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 3 — Top 10 locations by total confirmed cases"),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10, y="location", x="confirmed", hue="location",
            palette="Blues_r", legend=False, ax=ax)
ax.set_title(f"Top 10 Locations by Confirmed Cases (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 4 — Recovery rate of the 10 worst-hit locations"),
        (
            "code",
            """top10_sorted = top10.sort_values("recovery_rate")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10_sorted, y="location", x="recovery_rate", hue="location",
            palette="Greens", legend=False, ax=ax)
ax.set_title("Recovery Rate (%) — Top 10 Worst-Hit Locations")
ax.set_xlabel("Recovery rate (%)"); ax.set_ylabel("")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 5 — Confirmed-case trend for the top 5 locations"),
        (
            "code",
            """top5_locations = top10["location"].head(5).tolist()

fig, ax = plt.subplots(figsize=(8, 4.5))
for loc in top5_locations:
    sub = (df[df["location"] == loc]
           .sort_values("last_update")
           .drop_duplicates("date", keep="last"))
    ax.plot(sub["date"], sub["confirmed"], label=loc, lw=2)
ax.set_title("Confirmed Cases Over Time — Top 5 Locations")
ax.set_xlabel("Date"); ax.set_ylabel("Confirmed cases")
ax.legend(fontsize=8)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 6 — Share of global cases (pie)"),
        (
            "code",
            """hubei = int(latest.loc[latest["province"] == "Hubei", "confirmed"].sum())
other_china = int(latest.loc[latest["is_china"] & (latest["province"] != "Hubei"), "confirmed"].sum())
rest = int(latest.loc[~latest["is_china"], "confirmed"].sum())

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.pie(
    [hubei, other_china, rest],
    labels=["Hubei", "Rest of Mainland China", "Rest of world"],
    autopct="%1.1f%%",
    startangle=90,
    colors=sns.color_palette("Blues_r", 3),
)
ax.set_title("Share of Total Confirmed Cases")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 7 — Global recovery-rate vs death-rate trend"),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(daily["date"], daily["recovery_rate"], color="#16a34a", lw=2, label="Recovery rate")
ax.plot(daily["date"], daily["death_rate"], color="#dc2626", lw=2, label="Death rate")
ax.set_title("Global Recovery Rate vs Death Rate Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Rate (%)")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 8 — Deaths vs confirmed cases by location (bubble = case count)"),
        (
            "code",
            """sizeable = latest[latest["confirmed"] >= 1]

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=sizeable, x="confirmed", y="deaths", size="confirmed",
                sizes=(40, 600), hue="confirmed", palette="Reds", legend=False, ax=ax)
for _, r in sizeable.nlargest(6, "confirmed").iterrows():
    ax.annotate(r["location"].split(",")[0], (r["confirmed"], r["deaths"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_title("Deaths vs Confirmed Cases by Location (bubble = case count)")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("Deaths")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 9 — Correlation heatmap of global metrics"),
        (
            "code",
            """corr_cols = ["confirmed", "deaths", "recovered", "new_confirmed", "recovery_rate", "death_rate"]
corr = daily[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Between Global 2019-nCoV Metrics")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 10 — Spread of location-level confirmed counts, by day (boxplot)"),
        (
            "code",
            """box = df.copy()
box["day"] = box["date"].dt.strftime("%d Jan")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=box[box["confirmed"] > 0], x="day", y="confirmed",
            hue="day", palette="Blues", legend=False, ax=ax)
ax.set_title("Spread of Location-Level Confirmed Counts by Day")
ax.set_xlabel("Day"); ax.set_ylabel("Confirmed cases")
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 11 — Mainland China vs the rest of the world"),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(daily))
w = 0.38
ax.bar(x - w/2, daily["china_confirmed"], width=w, color="#2563eb", label="Mainland China")
ax.bar(x + w/2, daily["row_confirmed"], width=w, color="#ea580c", label="Rest of world")
ax.set_xticks(x, [d.strftime("%d Jan") for d in daily["date"]])
ax.set_title("Mainland China vs Rest of World — Confirmed Cases")
ax.set_xlabel("Date"); ax.set_ylabel("Confirmed cases")
ax.legend()
plt.tight_layout()
plt.show()""",
        ),
        ("markdown", "### Chart 12 — International cases outside Mainland China"),
        (
            "code",
            """intl = latest[~latest["is_china"]].sort_values("confirmed", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=intl, y="location", x="confirmed", hue="location",
            palette="Oranges_r", legend=False, ax=ax)
ax.set_title(f"Cases Outside Mainland China (as of {latest_date.date()})")
ax.set_xlabel("Confirmed cases"); ax.set_ylabel("")
plt.tight_layout()
plt.show()""",
        ),
        (
            "markdown",
            """## 8. Save the Cleaned Dataset

One more File I/O practice rep — save the cleaned table so it can be reused without repeating the cleaning steps.""",
        ),
        (
            "code",
            """df.to_csv("cleaned_ncov_summary.csv", index=False)
print("Saved cleaned_ncov_summary.csv —", df.shape[0], "rows.")""",
        ),
        (
            "markdown",
            """## 9. Written Summary

**Coverage.** The dataset stacks Johns Hopkins CSSE situation reports from **21 January 2020 to 26 January 2020** — the first six published days of the 2019-nCoV outbreak. After cleaning trailing spaces and missing counts, we have reports for **19 countries** and about **40 Chinese provinces / administrative regions**.

**Global picture.** Confirmed cases rose from **332** on 21 January to **2,794** on 26 January — roughly an **8×** jump in six days. Recoveries lagged far behind (**54** people, a **1.93%** recovery rate) because almost every patient was still in hospital. Deaths rose from **6** to **80** (a **2.86%** crude death rate). Daily new cases accelerated: **+223 → +98 → +288 → +879 → +974**.

**Geographic concentration.** The outbreak was not evenly spread. **Hubei province alone held 1,423 confirmed cases (~51%)** on 26 January, and **Mainland China as a whole held 2,737 (~98%)**. Guangdong, Zhejiang, Henan, Chongqing and Hunan filled out the rest of the top six. Hubei also accounted for **76 of the 80 deaths**.

**Recovery was still tiny.** Even among the worst-hit provinces, recovery rates sat in the low single digits (Hubei ~3.1%). That is *not* a clinical failure — it is a timing effect. Six days is too short for most patients to be discharged, so the recovered column had barely started to move.

**International spillover had already started.** By 26 January, **14 countries/territories outside Mainland China** had confirmed cases, but the counts were still single-digit: Hong Kong (8), Thailand (8), Macau (6), the US (5), then Australia, Japan, Taiwan, Singapore and Malaysia on 4 each. These are imported / traveller cases, not community epidemics — yet.

**Caveats.** Early 2019-nCoV bulletins inherit every reporting quirk of a brand-new pathogen: missing numeric fields, trailing spaces in country names, mixed timestamp formats, and "watchlist" countries (Brazil, Mexico, Colombia, Philippines) that appear with **zero** confirmed cases. Recovery and death rates here are simple ratios of cumulative totals, not lag-adjusted outcome rates, so they should be read as directional trends.

**What this window cannot tell us.** India's first laboratory-confirmed case was reported on **30 January 2020**, after this file ends. Vaccination data does not exist for January 2020. The value of this EDA is the *opening chapter*: how a provincial pneumonia cluster became a multi-country outbreak in less than a week.""",
        ),
    ]


def write_notebook() -> None:
    cells = []
    for kind, src in notebook_source():
        src = src.strip("\n") + "\n"
        cells.append(new_markdown_cell(src) if kind == "markdown" else new_code_cell(src))
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    NOTEBOOK_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print("Wrote", NOTEBOOK_PATH)


NAVY_COLOR = colors.HexColor(NAVY)
BLUE_COLOR = colors.HexColor(BLUE)
LIGHT = colors.HexColor("#F4F7FB")
RULE = colors.HexColor("#D6DEEA")
TEXT = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#475569")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverKicker",
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE_COLOR,
            alignment=TA_CENTER,
            tracking=1.2,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            fontName="Times-Bold",
            fontSize=22,
            leading=28,
            textColor=NAVY_COLOR,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSub",
            fontName="Times-Italic",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverMeta",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            textColor=TEXT,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "H1Custom",
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY_COLOR,
            spaceBefore=12,
            spaceAfter=8,
            borderPadding=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "H2Custom",
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE_COLOR,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyJust",
            fontName="Times-Roman",
            fontSize=11,
            leading=15.5,
            textColor=TEXT,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "FooterS",
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletBody",
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            textColor=TEXT,
            leftIndent=12,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            textColor=TEXT,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableHead",
            fontName="Times-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
    )
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY_COLOR)
    canvas.rect(0, height - 16, width, 16, fill=1, stroke=0)
    canvas.setFillColor(BLUE_COLOR)
    canvas.rect(0, height - 18, width, 2.2, fill=1, stroke=0)
    canvas.setFillColor(NAVY_COLOR)
    canvas.rect(0, 0, width, 28, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(18 * mm, 11, f"{STUDENT}  |  {REG_NO}")
    canvas.drawRightString(width - 18 * mm, 11, f"Page {doc.page}")
    canvas.setFont("Times-Italic", 8)
    canvas.drawCentredString(width / 2, 11, "Mini Project — 2019-nCoV EDA")
    canvas.restoreState()


def cover_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY_COLOR)
    canvas.rect(0, height - 72, width, 72, fill=1, stroke=0)
    canvas.setFillColor(BLUE_COLOR)
    canvas.rect(0, height - 78, width, 6, fill=1, stroke=0)
    canvas.setFillColor(NAVY_COLOR)
    canvas.rect(0, 0, width, 48, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Bold", 11)
    canvas.drawCentredString(width / 2, height - 38, "DATA SCIENCE AND MACHINE LEARNING")
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(width / 2, height - 54, "Mini Project Report")
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(width / 2, 22, f"{STUDENT}  ·  Registration No. {REG_NO}")
    canvas.restoreState()


def p(styles, text, style="BodyJust"):
    return Paragraph(text, styles[style])


def fig_block(styles, path: Path, caption: str, width=6.3 * inch, max_height=3.7 * inch):
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    height = width * (h_px / float(w_px))
    if height > max_height:
        width = width * (max_height / height)
        height = max_height
    img = Image(str(path), width=width, height=height)
    img.hAlign = "CENTER"
    return KeepTogether([img, Paragraph(caption, styles["Caption"])])


def styled_table(header, rows, col_widths, styles):
    head = [Paragraph(h, styles["TableHead"]) for h in header]
    body = []
    for row in rows:
        cells = []
        for i, val in enumerate(row):
            cells.append(Paragraph(str(val), styles["TableCell"]))
        body.append(cells)
    data = [head] + body
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_pdf(df, daily, latest, by_country, figs):
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=24 * mm,
        bottomMargin=18 * mm,
        title="Mini Project — Exploratory Data Analysis of the Early 2019-nCoV Outbreak",
        author=STUDENT,
    )

    last = daily.iloc[-1]
    first = daily.iloc[0]
    hubei_row = latest[latest["province"] == "Hubei"].iloc[0]
    china_conf = int(latest.loc[latest["is_china"], "confirmed"].sum())
    rest_conf = int(latest.loc[~latest["is_china"], "confirmed"].sum())
    n_intl = int((~latest["is_china"]).sum())
    hubei_share = hubei_row["confirmed"] / last["confirmed"] * 100
    china_share = china_conf / last["confirmed"] * 100
    growth = last["confirmed"] / first["confirmed"]

    story = []

    # ----- COVER -----
    story.append(Spacer(1, 28 * mm))
    story.append(p(styles, "MINI PROJECT REPORT", "CoverKicker"))
    story.append(
        p(
            styles,
            "Exploratory Data Analysis of the Early<br/>2019-nCoV Outbreak (21–26 January 2020)",
            "CoverTitle",
        )
    )
    story.append(
        p(
            styles,
            "A visual EDA of the Johns Hopkins CSSE summary file<br/>"
            "<b>2019_nC0v_20200121_20200126 - SUMMARY.csv</b>",
            "CoverSub",
        )
    )
    story.append(Spacer(1, 14 * mm))

    meta = [
        ["Submitted by", STUDENT],
        ["Registration Number", REG_NO],
        ["Programme", "Data Science and Machine Learning (DSML)"],
        ["Project type", "Module 1 Mini Project — Exploratory Data Analysis"],
        ["Tools", "Python, NumPy, Pandas, Matplotlib, Seaborn"],
        ["Dataset window", "21 January 2020 – 26 January 2020"],
        ["Date of report", "September 2026"],
    ]
    meta_table = Table(
        [[Paragraph(f"<b>{a}</b>", styles["CoverMeta"]), Paragraph(b, styles["CoverMeta"])] for a, b in meta],
        colWidths=[70 * mm, 95 * mm],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, NAVY_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 18 * mm))
    story.append(
        p(
            styles,
            "This report accompanies the Jupyter notebook "
            "<b>MiniProject_COVID19_nCoV_EDA.ipynb</b>. "
            "The analysis follows the same Load → Inspect → Clean → Explore → "
            "Visualise (12 charts) → Summarise workflow used in the course mini-project "
            "on India's COVID-19 data, applied here to the early 2019-nCoV summary file.",
            "CoverSub",
        )
    )
    story.append(PageBreak())

    # ----- ABSTRACT -----
    story.append(p(styles, "1. Abstract", "H1Custom"))
    story.append(
        p(
            styles,
            "This mini project performs an exploratory data analysis of the earliest "
            "publicly collated 2019 novel coronavirus (2019-nCoV) situation reports. "
            "The source file, <i>2019_nC0v_20200121_20200126 - SUMMARY.csv</i>, is an "
            "aggregated version of the Johns Hopkins University CSSE dataset covering "
            "21–26 January 2020. Using only NumPy, Pandas and Matplotlib/Seaborn, the "
            f"analysis cleans mixed timestamps and missing counts, then tracks how "
            f"confirmed cases grew from {int(first['confirmed']):,} to {int(last['confirmed']):,} "
            f"in six days. Hubei province alone held {hubei_share:.1f}% of confirmed cases "
            f"by 26 January, Mainland China held {china_share:.1f}%, and {n_intl} locations "
            "outside Mainland China had already reported imported cases. Recovery rates "
            "were still below 2% because almost every patient was newly admitted. The "
            "findings are descriptive, not causal: they document the opening week of the "
            "outbreak and the data-quality problems that come with a brand-new pathogen.",
        )
    )

    story.append(p(styles, "2. Introduction and Objectives", "H1Custom"))
    story.append(
        p(
            styles,
            "On 31 December 2019 the Wuhan Municipal Health Commission reported a cluster "
            "of pneumonia cases of unknown cause. By the third week of January 2020 the "
            "pathogen had a name — 2019-nCoV — and Johns Hopkins CSSE had begun publishing "
            "province-level situation reports. Those first reports are noisy: blank numeric "
            "fields, trailing spaces in country names, and clocks written as “10pm”, “12am” "
            "or “23:00”. They are also historically important. They show the outbreak before "
            "it had a pandemic label, before India recorded its first case (30 January 2020), "
            "and long before vaccines existed.",
        )
    )
    story.append(
        p(
            styles,
            "The assignment is a Module 1 mini project in the same spirit as the course "
            "notebook <i>MiniProject_COVID19_India_EDA.ipynb</i>: load a real CSV, inspect "
            "it before touching it, clean it, aggregate with <b>groupby</b> / <b>pivot_table</b>, "
            "draw twelve charts, and write down what the charts actually say. The India "
            "notebook used MoHFW state-wise counts from the first wave (Jan–Aug 2020) plus "
            "OWID vaccination totals. This project uses the earlier JHU summary file specified "
            "for this submission, so the geographic grain is Chinese provinces plus the first "
            "international detections rather than Indian states.",
        )
    )
    story.append(p(styles, "Objectives", "H2Custom"))
    bullets = [
        "Load and profile <b>2019_nC0v_20200121_20200126 - SUMMARY.csv</b>, including missingness and inconsistent labels.",
        "Clean dtypes, dates, country names and missing counts without leaking future information into the snapshot.",
        "Build daily global totals with groupby, and a country × date pivot of confirmed cases.",
        "Produce 12 visualisations covering cumulative trends, daily incidence, Hubei’s share, recovery/death rates, correlations, and international spread.",
        "Summarise what the first six days of reporting can — and cannot — tell us.",
    ]
    for b in bullets:
        story.append(p(styles, f"•  {b}", "BulletBody"))

    story.append(p(styles, "3. Dataset Description", "H1Custom"))
    story.append(
        p(
            styles,
            "The file is an aggregated stack of CSSE daily case updates from 21 through "
            "26 January 2020 (ten snapshots, because some days published more than one "
            "bulletin). Each row is one location at one timestamp.",
        )
    )

    col_table = styled_table(
        ["Column", "Meaning", "Notes after inspection"],
        [
            ["Province/State", "Chinese province or sub-national area", "Blank for national reports (Thailand, Japan, …)"],
            ["Country", "Country / region name", "Trailing spaces on Singapore, Malaysia"],
            ["Date last updated", "Bulletin timestamp", "Mixed text formats; parsed to datetime"],
            ["Confirmed", "Cumulative confirmed cases", "Missing treated as 0"],
            ["Suspected", "Cases under investigation", "Mostly missing after 25 Jan"],
            ["Recovered", "Cumulative recoveries", "Very sparse — outbreak too new"],
            ["Deaths", "Cumulative deaths", "Almost all in Hubei"],
        ],
        [32 * mm, 48 * mm, 85 * mm],
        styles,
    )
    story.append(col_table)
    story.append(Paragraph("Table 1. Raw columns in the 2019-nCoV summary file.", styles["Caption"]))

    story.append(
        p(
            styles,
            f"After stacking the bulletins the working table has <b>{len(df):,} rows</b>, "
            f"<b>{df['country'].nunique()} countries/territories</b> and "
            f"<b>{df['province'].replace('', np.nan).nunique()} named provinces</b>. "
            "A handful of Latin-American countries (Brazil, Mexico, Colombia) and the "
            "Philippines appear on the 23 January watchlist with zero confirmed cases; "
            "they are retained in the cleaned file but drop out of “latest confirmed” "
            "rankings because they still had no cases by 26 January.",
        )
    )

    story.append(p(styles, "4. Methodology", "H1Custom"))
    story.append(
        p(
            styles,
            "The workflow is the same six-step pipeline used in the course India EDA "
            "notebook, implemented in Python 3 with Pandas for tabular work and "
            "Matplotlib/Seaborn for charts.",
        )
    )
    story.append(p(styles, "4.1 Cleaning", "H2Custom"))
    story.append(
        p(
            styles,
            "Columns were renamed to snake_case. Country and province strings were stripped. "
            "Count fields were coerced with <font face='Courier'>pd.to_numeric(..., errors='coerce')</font> "
            "and missing values filled with 0 — a count that was not reported is treated as "
            "zero in these bulletins, not as “unknown but possibly large”. Timestamps were "
            "parsed with <font face='Courier'>format='mixed'</font> so that “10pm”, “12am” and "
            "24-hour clocks land on the same datetime axis. An <b>active</b> column was derived "
            "as confirmed − deaths − recovered, floored at zero.",
        )
    )
    story.append(p(styles, "4.2 Aggregation", "H2Custom"))
    story.append(
        p(
            styles,
            "Because the CSV stacks every bulletin, summing every row would triple-count "
            "provinces that were updated twice on the same day. For each calendar date we "
            "keep only the last report per (province, country) pair, then sum. Daily new "
            "cases are the first difference of that cumulative series. A pivot table of "
            "confirmed cases by country and date shows when each country entered the file.",
        )
    )
    story.append(p(styles, "4.3 Visualisation design", "H2Custom"))
    story.append(
        p(
            styles,
            "Twelve charts were drawn to mirror the India mini-project layout: a national "
            "(here: global) cumulative line, a daily-new bar chart, a top-10 ranking, a "
            "recovery-rate ranking, a top-5 trend overlay, a share pie, a rate trend, a "
            "bubble scatter, a correlation heatmap, a boxplot of the distribution, a "
            "grouped comparison, and a “least / outside-core” bar chart. Colour is reserved "
            "for meaning (blue = confirmed, green = recovered, red = deaths, orange = "
            "outside China).",
        )
    )

    story.append(p(styles, "5. Data Cleaning Findings", "H1Custom"))
    story.append(
        p(
            styles,
            "Inspection of the raw file reproduced the same class of problems the India "
            "notebook had to fix (a text-typed death column, duplicate state spellings). "
            "Here the analogues were:",
        )
    )
    for b in [
        "<b>Missing counts.</b> Confirmed, suspected, recovered and deaths are blank on many early rows. Filling with 0 is conservative and matches how CSSE later published the series.",
        "<b>Inconsistent country strings.</b> “Singapore ” and “Malaysia ” with trailing spaces would have split those countries in a groupby.",
        "<b>Mixed clocks.</b> Pandas cannot infer a single format for “1/21/2020 10pm” and “1/26/20 23:00”; mixed parsing is required.",
        "<b>Watchlist rows.</b> Brazil, Mexico, Colombia and the Philippines appear with empty confirmed counts. They are not outbreaks; they are “no cases reported yet”.",
        "<b>Suspected column decays.</b> After 25 January the suspected field is largely abandoned as testing shifted the definition toward laboratory-confirmed cases.",
    ]:
        story.append(p(styles, f"•  {b}", "BulletBody"))

    story.append(p(styles, "6. Exploratory Analysis", "H1Custom"))
    story.append(p(styles, "6.1 Global snapshot", "H2Custom"))

    snap_rows = [
        ["Metric", "21 Jan 2020", "26 Jan 2020"],
        ["Confirmed cases", f"{int(first['confirmed']):,}", f"{int(last['confirmed']):,}"],
        ["Deaths", f"{int(first['deaths']):,}", f"{int(last['deaths']):,}"],
        ["Recovered", f"{int(first['recovered']):,}", f"{int(last['recovered']):,}"],
        ["Active (derived)", f"{int(first['active']):,}", f"{int(last['active']):,}"],
        ["Recovery rate", f"{first['recovery_rate']:.2f}%", f"{last['recovery_rate']:.2f}%"],
        ["Death rate", f"{first['death_rate']:.2f}%", f"{last['death_rate']:.2f}%"],
        ["Locations in bulletin", f"{int(first['locations'])}", f"{int(last['locations'])}"],
        ["Countries / territories", f"{int(first['countries'])}", f"{int(last['countries'])}"],
        ["Mainland China share of cases", "—", f"{china_share:.1f}%"],
        ["Hubei share of cases", "—", f"{hubei_share:.1f}%"],
    ]
    story.append(
        styled_table(
            snap_rows[0],
            snap_rows[1:],
            [70 * mm, 45 * mm, 50 * mm],
            styles,
        )
    )
    story.append(Paragraph("Table 2. Six-day change in the cleaned daily totals.", styles["Caption"]))
    story.append(
        p(
            styles,
            f"Confirmed cases grew by a factor of about {growth:.1f} in six days. "
            f"New cases per day were already approaching 1,000 by 26 January "
            f"(+{int(last['new_confirmed']):,}). "
            "Recoveries barely moved, so the active-case curve tracks the confirmed curve. "
            "That is the signature of an outbreak that is still accelerating, not one that "
            "has peaked.",
        )
    )

    daily_tbl = []
    for _, r in daily.iterrows():
        daily_tbl.append(
            [
                r["date"].strftime("%d %b %Y"),
                f"{int(r['confirmed']):,}",
                f"{int(r['new_confirmed']):,}",
                f"{int(r['deaths']):,}",
                f"{int(r['recovered']):,}",
                f"{r['death_rate']:.2f}%",
            ]
        )
    story.append(
        styled_table(
            ["Date", "Confirmed", "New", "Deaths", "Recovered", "CFR"],
            daily_tbl,
            [32 * mm, 28 * mm, 24 * mm, 26 * mm, 28 * mm, 27 * mm],
            styles,
        )
    )
    story.append(Paragraph("Table 3. Daily global series after last-snapshot aggregation.", styles["Caption"]))

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.2 Chart 1 — Cumulative global trend", "H2Custom"))
    story.append(fig_block(styles, figs["c1"], "Figure 1. Cumulative confirmed, recovered and death counts, 21–26 January 2020."))
    story.append(
        p(
            styles,
            "The confirmed series is steep and convex. Recovered and death series are almost "
            "flat on this scale, not because outcomes were good, but because the time window "
            "is shorter than a typical hospital stay. This is the same cumulative-line chart "
            "used for India’s first wave in the course notebook; here the x-axis is days "
            "rather than months.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.3 Chart 2 — Daily new confirmed cases", "H2Custom"))
    story.append(fig_block(styles, figs["c2"], "Figure 2. First difference of the global confirmed series."))
    story.append(
        p(
            styles,
            f"Incidence is not smooth. 23 January is a quieter day (+{int(daily.loc[2, 'new_confirmed'])}), "
            f"then the next three bulletins jump to +{int(daily.loc[3, 'new_confirmed'])}, "
            f"+{int(daily.loc[4, 'new_confirmed']):,} and +{int(daily.loc[5, 'new_confirmed']):,}. "
            "Part of that jump is genuine spread inside Hubei; part is improved case-finding after "
            "Wuhan’s lockdown was announced on 23 January. EDA cannot separate those two effects, "
            "and the report does not pretend to.",
        )
    )

    story.append(p(styles, "6.4 Chart 3 — Top 10 locations", "H2Custom"))
    story.append(fig_block(styles, figs["c3"], "Figure 3. Top 10 locations by confirmed cases on 26 January 2020."))

    top_rows = []
    for _, r in latest.head(10).iterrows():
        top_rows.append(
            [
                r["location"],
                f"{int(r['confirmed']):,}",
                f"{int(r['deaths']):,}",
                f"{int(r['recovered']):,}",
                f"{r['death_rate']:.2f}%",
            ]
        )
    story.append(
        styled_table(
            ["Location", "Confirmed", "Deaths", "Recovered", "CFR"],
            top_rows,
            [62 * mm, 28 * mm, 26 * mm, 28 * mm, 21 * mm],
            styles,
        )
    )
    story.append(Paragraph("Table 4. Ten worst-hit locations on the last bulletin of 26 January.", styles["Caption"]))
    story.append(
        p(
            styles,
            f"Hubei ({int(hubei_row['confirmed']):,} confirmed, {int(hubei_row['deaths']):,} deaths) "
            "sits in a different league from every other row. The next province, Guangdong, "
            "has roughly one-tenth as many cases and two recoveries. This is the provincial "
            "analogue of Maharashtra’s dominance in the India mini-project.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.5 Chart 4 — Recovery rates of the worst-hit locations", "H2Custom"))
    story.append(fig_block(styles, figs["c4"], "Figure 4. Recovery rate among the ten locations with the most confirmed cases."))
    story.append(
        p(
            styles,
            "Recovery rates are all in the low single digits. Beijing (2 recoveries / 68 cases) "
            "and Guangdong look “better” than Henan or Chongqing only because a couple of "
            "early patients had been discharged. With so few recoveries, ranking locations "
            "by recovery rate is unstable — a useful caution against over-interpreting ratios "
            "built on small numerators.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.6 Chart 5 — Top-five trajectories", "H2Custom"))
    story.append(fig_block(styles, figs["c5"], "Figure 5. Confirmed-case trajectories for the five worst-hit locations."))
    story.append(
        p(
            styles,
            "Hubei’s curve bends upward after 24 January; the other four provinces rise "
            "linearly and stay an order of magnitude lower. If a single chart has to explain "
            "the first week of 2019-nCoV, it is this one: a provincial epicentre pulling away "
            "from the rest of the country.",
        )
    )

    story.append(p(styles, "6.7 Chart 6 — Share of global cases", "H2Custom"))
    story.append(fig_block(styles, figs["c6"], "Figure 6. Hubei vs the rest of Mainland China vs the rest of the world.", width=4.6 * inch, max_height=4.4 * inch))
    story.append(
        p(
            styles,
            f"On 26 January Hubei held {hubei_share:.1f}% of confirmed cases, other Mainland "
            f"Chinese provinces held {100 - hubei_share - (rest_conf / last['confirmed'] * 100):.1f}%, "
            f"and the rest of the world held only {rest_conf / last['confirmed'] * 100:.1f}% "
            f"({rest_conf:,} people). The pie is the counterpart of the “top 6 Indian states” "
            "share chart in the course notebook: the epidemic was geographically concentrated, not national-even.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.8 Chart 7 — Recovery rate vs death rate over time", "H2Custom"))
    story.append(fig_block(styles, figs["c7"], "Figure 7. Crude recovery rate and crude death rate, global totals."))
    story.append(
        p(
            styles,
            f"The crude death rate ends at {last['death_rate']:.2f}% and the recovery rate at "
            f"{last['recovery_rate']:.2f}%. Both numbers will move as the denominator (confirmed) "
            "and the numerator (deaths, discharges) catch up with reality. They are reported "
            "here as the ratios the file actually contains, not as estimates of infection fatality.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.9 Chart 8 — Deaths against confirmed cases", "H2Custom"))
    story.append(fig_block(styles, figs["c8"], "Figure 8. Location-level deaths vs confirmed cases. Bubble size is case count."))
    story.append(
        p(
            styles,
            "Hubei is the only point with a large death count. Everywhere else is still near "
            "the origin. Henan is the only other Mainland province with a recorded death in "
            "this window. The scatter is therefore a picture of concentration, not of a "
            "well-populated risk relationship.",
        )
    )

    story.append(PageBreak())
    story.append(p(styles, "6.10 Chart 9 — Correlation heatmap", "H2Custom"))
    story.append(fig_block(styles, figs["c9"], "Figure 9. Pearson correlations among the six-day global series.", width=5.4 * inch))
    story.append(
        p(
            styles,
            "Confirmed, deaths, recovered and new_confirmed are strongly positively correlated "
            "because they are all rising together on a six-point series. Recovery rate is "
            "negatively related to the size metrics: as the confirmed denominator exploded, "
            "the recovered numerator did not keep up, so the ratio fell. With only six daily "
            "points the coefficients are descriptive of this window, not a general law.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.11 Chart 10 — Distribution of location-level counts", "H2Custom"))
    story.append(fig_block(styles, figs["c10"], "Figure 10. Boxplot of confirmed counts across locations, one box per day."))
    story.append(
        p(
            styles,
            "Each box is right-skewed with a high outlier — Hubei. The median location still "
            "has a modest count even on 26 January, which is another way of saying the mean "
            "is a Hubei story. Boxplots are used here the same way the India notebook used "
            "them for daily new cases by month: to show spread, not just the total.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.12 Chart 11 — Mainland China versus the rest of the world", "H2Custom"))
    story.append(fig_block(styles, figs["c11"], "Figure 11. Grouped bars of confirmed cases inside vs outside Mainland China."))
    story.append(
        p(
            styles,
            "Outside Mainland China the series is almost invisible on the same axis. That is "
            "the point. On 26 January this was still a Chinese epidemic with a thin ring of "
            "imported detections, not a balanced global one. Replacing the vaccination chart "
            "from the India mini-project, this comparison is the “second chapter” the January "
            "2020 file actually supports.",
        )
    )

    story.append(CondPageBreak(110 * mm))
    story.append(p(styles, "6.13 Chart 12 — International detections", "H2Custom"))
    story.append(fig_block(styles, figs["c12"], "Figure 12. Confirmed cases outside Mainland China on 26 January 2020."))

    intl = latest[~latest["is_china"]].sort_values("confirmed", ascending=False)
    intl_rows = [
        [r["location"], f"{int(r['confirmed'])}", f"{int(r['deaths'])}", f"{int(r['recovered'])}"]
        for _, r in intl.iterrows()
    ]
    story.append(
        styled_table(
            ["Location", "Confirmed", "Deaths", "Recovered"],
            intl_rows,
            [80 * mm, 32 * mm, 32 * mm, 32 * mm],
            styles,
        )
    )
    story.append(Paragraph("Table 5. Every location outside Mainland China with a 26 January bulletin.", styles["Caption"]))
    story.append(
        p(
            styles,
            "The largest non-Mainland counts are Hong Kong and Thailand (8 each). The United "
            "States has 5, split across Washington, Illinois, California and Arizona in the "
            "underlying rows. Nepal’s single case is the first detection on the subcontinent "
            "in this file; India itself is not yet present. No deaths are recorded outside "
            "Mainland China in this window.",
        )
    )

    story.append(PageBreak())
    story.append(p(styles, "7. Key Findings", "H1Custom"))
    findings = [
        f"<b>Eight-fold growth in six days.</b> Confirmed cases moved from {int(first['confirmed']):,} (21 Jan) to {int(last['confirmed']):,} (26 Jan), with daily new cases reaching {int(last['new_confirmed']):,} on the last day.",
        f"<b>Hubei was the outbreak.</b> {int(hubei_row['confirmed']):,} of {int(last['confirmed']):,} confirmed cases ({hubei_share:.1f}%) and {int(hubei_row['deaths'])} of {int(last['deaths'])} deaths sat in one province.",
        f"<b>Mainland China held {china_share:.1f}% of cases.</b> The rest of the world had {rest_conf:,} confirmed people, all of them living, all of them in single-digit national totals.",
        f"<b>Recovery had not started in earnest.</b> Only {int(last['recovered'])} recoveries against {int(last['confirmed']):,} confirmed cases ({last['recovery_rate']:.2f}%). Low recovery here is a lag, not a clinical conclusion.",
        f"<b>The crude death rate was {last['death_rate']:.2f}%.</b> It is dominated by Hubei and will be biased by under-testing of mild cases.",
        "<b>International spread was already real, and still tiny.</b> 14 locations outside Mainland China were on the board; none had reached double-digit deaths, and India had not yet reported a case.",
        "<b>The raw CSV needed the same cleaning discipline as the India dataset:</b> missing counts, dirty strings, mixed dates, and watchlist rows that are not cases.",
    ]
    for b in findings:
        story.append(p(styles, f"•  {b}", "BulletBody"))

    story.append(p(styles, "8. Comparison with the Course India EDA", "H1Custom"))
    story.append(
        p(
            styles,
            "The course notebook <i>MiniProject_COVID19_India_EDA.ipynb</i> analysed MoHFW "
            "state-wise counts from 30 January to 6 August 2020 (India’s first wave) and an "
            "OWID vaccination series from 2021–24. This submission keeps that pedagogical "
            "skeleton and swaps in the specified early-outbreak file.",
        )
    )
    cmp_rows = [
        ["Item", "Course India EDA", "This 2019-nCoV EDA"],
        ["File", "complete.csv + India OWID vax", "2019_nC0v_20200121_20200126 - SUMMARY.csv"],
        ["Window", "30 Jan – 6 Aug 2020", "21–26 Jan 2020"],
        ["Grain", "Indian state / UT", "Chinese province + country"],
        ["Cleaning", "Death as text; Telangana spellings", "Missing counts; trailing spaces; mixed clocks"],
        ["Headline concentration", "Maharashtra ~¼ of India", "Hubei ~51% of the world file"],
        ["Recovery", "~68% by Aug 2020", "1.93% (too early to discharge)"],
        ["Vaccination chart", "2.2 bn doses by 2024", "Not applicable — replaced by China vs world"],
        ["India in the file", "Entire subject", "Not yet present (first case 30 Jan)"],
    ]
    story.append(
        styled_table(
            cmp_rows[0],
            cmp_rows[1:],
            [38 * mm, 68 * mm, 68 * mm],
            styles,
        )
    )
    story.append(Paragraph("Table 6. How this mini project maps onto the course EDA template.", styles["Caption"]))

    story.append(p(styles, "9. Limitations", "H1Custom"))
    story.append(
        p(
            styles,
            "These bulletins are not a complete epidemic curve. Testing capacity in mid-January "
            "2020 was low, case definitions were still moving, and suspected counts disappear "
            "from the schema within the week. Crude rates ignore reporting lag. Locations with "
            "zero confirmed cases remain in the raw file and must not be plotted as outbreaks. "
            "Nothing in this analysis is a forecast, a treatment effect, or a ranking of "
            "health-system quality.",
        )
    )

    story.append(p(styles, "10. Conclusion", "H1Custom"))
    story.append(
        p(
            styles,
            "The first six published days of 2019-nCoV data already contain the shape of what "
            "followed: exponential growth, extreme geographic concentration in Hubei, a death "
            "count that had started to move, a recovery count that had not, and a thin but "
            "real ring of international importations. Cleaning the summary file is not a "
            "preface to the analysis — it is part of the analysis. Once the dates parse and "
            "the country names stop splitting, twelve charts are enough to tell that story "
            "with the Module 1 toolkit.",
        )
    )
    story.append(
        p(
            styles,
            f"Submitted by <b>{STUDENT}</b>, registration number <b>{REG_NO}</b>. "
            "The executable companion to this report is "
            "<b>MiniProject_COVID19_nCoV_EDA.ipynb</b>.",
        )
    )

    story.append(p(styles, "11. References", "H1Custom"))
    refs = [
        "Johns Hopkins University Center for Systems Science and Engineering (CSSE). 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository. Archived daily case updates, 21–26 January 2020.",
        "Dong, E., Du, H. &amp; Gardner, L. (2020). An interactive web-based dashboard to track COVID-19 in real time. <i>The Lancet Infectious Diseases</i>.",
        "Dey, S. K., Rahman, M. M., Siddiqi, U. R. &amp; Howlader, A. (2020). Analyzing the epidemiological outbreak of COVID-19: A visual exploratory data analysis approach. <i>Journal of Medical Virology</i>. Uses 2019_nC0v_20200121_20200126-SUMMARY.csv as a primary source.",
        "Course material: MiniProject_COVID19_India_EDA.ipynb — Exploratory Data Analysis on India's COVID-19 Data (Module 1 template: NumPy, Pandas, Matplotlib/Seaborn).",
        "Our World in Data; Ministry of Health and Family Welfare, Government of India — referenced only as the data sources of the course India notebook, not used in this run.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(p(styles, f"[{i}]  {r}", "BulletBody"))

    story.append(Spacer(1, 8 * mm))
    story.append(p(styles, "Appendix A. Reproducibility", "H1Custom"))
    story.append(
        p(
            styles,
            "Place <font face='Courier'>2019_nC0v_20200121_20200126 - SUMMARY.csv</font> next to "
            "the notebook and run all cells top to bottom. Dependencies are listed in "
            "<font face='Courier'>requirements.txt</font> (pandas, numpy, matplotlib, seaborn). "
            "The notebook writes <font face='Courier'>cleaned_ncov_summary.csv</font> as a "
            "reusable output. This PDF was generated from the same cleaned tables and the "
            "twelve figures in the <font face='Courier'>figures/</font> folder.",
        )
    )

    story.append(Spacer(1, 10 * mm))
    sign = Table(
        [
            [
                Paragraph("<b>Student</b><br/>Shivansh Srivastava<br/>RA2411056030011", styles["CoverMeta"]),
                Paragraph("<b>Declaration</b><br/>This mini project is my own work, prepared with the specified dataset and the course EDA template.", styles["CoverMeta"]),
            ]
        ],
        colWidths=[80 * mm, 85 * mm],
    )
    sign.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, NAVY_COLOR),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(sign)

    def first_page(canvas, doc):
        cover_header_footer(canvas, doc)

    def later_pages(canvas, doc):
        header_footer(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    print("Wrote", PDF_PATH, "size", PDF_PATH.stat().st_size)


def main():
    df, daily, latest, by_country = load_and_clean()
    df.to_csv(CLEANED_PATH, index=False)
    figs = make_figures(df, daily, latest)
    write_notebook()
    build_pdf(df, daily, latest, by_country, figs)
    print("figures:", len(list(FIG_DIR.glob('*.png'))))


if __name__ == "__main__":
    main()
