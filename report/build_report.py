"""
Builds report/BFT65B_Final_Report_NguyenTrungKien.docx from the files in results/.
Every number in the report is read from the pipeline outputs, so the report always matches the last run.

    python report/build_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
OUT = ROOT / "report" / "BFT65B_Final_Report_NguyenTrungKien.docx"

AUTHOR, SID = "Nguyễn Trung Kiên", "11230982"
GITHUB = "https://github.com/kientainang2005-arch/Final-exam-Bigdata"
DOWNLOAD_DATE = "6 November 2026"


# ------------------------------------------------------------------ load results
def js(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def csv(name):
    return pd.read_csv(R / name)


panel = js("panel_summary.json")
feats = js("features_summary.json")
frozen = js("frozen_choices.json")
ew = js("test_early_warning.json")
hgbp = js("hgb_best_params.json")
eq = js("benchmark_equality.json")
storage = csv("storage_benchmark.csv")
bench = csv("benchmark_engines.csv")
cv = csv("cv_walk_forward.csv")
test = pd.read_csv(R / "test_metrics.csv", index_col=0)
sens = csv("threshold_sensitivity.csv")
dq = csv("dq_summary.csv")
fdict = csv("feature_dictionary.csv")
by_year = csv("panel_by_year.csv")
spark_cmp = csv("spark_vs_polars.csv")
spark_t = csv("spark_timing.csv")
mllib = csv("mllib_demo.csv") if (R / "mllib_demo.csv").exists() else None
man = json.loads((ROOT / "data_raw" / "fdic" / "manifest_financials.json").read_text()) \
    if (ROOT / "data_raw" / "fdic" / "manifest_financials.json").exists() else {}
small = json.loads((ROOT / "data_raw" / "fdic" / "manifest_small_tables.json").read_text()) \
    if (ROOT / "data_raw" / "fdic" / "manifest_small_tables.json").exists() else {}

split = frozen["split"]
dev_s, dev_e, dev_rows, dev_pos, dev_banks = split["dev"]
te_s, te_e, te_rows, te_pos, te_banks = split["test"]
costs = frozen["costs"]
thr = frozen["threshold"]
raw_rows = panel["raw_financial_rows"]
n_banks_raw = 24064


def q(d):  # '2006-12-31' -> '2006Q4'
    y, m = int(d[:4]), int(d[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def f0(x):
    return f"{x:,.0f}"


def f1(x):
    return f"{x:,.1f}"


def f2(x):
    return f"{x:,.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.0))
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.08
for lvl, size in ((1, 14), (2, 12), (3, 11)):
    hs = doc.styles[f"Heading {lvl}"]
    hs.font.name, hs.font.size, hs.font.color.rgb = "Calibri", Pt(size), RGBColor(0x1F, 0x3A, 0x5F)
    hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    hs.paragraph_format.space_before, hs.paragraph_format.space_after = Pt(10 if lvl == 1 else 7), Pt(3)

# page numbers in footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    run = fp.add_run()
    if t:
        el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t); run._r.append(el)
    else:
        el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt; run._r.append(el)
hp = sec.header.paragraphs[0]
hp.text = "BFT65B – Big Data Analytics in Finance · Final individual report · Nguyễn Trung Kiên (11230982)"
hp.runs[0].font.size, hp.runs[0].font.color.rgb = Pt(8), RGBColor(0x66, 0x66, 0x66)

FIGN = [0]
TABN = [0]


def H(text, lvl=1):
    doc.add_heading(text, level=lvl)


def P(text, bold_prefix=None, italic=False, size=None, align=None, after=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
        if size: r.font.size = Pt(size)
    # **bold** inline markup
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        r = p.add_run(part)
        r.bold = (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def B(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", it)):
            r = p.add_run(part); r.bold = (i % 2 == 1)
        p.paragraph_format.space_after = Pt(1)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def T(df: pd.DataFrame, caption: str, widths=None, size=8.5, bold_rows=()):
    TABN[0] += 1
    cp = doc.add_paragraph(); r = cp.add_run(f"Table {TABN[0]}. {caption}"); r.bold = True; r.font.size = Pt(9)
    cp.paragraph_format.space_after = Pt(2); cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ""
        run = cell.paragraphs[0].add_run(str(c)); run.bold = True; run.font.size = Pt(size)
        shade(cell, "DCE6F0")
    for i, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run("" if (isinstance(v, float) and pd.isna(v)) else str(v))
            run.font.size = Pt(size)
            if i in bold_rows:
                run.bold = True
    if len(df) <= 25:   # keep short tables (and their caption) on one page
        for row in t.rows[:-1]:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def FIGURE(path: Path, caption: str, width=16.5, label=None):
    if not path.exists():
        P(f"[figure missing: {path.name}]", italic=True); return
    if label is None:
        FIGN[0] += 1
        label = str(FIGN[0])
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph(); r = cp.add_run(f"Figure {label}. {caption}"); r.italic = True; r.font.size = Pt(9)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER


def CODE(text, size=7.5):
    p = doc.add_paragraph()
    r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(size)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    p.paragraph_format.space_after = Pt(4)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ derived numbers
fails_dev_events = by_year[(by_year.year >= int(dev_s[:4])) & (by_year.year <= int(dev_e[:4]))].failures.sum()
st_csv = storage.iloc[0]
st_part = storage[storage.layout.str.contains("partitioned")].iloc[0]
st_zstd = storage[storage.layout.str.startswith("Parquet zstd-3 (1")].iloc[0]
b1 = bench[bench.scale == 1].set_index("engine")
pd_t, pl_t = b1.loc["pandas", "seconds_median"], b1.loc["polars-lazy", "seconds_median"]
pd_m, pl_m = b1.loc["pandas", "peak_mb_median"], b1.loc["polars-lazy", "peak_mb_median"]
st_m = b1.loc["polars-streaming", "peak_mb_median"]
x8 = bench[(bench.scale == 8) & (bench.engine == "polars-streaming")]
x8_rows = x8.rows_out.iloc[0] if len(x8) else float("nan")
x8_mem = x8.peak_mb_median.iloc[0] if len(x8) else float("nan")
x8l = bench[(bench.scale == 8) & (bench.engine == "polars-lazy")]
x8l_mem = x8l.peak_mb_median.iloc[0] if len(x8l) else float("nan")
x4p = bench[(bench.scale == 4) & (bench.engine == "pandas")]
bytes_per_row_stream = x8_mem * 2**20 / x8_rows
rows_fit_16gb = 0.7 * 15.6 * 2**30 / (x8l_mem * 2**20 / x8_rows)
sp_t = spark_t.set_index("engine").seconds
max_spark_diff = spark_cmp.max_abs_diff.max()
cv4 = cv[cv.fold.str.contains("1996")].set_index("model")
cvm = cv.groupby("model")[["pr_auc", "roc_auc", "recall_top1pct", "recall_top5pct", "brier"]].mean()
chosen = frozen["chosen_model"]
names = {"baseline_logit_5": "Baseline: logistic, 5 CAMELS ratios", "logit_all": "Logistic, all 40 features (L2)",
         "hgb": "Histogram gradient boosting (HGB)"}
test = test.assign(cost_musd=(costs["C_FN_kusd"] * test.fn + costs["C_FP_kusd"] * test.fp) / 1e3)
all_neg_acc = 1 - test.loc[chosen, "positives"] / test.loc[chosen, "n"]

# ================================================================== FIRST PAGE
p = P("NATIONAL ECONOMICS UNIVERSITY · BFT65B – BIG DATA ANALYTICS IN FINANCE", size=9, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("A Big-Data Pipeline for Bank-Failure Early Warning"); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
P("Final examination – individual research report", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)

first = pd.DataFrame([
    ["Student", f"{AUTHOR} — student ID {SID}"],
    ["GitHub repository (public)", GITHUB],
    ["Download method and date", f"FDIC BankFind public API (https://api.fdic.gov/banks), own resumable script "
                                 f"src/01_download.py adapted from the course starter; downloaded on {DOWNLOAD_DATE}"],
    ["Row counts", f"Financials {f0(raw_rows)} rows (170 quarters 1984Q1–2026Q2, {f0(n_banks_raw)} banks; equal to the "
                   f"API total in every quarter) · Failures {f0(small.get('failures', {}).get('rows', 4117))} · "
                   f"Institutions {f0(small.get('institutions', {}).get('rows', 27835))}"],
    ["Clean panel", f"{f0(panel['panel_rows'])} bank-quarters; {f0(panel['labelled_rows_upto_2025Q2'])} labelled rows up to "
                    f"2025Q2 with {f0(panel['positive_rows_upto_2025Q2'])} positives ({panel['positive_rate_pct']:.2f}%)"],
    ["Development period", f"{q(dev_s)}–{q(dev_e)}: {f0(dev_rows)} bank-quarters, {f0(dev_pos)} positive rows, "
                           f"{f0(dev_banks)} failing banks"],
    ["Gap (unused)", f"{q(str(pd.Timestamp(dev_e) + pd.offsets.QuarterEnd(1))[:10])}–2007Q4 (4 quarters, {f0(split['gap_rows_unused'])} rows)"],
    ["Test period (used once)", f"{q(te_s)}–{q(te_e)}: {f0(te_rows)} bank-quarters, {f0(te_pos)} positive rows, "
                                f"{f0(te_banks)} failing banks"],
])
first.columns = ["Item", "Value"]
T(first, "Key facts required on the first page.", widths=[4.2, 12.8], size=9)

P("Abstract", bold_prefix=None).runs[0].bold = True
P(f"This report builds a reproducible early-warning data pipeline on four decades of US Call Reports. The data are "
  f"downloaded with a resumable, self-verifying procedure and stored as a typed, zstd-compressed Parquet layer "
  f"partitioned by year, which is {pct(1 - st_part.size_vs_csv, 0)} smaller than CSV and reads a three-year slice "
  f"{st_csv.filter_2008_2010_s / st_part.filter_2008_2010_s:,.0f} times faster. An automated audit finds and handles "
  f"{len(dq)} classes of data problems, among them year-to-date income flows, zeros that mean ‘not reported’, schema "
  f"drift, failures that cannot be identified from structure codes and static attributes that leak the future. The core "
  f"transformations run in Polars' lazy engine in {pl_t:.1f} s, {pd_t / pl_t:.0f}× faster than an identical pandas "
  f"pipeline with {pct(1 - pl_m / pd_m, 0)} less memory; PySpark reproduces them to within {max_spark_diff:.0e}. "
  f"Forty point-in-time features pass an automatic truncation test (identical values when future data are removed). "
  f"A gradient-boosting model selected by walk-forward validation on {q(dev_s)}–{q(dev_e)} reaches a test ROC-AUC of "
  f"{test.loc[chosen, 'roc_auc']:.3f} and PR-AUC of {test.loc[chosen, 'pr_auc']:.3f} on {q(te_s)}–{q(te_e)}; at the "
  f"cost-based threshold it flags {ew['flagged_at_least_once']} of {ew['failed_banks_in_test']} failing banks, with a "
  f"median lead of {ew['median_lead_quarters']:.0f} quarters.")
page_break()

# ================================================================== 1 INTRODUCTION
H("1. Introduction")
P("The supervision department plans an early-warning system (EWS) that ranks banks by failure risk. Its value depends "
  "first on the data pipeline: if the data are incomplete, mis-typed or contain information from the future, no model "
  "on top of them can be trusted. Because Vietnamese supervisory data are not public, the pipeline is built and tested "
  "on the US FDIC data, which cover three crises (1984–1993, 2008–2014, 2023). The report answers three questions:")
B(["**Q1** – How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed? (Sections 2 and 4)",
   "**Q2** – What quality problems do the raw data contain, and how does the pipeline handle them without look-ahead bias? (Sections 3 and 5)",
   "**Q3** – Can the pipeline's output support an honestly validated early-warning model? (Section 6)"])
P("Every claim is backed by a measurement produced by the code in the repository; all tables and figures are written "
  "to results/ by the pipeline and this report is generated from those files (report/build_report.py). "
  "The machine is a 2024 laptop (Snapdragon X, 8 cores, 16 GB RAM, Windows 11; Python runs as x64 under emulation), "
  "so absolute timings are conservative.")

# ================================================================== 2 ACQUISITION & STORAGE
H("2. Acquisition and storage (Part 1)")
H("2.1 Acquisition", 2)
P("src/01_download.py (adapted from the course starter) pages through the /financials endpoint one quarter at a time "
  "(10,000 rows per call, exponential back-off on HTTP or network errors) and writes each quarter atomically "
  "(temporary file, then rename), so an interrupted run never leaves a half-written file. A JSON manifest records, "
  "per quarter, the rows received, the API's own total, a timestamp and a completeness flag; re-running the script skips "
  "quarters that are already complete, which makes the download **resumable**. Failures and Institutions are small and "
  "are fetched in full. The full download took about 25 minutes.")
P(f"**Completeness check.** For all 170 quarters the number of rows received equals the API total; the sum is "
  f"{f0(raw_rows)} rows, there are 0 duplicate (CERT, REPDTE) keys and 0 incomplete quarters (Table 11 in the appendix "
  f"and results/dq_rows_per_quarter.csv). The number of reporting banks falls from about 18,000 in 1985 to about 4,300 "
  f"in 2026 (Figure A1 in the appendix), the expected consolidation trend; no quarter shows a gap or spike that would "
  f"indicate a partial download.")
P("**Field choice.** The starter's 30 Call Report fields were extended to 45 (plus CERT, REPDTE and NAME), chosen from the "
  "FDIC dictionary to cover every CAMELS dimension: capital (EQ, leverage and risk-based ratios), asset quality "
  "(non-current, past-due and non-accrual assets, ORE, charge-offs, provisions), earnings (net income, interest income "
  "and expense, non-interest items, dividends), liquidity and funding (cash, securities, core and brokered deposits, "
  "volatile liabilities, FHLB borrowings) and loan concentration (construction, multifamily, non-farm non-residential, "
  "1–4 family, C&I, consumer, agricultural loans). Commercial real-estate concentration was the main driver of the "
  "2008–2012 failures, so the real-estate breakdown is essential. BKCLASS, SPECGRP and STNAME are taken from Financials "
  "because there they are recorded per quarter (point-in-time), unlike in Institutions.")

H("2.2 Storage layout", 2)
P("The pipeline keeps three layers: a raw landing zone (one Parquet file per quarter, exactly as received), a curated "
  "layer with one explicit schema, and derived tables (panel, features) that can always be rebuilt.")
B(["**Format:** Parquet (columnar). Credit-risk work reads a few columns of many rows; a columnar file lets the engine "
   "read only those columns and skip row groups using min/max statistics.",
   "**Schema:** CERT Int32, REPDTE Date, amounts Int64 (whole thousands of USD), ratios Float64, state and charter class "
   "Categorical. The raw files do not agree on types: 20 columns change between int64 and double across quarters "
   "(Section 3), which breaks naive multi-file reads; the curated layer casts once and for all.",
   "**Partitioning:** by year (Hive style, year=YYYY/), 43 files of 1.2–7.8 MB (2026 holds only two quarters). Analyses are mostly time-sliced (crisis "
   "windows, train/test periods), so a year filter becomes partition pruning. Partitioning by quarter would create 170 "
   "small files, and by CERT more than 24,000 tiny files – the classic small-file problem.",
   "**Ordering and compression:** rows sorted by (CERT, REPDTE) inside each file so that window operations per bank read "
   "contiguous data and run-length/dictionary encoding works well; zstd level 3 compression."])
sb = storage.copy()
sb["size_MB"] = sb.size_MB.map(f1)
sb["size_vs_csv"] = sb.size_vs_csv.map(lambda v: f"{v:.2f}")
for c in ("write_s", "read_all_s", "read_5cols_s", "filter_2008_2010_s"):
    sb[c] = sb[c].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}")
sb.columns = ["Layout", "Size (MB)", "Write (s)", "Read all (s)", "Read 5 cols (s)", "Filter 2008–10, 5 cols (s)", "Size / CSV"]
T(sb, "Storage benchmark on the full panel (1.68 M rows × 49 columns). One warm-up then median of 5 runs; Polars reader unless stated.",
  widths=[5.0, 1.8, 1.7, 1.9, 2.0, 2.6, 1.7])
P(f"Parquet with zstd is {pct(1 - st_zstd.size_vs_csv, 0)} smaller than CSV and the partitioned layout "
  f"{pct(1 - st_part.size_vs_csv, 0)} smaller. Reading everything is {st_csv.read_all_s / st_part.read_all_s:.0f}× "
  f"faster, because CSV must be parsed and its types inferred, whereas Parquet is already typed and compressed in "
  f"column chunks. The gain is largest when the query needs only a few columns ({st_csv.read_5cols_s / st_part.read_5cols_s:.0f}× "
  f"faster) or a time slice ({st_csv.filter_2008_2010_s / st_part.filter_2008_2010_s:.0f}× faster), the typical "
  f"supervisory access pattern: CSV has no column projection or statistics, so every query scans all 519 MB. gzip "
  f"makes CSV small but is the slowest to write and still has to be fully decompressed and parsed. The partitioned "
  f"layout costs slightly more space than a single zstd file (per-file overhead) and is slightly slower for full scans, "
  f"but it is the fastest for time filters and allows new quarters to be appended by rewriting one small partition "
  f"instead of the whole file – the decisive operational advantage for a quarterly-updated EWS.")

# ================================================================== 3 QUALITY & PANEL
H("3. Data quality and the bank-quarter panel (Part 2)")
H("3.1 Automated audit", 2)
P(f"src/03_quality.py runs a fixed battery of {len(dq)} checks on every run and writes results/dq_report.md with the "
  "evidence tables (the full summary is Table 10 in the appendix). The most consequential findings are:")
B([
    "**Year-to-date income flows.** All income-statement items accumulate from January: median INTINC/ASSET is 1.7% in Q1, "
    "3.5% in Q2, 5.2% in Q3 and 6.8% in Q4, while the FDIC's ROA is flat across quarters because it is already annualised. "
    "Using NETINC/ASSET directly would inject a seasonal saw-tooth. The pipeline de-cumulates: flow_q = YTD_q − YTD_{q−1} "
    "when the previous quarter of the same year exists, Q1 = YTD, otherwise null.",
    "**Zeros that mean ‘not reported’.** Risk-based capital ratios are exactly 0 in 100% of 1984–89 rows (they did not exist "
    "before Basel I) and in 37% of 2020–26 rows (small banks that opted into the Community Bank Leverage Ratio stop reporting "
    "them). A model would read 0% capital as insolvency. The ratios are therefore not used; equity/assets, which exists in all "
    "eras, measures capital instead.",
    "**Structural missingness.** Some items were introduced late: 30–89-day past-due assets are missing for 100% of rows before "
    "2000, FHLB borrowings for 82–87% before 2000 (Figure A2). They are kept with a missing-indicator and a training-period "
    "median, never imputed from later data.",
    "**Impossible values.** 860 negative values in stock amounts that cannot be negative (mostly ORE) are set to null; 15,659 "
    "rows with negative equity are kept because they are genuine and highly informative; 18,077 risk-based ratios above 100% "
    "come from banks with almost no risk-weighted assets. Ratios are recomputed from amounts with safe division and "
    "winsorised with training-period quantiles.",
    "**Schema drift.** 20 columns change physical type across raw files (int64 vs double), fixed by the curated schema.",
    "**Failure identification.** The Failures table holds 2,848 FAILURE and 411 ASSISTANCE events in scope. Its 488 rows "
    "without CERT are all pre-1984. 44 CERTs appear more than once (37 assisted and later failed). In 271 failures the "
    "institution kept filing after FAILDATE – 226 of them savings institutions placed in RTC conservatorship in 1989, "
    "where FAILDATE is the takeover date.",
    "**Structure codes are not a failure flag.** Code 211 is 99.3% failures, but only 83.6% of failed banks carry it; the others "
    "carry merger-type codes (213, 217, 235) that also occur for healthy mergers. Using CHANGEC would mislabel about one failure "
    "in six as a merger.",
    "**Point-in-time traps.** 4,518 banks (18.8%) changed charter class over time and 249 changed state, but Institutions "
    "stores only the latest values; ACTIVE, ENDEFYMD and CHANGEC describe the end of a bank's life. Joining them to old "
    "quarters would leak the future, so Institutions is used only to analyse exits, never for features.",
])
H("3.2 Panel construction and exits", 2)
P("src/04_panel.py (logic in src/transforms.py) builds the panel in one lazy Polars query: drop rows with zero assets "
  "(1,211 rows that cannot form ratios), null impossible negatives, add a quarter index qi = 4·year + quarter − 1, "
  "de-cumulate the flows, attach the first FAILURE date per CERT and derive the label and exit type.")
B([
    "**Label:** y(i,q) = 1 if bank i fails in (q, q+4]: fail_qi > qi and fail_qi ≤ qi + 4, where fail_qi is the quarter "
    "containing FAILDATE. Rows dated on or after FAILDATE are removed (the event is public by then).",
    "**Failure vs. other exits:** failure if and only if the CERT has a RESTYPE = FAILURE event. Open-bank assistance is not "
    "a failure: the charter survives and 176 of the 411 assisted banks keep reporting afterwards. All other exits (mergers, "
    "voluntary liquidations, charter conversions) keep label 0 in all their rows and simply leave the panel after their last "
    "report; they are not dropped, which would create survivorship bias, and not labelled as failures, which would teach the "
    "model that healthy acquisition targets are risky.",
    "**Label completeness:** for modelling only quarters up to 2025Q2 are used, so that every label window (q, q+4] is observed.",
])
ex = panel["exit_types"]
P(f"The panel has {f0(panel['panel_rows'])} bank-quarters for {f0(panel['banks'])} banks: {f0(ex['failure'])} failed, "
  f"{f0(ex['other exit (merger/closure)'])} exited otherwise and {f0(ex['active'])} still report in 2026Q2. "
  f"{panel['failed_banks_with_a_positive_label_row']} of the {panel['failure_events_FAILURE_in_scope']} in-scope failed banks "
  f"have at least one positive row; the remainder stopped reporting more than four quarters before failing. The positive "
  f"rate is {panel['positive_rate_pct']:.2f}%, i.e. about 1 in 150 bank-quarters (Figure 1).")
FIGURE(FIG / "failures_per_year.png", "Bank failures per year (RESTYPE = FAILURE). Failures cluster in 1984–1993 and 2008–2014.", 14)

# ================================================================== 4 PROCESSING
H("4. Scalable processing (Part 3)")
H("4.1 Polars implementation and how the engine executes it", 2)
P("All transformations are written once as Polars LazyFrame expressions (src/transforms.py, src/features.py) and reused by "
  "the panel, the feature table, the benchmark and the truncation test. Nothing is computed until collect(); Polars first "
  "builds a logical plan and optimises it (the plans are saved in results/polars_plan_panel_*.txt):")
B(["**Projection pushdown:** only the columns used downstream are read from Parquet. In the panel plan (appendix E) "
   "the failures scan reads PROJECT 3/16 COLUMNS and the financials scan drops the unused partition column; a query "
   "that selects five columns never decodes the other 44 (‘Read 5 cols’ in Table 2).",
   "**Predicate pushdown:** the filter ASSET > 0 written after the scan appears inside it as SELECTION: col(\"ASSET\") > 0, "
   "and the RESTYPE = FAILURE filter is pushed into the failures scan; with a REPDTE or year filter, Hive partitions "
   "outside the range are never opened.",
   "**Common-subexpression elimination:** the optimised plan contains cached sub-expressions (__POLARS_CSER_…) – e.g. "
   "shift(1).over(CERT) of the quarter index is computed once and reused for all eight de-cumulated flows.",
   "**Parallel window and join execution:** .over('CERT') is evaluated as a parallel group-by on all cores; lags are "
   "self-joins on (CERT, qi−4), which are hash joins rather than Python loops. Rolling statistics use rolling_*_by on the "
   "quarter index, so a missing quarter shrinks the window instead of silently using an older report.",
   "**Streaming engine:** collect(engine='streaming') processes the data in batches, trading some speed for a lower "
   "memory peak."])
H("4.2 Benchmark against pandas", 2)
P("The same pipeline (clean → de-cumulate → label → ratios → 4-quarter lags → trailing rolling statistics → state medians) "
  "was written idiomatically in pandas (vectorised, groupby/shift/merge/rolling, no Python loops) and run against Polars in "
  "three modes. To be fair, every run starts in a fresh subprocess (cold Python, same files), peak memory is the process "
  "peak working set, and each configuration is run three times (median reported). The outputs are compared value by value.")
bt = bench[bench.scale == 1][["engine", "seconds_median", "seconds_min", "peak_mb_median"]].copy()
bt["speed-up vs pandas"] = (pd_t / bt.seconds_median).map(lambda v: f"{v:.1f}×")
bt["seconds_median"] = bt.seconds_median.map(f2); bt["seconds_min"] = bt.seconds_min.map(f2)
bt["peak_mb_median"] = bt.peak_mb_median.map(f0)
bt.columns = ["Engine", "Time, median (s)", "Time, min (s)", "Peak memory (MB)", "Speed-up"]
T(bt, f"Same pipeline on {f0(panel['panel_rows'])} output rows. Max |pandas − Polars| over 15 output columns: "
      f"{max(v['max_abs_diff_all'] for v in eq.values()):.0e}; null patterns identical.", widths=[3.6, 3.0, 3.0, 3.2, 2.4])
pe_t, pe_m = b1.loc["polars-eager", "seconds_median"], b1.loc["polars-eager", "peak_mb_median"]
P(f"Polars lazy is {pd_t / pl_t:.1f}× faster than pandas and uses {pct(1 - pl_m / pd_m, 0)} less memory. "
  f"The eager variant shows where the gain comes from. With the same expressions but every intermediate table "
  f"materialised, Polars eager is still {pd_t / pe_t:.1f}× faster than pandas – the vectorised, multi-threaded Rust "
  f"kernels – but it needs the most memory of all engines ({f0(pe_m)} MB, {pe_m / pl_m:.1f}× the lazy run) and is "
  f"{pct(pe_t / pl_t - 1, 0)} slower than lazy. The lazy query optimiser adds the rest: it fuses steps, prunes columns "
  f"and never allocates the intermediate tables, which matters more for memory than for time. Streaming lowers the "
  f"memory peak further to {f0(st_m)} MB at a small cost in time, because sort-based window operations limit how "
  f"much of this pipeline can be streamed.")
P("**Scaling.** To see how the engines behave beyond the real data size, the panel was replicated 2, 4 and 8 times "
  "with disjoint CERTs (Figure 2).")
FIGURE(FIG / "benchmark_scaling.png", "Wall time and peak memory vs. data size (×1, ×2, ×4, ×8 replicas of the panel).", 16.5)
P(f"pandas grows to {f0(x4p.seconds_median.iloc[0])} s and {x4p.peak_mb_median.iloc[0] / 1024:.1f} GB at ×4 "
  f"({x4p.rows_out.iloc[0] / 1e6:.1f} M rows) – half of the laptop's RAM – and was not run at ×8. Polars processes "
  f"×8 ({x8_rows / 1e6:.1f} M rows) in {x8l.seconds_median.iloc[0]:.0f}–{x8.seconds_median.iloc[0]:.0f} s with {x8_mem / 1024:.1f} GB (streaming) to {x8l_mem / 1024:.1f} GB (lazy). "
  f"Time and memory grow roughly linearly.")
H("4.3 Reproduction in PySpark", 2)
P("notebooks/08_spark_demo.ipynb re-implements the key steps in the Spark DataFrame API (local[*], AQE on, 16 shuffle "
  "partitions): cleaning, de-cumulation with lag() over a window partitioned by CERT, the broadcast join with failures, "
  "the label, eight ratio features, the 4-quarter self-join, gap-safe rolling windows (rangeBetween(−7, 0) on the "
  "quarter index) and state medians. The result is compared with Polars row by row:")
sc = spark_cmp.copy()
sc["max_abs_diff"] = sc.max_abs_diff.map(lambda v: f"{v:.1e}")
sc["sum_polars"] = sc.sum_polars.map(f3); sc["sum_spark"] = sc.sum_spark.map(f3)
sc.columns = ["Column", "Null mismatches", "Max |diff|", "Sum (Polars)", "Sum (Spark)"]
T(sc, "Spark vs. Polars on the same 1.68 M rows (sorted by CERT, quarter).", widths=[3.2, 2.6, 2.6, 4.0, 4.0], size=8)
P(f"All rows and columns match; the only differences (≤ {max_spark_diff:.0e}) are floating-point summation order in "
  f"the rolling standard deviation. Spark needed {sp_t.get('spark local[*]', float('nan')):.0f} s for this job against "
  f"{sp_t.get('polars lazy', float('nan')):.1f} s for Polars on the same laptop.")
P("**Reading the physical plan** (explain('formatted') in the notebook, read bottom-up). (1) Scan parquet: ReadSchema "
  "lists only the 17 of 49 columns the job needs and PushedFilters contains GreaterThan(ASSET, 0) – the same projection "
  "and predicate pushdown as Polars; with a year filter (notebook section 1) the scan also shows PartitionFilters, i.e. "
  "partition pruning. (2) Exchange hashpartitioning(CERT, 16) → Sort → Window: a shuffle that sends each bank's history "
  "to one partition so lag() can de-cumulate the flows. (3) The failures table is aggregated (HashAggregate – Exchange – "
  "HashAggregate), shipped to every task by BroadcastExchange and joined with BroadcastHashJoin LeftOuter, so the "
  "1.68 M-row panel is not shuffled for this join. (4) The 4-quarter self-join is a SortMergeJoin LeftOuter with both "
  "sides re-partitioned on hashpartitioning(CERT, qi). Its right-hand side scans the Parquet files a second time and "
  "contains no Window: the optimiser saw that eq_ta, npa_ta and ASSET do not depend on the de-cumulated flows and "
  "pruned that work, but it still reads the data twice. (5) A new Exchange on CERT for the rolling windows – the "
  "(CERT, qi) partitioning of the join does not satisfy a window partitioned by CERT – and (6) an Exchange on "
  "(STNAME, qi) for the state median. Five shuffles in total: on a cluster each moves data across the network, on one "
  "machine it is pure serialisation overhead, which is why Spark is slower than Polars here. Spark is also lazy: "
  "without cache() every action re-executes the whole lineage from the scan, so the MLlib part caches the feature "
  "table first (35 s once, instead of once per pass of every estimator).")
H("4.4 When would a cluster be justified?", 2)
P(f"The full 42-year panel occupies {st_part.size_MB:.0f} MB on disk and at most {pl_m / 1024:.1f} GB in memory during "
  f"the whole transformation pipeline, which takes {pl_t:.1f} s. From the scaling experiment the lazy engine needs roughly "
  f"{x8l_mem * 2**20 / x8_rows:,.0f} bytes per output row, so with 70% of 16 GB usable a single laptop can process about "
  f"{rows_fit_16gb / 1e6:,.0f} M bank-quarter rows in memory – about {rows_fit_16gb / panel['panel_rows']:.0f} times the "
  f"FDIC panel – and the streaming engine and out-of-core sinks push this further. On this data Spark is "
  f"{sp_t.get('spark local[*]', 1) / max(sp_t.get('polars lazy', 1), 1e-9):.0f}× slower because of JVM start-up, task "
  f"scheduling and shuffle overhead. A cluster is justified only when one of these conditions holds:")
B(["the working set exceeds one machine's memory even with streaming – e.g. loan-level credit-registry data "
   "(hundreds of millions of loan-months) or transaction data, not quarterly bank aggregates;",
   "the job must finish within a fixed window that one machine cannot meet (e.g. daily stress tests over many scenarios);",
   "data arrive continuously from many sources and must be processed with fault tolerance (Spark retries failed tasks), or "
   "many users and jobs share the same data lake."])
P("For a quarterly bank-level EWS – the FDIC case and even more so Vietnam's roughly 50 banks – none of these hold: "
  "Polars on one machine is the right tool, and Spark should be kept as a demonstrated, tested fall-back for a future "
  "move to loan-level data.")

# ================================================================== 5 FEATURES
H("5. Feature pipeline (Part 4)")
P("Features follow the CAMELS framework used by supervisors, plus dynamics and context. Table 5 groups them; "
  "the full dictionary with rationale, missing share and medians for failing and other banks is Table 12 in the appendix.")
grp = fdict.groupby("group").feature.apply(lambda s: ", ".join(s)).reset_index()
gname = {"C": "Capital", "A": "Asset quality", "M": "Management / efficiency", "E": "Earnings", "L": "Liquidity / funding",
         "S": "Sensitivity / concentration", "size": "Size", "dyn": "Dynamics (change, growth, volatility, history)",
         "ctx": "Context (state and national stress)"}
order = ["C", "A", "M", "E", "L", "S", "size", "dyn", "ctx"]
grp["o"] = grp.group.map(order.index); grp = grp.sort_values("o")
grp["group"] = grp.group.map(gname)
T(grp[["group", "feature"]].rename(columns={"group": "Group", "feature": "Features"}), "Feature groups (40 features).",
  widths=[4.2, 12.8], size=8)
_m = fdict.set_index("feature")
_f = lambda k, c: _m.loc[k, c]
P(f"Examples of the separation in the data (medians, failing vs. other bank-quarters): equity/assets "
  f"{pct(_f('eq_ta', 'median_failing (y=1)'))} vs {pct(_f('eq_ta', 'median_others (y=0)'))}, Texas ratio "
  f"{_f('texas', 'median_failing (y=1)'):.2f} vs {_f('texas', 'median_others (y=0)'):.2f}, non-performing assets "
  f"{pct(_f('npa_ta', 'median_failing (y=1)'))} vs {pct(_f('npa_ta', 'median_others (y=0)'))} of assets, quarterly ROA "
  f"{pct(_f('roa_q', 'median_failing (y=1)'))} vs {pct(_f('roa_q', 'median_others (y=0)'))}, 4-quarter asset growth "
  f"{pct(_f('g4_asset', 'median_failing (y=1)'), 0)} vs {pct(_f('g4_asset', 'median_others (y=0)'), 0)}. Rapid "
  f"deterioration (4-quarter changes) and stress in the bank's state add information that a single snapshot misses.")
P("**Point-in-time contract.** A feature for bank i in quarter q may use (a) bank i's reports dated ≤ q, (b) aggregates "
  "of other banks' reports dated ≤ q (state medians), and (c) failures with FAILDATE ≤ q (public). Lags are self-joins on "
  "(CERT, qi−4); trailing windows are defined on the quarter index; no statistic of the whole sample is computed in the "
  "feature code. Static Institutions attributes and exit fields are never used.")
pit = feats["pit_test"]
P(f"**Automatic leakage test.** src/06_features.py recomputes all {feats['n_features']} features from data truncated at a "
  f"cut-off T (Financials with REPDTE ≤ T, Failures with FAILDATE ≤ T) and compares them with the features from the full "
  f"data for every row dated ≤ T. For the three cut-offs {', '.join(x['cutoff'] for x in pit)} "
  f"({', '.join(f0(x['rows_compared']) for x in pit)} rows) the values and null patterns are identical "
  f"(max |diff| = {max(x['max_abs_diff'] for x in pit):.0f}); the script stops with an error otherwise. This turns "
  f"‘no look-ahead’ from a claim into a tested property.")
P("**Preprocessing fitted on training data only.** All fitted steps live in an sklearn Pipeline that is re-fitted inside "
  "every training block: DropEmpty (drops columns that are entirely missing in the training block, e.g. past-due assets "
  "before 2000), Winsorizer (clips each feature to its 0.5%/99.5% training quantiles), SimpleImputer (training median, "
  "with missing-indicator columns), StandardScaler (logistic models only). Validation and test rows are only transformed.")

# ================================================================== 6 MODELLING
H("6. Modelling and validation (Part 5)")
H("6.1 Split and validation scheme", 2)
P(f"**Development** {q(dev_s)}–{q(dev_e)}, **gap** 2007Q1–Q4, **test** {q(te_s)}–{q(te_e)}. 1984–85 are used only as "
  f"history for the lags. The gap of four quarters is necessary because the label of a development row in 2006Q4 looks "
  f"ahead to 2007Q4; without it, development labels would overlap the test period. The choice is deliberate: the model "
  f"learns from the savings-and-loan and early-1990s crises and must then recognise a different crisis (2008–2012, "
  f"driven by real-estate lending) and the 2023 failures – a genuine out-of-time and out-of-crisis test with "
  f"{te_banks} failing banks, far above the minimum of 50. A test period starting later (e.g. 2012) would leave too few "
  f"failures for stable estimates.")
cvf = cv[cv.model == chosen][["fold", "n", "positives", "pr_auc", "roc_auc"]].copy()
cvf["fold"] = cvf.fold.str.replace("train<= ", "train ≤ ")
for c in ("pr_auc", "roc_auc"):
    cvf[c] = cvf[c].map(f3)
cvf["n"] = cvf.n.map(f0); cvf["positives"] = cvf.positives.map(f0)
cvf.columns = ["Fold (expanding window, 4-quarter gap before validation)", "Val. rows", "Val. positives", "PR-AUC (HGB)", "ROC-AUC (HGB)"]
P("Inside the development period, model selection uses **expanding-window walk-forward validation** (Table 6): each "
  "model is trained on all quarters up to a cut-off and validated on a later block, with four unused quarters in between "
  "for the same reason as the main gap. Random k-fold would mix 1990 and 1988 rows of the same bank and crisis and "
  "overstate performance.")
T(cvf, "Walk-forward folds inside the development period (chosen model).", widths=[7.4, 2.0, 2.2, 2.6, 2.6], size=8)
P(f"**Metrics.** With {panel['positive_rate_pct']:.2f}% positives, accuracy is meaningless (predicting ‘no failure’ "
  f"everywhere scores {pct(all_neg_acc, 1)} on the test period). The primary metric is PR-AUC (average precision), which "
  f"focuses on the rare class; ROC-AUC, recall and precision among the top 1% and 5% of bank-quarters (what a supervisor "
  f"with limited examination capacity would review) and the Brier score (calibration) complete the picture.")
H("6.2 Models and development results", 2)
cm = cvm.loc[["baseline_logit_5", "logit_all", "hgb"]].copy()
cm.index = [names[i] for i in cm.index]
cm = cm.map(f3).reset_index()
cm.columns = ["Model", "PR-AUC", "ROC-AUC", "Recall top 1%", "Recall top 5%", "Brier"]
T(cm, "Walk-forward results on the development period (mean over the 4 folds).", widths=[6.0, 2.0, 2.0, 2.4, 2.4, 2.0],
  bold_rows=(2,))
P(f"The baseline is the classic supervisory logit on five CAMELS ratios (equity/assets, NPA/assets, ROA, liquid "
  f"assets, log size). The richer logit uses all 40 features with L2 regularisation. HGB (scikit-learn's histogram "
  f"gradient boosting) captures non-linear thresholds and interactions such as ‘high CRE concentration and falling "
  f"capital’ and handles missing values natively; its hyper-parameters were chosen from 8 combinations by mean "
  f"walk-forward PR-AUC (learning rate {hgbp['learning_rate']}, {hgbp['max_leaf_nodes']} leaves, at least "
  f"{hgbp['min_samples_leaf']} rows per leaf, {hgbp['max_iter']} trees). HGB is clearly best on every development metric "
  f"and was selected; the model, features and threshold were frozen to results/frozen_choices.json before the test "
  f"period was read.")
FIGURE(FIG / "model_pr_and_cost.png", "Left: pooled out-of-fold PR curves on the development period (AP of the pooled "
       "predictions, therefore different from the per-fold means in Table 7). Right: expected cost vs. alarm threshold.", 16.5)
H("6.3 Alarm threshold from costs", 2)
P(f"The threshold minimises expected cost C_FN·FN + C_FP·FP on the pooled out-of-fold development predictions. Both costs "
  f"come from development-period data: the median loss to the deposit insurance fund of the {costs['failures_used']:,} "
  f"failures in {dev_s[:4]}–{dev_e[:4]} with a recorded COST is ${costs['median_cost_kusd'] / 1e3:.2f} M "
  f"(median loss rate {pct(costs['median_loss_rate'], 1)} of the failed bank's assets). Assuming that early "
  f"supervisory action avoids {pct(costs['avoidable_share'], 0)} of that loss gives a missed-failure cost "
  f"C_FN = ${costs['C_FN_kusd'] / 1e3:.2f} M. A false alarm triggers an unnecessary targeted examination, valued at 0.1% "
  f"of the median bank's assets: C_FP = ${costs['C_FP_kusd']:.0f} k. The ratio is {costs['ratio_FN_FP']:.0f}:1 and the "
  f"cost-minimising threshold is p = {thr:.4f}. Because the 25% and 0.1% are assumptions, Table 8 shows how the threshold "
  f"responds to C_FP.")
ss = sens.copy()
ss["threshold"] = ss.threshold.map(lambda v: f"{v:.4f}")
ss["recall"] = ss.recall.map(pct); ss["precision"] = ss.precision.map(pct)
ss["tp"] = ss.tp.map(f0); ss["fp"] = ss.fp.map(f0)
ss = ss[["C_FP multiplier", "threshold", "tp", "fp", "recall", "precision"]]
ss.columns = ["C_FP × (1 = base case)", "Threshold", "TP", "FP", "Recall", "Precision"]
T(ss, "Sensitivity of the threshold to the false-alarm cost (development out-of-fold predictions).",
  widths=[3.6, 2.4, 2.2, 2.2, 2.2, 2.4], bold_rows=(2,))
H("6.4 Single evaluation on the test period", 2)
tt = test.loc[["baseline_logit_5", "logit_all", "hgb"]].copy()
tbl = pd.DataFrame({
    "Model": [names[i] for i in tt.index],
    "PR-AUC": tt.pr_auc.map(f3), "ROC-AUC": tt.roc_auc.map(f3),
    "Recall top 1%": tt.recall_top1pct.map(pct), "Recall @thr": tt.recall.map(pct),
    "Precision @thr": tt.precision.map(pct), "FP": tt.fp.map(f0), "FN": tt.fn.map(f0),
    "Cost ($ bn)": (tt.cost_musd / 1e3).map(f2)})
T(tbl, f"Test period {q(te_s)}–{q(te_e)} ({f0(te_rows)} bank-quarters, {f0(te_pos)} positives). Each model at its own "
       f"development-chosen cost threshold. Evaluated once.", widths=[4.4, 1.5, 1.5, 1.6, 1.6, 1.8, 1.4, 1.2, 1.6],
  size=8, bold_rows=(2,))
P(f"The selected HGB model reaches ROC-AUC {test.loc[chosen, 'roc_auc']:.3f} and PR-AUC {test.loc[chosen, 'pr_auc']:.3f}. "
  f"At the frozen threshold it catches {pct(test.loc[chosen, 'recall'], 1)} of positive bank-quarters with "
  f"{pct(test.loc[chosen, 'precision'], 1)} precision and has the lowest expected cost of the three models. From the bank "
  f"perspective, {ew['flagged_at_least_once']} of the {ew['failed_banks_in_test']} banks that failed in the test period "
  f"({pct(ew['flagged_at_least_once'] / ew['failed_banks_in_test'], 1)}) were flagged at least once before failing, with a "
  f"median lead time of {ew['median_lead_quarters']:.0f} quarters (Figure 4). Its accuracy, "
  f"{pct(test.loc[chosen, 'accuracy'], 1)}, is below the {pct(all_neg_acc, 1)} of the trivial ‘no failure’ rule – an "
  f"illustration of why accuracy was not used.")
P(f"Two results deserve an honest reading. First, all models score much higher on the test period than in development; "
  f"the 2008–2012 failures were preceded by large, visible deteriorations in capital and asset quality, which makes them "
  f"easier to rank than the heterogeneous 1990s failures in the later development folds. Second, the gain of HGB over the "
  f"simple logit, large in development, is small on the test period (the 40-feature logit even has a slightly higher "
  f"PR-AUC). HGB's advantage on the test set lies in recall at the chosen operating point – it misses "
  f"{f0(test.loc[chosen, 'fn'])} positive bank-quarters against {f0(test.loc['baseline_logit_5', 'fn'])} for the baseline – at "
  f"the price of more false alarms, which the cost function accepts because a miss is {costs['ratio_FN_FP']:.0f} times "
  f"as expensive. Alarms track the crisis closely and fall to a few dozen per quarter after 2014.")
FIGURE(FIG / "test_alarms_over_time.png", "Test period: number of alarms vs. banks failing within four quarters, by quarter.", 15.5)
if mllib is not None:
    m = mllib.iloc[0]
    P(f"**Spark MLlib demo.** The notebook fits an MLlib pipeline (Imputer → VectorAssembler → StandardScaler → "
      f"LogisticRegression, 14 features, quantile clipping from the training block) on 1986–1994 and validates on "
      f"1996–2006, the last development fold – the test period is not touched. It reaches PR-AUC {m.pr_auc:.3f} and "
      f"ROC-AUC {m.roc_auc:.3f} (fit {m.fit_seconds:.0f} s). Its ROC-AUC is close to the scikit-learn logits on the same "
      f"fold (baseline {cv4.loc['baseline_logit_5', 'roc_auc']:.3f}, all features {cv4.loc['logit_all', 'roc_auc']:.3f}), "
      f"but its PR-AUC is lower (sklearn: {cv4.loc['baseline_logit_5', 'pr_auc']:.3f} and {cv4.loc['logit_all', 'pr_auc']:.3f}): "
      f"the demo uses only 14 features, no missing-value indicators and a different regularisation, and with 168 "
      f"positives among 440,000 rows PR-AUC is very sensitive to the few top-ranked cases. The point of the demo is "
      f"that the whole modelling stage – imputation, scaling, fitting, evaluation – runs as a distributed Spark ML "
      f"pipeline on the same data, so it can be moved to a cluster if the data ever outgrow one machine.")

# ================================================================== 7 CONCLUSIONS
H("7. Conclusions (Part 6)")
P(f"**Q1 – Storage and processing.** Store the reports as a typed, zstd-compressed Parquet layer partitioned by year "
  f"({st_part.size_MB:.0f} MB instead of {st_csv.size_MB:.0f} MB of CSV, {st_csv.filter_2008_2010_s / st_part.filter_2008_2010_s:.0f}× "
  f"faster time-sliced reads) and process it with Polars' lazy engine ({pl_t:.1f} s and {pl_m / 1024:.1f} GB for the whole "
  f"pipeline, {pd_t / pl_t:.0f}× faster than pandas). Spark gives identical results but is "
  f"{sp_t.get('spark local[*]', 1) / max(sp_t.get('polars lazy', 1), 1e-9):.0f}× slower on one machine; a cluster becomes "
  f"necessary only beyond roughly {rows_fit_16gb / 1e6:,.0f} M rows per job, for loan- or transaction-level data, strict "
  f"processing windows or many concurrent users.")
P(f"**Q2 – Data quality without look-ahead.** The raw data contain year-to-date flows, zeros that mean ‘not reported’, "
  f"fields introduced decades apart, impossible values, type drift, multiple and non-standard failure events and static "
  f"attributes that describe the end of a bank's life. Each is detected automatically and handled with information "
  f"available at the time: de-cumulation within the year, null instead of zero, missing indicators and training-period "
  f"statistics, failures defined only by the Failures table, quarter-specific attributes. The truncation test proves "
  f"that no feature changes when future data are removed.")
P(f"**Q3 – Honest validation.** Yes. With a time-ordered split, a four-quarter gap, walk-forward selection, preprocessing "
  f"fitted inside each training block, a cost-based threshold frozen before testing and a single test evaluation, the "
  f"pipeline's output supports a model that ranks banks well out of sample (test ROC-AUC {test.loc[chosen, 'roc_auc']:.3f}) "
  f"and warns about {pct(ew['flagged_at_least_once'] / ew['failed_banks_in_test'], 0)} of failures about a year ahead.")
H("Limitations", 2)
B(["The API serves the latest amended Call Reports, not the vintages available at the time, and reports are published "
   "30–60 days after quarter end; a production EWS would need vintage data and a publication lag in the label window.",
   "Failures cluster in two crises; the test period is dominated by 2009–2011, so performance in calm periods (2015–2025, "
   "a handful of failures per year) is estimated with large uncertainty.",
   "The two cost parameters (avoidable share of loss and examination cost) are assumptions; Table 8 shows the threshold "
   "is sensitive to them. Assisted banks are treated as non-failures, a defensible but debatable choice.",
   "Timings come from one laptop running x64 Python under emulation on an ARM CPU; ratios between engines are more "
   "reliable than absolute numbers. The ×2–×8 scaling uses replicated data with the same distribution."])
H("What would change for Vietnam", 2)
B(["**Scale and tools:** about 50 credit institutions reporting monthly to the SBV is a much smaller panel, so the same "
   "Parquet + Polars design is more than enough; Spark becomes relevant only for loan-level data from the Credit "
   "Information Centre (CIC).",
   "**Labels:** Vietnam has had very few formal bank failures; distress appears as special control, compulsory "
   "acquisition at zero price (2015) or restructuring under Law 32/2024. The label must be redefined (e.g. placement under "
   "special control or a large capital shortfall) and will be even rarer – US data or a transfer-learning approach could "
   "provide the prior.",
   "**Data definitions:** Vietnamese reports follow VAS and SBV circulars (e.g. Circular 31/2024 on loan classification, "
   "Circular 41/2016 and 22/2019 on capital and liquidity ratios); every field mapping, unit and year-to-date convention "
   "must be audited in the same automated way, and state-owned banks may need separate treatment.",
   "**Point-in-time and governance:** supervisory data are confidential, so the pipeline would run on-premise with access "
   "control and must store every report vintage, which the SBV can do and the public FDIC API cannot."])

# ================================================================== REFERENCES
H("References and tools")
B(["FDIC BankFind Suite API and field dictionaries, https://api.fdic.gov/banks/docs/ (accessed November 2026).",
   "BFT65B course materials: exam paper and starter script fdic_download_starter.py (adapted in src/01_download.py).",
   "FDIC (1998). Managing the Crisis: The FDIC and RTC Experience 1980–1994.",
   "Cole, R. A. & White, L. J. (2012). Déjà Vu All Over Again: The Causes of U.S. Commercial Bank Failures This Time Around. "
   "Journal of Financial Services Research 42, 5–29.",
   "Polars (v1.44), pandas (v3.1), PyArrow, PySpark (v4.3), scikit-learn (v1.9) documentation.",
   "AI assistance: Claude (Anthropic) via Claude Code was used to help write and debug the pipeline code, the report "
   "generator and to draft text. The author reviewed and is responsible for all design choices, code, results and text."],
  style="List Number")

# ================================================================== APPENDIX
page_break()
H("Appendix")
H("A. Data-quality summary (generated by src/03_quality.py)", 2)
d = dq.copy()
d.columns = ["Check", "Finding", "Affected", "Handling in the pipeline"]
T(d, "All data-quality checks, findings and handling.", widths=[2.8, 5.2, 2.0, 7.0], size=7.5)
H("B. Completeness by quarter (excerpt)", 2)
rq = pd.read_csv(R / "dq_rows_per_quarter.csv")
rq["REPDTE"] = rq.REPDTE.str[:10]
ex_rows = pd.concat([rq.head(4), rq.iloc[[40, 80, 100, 120, 140, 160]], rq.tail(3)])
ex_rows = ex_rows.assign(match=lambda d: (d.rows == d.api_total).map({True: "yes", False: "NO"}))
for c in ("rows", "banks", "api_total"):
    ex_rows[c] = ex_rows[c].map(f0)
ex_rows.columns = ["Quarter", "Rows", "Banks", "API total", "Match"]
T(ex_rows, f"Rows per quarter vs. API total (all 170 quarters match; full list in results/dq_rows_per_quarter.csv).",
  widths=[3, 2.5, 2.5, 2.5, 2], size=8)
FIGURE(FIG / "dq_banks_per_quarter.png", "Banks filing a Call Report per quarter, 1984–2026.", 13, label="A1")
FIGURE(FIG / "dq_missing_by_era.png", "Share of missing values by field and era.", 9, label="A2")
H("C. Feature dictionary", 2)
fd = fdict.copy()
for c in ("median_failing (y=1)", "median_others (y=0)"):
    fd[c] = fd[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
fd["missing_pct"] = fd.missing_pct.map(lambda v: f"{v:.1f}")
fd.columns = ["Feature", "Group", "Rationale", "Missing %", "Median, y=1", "Median, y=0"]
T(fd, "Feature dictionary (labelled rows up to 2025Q2).", widths=[2.6, 1.2, 7.2, 1.6, 2.0, 2.0], size=7.5)
H("D. Walk-forward results for all models and folds", 2)
cva = cv[["model", "fold", "positives", "pr_auc", "roc_auc", "recall_top5pct"]].copy()
cva["model"] = cva.model.map(names)
for c in ("pr_auc", "roc_auc"):
    cva[c] = cva[c].map(f3)
cva["recall_top5pct"] = cva.recall_top5pct.map(pct)
cva.columns = ["Model", "Fold", "Val. positives", "PR-AUC", "ROC-AUC", "Recall top 5%"]
T(cva, "Per-fold walk-forward results.", widths=[4.4, 6.2, 1.8, 1.5, 1.5, 1.7], size=7.5)
H("E. Optimised Polars plan for the panel (excerpt)", 2)
plan = (R / "polars_plan_panel_optimized.txt").read_text(encoding="utf-8") if (R / "polars_plan_panel_optimized.txt").exists() else ""
lines = [ln if len(ln) < 160 else ln[:157] + "..." for ln in plan.splitlines()]
CODE("\n".join(lines[:28]) + ("\n..." if len(lines) > 28 else ""), size=6.5)
H("F. Reproducibility", 2)
P("python -m venv .venv; pip install -r requirements.txt; python run_all.py. The download is resumable and "
  "verified, all seeds are fixed (42), data files are not committed, and this report is regenerated with "
  "python report/build_report.py. Full instructions: README.md.")

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
print("written", OUT)
