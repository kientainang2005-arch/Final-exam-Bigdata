"""
Step 1 - Acquisition: download FDIC BankFind data via the public API (no key needed).

Adapted from the course starter script `fdic_download_starter.py` (BFT65B course page).
Changes vs. the starter:
  * extended field list (CRE loan mix, funding structure, point-in-time BKCLASS/SPECGRP ...)
  * raw "landing zone": one Parquet file per quarter, written atomically (tmp -> rename)
  * resumable: quarters already complete in the manifest are skipped
  * manifest records rows, API total, timestamp and duration for every quarter
  * all failure fields and the structure-change fields of /institutions are kept

Usage
    python src/01_download.py --start 2023-03-31 --end 2023-12-31   # quick test
    python src/01_download.py                                        # full scope 1984Q1-2026Q2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000  # maximum rows the API returns per call

# Field codes verified against https://api.fdic.gov/banks/docs/risview_properties.yaml
FIN_FIELDS = [
    # keys + point-in-time descriptors (as reported in that quarter)
    "CERT", "REPDTE", "NAME", "STNAME", "BKCLASS", "SPECGRP",
    # balance sheet & liquidity
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "COREDEP", "VOLIAB",
    "FREPP", "OTHBFHLB",
    # loans & reserves
    "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNREMULT", "LNRENRES", "LNRERES",
    "LNCI", "LNCON", "LNAG", "LNATRES",
    # asset quality
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    # earnings (flows are year-to-date in the Call Report!)
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "EQCDIV",
    # ratios (%)
    "ROA", "ROE", "NIMY", "EQV",
    # regulatory capital ratios (%)
    "RBC1AAJ", "RBCRWAJ", "IDT1RWAJR",
]
FAIL_FIELDS = ["CERT", "NAME", "CITYST", "PSTALP", "FIN", "FAILDATE", "FAILYR", "RESDATE",
               "SAVR", "FUND", "CHCLASS1", "RESTYPE", "RESTYPE1", "COST", "QBFASSET", "QBFDEP"]
INST_FIELDS = (["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "INACTIVE", "ESTYMD", "INSDATE",
                "ENDEFYMD", "EFFDATE", "INSDROPDATE", "NEWCERT", "REPDTE", "SPECGRP", "CHRTAGNT"]
               + [f"CHANGEC{i}" for i in range(1, 16)])

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-student-pipeline/1.0"})


def get_json(endpoint: str, params: dict, max_retries: int = 8, pause: float = 0.5) -> dict:
    """GET {BASE}/{endpoint}; retry with exponential back-off on HTTP or network errors."""
    url = f"{BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)
                return r.json()
            print(f"    HTTP {r.status_code} (attempt {attempt}) - retrying", flush=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"    {type(exc).__name__} (attempt {attempt}) - retrying", flush=True)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Giving up on {url} with {params}")


def fetch_all(endpoint: str, fields: list[str], filters: str | None = None,
              sort_by: str = "CERT") -> tuple[pd.DataFrame, int]:
    """Page through an endpoint with limit/offset. Returns (DataFrame, total reported by API)."""
    rows, offset, total = [], 0, None
    while True:
        params = {"fields": ",".join(fields), "limit": PAGE_SIZE, "offset": offset,
                  "sort_by": sort_by, "sort_order": "ASC", "format": "json"}
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params)
        total = js["meta"]["total"]
        page = [rec["data"] for rec in js["data"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    df = pd.DataFrame(rows)
    # make sure every requested column exists (API omits fields that are null for all rows)
    for f in fields:
        if f not in df.columns:
            df[f] = None
    return df[fields], total


def quarter_ends(start: date, end: date) -> list[str]:
    ends = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                ends.append(q.strftime("%Y%m%d"))
    return ends


def write_atomic(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)


def download_financials(out: Path, quarters: list[str]) -> None:
    fin_dir = out / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest_financials.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for i, q in enumerate(quarters, 1):
        target = fin_dir / f"REPDTE={q}.parquet"
        if target.exists() and manifest.get(q, {}).get("complete"):
            print(f"[{i}/{len(quarters)}] {q}: already complete, skipped", flush=True)
            continue
        t0 = time.time()
        df, total = fetch_all("financials", FIN_FIELDS, filters=f"REPDTE:{q}")
        if df.empty:
            print(f"[{i}/{len(quarters)}] {q}: no data (not published yet?)", flush=True)
            continue
        write_atomic(df, target)
        complete = len(df) == total and not df.duplicated(["CERT", "REPDTE"]).any()
        manifest[q] = {"rows": len(df), "api_total": total, "complete": bool(complete),
                       "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "seconds": round(time.time() - t0, 1)}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{i}/{len(quarters)}] {q}: {len(df):>6,} rows (API total {total:,}) "
              f"{'OK' if complete else 'INCOMPLETE'} {time.time() - t0:5.1f}s", flush=True)


def download_small_tables(out: Path) -> None:
    meta = {}
    for name, fields, sort_by in (("failures", FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", INST_FIELDS, "CERT")):
        t0 = time.time()
        df, total = fetch_all(name, fields, sort_by=sort_by)
        write_atomic(df, out / f"{name}.parquet")
        meta[name] = {"rows": len(df), "api_total": total,
                      "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"{name:<12}: {len(df):>6,} rows (API total {total:,}) {time.time() - t0:5.1f}s", flush=True)
    (out / "manifest_small_tables.json").write_text(json.dumps(meta, indent=1))


def check(out: Path) -> bool:
    manifest = json.loads((out / "manifest_financials.json").read_text())
    panel = pd.read_parquet(out / "financials", columns=["CERT", "REPDTE"])
    n_dup = int(panel.duplicated(["CERT", "REPDTE"]).sum())
    bad = [q for q, m in manifest.items() if not m["complete"]]
    fails = pd.read_parquet(out / "failures.parquet")
    print("\n=== Completeness check ===")
    print(f"quarters downloaded     : {len(manifest)}")
    print(f"total panel rows        : {len(panel):,}  (sum of API totals: "
          f"{sum(m['api_total'] for m in manifest.values()):,})")
    print(f"distinct banks (CERT)   : {panel['CERT'].nunique():,}")
    print(f"duplicate CERT-REPDTE   : {n_dup}")
    print(f"incomplete quarters     : {bad if bad else 'none'}")
    print(f"failures in list        : {len(fails):,}")
    return n_dup == 0 and not bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1984-03-31")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out", default="data_raw/fdic")
    ap.add_argument("--skip-small", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quarters = quarter_ends(date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"Downloading {len(quarters)} quarters: {quarters[0]} -> {quarters[-1]} into {out.resolve()}\n")
    download_financials(out, quarters)
    if not args.skip_small:
        print()
        download_small_tables(out)
    ok = check(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
