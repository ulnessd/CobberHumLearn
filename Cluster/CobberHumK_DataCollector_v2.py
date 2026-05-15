#!/usr/bin/env python3
"""
CobberHumK_DataCollector_v2.py

Robust terminal-only data collector for CobberHumK.

This version fixes two issues that can occur with the World Bank API:
1. Large per_page requests can time out.
   v2 uses smaller paginated requests with retries.
2. Indicator rows should be merged by `countryiso3code`.
   v2 uses countryiso3code when available, which avoids 0-row merges.

Run:
    python CobberHumK_DataCollector_v2.py --start-year 2015 --end-year 2023 --min-feature-count 8

Output:
    CobberHumKData/wdi_human_development_cluster.csv
    CobberHumKData/wdi_human_development_cluster_full.csv
    CobberHumKData/wdi_human_development_metadata.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests


WORLD_BANK_BASE = "https://api.worldbank.org/v2"

INDICATORS = [
    {
        "code": "SE.ADT.LITR.FE.ZS",
        "short": "female_adult_literacy_pct",
        "label": "Female adult literacy (%)",
        "direction": "higher_is_better",
        "theme": "education_gender",
    },
    {
        "code": "SE.ADT.1524.LT.FE.ZS",
        "short": "female_youth_literacy_pct",
        "label": "Female youth literacy (%)",
        "direction": "higher_is_better",
        "theme": "education_gender",
    },
    {
        "code": "SE.PRM.NENR.FE",
        "short": "female_primary_net_enrollment_pct",
        "label": "Female primary net enrollment (%)",
        "direction": "higher_is_better",
        "theme": "education_gender",
    },
    {
        "code": "SE.SEC.ENRR.FE",
        "short": "female_secondary_gross_enrollment_pct",
        "label": "Female secondary gross enrollment (%)",
        "direction": "higher_is_better",
        "theme": "education_gender",
    },
    {
        "code": "SE.TER.ENRR.FE",
        "short": "female_tertiary_gross_enrollment_pct",
        "label": "Female tertiary gross enrollment (%)",
        "direction": "higher_is_better",
        "theme": "education_gender",
    },
    {
        "code": "SE.ENR.PRSC.FM.ZS",
        "short": "primary_secondary_gender_parity",
        "label": "Primary/secondary enrollment gender parity index",
        "direction": "near_1_is_parity",
        "theme": "education_gender",
    },
    {
        "code": "SP.DYN.LE00.IN",
        "short": "life_expectancy_years",
        "label": "Life expectancy at birth (years)",
        "direction": "higher_is_better",
        "theme": "health",
    },
    {
        "code": "SH.STA.MMRT",
        "short": "maternal_mortality_per_100k",
        "label": "Maternal mortality ratio (per 100,000 live births)",
        "direction": "lower_is_better",
        "theme": "health_gender",
    },
    {
        "code": "SP.DYN.TFRT.IN",
        "short": "fertility_rate_births_per_woman",
        "label": "Fertility rate (births per woman)",
        "direction": "context_dependent",
        "theme": "demography",
    },
    {
        "code": "NY.GDP.PCAP.CD",
        "short": "gdp_per_capita_usd",
        "label": "GDP per capita (current US$)",
        "direction": "higher_is_better",
        "theme": "economy",
    },
    {
        "code": "IT.NET.USER.ZS",
        "short": "internet_users_pct",
        "label": "Individuals using the Internet (%)",
        "direction": "higher_is_better",
        "theme": "digital_access",
    },
    {
        "code": "SL.TLF.CACT.FE.ZS",
        "short": "female_labor_force_participation_pct",
        "label": "Female labor force participation (%)",
        "direction": "context_dependent",
        "theme": "work_gender",
    },
]


def get_json(url: str, params: Dict[str, Any], retries: int = 4, timeout: int = 90) -> Any:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            wait = 2.0 * attempt
            print(f"    request attempt {attempt}/{retries} failed; retrying in {wait:.1f}s")
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"Request failed: {url} params={params} error={last_error}")


def fetch_country_metadata() -> pd.DataFrame:
    print("Fetching country metadata...")
    url = f"{WORLD_BANK_BASE}/country"

    # Country metadata is small, but still paginate defensively.
    first = get_json(url, {"format": "json", "per_page": 100, "page": 1})
    if not isinstance(first, list) or len(first) < 2:
        raise RuntimeError("Unexpected country metadata response.")

    meta = first[0]
    pages = int(meta.get("pages", 1))
    all_rows = first[1]

    for page in range(2, pages + 1):
        data = get_json(url, {"format": "json", "per_page": 100, "page": page})
        all_rows.extend(data[1])

    rows = []
    for c in all_rows:
        region = c.get("region") or {}
        income = c.get("incomeLevel") or {}

        # region id NA means aggregate/non-country in the World Bank metadata.
        if region.get("id") == "NA":
            continue
        if income.get("id") in {"", "NA", None}:
            continue

        rows.append({
            "country_code": c.get("id"),  # ISO3 code in the World Bank country endpoint.
            "country_name": c.get("name"),
            "region": region.get("value", ""),
            "region_id": region.get("id", ""),
            "income_group": income.get("value", ""),
            "income_group_id": income.get("id", ""),
            "capital_city": c.get("capitalCity", ""),
            "longitude": c.get("longitude", ""),
            "latitude": c.get("latitude", ""),
        })

    df = pd.DataFrame(rows).sort_values("country_name").reset_index(drop=True)
    print(f"  countries retained: {len(df)}")
    return df


def fetch_indicator(indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    print(f"Fetching {indicator_code}...")
    url = f"{WORLD_BANK_BASE}/country/all/indicator/{indicator_code}"

    # Smaller page sizes avoid the timeout you saw with per_page=20000.
    per_page = 1000
    first = get_json(url, {
        "format": "json",
        "per_page": per_page,
        "page": 1,
        "date": f"{start_year}:{end_year}",
    })

    if not isinstance(first, list) or len(first) < 2:
        print(f"  warning: unexpected response for {indicator_code}")
        return pd.DataFrame(columns=["country_code", "country_name_api", "year", "value", "indicator_code"])

    meta = first[0] or {}
    pages = int(meta.get("pages", 1))
    rows_raw = first[1] or []

    print(f"  pages: {pages}")

    for page in range(2, pages + 1):
        data = get_json(url, {
            "format": "json",
            "per_page": per_page,
            "page": page,
            "date": f"{start_year}:{end_year}",
        })
        rows_raw.extend(data[1] or [])

    rows = []
    for item in rows_raw:
        country = item.get("country") or {}
        code = item.get("countryiso3code") or country.get("id")
        val = item.get("value")
        if code in {"", None}:
            continue
        rows.append({
            "country_code": code,
            "country_name_api": country.get("value"),
            "year": int(item.get("date")),
            "value": val,
            "indicator_code": indicator_code,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  no rows returned")
        return df

    df = df[df["value"].notna()].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()

    print(f"  non-missing rows: {len(df)}")
    return df


def latest_values(df: pd.DataFrame, short_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["country_code", short_name, short_name + "_year"])
    latest = (
        df.sort_values(["country_code", "year"])
          .groupby("country_code", as_index=False)
          .tail(1)
          [["country_code", "value", "year"]]
          .copy()
    )
    latest = latest.rename(columns={"value": short_name, "year": short_name + "_year"})
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-feature-count", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="CobberHumKData")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    args = parser.parse_args()

    print("CobberHumK WDI data collector v2")
    print("===============================")
    print(f"Year search window: {args.start_year}-{args.end_year}")
    print(f"Minimum feature count: {args.min_feature_count}")
    print()

    countries = fetch_country_metadata()
    dataset = countries.copy()

    metadata_indicators: List[Dict[str, Any]] = []

    for ind in INDICATORS:
        short = ind["short"]
        try:
            raw = fetch_indicator(ind["code"], args.start_year, args.end_year)
            latest = latest_values(raw, short)
            dataset = dataset.merge(latest, on="country_code", how="left")
            count = int(dataset[short].notna().sum())
            print(f"  latest values for {short}: {count}/{len(dataset)}")
            meta = dict(ind)
            meta["non_missing_country_count"] = count
            meta["fetch_status"] = "ok"
        except Exception as exc:
            print(f"  WARNING: failed to fetch {ind['code']}: {exc}")
            dataset[short] = pd.NA
            dataset[short + "_year"] = pd.NA
            meta = dict(ind)
            meta["non_missing_country_count"] = 0
            meta["fetch_status"] = "failed"
            meta["error"] = str(exc)
            if not args.continue_on_error:
                raise
        metadata_indicators.append(meta)
        print()

    feature_cols = [ind["short"] for ind in INDICATORS]
    dataset["feature_count_available"] = dataset[feature_cols].notna().sum(axis=1)
    dataset["feature_count_missing"] = len(feature_cols) - dataset["feature_count_available"]
    dataset["included_for_default_gui"] = dataset["feature_count_available"] >= args.min_feature_count

    default_df = dataset[dataset["included_for_default_gui"]].copy()

    print("Missingness summary:")
    print(dataset[feature_cols].isna().sum().sort_values(ascending=False).to_string())
    print()
    print(f"Countries included in default GUI dataset: {len(default_df)}/{len(dataset)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_path = outdir / "wdi_human_development_cluster_full.csv"
    default_path = outdir / "wdi_human_development_cluster.csv"
    meta_path = outdir / "wdi_human_development_metadata.json"

    dataset.to_csv(full_path, index=False)
    default_df.to_csv(default_path, index=False)

    metadata = {
        "description": "Curated WDI dataset for CobberHumK clustering/classification lab.",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "min_feature_count": args.min_feature_count,
        "feature_columns": feature_cols,
        "indicators": metadata_indicators,
        "rows_full": int(len(dataset)),
        "rows_default_gui": int(len(default_df)),
        "source_note": "Fetched from the World Bank API. The student GUI ships with the resulting CSV.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("Saved:")
    print(f"  {full_path}")
    print(f"  {default_path}")
    print(f"  {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
