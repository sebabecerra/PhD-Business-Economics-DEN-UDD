# ============================================================
# AI Entrepreneurship Paper - Empirical Pipeline
# Author: Sebastián Becerra
# Purpose:
#   1) Download BFS monthly state-level data from Census API
#   2) Merge with a pre-built AI exposure measure
#   3) Estimate baseline DiD
#   4) Prepare skeleton for BDS annual outcomes
#
# Requirements:
#   pip install pandas requests statsmodels linearmodels numpy
# ============================================================

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BFS_BASE_URL = "https://api.census.gov/data/timeseries/eits/bfs"
BDS_BASE_URL = "https://api.census.gov/data/timeseries/bds"

# ChatGPT public release: Nov 2022
AI_SHOCK_DATE = pd.Timestamp("2022-11-30")

# Optional Census API key. Most small pulls work without one.
CENSUS_API_KEY = None  # e.g. "YOUR_KEY"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

FIPS_TO_STATE_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS.items()}


def _safe_get_json(url: str, params: dict, sleep: float = 0.2) -> list:
    """GET JSON with basic error handling."""
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    time.sleep(sleep)
    return r.json()


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


# ------------------------------------------------------------
# BFS Downloader
# ------------------------------------------------------------

def fetch_bfs_state_monthly(
    series: str = "BA_BA",   # total business applications; update if needed
    start_year: int = 2019,
    end_year: int = 2025,
    seasonal_adj: str = "SA",
) -> pd.DataFrame:
    """
    Download BFS monthly state-level data.
    
    Notes:
    - BFS variable naming can evolve. This function is written to be easy to adapt.
    - If a series code fails, inspect the API variable list and replace `series`.
    """

    frames = []

    for year in range(start_year, end_year + 1):
        params = {
            "get": "cell_value,time_slot_id,seasonally_adj,geo_level_code,state_fips",
            "for": "state:*",
            "YEAR": str(year),
            "series_code": series,
            "seasonally_adj": seasonal_adj,
        }
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY

        try:
            raw = _safe_get_json(BFS_BASE_URL, params)
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Failed BFS pull for year={year}, series={series}. "
                f"Check series_code / filters. Original error: {e}"
            ) from e

        df = pd.DataFrame(raw[1:], columns=raw[0])
        df["YEAR"] = year
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = _to_numeric(out, ["cell_value", "time_slot_id", "YEAR"])

    out.rename(columns={
        "cell_value": "value",
        "time_slot_id": "month",
        "state": "state_name",  # if returned
        "state_fips": "state_fips",
    }, inplace=True)

    # Build a monthly date
    out["date"] = pd.to_datetime(
        dict(year=out["YEAR"], month=out["month"], day=1),
        errors="coerce"
    )

    # Keep clean fields
    keep = [c for c in [
        "date", "YEAR", "month", "value", "seasonally_adj",
        "geo_level_code", "state_fips", "series_code"
    ] if c in out.columns]
    out = out[keep].copy()

    out["state_abbr"] = out["state_fips"].map(FIPS_TO_STATE_ABBR)
    out["series"] = series

    return out.sort_values(["state_fips", "date"]).reset_index(drop=True)


# ------------------------------------------------------------
# Example BFS panel builder
# ------------------------------------------------------------

def build_bfs_panel() -> pd.DataFrame:
    """
    Pull several BFS series and merge them into one state-month panel.
    Series codes below are placeholders that may need adjustment depending
    on current API naming. Keep the structure and swap codes if needed.
    """

    series_map = {
        "bfs_total_applications": "BA_BA",
        "bfs_high_propensity": "HBA_BA",
        "bfs_projected_formations": "PBF4Q_BA",
        "bfs_employer_formations": "BF4Q_BA",
    }

    dfs = []
    for name, code in series_map.items():
        df = fetch_bfs_state_monthly(series=code, start_year=2019, end_year=2025)
        df = df[["state_fips", "state_abbr", "date", "value"]].rename(columns={"value": name})
        dfs.append(df)

    panel = dfs[0]
    for df in dfs[1:]:
        panel = panel.merge(df, on=["state_fips", "state_abbr", "date"], how="outer")

    panel = panel.sort_values(["state_fips", "date"]).reset_index(drop=True)

    # Post-treatment dummy
    panel["post_ai"] = (panel["date"] >= AI_SHOCK_DATE).astype(int)

    # Basic time fields
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month

    return panel


# ------------------------------------------------------------
# AI Exposure
# ------------------------------------------------------------

def load_ai_exposure_csv(filepath: str) -> pd.DataFrame:
    """
    Load a precomputed state-level AI exposure file.
    
    Expected columns:
      - state_fips
      - ai_exposure
    
    You should build this from pre-2022 occupational/industry composition.
    """
    df = pd.read_csv(filepath, dtype={"state_fips": str})
    required = {"state_fips", "ai_exposure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in AI exposure file: {missing}")

    # standardize
    df = df[["state_fips", "ai_exposure"]].copy()
    df["ai_exposure_z"] = (df["ai_exposure"] - df["ai_exposure"].mean()) / df["ai_exposure"].std(ddof=0)
    return df


# ------------------------------------------------------------
# Merge + DiD
# ------------------------------------------------------------

def make_baseline_panel(ai_exposure_path: str) -> pd.DataFrame:
    bfs = build_bfs_panel()
    ai = load_ai_exposure_csv(ai_exposure_path)
    panel = bfs.merge(ai, on="state_fips", how="left")

    if panel["ai_exposure"].isna().any():
        missing_states = panel.loc[panel["ai_exposure"].isna(), "state_abbr"].dropna().unique().tolist()
        raise ValueError(f"AI exposure missing for states: {missing_states}")

    # Logs: add 1 to avoid dropping zeros
    outcome_cols = [
        "bfs_total_applications",
        "bfs_high_propensity",
        "bfs_projected_formations",
        "bfs_employer_formations",
    ]
    for c in outcome_cols:
        if c in panel.columns:
            panel[f"log_{c}"] = np.log1p(panel[c])

    panel["state_fe"] = panel["state_fips"]
    panel["time_fe"] = panel["date"].dt.to_period("M").astype(str)

    return panel


def run_baseline_did(
    panel: pd.DataFrame,
    outcome: str = "log_bfs_total_applications",
) -> object:
    """
    Baseline TWFE DiD:
    outcome ~ post_ai * ai_exposure_z + state FE + month FE
    """
    formula = f"{outcome} ~ post_ai * ai_exposure_z + C(state_fe) + C(time_fe)"
    model = smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["state_fips"]}
    )
    return model


# ------------------------------------------------------------
# Event study
# ------------------------------------------------------------

def add_event_time(panel: pd.DataFrame, shock_month: str = "2022-11-01") -> pd.DataFrame:
    shock = pd.Period(pd.Timestamp(shock_month), freq="M")
    panel = panel.copy()
    panel["period_m"] = panel["date"].dt.to_period("M")
    panel["event_time"] = (panel["period_m"] - shock).astype(int)
    return panel


def run_event_study(
    panel: pd.DataFrame,
    outcome: str = "log_bfs_total_applications",
    min_k: int = -24,
    max_k: int = 24,
    ref_k: int = -1,
) -> object:
    """
    Event study with interaction between event time and ai_exposure_z.
    """
    df = add_event_time(panel)
    df = df[(df["event_time"] >= min_k) & (df["event_time"] <= max_k)].copy()

    # Create dummies except reference period
    for k in range(min_k, max_k + 1):
        if k == ref_k:
            continue
        df[f"event_{k}"] = (df["event_time"] == k).astype(int)

    interaction_terms = " + ".join(
        [f"event_{k}:ai_exposure_z" for k in range(min_k, max_k + 1) if k != ref_k]
    )

    formula = f"{outcome} ~ {interaction_terms} + C(state_fe) + C(time_fe)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["state_fips"]}
    )
    return model


# ------------------------------------------------------------
# BDS Skeleton
# ------------------------------------------------------------

def fetch_bds_annual(
    variable_list: list[str],
    year: int,
    state_level: bool = True,
) -> pd.DataFrame:
    """
    Skeleton for BDS annual pulls.
    
    Example variables depend on the specific BDS table you use.
    You will likely pull startup firms, establishment births,
    employment, or job creation by geography/year.
    """
    get_vars = ",".join(variable_list)
    params = {
        "get": get_vars,
        "YEAR": str(year),
    }

    if state_level:
        params["for"] = "state:*"

    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY

    raw = _safe_get_json(BDS_BASE_URL, params)
    df = pd.DataFrame(raw[1:], columns=raw[0])
    return df


# ------------------------------------------------------------
# Output tables
# ------------------------------------------------------------

def summarize_model(model: object, title: str = "Baseline DiD") -> pd.DataFrame:
    out = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
    })
    out["title"] = title
    return out.reset_index().rename(columns={"index": "term"})


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # 1) Build your AI exposure file separately and point to it here
    ai_exposure_path = "data/ai_exposure_state_pre2022.csv"

    # 2) Build state-month panel
    panel = make_baseline_panel(ai_exposure_path)
    panel.to_csv("bfs_ai_panel.csv", index=False)

    # 3) Run baseline DiD
    did_model = run_baseline_did(panel, outcome="log_bfs_total_applications")
    print(did_model.summary())

    # 4) Save coefficients
    did_table = summarize_model(did_model, title="BFS Total Applications")
    did_table.to_csv("did_results_total_applications.csv", index=False)

    # 5) Event study
    es_model = run_event_study(panel, outcome="log_bfs_total_applications")
    es_table = summarize_model(es_model, title="Event Study")
    es_table.to_csv("event_study_results.csv", index=False)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()