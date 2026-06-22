
"""
dashboard_queries.py — Weekly performance dashboard engine.

Architecture:
    DataLayer         — SQL fetches (filter + project, no business logic)
    Metric functions  — Pure Python/pandas on DataFrames (testable in isolation)
    DashboardService  — Orchestrator: fetch once, pass to all metric functions

Design rules:
    SQL  → heavy aggregation, date windowing, column projection
    Python → business definitions, % splits, growth calc, warnings
    Each metric function accepts DataFrames + date range, returns typed dict/DF
"""

import os
from datetime import date, timedelta
from typing import List, Optional, Set

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ── Pydantic models (typed, serialisable, importable) ─────────────────────────

class TotalSalesMetric(BaseModel):
    value:        float
    prior_value:  float
    variance:     float
    variance_pct: Optional[float]
    coverage_pct: float
    warnings:     List[str] = []


class RepeatCustomerMetric(BaseModel):
    total_customers:  int
    repeat_customers: int
    new_customers:    int
    repeat_pct:       float
    warnings:         List[str] = []


class SpeciesRow(BaseModel):
    species:            str
    this_week_revenue:  float
    prior_3wk_avg:      float
    pct_contribution:   float
    growth_pct:         Optional[float]


class CategoryRow(BaseModel):
    category:           str
    this_week_revenue:  float
    prior_3wk_avg:      float
    pct_contribution:   float
    growth_pct:         Optional[float]


class InvoiceCountRow(BaseModel):
    species:          str
    bill_count:       int
    prior_3wk_avg:    float
    pct_contribution: float
    growth_pct:       Optional[float]


class DayNightRow(BaseModel):
    time_band:        str
    revenue:          float
    prior_3wk_avg:    float
    pct_contribution: float
    growth_pct:       Optional[float]


class CustomerBySpeciesRow(BaseModel):
    customer_type: str
    species:       str
    client_count:  int
    revenue:       float


class NewExistingRevenueResult(BaseModel):
    new_revenue:      float
    existing_revenue: float
    total_revenue:    float
    new_pct:          float
    existing_pct:     float
    warnings:         List[str] = []


class InventoryRow(BaseModel):
    inventory_type: str
    stock_value:    float
    pct_of_total:   float


class LifeStageRow(BaseModel):
    species:       str
    life_stage:    str
    revenue:       float
    invoice_count: int


class WeeklyDashboard(BaseModel):
    week:                      str
    total_sales:               TotalSalesMetric
    repeat_customer_pct:       RepeatCustomerMetric
    species_split:             List[SpeciesRow]
    category_top5:             List[CategoryRow]
    invoice_count_by_species:  List[InvoiceCountRow]
    day_night_split:           List[DayNightRow]
    new_vs_existing_customers: List[CustomerBySpeciesRow]
    new_vs_existing_revenue:   NewExistingRevenueResult
    inventory_by_type:         List[InventoryRow]
    life_stage:                List[LifeStageRow]


# ── Date helpers ──────────────────────────────────────────────────────────────

PHANTOM_CLIENT = "13"


def _excl(d: date) -> date:
    """Inclusive Sunday → exclusive Monday for DATETIME < comparisons."""
    return d + timedelta(days=1)



def _growth_pct(current: float, prior: float) -> Optional[float]:
    if not prior:
        return None
    return round(100.0 * (current - prior) / prior, 1)


def _iso_week_label(ts_series: pd.Series) -> pd.Series:
    """YYYY-WW string — sortable, handles year boundaries correctly."""
    cal = ts_series.dt.isocalendar()
    return cal.year.astype(str).str.zfill(4) + "-" + cal.week.astype(str).str.zfill(2)


def _coverage_pct(invoices: pd.DataFrame, ws: date, we: date) -> float:
    """% of the 7-day window that has at least one invoice."""
    mask = (invoices["performed_date"] >= pd.Timestamp(ws)) & \
           (invoices["performed_date"] <  pd.Timestamp(we))
    days = invoices.loc[mask, "performed_date"].dt.date.nunique()
    return round(100.0 * days / 7, 1)


def _prior_weekly_avg(series_by_isoweek: pd.Series) -> float:
    """Mean of per-week totals. Correct even if a week has zero invoices."""
    return float(series_by_isoweek.groupby(level=0).mean().mean()) if not series_by_isoweek.empty else 0.0


# ── Data fetch layer (SQL only — no business logic) ───────────────────────────

class DataLayer:

    def __init__(self, engine: Engine):
        self.engine = engine

    def fetch_invoices(self, start: date, end_excl: date) -> pd.DataFrame:
        """
        Pull all invoice rows for the analysis window (current week + prior 3 weeks).
        SQL does: date filter, cancelled = 0, column projection.
        Returns DataFrame with dtypes coerced.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    invoice_id,
                    client_id,
                    patient_id,
                    patient_species,
                    plan_category_name,
                    total,
                    performed_date
                FROM allpets_new_invoices
                WHERE cancelled = 0
                  AND performed_date IS NOT NULL
                  AND performed_date >= :start
                  AND performed_date <  :end
            """), {"start": start, "end": end_excl}).fetchall()

        df = pd.DataFrame(rows, columns=[
            "invoice_id", "client_id", "patient_id", "patient_species",
            "plan_category_name", "total", "performed_date",
        ])
        df["performed_date"] = pd.to_datetime(df["performed_date"])
        df["total"]          = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
        return df

    # Stock items excluded from inventory valuation — phantom entries with no ETL
    # update history (updated_on IS NULL) and anomalously high quantities that
    # are not physical stock. Add new names here if more phantoms are identified.
    _PHANTOM_STOCK_PATTERNS = (
        "cytopoint",       # 13,850 qty × ₹3,850 = ₹5.33Cr — never updated, never sold
        "lasix 10mg",      # 10,127 qty × ₹127.60 = ₹12.92L — same NULL updated_on signature
    )

    def fetch_stocks(self) -> pd.DataFrame:
        """Current stock snapshot — phantoms excluded, negatives clipped to 0."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    stock_id,
                    stock_name,
                    plan_category_name,
                    updated_on,
                    GREATEST(COALESCE(onhand_qty,   0), 0) AS onhand_qty,
                    COALESCE(purchase_cost, 0)              AS purchase_cost
                FROM allpets_new_stocks
            """)).fetchall()

        df = pd.DataFrame(rows, columns=["stock_id", "stock_name", "plan_category_name",
                                          "updated_on", "onhand_qty", "purchase_cost"])
        df["onhand_qty"]    = pd.to_numeric(df["onhand_qty"],    errors="coerce").fillna(0.0)
        df["purchase_cost"] = pd.to_numeric(df["purchase_cost"], errors="coerce").fillna(0.0)

        # Exclude known phantom items by name (case-insensitive prefix/substring match)
        mask = df["stock_name"].str.lower().apply(
            lambda n: any(pat in n for pat in self._PHANTOM_STOCK_PATTERNS)
        )
        return df.loc[~mask].reset_index(drop=True)

    def fetch_patients(self) -> pd.DataFrame:
        """Patient birth_date for life stage computation."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT patient_id, birth_date
                FROM allpets_new_patients
                WHERE birth_date IS NOT NULL
            """)).fetchall()
        df = pd.DataFrame(rows, columns=["patient_id", "birth_date"])
        df["birth_date"] = pd.to_datetime(df["birth_date"])
        return df

    def fetch_prior_client_ids(self, before_date: date) -> Set[str]:
        """
        All client_ids who existed before before_date.
        Primary source: allpets_new_clients.first_activity (covers full history back to 2010).
        Union with invoices to catch any client in invoices but missing from clients table.
        Pre-2025 clients who return after a gap are correctly classified as existing/repeat.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT client_id FROM allpets_new_clients
                WHERE client_id != '13'
                  AND first_activity IS NOT NULL
                  AND first_activity < :before
                UNION
                SELECT DISTINCT client_id FROM allpets_new_invoices
                WHERE cancelled = 0
                  AND performed_date IS NOT NULL
                  AND performed_date < :before
                  AND client_id != '13'
            """), {"before": before_date}).fetchall()
        return {str(r[0]) for r in rows}

    def fetch_new_client_ids(self, week_start: date, week_end_excl: date) -> Set[str]:
        """
        Client IDs whose first_activity in allpets_new_clients falls in this week.
        REQUIRES clients ETL backfill from 2010 for accuracy.
        Returns empty set until the backfill runs.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT client_id
                FROM allpets_new_clients
                WHERE client_id != '13'
                  AND first_activity IS NOT NULL
                  AND first_activity >= :ws
                  AND first_activity <  :we
            """), {"ws": week_start, "we": week_end_excl}).fetchall()
        return {str(r[0]) for r in rows}


# ── Metric functions (pure Python / pandas) ───────────────────────────────────

def compute_total_sales(
    invoices:    pd.DataFrame,
    week_start:  date,
    week_end:    date,          # inclusive Sunday
    prior_weeks: int = 3,
) -> TotalSalesMetric:
    we  = _excl(week_end)
    ps  = week_start - timedelta(weeks=prior_weeks)
    ws  = pd.Timestamp(week_start)

    this_mask  = (invoices["performed_date"] >= ws) & (invoices["performed_date"] < pd.Timestamp(we))
    prior_mask = (invoices["performed_date"] >= pd.Timestamp(ps)) & (invoices["performed_date"] < ws)

    this_week = float(invoices.loc[this_mask, "total"].sum())

    prior = invoices.loc[prior_mask].copy()
    prior["iso_week"] = _iso_week_label(prior["performed_date"])
    prior_avg = float(prior.groupby("iso_week")["total"].sum().mean()) if not prior.empty else 0.0

    cov      = _coverage_pct(invoices, week_start, we)
    warnings = []
    if cov < 100:
        warnings.append(f"Only {cov}% of week days have data — partial week.")

    return TotalSalesMetric(
        value        = round(this_week, 0),
        prior_value  = round(prior_avg, 0),
        variance     = round(this_week - prior_avg, 0),
        variance_pct = _growth_pct(this_week, prior_avg),
        coverage_pct = cov,
        warnings     = warnings,
    )


def compute_repeat_customer_pct(
    invoices:         pd.DataFrame,
    week_start:       date,
    week_end:         date,
    prior_client_ids: Set[str],     # full history — from DataLayer.fetch_prior_client_ids()
) -> RepeatCustomerMetric:
    """
    Repeat = client who visited at least once before this week (full DB history).
    prior_client_ids must be fetched separately to cover the complete invoice history,
    not just the 4-week analysis window.
    """
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))

    this_clients = set(invoices.loc[
        (invoices["performed_date"] >= ws) &
        (invoices["performed_date"] < we) &
        (invoices["client_id"] != PHANTOM_CLIENT),
        "client_id"
    ].unique())

    total  = len(this_clients)
    repeat = len(this_clients & prior_client_ids)

    return RepeatCustomerMetric(
        total_customers  = total,
        repeat_customers = repeat,
        new_customers    = total - repeat,
        repeat_pct       = round(100.0 * repeat / total, 1) if total else 0.0,
        warnings         = ["No customer data for this week."] if total == 0 else [],
    )


def compute_species_sales(
    invoices:    pd.DataFrame,
    week_start:  date,
    week_end:    date,
    prior_weeks: int = 3,
) -> pd.DataFrame:
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))
    ps = pd.Timestamp(week_start - timedelta(weeks=prior_weeks))

    this_week = (
        invoices.loc[(invoices["performed_date"] >= ws) & (invoices["performed_date"] < we)]
        .groupby("patient_species")["total"].sum()
        .rename("this_week")
    )
    grand_total = this_week.sum()

    prior = invoices.loc[
        (invoices["performed_date"] >= ps) & (invoices["performed_date"] < ws)
    ].copy()
    prior["iso_week"] = _iso_week_label(prior["performed_date"])
    prior_avg = (
        prior.groupby(["patient_species", "iso_week"])["total"].sum()
        .groupby(level=0).mean()
        .rename("prior_avg")
    )

    df = pd.concat([this_week, prior_avg], axis=1).fillna(0.0)
    df["pct_contribution"] = (df["this_week"] / grand_total * 100).round(1) if grand_total else 0.0
    df["growth_pct"]       = df.apply(lambda r: _growth_pct(r["this_week"], r["prior_avg"]), axis=1)
    df = df.reset_index().rename(columns={
        "patient_species": "species",
        "this_week":       "this_week_revenue",
        "prior_avg":       "prior_3wk_avg",
    })
    df["this_week_revenue"] = df["this_week_revenue"].round(0)
    df["prior_3wk_avg"]     = df["prior_3wk_avg"].round(0)
    return df.sort_values("this_week_revenue", ascending=False).reset_index(drop=True)


def compute_category_sales(
    invoices:    pd.DataFrame,
    week_start:  date,
    week_end:    date,
    top_n:       int = 5,
    prior_weeks: int = 3,
) -> pd.DataFrame:
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))
    ps = pd.Timestamp(week_start - timedelta(weeks=prior_weeks))

    this_week = (
        invoices.loc[(invoices["performed_date"] >= ws) & (invoices["performed_date"] < we)]
        .groupby("plan_category_name")["total"].sum()
        .nlargest(top_n)
        .rename("this_week")
    )
    grand_total = this_week.sum()
    top_cats    = set(this_week.index)

    prior = invoices.loc[
        (invoices["performed_date"] >= ps) &
        (invoices["performed_date"] < ws) &
        (invoices["plan_category_name"].isin(top_cats))
    ].copy()
    prior["iso_week"] = _iso_week_label(prior["performed_date"])
    prior_avg = (
        prior.groupby(["plan_category_name", "iso_week"])["total"].sum()
        .groupby(level=0).mean()
        .rename("prior_avg")
    )

    df = pd.concat([this_week, prior_avg], axis=1).fillna(0.0)
    df["pct_contribution"] = (df["this_week"] / grand_total * 100).round(1) if grand_total else 0.0
    df["growth_pct"]       = df.apply(lambda r: _growth_pct(r["this_week"], r["prior_avg"]), axis=1)
    df = df.reset_index().rename(columns={
        "plan_category_name": "category",
        "this_week":          "this_week_revenue",
        "prior_avg":          "prior_3wk_avg",
    })
    df["this_week_revenue"] = df["this_week_revenue"].round(0)
    df["prior_3wk_avg"]     = df["prior_3wk_avg"].round(0)
    return df.sort_values("this_week_revenue", ascending=False).reset_index(drop=True)


def compute_invoice_count_by_species(
    invoices:    pd.DataFrame,
    week_start:  date,
    week_end:    date,
    prior_weeks: int = 3,
) -> pd.DataFrame:
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))
    ps = pd.Timestamp(week_start - timedelta(weeks=prior_weeks))

    this_week = (
        invoices.loc[(invoices["performed_date"] >= ws) & (invoices["performed_date"] < we)]
        .groupby("patient_species")["invoice_id"].nunique()
        .rename("this_week")
    )
    grand_total = this_week.sum()

    prior = invoices.loc[
        (invoices["performed_date"] >= ps) & (invoices["performed_date"] < ws)
    ].copy()
    prior["iso_week"] = _iso_week_label(prior["performed_date"])
    prior_avg = (
        prior.groupby(["patient_species", "iso_week"])["invoice_id"].nunique()
        .groupby(level=0).mean()
        .rename("prior_avg")
    )

    df = pd.concat([this_week, prior_avg], axis=1).fillna(0.0)
    df["pct_contribution"] = (df["this_week"] / grand_total * 100).round(1) if grand_total else 0.0
    df["growth_pct"]       = df.apply(lambda r: _growth_pct(r["this_week"], r["prior_avg"]), axis=1)
    df = df.reset_index().rename(columns={
        "patient_species": "species",
        "this_week":       "bill_count",
        "prior_avg":       "prior_3wk_avg",
    })
    df["bill_count"]    = df["bill_count"].astype(int)
    df["prior_3wk_avg"] = df["prior_3wk_avg"].round(1)
    return df.sort_values("bill_count", ascending=False).reset_index(drop=True)


def compute_day_night_split(
    invoices:    pd.DataFrame,
    week_start:  date,
    week_end:    date,
    prior_weeks: int = 3,
) -> pd.DataFrame:
    """Day = HOUR 9–20 (9AM up to 9PM). Night = HOUR 21–23 + 0–8."""
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))
    ps = pd.Timestamp(week_start - timedelta(weeks=prior_weeks))

    def _band(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["time_band"] = df["performed_date"].dt.hour.map(
            lambda h: "Day (9AM-9PM)" if 9 <= h < 21 else "Night (9PM-9AM)"
        )
        return df

    this_week = (
        _band(invoices.loc[(invoices["performed_date"] >= ws) & (invoices["performed_date"] < we)])
        .groupby("time_band")["total"].sum()
        .rename("this_week")
    )
    grand_total = this_week.sum()

    prior = _band(invoices.loc[
        (invoices["performed_date"] >= ps) & (invoices["performed_date"] < ws)
    ]).copy()
    prior["iso_week"] = _iso_week_label(prior["performed_date"])
    prior_avg = (
        prior.groupby(["time_band", "iso_week"])["total"].sum()
        .groupby(level=0).mean()
        .rename("prior_avg")
    )

    df = pd.concat([this_week, prior_avg], axis=1).fillna(0.0)
    df["pct_contribution"] = (df["this_week"] / grand_total * 100).round(1) if grand_total else 0.0
    df["growth_pct"]       = df.apply(lambda r: _growth_pct(r["this_week"], r["prior_avg"]), axis=1)
    df = df.reset_index().rename(columns={
        "this_week": "revenue",
        "prior_avg": "prior_3wk_avg",
    })
    df["revenue"]       = df["revenue"].round(0)
    df["prior_3wk_avg"] = df["prior_3wk_avg"].round(0)
    return df.sort_values("revenue", ascending=False).reset_index(drop=True)


def compute_new_vs_existing_customers(
    invoices:       pd.DataFrame,
    new_client_ids: Set[str],
    week_start:     date,
    week_end:       date,
) -> pd.DataFrame:
    """
    New = client_id in new_client_ids (first_activity this week from allpets_new_clients).
    Existing = everyone else — correctly includes pre-2025 clients with no DB record.
    Dog / Cat cross-tab.
    """
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))

    week_inv = invoices.loc[
        (invoices["performed_date"] >= ws) &
        (invoices["performed_date"] < we) &
        (invoices["client_id"] != PHANTOM_CLIENT)
    ].copy()

    week_inv["species_bucket"] = week_inv["patient_species"].apply(
        lambda s: s if s in ("Canine", "Feline") else "Others (Exotics)"
    )
    week_inv["customer_type"] = week_inv["client_id"].apply(
        lambda cid: "New" if cid in new_client_ids else "Existing"
    )

    result = (
        week_inv.groupby(["customer_type", "species_bucket"])
        .agg(client_count=("client_id", "nunique"), revenue=("total", "sum"))
        .reset_index()
        .rename(columns={"species_bucket": "species"})
    )
    result["revenue"] = result["revenue"].round(0)
    species_order = {"Canine": 0, "Feline": 1, "Others (Exotics)": 2}
    result["_ord"] = result["species"].map(species_order).fillna(3)
    return result.sort_values(["customer_type", "_ord"],
                               ascending=[False, True]).drop(columns="_ord").reset_index(drop=True)


def compute_new_vs_existing_revenue(
    invoices:       pd.DataFrame,
    new_client_ids: Set[str],
    week_start:     date,
    week_end:       date,
) -> NewExistingRevenueResult:
    """
    Total-minus-new pattern. new + existing always = total.
    No explicit old-client join — pre-2025 clients correctly fall into existing.
    """
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))

    week_inv  = invoices.loc[
        (invoices["performed_date"] >= ws) &
        (invoices["performed_date"] < we) &
        (invoices["client_id"] != PHANTOM_CLIENT)
    ]
    total_rev = float(week_inv["total"].sum())
    new_rev   = float(week_inv.loc[week_inv["client_id"].isin(new_client_ids), "total"].sum())
    exist_rev = total_rev - new_rev

    warnings = []
    if not new_client_ids:
        warnings.append("new_client_ids is empty — clients ETL backfill (from 2010) has not run yet.")

    return NewExistingRevenueResult(
        new_revenue      = round(new_rev, 0),
        existing_revenue = round(exist_rev, 0),
        total_revenue    = round(total_rev, 0),
        new_pct          = round(100.0 * new_rev / total_rev, 1) if total_rev else 0.0,
        existing_pct     = round(100.0 * exist_rev / total_rev, 1) if total_rev else 0.0,
        warnings         = warnings,
    )


# ── Inventory buckets — defined as Python business rules, not SQL CASE ────────

_PHARMACY_EXACT = {
    "Parenteral Medication", "Parenteral Fluids",
    "Preventive Medicine",   "Anesthesia Gas",
    "Anesthesia Parenteral", "Sedation",
    "Medvet",
}

def _inventory_bucket(category: str) -> str:
    cat = (category or "").strip()
    if cat in ("Diets/Food", "Diets\\Food"):
        return "Food"
    if cat.startswith("Prescription") or cat in _PHARMACY_EXACT or cat.startswith("Preventive Medicine"):
        return "Pharmacy"
    return "Non-Food"


def compute_inventory_value(stocks: pd.DataFrame) -> pd.DataFrame:
    df = stocks.copy()
    df["stock_value"]    = df["onhand_qty"] * df["purchase_cost"]
    df["inventory_type"] = df["plan_category_name"].fillna("").apply(_inventory_bucket)

    grouped     = df.groupby("inventory_type")["stock_value"].sum().reset_index()
    grand_total = grouped["stock_value"].sum()
    grouped["pct_of_total"] = (grouped["stock_value"] / grand_total * 100).round(1) if grand_total else 0.0
    grouped["stock_value"]  = grouped["stock_value"].round(0)
    return grouped.sort_values("stock_value", ascending=False).reset_index(drop=True)


def compute_life_stage_metrics(
    invoices:   pd.DataFrame,
    patients:   pd.DataFrame,
    week_start: date,
    week_end:   date,
) -> pd.DataFrame:
    """
    Life stage revenue for Canine + Feline.
    Canine:  Puppy <12m | Adult 12-83m | Senior >=84m
    Feline:  Kitten <12m | Adult 12-119m | Senior >=120m
    """
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(_excl(week_end))

    week_inv = invoices.loc[
        (invoices["performed_date"] >= ws) &
        (invoices["performed_date"] < we) &
        (invoices["patient_species"].isin(["Canine", "Feline"]))
    ].merge(patients, on="patient_id", how="left")

    week_inv = week_inv.dropna(subset=["birth_date"]).copy()
    week_inv["age_months"] = (
        (week_inv["performed_date"] - week_inv["birth_date"]) / pd.Timedelta(days=30.44)
    ).astype(int)

    def _stage(row) -> str:
        a, sp = row["age_months"], row["patient_species"]
        if sp == "Canine":
            return "Puppy"  if a < 12 else "Adult" if a < 84  else "Senior"
        return     "Kitten" if a < 12 else "Adult" if a < 120 else "Senior"

    week_inv["life_stage"] = week_inv.apply(_stage, axis=1)

    result = (
        week_inv.groupby(["patient_species", "life_stage"])
        .agg(revenue=("total", "sum"), invoice_count=("invoice_id", "nunique"))
        .reset_index()
        .rename(columns={"patient_species": "species"})
    )
    result["revenue"] = result["revenue"].round(0)
    return result.sort_values(["species", "revenue"],
                               ascending=[True, False]).reset_index(drop=True)


# ── Dashboard service (orchestrator) ─────────────────────────────────────────

class DashboardService:
    """
    Fetch once, compute all metrics.
    Extend by adding new compute_* functions — no SQL changes needed.
    """

    def __init__(self, engine: Engine):
        self._data = DataLayer(engine)

    def run_weekly(
        self,
        week_start_str: str,
        week_end_str:   str,
        prior_weeks:    int = 3,
    ) -> WeeklyDashboard:
        ws      = date.fromisoformat(week_start_str)
        we_incl = date.fromisoformat(week_end_str)
        we_excl = _excl(we_incl)
        ps      = ws - timedelta(weeks=prior_weeks)

        # ── One fetch covers current week + prior N weeks ─────────────────
        invoices         = self._data.fetch_invoices(ps, we_excl)
        stocks           = self._data.fetch_stocks()
        patients         = self._data.fetch_patients()
        new_client_ids   = self._data.fetch_new_client_ids(ws, we_excl)
        prior_client_ids = self._data.fetch_prior_client_ids(ws)

        def _rows(df: pd.DataFrame, model):
            return [model.model_validate(r) for r in df.to_dict("records")]

        pw = prior_weeks
        # ── All metrics computed in Python from those DataFrames ──────────
        return WeeklyDashboard(
            week                      = f"{week_start_str} to {week_end_str}",
            total_sales               = compute_total_sales(invoices, ws, we_incl, pw),
            repeat_customer_pct       = compute_repeat_customer_pct(invoices, ws, we_incl, prior_client_ids),
            species_split             = _rows(compute_species_sales(invoices, ws, we_incl, pw), SpeciesRow),
            category_top5             = _rows(compute_category_sales(invoices, ws, we_incl, prior_weeks=pw), CategoryRow),
            invoice_count_by_species  = _rows(compute_invoice_count_by_species(invoices, ws, we_incl, pw), InvoiceCountRow),
            day_night_split           = _rows(compute_day_night_split(invoices, ws, we_incl, pw), DayNightRow),
            new_vs_existing_customers = _rows(compute_new_vs_existing_customers(invoices, new_client_ids, ws, we_incl), CustomerBySpeciesRow),
            new_vs_existing_revenue   = compute_new_vs_existing_revenue(invoices, new_client_ids, ws, we_incl),
            inventory_by_type         = _rows(compute_inventory_value(stocks), InventoryRow),
            life_stage                = _rows(compute_life_stage_metrics(invoices, patients, ws, we_incl), LifeStageRow),
        )


# ── Stand-alone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from urllib.parse import quote_plus
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    engine = create_engine(
        "mysql+pymysql://" + os.getenv("DB_USER", "") + ":"
        + quote_plus(os.getenv("DB_PASSWORD", ""))
        + "@" + os.getenv("DB_HOST", "") + ":" + os.getenv("DB_PORT", "3306")
        + "/" + os.getenv("DB_NAME", "cohort_main")
    )

    svc    = DashboardService(engine)
    result = svc.run_weekly("2026-05-19", "2026-05-25")

    for section, data in result.model_dump().items():
        print(f"\n{'='*60}")
        print(f"  {section.upper()}")
        print(f"{'='*60}")
        if isinstance(data, list):
            if data:
                cols = list(data[0].keys())
                print("  " + "  |  ".join(f"{c:<22}" for c in cols))
                print("  " + "-" * (27 * len(cols)))
                for row in data:
                    print("  " + "  |  ".join(f"{str(v):<22}" for v in row.values()))
            else:
                print("  (no data)")
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k:<30} {v}")
        else:
            print(f"  {data}")
