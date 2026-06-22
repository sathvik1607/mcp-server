#!/usr/bin/env python3
import sys, io
if __name__ == '__main__' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
sql_dynamic_agent.py — SQL Generation + Execution Sub-Agent
============================================================
A self-contained LangGraph agent responsible for:
  1. SQL Generator  — writes one SQL query per plan step (with retry context)
  2. SQL Executor   — runs each query, captures results as DataFrames
  3. Retry Loop     — up to MAX_RETRIES self-corrections on failed queries

Designed to be imported by nl2sql_agent.py (or any orchestrator) via:
    from sql_dynamic_agent import SQLDynamicAgent
    results = SQLDynamicAgent(llm, engine, debug=True).run(schema, plan, query)

Stand-alone test:
    python sql_dynamic_agent.py
"""

import json
import re
import time
import traceback
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from sqlalchemy import create_engine, sql, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
import pandas as pd

from openai import RateLimitError as _OpenAIRateLimitError


def _llm_invoke_with_retry(llm, messages):
    """Invoke the LLM with exponential-backoff retry on rate limit errors."""
    _delays = [0, 5, 10, 20]
    last_exc: Exception = RuntimeError("LLM call failed after retries")
    for _, delay in enumerate(_delays):
        if delay:
            print(f"   ⚠️  Rate limit — retrying in {delay}s…")
            time.sleep(delay)
        try:
            return llm.invoke(messages)
        except _OpenAIRateLimitError as exc:
            last_exc = exc
    raise last_exc


# ═══════════════════════════════════════════════════════════════════════════════
#  SUB-AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class SQLAgentState(TypedDict):
    user_query:        str
    schema:            str
    query_plan:        Dict[str, Any]
    generated_queries: Optional[List[Dict[str, Any]]]
    query_results:     Optional[List[Dict[str, Any]]]
    failed_queries:    Optional[List[Dict[str, Any]]]
    retry_count:       int
    max_retries:       int
    error:             Optional[str]
    debug:             bool


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str, is_array: bool = False, debug: bool = False) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if debug:
            print(f"   [DEBUG] JSON direct parse failed — attempting boundary extraction…")
        start, end = (raw.find("["), raw.rfind("]") + 1) if is_array else (raw.find("{"), raw.rfind("}") + 1)
        if start != -1 and end > 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as exc:
                if debug:
                    print(f"   [DEBUG] Boundary extraction failed: {exc}\n   Raw:\n{raw[:600]}")
                raise ValueError(f"Could not parse JSON:\n{raw[:400]}")
        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


def _extract_mysql_error(exc: Exception) -> str:
    """
    Pull the clean MySQL error message out of a SQLAlchemy exception.

    SQLAlchemy wraps the driver error in its own class and appends verbose
    boilerplate ('[SQL: ...]', '[Background on this error...]') that wastes
    tokens and confuses the LLM on retry.  This function unwraps one level
    and returns only the actionable part.
    """
    # Try the driver-level exception first (pymysql.err.* attached as .orig)
    orig = getattr(exc, "orig", None)
    if orig is not None:
        args = getattr(orig, "args", ())
        if len(args) >= 2:
            # pymysql format: (error_code: int, "message": str)
            return f"MySQL {args[0]}: {args[1]}"
        return str(orig)[:400]

    # Fallback: strip SQLAlchemy's boilerplate from the string representation
    raw = str(exc)
    for marker in ("\n[SQL:", "\n(Background on this error"):
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]
    return raw.strip()[:400]


def _strip_sql_literals_and_comments(sql_text: str) -> str:
    """
    Remove comments and string contents so reference checks do not match
    harmless words inside literals.
    """
    sql_text = re.sub(r"/\*.*?\*/", " ", sql_text, flags=re.DOTALL)
    sql_text = re.sub(r"--[^\r\n]*", " ", sql_text)
    sql_text = re.sub(r"#[^\r\n]*", " ", sql_text)
    sql_text = re.sub(r"'(?:''|\\'|[^'])*'", "''", sql_text)
    sql_text = re.sub(r'"(?:""|\\\"|[^"])*"', '""', sql_text)
    return sql_text


def _identifier_pattern(name: str) -> str:
    escaped = re.escape(name)
    return rf"(?:`{escaped}`|{escaped})"


def _find_result_alias_references(sql_text: str, result_aliases: List[str]) -> List[str]:
    """
    Return result_alias values that are used as SQL relation names.

    result_alias is workflow metadata only. The executor does not create temp
    tables, views, registered DataFrames, or any other SQL-queryable relation.
    """
    if not result_aliases:
        return []

    searchable = _strip_sql_literals_and_comments(sql_text)
    matches: List[str] = []
    seen = set()
    relation_keywords = r"(?:FROM|JOIN|UPDATE|INTO|TABLE|DESC|DESCRIBE)"

    for alias in result_aliases:
        if not alias or not isinstance(alias, str):
            continue
        ident = _identifier_pattern(alias)
        relation_ref = (
            rf"\b{relation_keywords}\s+"
            rf"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?"
            rf"{ident}\b"
        )
        with_ref = rf"\bWITH\b[\s\S]*?\b{ident}\b\s+AS\s*\("

        if re.search(relation_ref, searchable, flags=re.IGNORECASE) or re.search(with_ref, searchable, flags=re.IGNORECASE):
            if alias not in seen:
                matches.append(alias)
                seen.add(alias)

    return matches


def _validate_no_result_alias_references(sql_text: str, result_aliases: List[str]) -> Optional[str]:
    references = _find_result_alias_references(sql_text, result_aliases)
    if not references:
        return None
    return (
        "Query rejected: result_alias is workflow metadata only and is not a "
        "database table/view/CTE. Do not reference generated result_alias values "
        f"in SQL relation clauses. Offending alias(es): {', '.join(references)}"
    )


def _execute_sql(sql: str, engine: Engine, debug: bool = False) -> Tuple[pd.DataFrame, Optional[str]]:
    if debug:
        print(f"\n   [DEBUG] Executing SQL:\n   {sql}")
    first_token = sql.strip().upper().split()[0] if sql.strip() else ""
    if first_token not in ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"):
        return pd.DataFrame(), f"Query rejected: only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN are permitted (got {first_token!r})"
    t0 = time.perf_counter()
    try:
        with engine.connect() as conn:
            # PyMySQL interprets bare % as Python format tokens (%m → error).
            # Pre-escape every % to %% so the driver passes the literal
            # character to MySQL unchanged.  This fixes STR_TO_DATE('%m/%d/%Y')
            # and DATE_FORMAT calls without altering the MySQL semantics.
            sql_for_exec = sql.replace("%", "%%")
            result = conn.exec_driver_sql(sql_for_exec)
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            elapsed = time.perf_counter() - t0
            if debug:
                print(f"   [DEBUG] {elapsed:.2f}s | {len(df)} row(s) | cols: {list(df.columns)}")
                if not df.empty:
                    null_cols = [c for c in df.columns if df[c].isnull().any()]
                    if null_cols:
                        print(f"   ⚠️  NULL values in: {null_cols} — aggregate over empty set")
            return df, None
    except SQLAlchemyError as exc:
        if debug:
            traceback.print_exc()
        return pd.DataFrame(), _extract_mysql_error(exc)
    except Exception as exc:
        if debug:
            traceback.print_exc()
        return pd.DataFrame(), str(exc)[:400]


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE FACTORIES  (closures capture llm / engine / debug from SQLDynamicAgent)
# ═══════════════════════════════════════════════════════════════════════════════

_SQL_GENERATOR_SYSTEM = """You are an expert analytics SQL developer for a veterinary clinic management platform.

Your primary responsibility is to answer business questions correctly.
SQL is only an implementation detail.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {
    "step_id": 1,
    "description": "what this query fetches",
    "sql": "SELECT …",
    "result_alias": "short_snake_case_name",
    "sql_metadata": {
      "metric_definition": "gross_revenue",
      "date_column_used": "performed_date",
      "auto_filters_applied": []
    }
  }
]

sql_metadata fields:
  metric_definition  — one of: gross_revenue, active_revenue, invoice_count,
                       patient_count, customer_count, stock_value, other
  date_column_used   — "performed_date" or "invoice_date"
  auto_filters_applied — list any filters added beyond what the user explicitly requested,
                         e.g. ["performed_date IS NOT NULL"]. Empty list [] if none added.

IMPORTANT: result_alias is workflow tracking metadata only.
It is not a database table, temp table, view, CTE, or registered DataFrame.
It must NEVER appear in FROM, JOIN, WITH, UPDATE, INSERT, DELETE, TABLE,
DESC, or DESCRIBE references. No previous query result can be queried by SQL.

════════════════════════════════════════════════════════════
RETRY BEHAVIOUR — MANDATORY WHEN MESSAGE STARTS WITH ⚠️ RETRY
════════════════════════════════════════════════════════════
When the user message begins with "⚠️ RETRY", a prior SQL attempt failed.
Apply these rules to every step listed under FAILED STEPS:

  1. READ the failed SQL and the exact database error message.
  2. DIAGNOSE the root cause: wrong column name, wrong table, missing
     join, type mismatch, syntax error, forbidden function, etc.
  3. REWRITE with a completely different strategy:
       • Choose a different join path, different function, or different
         WHERE clause structure than the one that failed.
       • Do NOT copy any fragment of the failed SQL for that step.
       • If the error is "Unknown column X", stop using column X.
       • If the error is a syntax error, simplify the query structure.
       • If the error is a type mismatch, add an explicit CAST.
  4. For steps NOT listed under FAILED STEPS: keep the same SQL.

Reproducing the same failing SQL is a critical correctness error.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
THINKING PROCESS — FOLLOW FOR EVERY REQUEST
════════════════════════════════════════════════════════════
Before writing any SQL, internally resolve these six steps in order:

  Step 1 — BUSINESS QUESTION
    What is the user really asking?
    Revenue trend / customer acquisition / customer retention /
    category growth / inventory health / SKU performance?

  Step 2 — KPI
    What is the single number being measured?
    Revenue / Customer Count / Invoice Count / Stock Value /
    Coverage Days / Units Sold

  Step 3 — DIMENSIONS
    What axes does the result need?
    Month / Week / Species / Category / SKU / Clinic

  Step 4 — REPORTING GRAIN
    One row per: Month / Week / Species / Category / SKU

  Step 5 — CORRECT MEASURE
    Revenue              → SUM(total)
    Customer Count       → COUNT(DISTINCT client_id)
    Invoice Count        → COUNT(DISTINCT invoice_id)
    Stock Value          → SUM(GREATEST(COALESCE(onhand_qty,0),0) * purchase_cost)

  Step 6 — BUSINESS RULE VALIDATION
    Check all analytics safety rules (see below) before generating SQL.
    If any rule is violated, fix the query before returning it.

Only after completing all six steps, generate the SQL.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
METRIC DEFINITIONS — MATCH USER INTENT BEFORE ADDING FILTERS
════════════════════════════════════════════════════════════
Before generating SQL, identify which metric the user is asking about.
The metric determines which filters are appropriate.

  gross_revenue    "total sales", "all sales", "gross revenue", "overall revenue",
                   "total revenue", "how much was sold", "sales figure"
                   → SUM(total) — NO cancelled filter

  active_revenue   "active sales", "net sales", "excluding cancelled",
                   "valid invoices", "net revenue"
                   → SUM(total) WHERE cancelled = 0

  invoice_count    "number of invoices", "invoice count", "how many invoices"
                   → COUNT(DISTINCT invoice_id) WHERE cancelled = 0

  patient_count    "number of patients", "how many patients"
                   → COUNT(DISTINCT patient_id)

  customer_count   "number of customers", "unique clients", "how many clients"
                   → COUNT(DISTINCT client_id) WHERE client_id != '13'

  stock_value      "stock value", "inventory value"
                   → SUM(GREATEST(COALESCE(onhand_qty,0),0) * purchase_cost)

Record your choice in sql_metadata.metric_definition.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
ANALYTICS SAFETY RULES — ALWAYS APPLY
════════════════════════════════════════════════════════════
Data Coverage — CHECK FROM DB, DO NOT HARDCODE:
  Never assume what date range the database contains. Coverage changes as new data is loaded.
  When the user asks for a recent period (this month, last month, last N weeks, today, etc.):
    Add a coverage-check step FIRST:
      SELECT MIN(performed_date) AS earliest_date, MAX(performed_date) AS latest_date
      FROM allpets_new_invoices
      WHERE performed_date IS NOT NULL AND cancelled = 0
    Use the returned latest_date to determine if the requested period has data.
    If the requested period is entirely after latest_date → tell the user: no data available yet for that period.
    If the requested period overlaps with available data → query only the available range.

Revenue:
  Always use SUM(total). Never use SUM(invoice_amount).

Cancelled Invoices:
  Add cancelled = 0 ONLY when metric_definition = active_revenue or invoice_count,
  OR when the user explicitly says "excluding cancelled" / "active only".
  Do NOT add cancelled = 0 for gross_revenue / total_sales / total_revenue queries.

Customer Analytics:
  Exclude client_id = '13' from customer counts and new/existing splits only.
  Include client_id = '13' in all revenue aggregates (total, species, category).

Category Analytics:
  Use plan_category_name. Never use bin_name for business categories.

Inventory:
  Negative stock is zero. Use GREATEST(COALESCE(onhand_qty,0),0).

Date Column for Filtering:
  Default: use performed_date (actual service date — aligns with VetBuddy reports).
  Override: use invoice_date when the execution plan step OR the user explicitly specifies
  "invoice_date", "invoice date", "billing date", or "by invoice".
  Never silently substitute performed_date when the plan or user specifies invoice_date.
  performed_date is NULLABLE — always add AND performed_date IS NOT NULL when filtering on it.
  invoice_date is NOT NULL — no IS NOT NULL check needed when filtering on it.
  ✓  WHERE performed_date IS NOT NULL AND YEAR(performed_date) = 2026 AND MONTH(performed_date) = 5
  ✓  WHERE YEAR(invoice_date) = 2026 AND MONTH(invoice_date) = 5  ← allowed when explicitly specified
  Exception: new customer definition uses allpets_new_clients.first_activity — not invoice_date.
  Record your choice in sql_metadata.date_column_used.
════════════════════════════════════════════════════════════

Strict rules:
• Use only table and column names that exist in the schema.
• Use clear column aliases (e.g. AS total_sales, AS month_label).
• Never use placeholder tokens like <table> or {{column}} — use real names.
• BANNED ALIAS — never use "year_month" as a column alias. It is a MySQL
  reserved word and causes a 1064 syntax error. Use "month_label", "ym",
  or "month_key" instead.
• Each query must run standalone; do not reference results from other steps inside SQL.
• result_alias values are metadata only. Never use any result_alias as a SQL
  relation in FROM/JOIN/WITH/UPDATE/INSERT/DELETE/TABLE/DESC/DESCRIBE.
• One array entry per plan step, same step_id values.
• Use COALESCE on aggregates only in LEFT JOIN queries where a dimension row may have zero
  matching fact rows (e.g. a clinic with no invoices). For standard GROUP BY revenue queries
  (FROM allpets_new_invoices WHERE … GROUP BY …), SUM(total) is sufficient — COALESCE adds
  no value and masks intent. Verified: SUM(total) without COALESCE matches VetBuddy exactly.
• CANCELLED INVOICES: Only add cancelled = 0 when metric_definition = active_revenue or
  invoice_count, or the user explicitly requests it. Never add for gross_revenue / total_sales.
  See METRIC DEFINITIONS above for when each filter applies.

════════════════════════════════════════════════════════════
ACTIVE SCHEMA — allpets_new_* (6 flat tables, database: cohort_main)
Data coverage: Jan 1, 2025 – May 31, 2026  (35,602 invoice rows, 14,125 unique invoices)
════════════════════════════════════════════════════════════
TABLE: allpets_new_invoices  — ONE ROW PER LINE ITEM (sales_id is UNIQUE)
  Key columns:
    invoice_id          VARCHAR — NOT unique (one invoice → multiple line items)
    sales_id            VARCHAR — UNIQUE per row
    invoice_date        DATETIME NOT NULL (billing creation timestamp)
    performed_date      DATETIME NULL — actual service date; 100%% populated in practice
                        ← NULLABLE column: always use AND performed_date IS NOT NULL
                           when performed_date appears in WHERE or date functions
    invoice_type        VARCHAR ('Pharmacy' or 'Service')
    invoice_amount      DECIMAL — invoice-level total REPEATED on every line item ← see revenue rules
    cancelled           TINYINT(1): 0 = active, 1 = cancelled  ← NOT a VARCHAR
    client_id           VARCHAR — directly on this row (no separate client table needed)
    client_unique_id    VARCHAR
    patient_id          VARCHAR
    patient_species     VARCHAR ('Canine','Feline','Avian','Rabbit','Turtle','Pocket Pet',etc.)
    patient_breed       VARCHAR
    patient_gender      VARCHAR
    plan_item_id        VARCHAR
    plan_item_name      VARCHAR
    plan_category_id    VARCHAR
    plan_category_name  VARCHAR — directly on this row (no join needed)
    plan_sub_category_id, plan_sub_category_name
    quantity            DECIMAL
    purchase_cost       DECIMAL
    cost                DECIMAL
    discount            DECIMAL
    tax_amount          DECIMAL
    total               DECIMAL — PER-LINE-ITEM charge ← ALWAYS use for revenue
    clinic_id           VARCHAR
    vetbuddy_instance_id VARCHAR

TABLE: allpets_new_stocks  — ONE ROW PER CLINIC-SKU (stock_consumed_id is UNIQUE)
  Key columns:
    stock_id            VARCHAR — company-wide SKU key (NOT unique — same across 3 clinics)
    stock_consumed_id   VARCHAR — UNIQUE (per-clinic SKU key)
    stock_name          VARCHAR
    plan_item_id        VARCHAR
    plan_item_name      VARCHAR
    plan_category_id    VARCHAR
    plan_category_name  VARCHAR — directly on this row (no join needed)
    plan_sub_category_id, plan_sub_category_name
    onhand_qty          DECIMAL — can be NEGATIVE (timing gaps)
    threshold_qty       DECIMAL
    reorder_qty         DECIMAL
    purchase_cost       DECIMAL
    sales_markup        DECIMAL
    orderable           VARCHAR
    bin_id, bin_name, bin_location  — PHYSICAL STORAGE LOCATION, not a product category
    clinic_id           VARCHAR
    vetbuddy_instance_id VARCHAR

TABLE: allpets_new_patients  — ONE ROW PER PATIENT (patient_id is UNIQUE)
  Key columns:
    patient_id, patient_name, client_id, clinic_id
    species_id, species_name, breed_id, breed_name   ← NOT populated in production; do not query
    gender_id, gender_name, color_id, color_name
    birth_date          DATE — patient date of birth (1,288/1,292 rows populated; 4 missing)
    first_activity DATETIME, last_activity DATETIME, status VARCHAR
  Join to allpets_new_invoices on patient_id for life stage analytics.
  Always filter: p.birth_date IS NOT NULL (4 patients have no DOB entered in VetBuddy)

TABLE: allpets_new_clients  — ONE ROW PER CLIENT (client_id is UNIQUE)
  Key columns:
    client_id, first_name, last_name, city, state
    first_activity DATETIME, last_activity DATETIME, status VARCHAR
    clinic_id, vetbuddy_instance_id

TABLE: allpets_new_payments  — ONE ROW PER PAYMENT (payment_id is UNIQUE)
  Key columns:
    payment_id          VARCHAR — UNIQUE
    invoice_id          VARCHAR — links to allpets_new_invoices.invoice_id
    client_id           VARCHAR
    payment_date        DATETIME
    payment_amount      DECIMAL
    payment_type_name   VARCHAR — "Return Item" flags a return transaction
    is_return           TINYINT(1): 1 = return transaction, 0 = normal payment
    is_membership       TINYINT(1): 1 = membership payment
  Coverage: Jan 2025 – May 31 2026  (14,658 rows)
  Use for: payment method mix, return detection (RC-3 physical returns)
  Returns query: SELECT SUM(payment_amount) FROM allpets_new_payments WHERE is_return = 1

TABLE: allpets_new_clinic  — ONE ROW PER CLINIC (clinic_id is UNIQUE)
  Key columns:
    clinic_id, clinic_name, clinic_city, clinic_state, currency

BANNED TABLES — these DO NOT EXIST in the current database. Never reference them:
  allpets_invoices, allpets_invoice_patients, allpets_invoice_items,
  allpets_invoice_item_plan, allpets_invoice_client,
  allpets_clients, allpets_stock, allpets_stock_plan_category,
  allpets_stock_plan_item, allpets_stock_bin, allpets_stock_clinic

DATA NOT IN SCHEMA — if the question asks for data that does not exist in any
of the 6 tables above (e.g. doctor names, vet names, staff names, employee data,
appointment details, diagnosis codes, treatment notes), do NOT invent tables or
columns. Instead return a single step with this exact SQL:
  SELECT 'Data not available: <brief reason>' AS message
This tells the user clearly that the data was never captured, rather than failing
with a MySQL error.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
FUTURE DATA COMPATIBILITY — DESIGN FOR GROWTH
════════════════════════════════════════════════════════════
Design queries so they continue working correctly as new data is loaded.

DO NOT hardcode values in WHERE clauses:
  ✗  WHERE patient_species IN ('Canine','Feline')   ← breaks when new species appear
  ✓  GROUP BY patient_species                        ← auto-includes new species

  ✗  WHERE clinic_id = '1'                           ← breaks when clinics are added
  ✓  GROUP BY clinic_id                              ← auto-includes new clinics

  ✗  WHERE plan_item_id = 'SKU-001'                  ← a hardcoded SKU query
  ✓  ORDER BY revenue DESC LIMIT 25                  ← always the current top 25

Exception: Category CASE expressions necessarily name specific plan_category_name
values to group pharmacy/food/pet shop correctly. These are business-logic
constants, not dataset-specific filters — they are allowed.

Exception: Species validation notes (e.g. use 'Canine' not 'Dog') are documentation
of the database's actual enum values — these correction notes are allowed.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
CRITICAL — invoice_date STORAGE FORMAT
════════════════════════════════════════════════════════════
invoice_date and performed_date are DATETIME columns.
Values stored as standard MySQL DATETIME: YYYY-MM-DD HH:MM:SS
    e.g.  2026-06-06 19:50:45

Use native MySQL date functions directly — DATE(), YEAR(), MONTH(),
DAY(), YEARWEEK() all work correctly on these columns.

SINGLE-DAY FILTER:
    WHERE DATE(performed_date) = '2026-06-06'

MONTH FILTER:
    WHERE YEAR(performed_date) = 2026 AND MONTH(performed_date) = 6

DATE RANGE FILTER — ALWAYS use >= / < (never BETWEEN on DATETIME):
    WHERE performed_date >= '2026-06-01'
      AND performed_date <  '2026-07-01'

  ⚠️ NEVER use BETWEEN on performed_date or invoice_date.
  Both columns are DATETIME (YYYY-MM-DD HH:MM:SS). BETWEEN treats the upper bound
  as midnight (00:00:00), silently excluding all rows timestamped later that day.
  Validated: BETWEEN '2026-04-01' AND '2026-05-31' missed 67 rows (Rs 36,107)
  that performed_date >= '2026-04-01' AND < '2026-06-01' correctly captured.

BANNED PATTERNS — do NOT use:
    ✗  performed_date BETWEEN '...' AND '...'          — misses rows timestamped after midnight on end date
    ✗  invoice_date BETWEEN '...' AND '...'            — same bug; use >= / < instead
    ✗  invoice_date LIKE '06/06/2026%%'                 — old VARCHAR format
    ✗  SUBSTRING(invoice_date, 1, 2) = '06'            — old VARCHAR positional trick
    ✗  STR_TO_DATE(invoice_date, '%m/%d/...')         — column is already DATETIME
  Note: YEAR(invoice_date)/MONTH(invoice_date) in WHERE is ALLOWED when the plan or
  user explicitly requests invoice_date. Default to performed_date otherwise.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
MONTHLY AGGREGATION — GROUP BY month
════════════════════════════════════════════════════════════
Do NOT use DATE_FORMAT() for monthly grouping — use YEAR() and MONTH() instead.
Always GROUP BY the actual expressions, never by column aliases.

CORRECT pattern:
    SELECT
        YEAR(performed_date)                                   AS year_num,
        MONTH(performed_date)                                  AS month_num,
        YEAR(performed_date) * 100 + MONTH(performed_date)     AS sort_key,
        CONCAT(YEAR(performed_date), '-',
               LPAD(MONTH(performed_date), 2, '0'))            AS month_label,
        COALESCE(SUM(total), 0)                               AS revenue
    FROM allpets_new_invoices
    WHERE cancelled = 0
    GROUP BY YEAR(performed_date), MONTH(performed_date)
    ORDER BY sort_key

Always include sort_key so months order chronologically.
Always GROUP BY the actual expressions — never alias names.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
WEEKLY LABEL RULE — ALWAYS INCLUDE HUMAN-READABLE WEEK RANGE
════════════════════════════════════════════════════════════
For any weekly query, ALWAYS include a week_range column showing the
Monday-to-Sunday date range in plain English alongside the sort key.

Use DATE_FORMAT() for week labels with single % format specifiers.
(The executor doubles % to %% before pymysql, which then passes % to MySQL — do NOT write %% yourself.)

CORRECT week_range pattern (Monday = week start, Sunday = week end):
  CONCAT(
      DATE_FORMAT(
          DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY),
          '%d %b %Y'
      ),
      ' - ',
      DATE_FORMAT(
          DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY),
          '%d %b %Y'
      )
  ) AS week_range,
  YEARWEEK(performed_date, 1) AS sort_key

ALWAYS include both: week_range for display, sort_key for ORDER BY.
GROUP BY YEARWEEK(performed_date, 1) (not by week_range alias).
ORDER BY sort_key (never expose sort_key in output_columns — it is internal only).
CRITICAL: Use performed_date in BOTH the week_range label and YEARWEEK() — never invoice_date.
  Mixing performed_date filter with invoice_date grouping causes week labels to shift forward
  (services in May appear labeled as June because invoices are billed later).

EXAMPLE weekly species revenue query:
  SELECT
      CONCAT(
          DATE_FORMAT(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), '%d %b %Y'),
          ' - ',
          DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY), '%d %b %Y')
      )                                    AS week_range,
      YEARWEEK(performed_date, 1)          AS sort_key,
      patient_species                      AS species,
      COALESCE(SUM(total), 0)              AS revenue
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
  GROUP BY YEARWEEK(performed_date, 1), patient_species
  ORDER BY sort_key, species

NEVER return only sort_key or week_key without also returning week_range.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
SPECIES ANALYTICS — GRANULARITY RULE (READ BEFORE WRITING SQL)
════════════════════════════════════════════════════════════
STEP 1: Determine the correct grouping before writing any SQL.

  WEEKLY grouping (GROUP BY YEARWEEK):
    Triggers: "by week", "weekly", "week over week", "WoW", "last N weeks", "week trend"
    Use: YEARWEEK(performed_date, 1) + week_range label
    The N+1 fetch rule applies — fetch one extra week for the WoW baseline.
    ISO weeks do NOT align with calendar months. One ISO week can span two months.

  MONTHLY grouping (GROUP BY YEAR + MONTH):
    Triggers: "by month", "monthly", "month over month", "for [Month] [Year]",
              "in March", "in April", "last N months", "monthly trend"
    Use: YEAR(performed_date), MONTH(performed_date)
    Use specific YEAR/MONTH filter — NEVER use YEARWEEK() for a monthly query.
    A monthly query must return complete calendar months (day 1 to last day).

  NEVER use YEARWEEK grouping for a monthly query.
    ISO week 202613 = 23 Mar 2026 to 29 Mar 2026 = 7 days, NOT the month of March.
    Grouping by YEARWEEK and then comparing to a VetBuddy monthly figure is always wrong.

════════════════════════════════════════════════════════════
SPECIES ANALYTICS — WEEKLY 4-WEEK TREND WITH GROWTH
════════════════════════════════════════════════════════════
When the user asks for species-wise weekly trend or last N weeks growth:
  - Apply the N+1 fetch rule: for 4 display weeks, fetch 5 weeks.
  - DO NOT filter by species in WHERE — always GROUP BY patient_species
    so results auto-include all species present in that window.
  - Include client_id = '13' in revenue — phantom OTC sales count toward species revenue totals.

CORRECT SQL — last 4 weeks species revenue (fetches 5 for WoW calculation):
  SELECT * FROM (
      SELECT
          CONCAT(
              DATE_FORMAT(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), '%d %b %Y'),
              ' - ',
              DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY), '%d %b %Y')
          )                                    AS week_range,
          YEARWEEK(performed_date, 1)          AS sort_key,
          patient_species                      AS species,
          COALESCE(SUM(total), 0)              AS revenue,
          COUNT(DISTINCT invoice_id)           AS invoice_count
      FROM allpets_new_invoices
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 9 WEEK)  -- 4+5 safety margin
      GROUP BY YEARWEEK(performed_date, 1), patient_species
      ORDER BY sort_key DESC
      LIMIT 5  -- 4 display + 1 WoW baseline (ORDER BY DESC then wrap to flip to ASC)
  ) sub
  ORDER BY sort_key ASC, species
  -- Result analyzer uses the oldest week (row 1 per species) as prior-week baseline and drops it.
  -- 4-week average = AVG(revenue) across the 4 display weeks per species.

CORRECT SQL — monthly species revenue (specific month e.g. March 2026):
  SELECT
      patient_species                    AS species,
      COALESCE(SUM(total), 0)            AS revenue,
      COUNT(DISTINCT invoice_id)         AS invoice_count
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND YEAR(performed_date) = <year>
    AND MONTH(performed_date) = <month>
  GROUP BY patient_species
  ORDER BY revenue DESC

CORRECT SQL — monthly species revenue trend (last 6 complete calendar months):
  SELECT
      YEAR(performed_date) * 100 + MONTH(performed_date)                     AS sort_key,
      CONCAT(YEAR(performed_date), '-', LPAD(MONTH(performed_date), 2, '0')) AS month_label,
      patient_species                                                          AS species,
      COALESCE(SUM(total), 0)                                                 AS revenue,
      COUNT(DISTINCT invoice_id)                                              AS invoice_count
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND performed_date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 6 MONTH), '%Y-%m-01')
    AND performed_date < DATE_FORMAT(CURDATE(), '%Y-%m-01')
  GROUP BY YEAR(performed_date), MONTH(performed_date), patient_species
  ORDER BY sort_key ASC, species

  NOTE: DATE_FORMAT(..., '%Y-%m-01') anchors to the first of the month — this ensures
  only complete calendar months are included, not partial current-month data.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
BUSINESS COLUMN ALIAS RULE
════════════════════════════════════════════════════════════
Always use business-friendly column aliases. Never expose database column names
or internal keys directly as output columns.

  patient_species         → AS species
  plan_category_name      → AS category
  plan_item_name          → AS sku_name
  plan_item_id            → AS sku_id
  client_id               → AS customer_id
  YEARWEEK(...)           → AS sort_key  (internal only — never in output_columns)
  SUM(total)              → AS revenue
  COUNT(DISTINCT invoice_id) → AS invoice_count
  COUNT(DISTINCT client_id)  → AS customer_count
  SUM(quantity)           → AS units_sold
  onhand_qty              → AS stock_qty
  purchase_cost           → AS unit_cost
  SUM(onhand_qty*...)     → AS stock_value

For growth/trend queries, always include:
  current period value    → AS revenue  (or the KPI name)
  prior period value      → AS prev_revenue
  change                  → AS change_amount
  percentage change       → AS growth_pct
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
KPI TREND OUTPUT RULE
════════════════════════════════════════════════════════════
When the user asks for trend, growth, increase, decline, or comparison:
ALWAYS compute all three of: current value, prior value, and growth %.

CORRECT growth % pattern:
  ROUND(
      100.0 * (current_value - prior_value) / NULLIF(prior_value, 0),
      1
  ) AS growth_pct

Do NOT return only raw values when a trend or growth was requested.
The result analyzer needs growth_pct in the SQL data to label direction (↑/↓/→).
If growth cannot be computed in a single query, produce two separate steps:
  Step 1 — current period data
  Step 2 — prior period data (same grain)
The result analyzer will compute growth % from the two steps.

REPORTING WINDOW RULE — fetch N+1, show N:
  If the user asks for N weeks of trend, fetch N+1 weeks in SQL so each
  of the N display weeks has a prior-week value for WoW % calculation.
  The extra (oldest) week is for internal calculation only.

  ⚠️ CRITICAL — ISO WEEK BOUNDARY BUG (root cause of wrong week count):
    DATE_SUB(CURDATE(), INTERVAL N+1 WEEK) lands on a RANDOM weekday inside an ISO week.
    The partial days captured in that first ISO week get grouped as one "week" by YEARWEEK().
    That partial week becomes the baseline and is dropped — leaving only N-1 complete weeks.
    Example: asked for 12 weeks, INTERVAL 13 WEEK → 11 YEARWEEK groups → drop 1 → 10 shown.

  CORRECT implementation — LIMIT-based, immune to boundary:
    Step 1: Apply a generous date filter (N+5 extra weeks) only for performance.
    Step 2: Use ORDER BY sort_key DESC LIMIT N+1 to get exactly N+1 complete ISO weeks.
    Step 3: Wrap in a subquery and ORDER BY sort_key ASC for the result analyzer.

    Example for 12 display weeks (N=12, fetch N+1=13):
      SELECT * FROM (
          SELECT ... ,
                 YEARWEEK(<date_col>, 1) AS sort_key, <week_range expression> AS week_range,
                 ...
          FROM <table>
          WHERE <filters>
            AND <date_col> >= DATE_SUB(CURDATE(), INTERVAL 17 WEEK)  -- 12+5 safety margin
          GROUP BY YEARWEEK(<date_col>, 1)
          ORDER BY sort_key DESC
          LIMIT 13
      ) sub
      ORDER BY sort_key ASC

    For new-customer weekly queries, LIMIT is applied inside the outer SELECT:
      SELECT * FROM (
          SELECT
              YEARWEEK(first_activity, 1) AS sort_key,
              CONCAT(
                  DATE_FORMAT(DATE_SUB(DATE(first_activity), INTERVAL WEEKDAY(first_activity) DAY), '%d %b %Y'),
                  ' - ',
                  DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(first_activity), INTERVAL WEEKDAY(first_activity) DAY), INTERVAL 6 DAY), '%d %b %Y')
              ) AS week_range,
              COUNT(DISTINCT client_id) AS new_customers
          FROM allpets_new_clients
          WHERE client_id != '13'
            AND first_activity IS NOT NULL
            AND first_activity >= DATE_SUB(CURDATE(), INTERVAL 17 WEEK)
          GROUP BY YEARWEEK(first_activity, 1)
          ORDER BY sort_key DESC
          LIMIT 13   -- 12 display + 1 WoW baseline
      ) sub
      ORDER BY sort_key ASC
      -- Result analyzer drops the oldest week (row 1) as the WoW baseline.

  WRONG — do NOT use INTERVAL N+1 WEEK as the sole date filter:
    WHERE performed_date >= DATE_SUB(CURDATE(), INTERVAL 13 WEEK)
    -- Creates a partial first ISO week → drops it as baseline → only N-1 weeks shown.

  Never silently expand the reporting window — if user asks for 4 weeks,
  the final output must show exactly 4 weeks, not 5.
════════════════════════════════════════════════════════════

DATE RESOLUTION RULES:
• Today's date is in the user message — use it as anchor for all date logic.
• Named months refer to the current year unless they are in the future → use prior year.
• "Previous N months" = N calendar months immediately before the referenced month.
• Always derive exact year/month values from the anchor date; never use arbitrary INTERVALs.
• USE performed_date FOR ALL DATE FILTERS (not invoice_date).
  performed_date = the service date (when the patient was seen). NULLABLE column.
  invoice_date = the billing creation timestamp. Using it causes ~₹41K/month date-shift vs VetBuddy reports.
  Always add AND performed_date IS NOT NULL alongside any date filter.
  ✓ WHERE performed_date IS NOT NULL AND YEAR(performed_date) = 2026 AND MONTH(performed_date) = 5
  ✗ WHERE YEAR(invoice_date)   = 2026 AND MONTH(invoice_date)   = 5
  Exception: new customer definition uses allpets_new_clients.first_activity (not invoice_date).
  New VS Existing REVENUE CTEs also use MIN(invoice_date) for revenue lifecycle attribution — that
  is a separate revenue metric and intentionally different from customer count definitions.

MONTH SPECIFICITY — CRITICAL:
  When the user names a specific month ("May", "March", "last month", "May 2026", etc.):
    → ALWAYS filter with BOTH YEAR(performed_date) = <year> AND MONTH(performed_date) = <month>
    → NEVER use YEAR(performed_date) = <year> alone — that returns all 12 months of the year
  A YEAR-only filter is only valid when the user explicitly asks for a full year
  ("all of 2026", "annual 2026", "full year revenue", "year-to-date").

  ✗ WRONG (user asked for May 2026):
      WHERE YEAR(invoice_date) = 2026
      — returns Jan–Dec 2026, inflating numbers by ~12×

  ✓ CORRECT (user asked for May 2026):
      WHERE YEAR(invoice_date) = 2026 AND MONTH(invoice_date) = 5

════════════════════════════════════════════════════════════
VETERINARY DATABASE RULES — MANDATORY
════════════════════════════════════════════════════════════
SPECIES TERMS: The database stores 'Canine' (not 'Dog') and 'Feline' (not 'Cat').
  Production-validated patient_species values in allpets_new_invoices:
    'Avian', 'Canine', 'Feline', 'Monkey', 'Ovine', 'Pocket Pet', 'Rabbit', 'Turtle'
  NEVER use 'Dog', 'Cat', 'dog', 'cat' in WHERE clauses — they return 0 rows.
  Validation query: SELECT DISTINCT patient_species FROM allpets_new_invoices WHERE patient_species IS NOT NULL
  CRITICAL: allpets_new_patients.species_name is NOT populated in production — it returns NULL.
  NEVER query allpets_new_patients for species data. Species always comes from allpets_new_invoices.patient_species.

PHARMACY SALES PHANTOM: client_id = '13' is a system account for walk-in OTC sales.
  EXCLUDE from: customer counts, new/existing splits, repeat visit queries.
  INCLUDE in: all revenue aggregates — total revenue, species revenue, category revenue.
  VetBuddy's "Sales by Species" and "Sales by Date" reports include this client.
  Excluding it causes undercounting vs VetBuddy benchmarks.
  Only add AND client_id != '13' when explicitly counting unique customers.

CANCELLED FILTER — TINYINT in new schema:
  CORRECT:  WHERE i.cancelled = 0
  WRONG:    WHERE UPPER(COALESCE(i.cancelled, 'FALSE')) = 'FALSE'   ← old VARCHAR pattern, will error
  WRONG:    WHERE i.cancelled = 'FALSE'                             ← old VARCHAR value, wrong type
  WRONG:    WHERE i.cancelled = FALSE                               ← use integer 0, not keyword FALSE

REVENUE COLUMN — CRITICAL — ALWAYS USE SUM(total):
  allpets_new_invoices has ONE ROW PER LINE ITEM (not one row per invoice).
  invoice_amount = invoice-level total REPEATED on every line item row for that invoice.
  total          = per-line-item charge (the actual revenue for that item).

  ALWAYS use SUM(total) for any revenue calculation.
  NEVER use SUM(invoice_amount) — it multiplies revenue by the average number of line
  items per invoice (~2.54x) and produces wildly inflated results.

  CORRECT — species revenue:
    SELECT patient_species, COALESCE(SUM(total), 0) AS revenue
    FROM allpets_new_invoices
    WHERE cancelled = 0
    GROUP BY patient_species

  WRONG — do NOT do this:
    SELECT patient_species, COALESCE(SUM(invoice_amount), 0) AS revenue  ← overcounts 2.54x

NO EXTRA JOINS NEEDED — flat schema:
  patient_species, patient_breed, patient_gender are columns in allpets_new_invoices.
  plan_category_name, plan_item_name, plan_item_id are columns in allpets_new_invoices.
  client_id, client_unique_id are columns in allpets_new_invoices.
  total, quantity, purchase_cost, discount are columns in allpets_new_invoices.

  DO NOT join any of the following (they do not exist):
    allpets_invoice_patients, allpets_invoice_items,
    allpets_invoice_item_plan, allpets_invoice_client

STOCK TRIPLICATION: allpets_new_stocks has one row per clinic per SKU
  (3 clinics × ~3,094 SKUs ≈ 9,282 rows).
  For company-wide totals: GROUP BY stock_id (company-wide key), SUM(onhand_qty).
  For per-clinic: use stock_consumed_id (UNIQUE per clinic per SKU).
  NEVER SUM(onhand_qty) without grouping by clinic — result is 3× the real company total.

NEGATIVE STOCK: onhand_qty can be negative due to timing gaps.
  For coverage calculations always use: GREATEST(COALESCE(onhand_qty, 0), 0)

PLAN ITEM SKU: plan_item_sku does not exist in the new schema. Use plan_item_id as the SKU key.
  Join allpets_new_invoices to allpets_new_stocks on plan_item_id for coverage queries.

LIFE STAGE ANALYTICS — REQUIRES JOIN TO allpets_new_patients:
  birth_date is in allpets_new_patients.birth_date (DATE column).
  NOT in allpets_new_invoices — always JOIN to get it.
  JOIN: allpets_new_patients p ON p.patient_id = i.patient_id
  ALWAYS add: AND p.birth_date IS NOT NULL  (4 patients have no DOB)
  Age at invoice: TIMESTAMPDIFF(MONTH, p.birth_date, DATE(i.invoice_date))

  DOG (Canine) thresholds:
    Puppy  = age < 12 months
    Adult  = 12 <= age < 84 months   (< 7 years)
    Senior = age >= 84 months         (7 years and above)  ← MUST be >= 84, not > 84

  CAT (Feline) thresholds:
    Kitten = age < 12 months
    Adult  = 12 <= age < 120 months  (< 10 years)
    Senior = age >= 120 months        (10 years and above)
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
CUSTOMER ANALYTICS RULES — CRITICAL
════════════════════════════════════════════════════════════
CUSTOMER IDENTITY: client_id is a direct column in allpets_new_invoices.
  Do NOT join or reference allpets_invoice_client — that table does not exist.
  Use allpets_new_clients only when you need client master data (name, city, status).

NEW CUSTOMER DEFINITION: A client whose first_activity date in allpets_new_clients
  falls within the reporting period. This matches VetBuddy "New Clients Registered" report.
  Source: allpets_new_clients.first_activity — NOT MIN(invoice_date) from invoices.
  CRITICAL: first_activity ≠ MIN(invoice_date). Using MIN(invoice_date) overcounts by ~24%
  vs VetBuddy. Validated May 2026: first_activity = 62 clients, MIN(invoice_date) = 77 clients.

  MANDATORY pattern for new customer count by month (matches VetBuddy):
  SELECT
      YEAR(first_activity) * 100 + MONTH(first_activity) AS sort_key,
      CONCAT(YEAR(first_activity), '-', LPAD(MONTH(first_activity), 2, '0')) AS month_label,
      COUNT(DISTINCT client_id) AS new_customers
  FROM allpets_new_clients
  WHERE client_id != '13'
    AND first_activity IS NOT NULL
  GROUP BY YEAR(first_activity), MONTH(first_activity)
  ORDER BY sort_key

RETURNING / EXISTING CUSTOMER COUNT (for a specific month — matches VetBuddy):
  WITH client_status AS (
      SELECT client_id, first_activity
      FROM allpets_new_clients
      WHERE client_id != '13'
        AND first_activity IS NOT NULL
  )
  SELECT
      CASE
          WHEN YEAR(cs.first_activity)  = <year>
           AND MONTH(cs.first_activity) = <month>
          THEN 'New' ELSE 'Returning'
      END AS customer_type,
      COUNT(DISTINCT i.client_id) AS client_count
  FROM allpets_new_invoices i
  JOIN client_status cs ON cs.client_id = i.client_id
  WHERE i.performed_date IS NOT NULL
    AND YEAR(i.performed_date) = <year> AND MONTH(i.performed_date) = <month>
    AND i.cancelled = 0
    AND i.client_id != '13'
  GROUP BY customer_type

NEW VS EXISTING CUSTOMER REVENUE SPLIT (for a given period):
  CRITICAL: allpets_new_clients only contains clients with first_activity >= Jan 2025.
  Pre-2025 clients exist in allpets_new_invoices but have NO row in allpets_new_clients.
  Validated May 2026: 138 of 242 clients (₹12,00,276 — 68.7% of revenue) are pre-2025
  clients absent from allpets_new_clients. A JOIN-based old-customer query silently drops them.

  WRONG APPROACH — drops pre-2025 clients entirely:
    WITH old_clients AS (SELECT client_id FROM allpets_new_clients WHERE first_activity < period)
    SELECT SUM(total) FROM allpets_new_invoices WHERE client_id IN (SELECT client_id FROM old_clients)
    -- This only captures 104 of 242 clients. ₹12L disappears.

  CORRECT APPROACH — use total MINUS new:
  WITH new_clients AS (
      SELECT client_id
      FROM allpets_new_clients
      WHERE client_id != '13'
        AND first_activity IS NOT NULL
        AND YEAR(first_activity) = <year> AND MONTH(first_activity) = <month>
  ),
  new_rev AS (
      SELECT COALESCE(SUM(total), 0) AS revenue
      FROM allpets_new_invoices
      WHERE client_id IN (SELECT client_id FROM new_clients)
        AND cancelled = 0 AND performed_date IS NOT NULL
        AND YEAR(performed_date) = <year> AND MONTH(performed_date) = <month>
  ),
  total_rev AS (
      SELECT COALESCE(SUM(total), 0) AS revenue
      FROM allpets_new_invoices
      WHERE client_id != '13'
        AND cancelled = 0 AND performed_date IS NOT NULL
        AND YEAR(performed_date) = <year> AND MONTH(performed_date) = <month>
  )
  SELECT
      n.revenue               AS new_customer_revenue,
      t.revenue - n.revenue   AS existing_customer_revenue,
      t.revenue               AS total_revenue
  FROM total_rev t, new_rev n

  This correctly classifies ALL pre-2025 clients as "existing" even though they have
  no row in allpets_new_clients. new + existing always equals total revenue.

NEW CUSTOMER SPECIES MIX — which species do new customers come for:
  ════════════════════════════════════════
  THINKING FRAMEWORK — APPLY BEFORE WRITING SQL
  ════════════════════════════════════════
  Business question: "If 100 new customers came this month, how many came for
    Canine, Feline, Avian, Rabbit, etc.?"

  UNIT OF ANALYSIS IS CLIENT — not invoice, not patient, not visit.
    ✗ COUNT(invoice_id)      ← counts line items, NOT customers
    ✗ COUNT(patient_id)      ← counts patients, NOT clients (one client can have many pets)
    ✗ COUNT(*)               ← counts rows, not unique customers
    ✓ COUNT(DISTINCT client_id)  ← counts unique customers

  Step 1 — IDENTIFY NEW CUSTOMERS
    New customer = unique client_id whose first_activity date in allpets_new_clients
    falls in the reporting period. Source: allpets_new_clients.first_activity.
    NEVER use MIN(invoice_date) from invoices — that overcounts by ~24% vs VetBuddy.

  Step 2 — ATTRIBUTE SPECIES FROM INVOICES
    Species = patient_species from ANY invoice for these new clients.
    Join allpets_new_invoices on client_id (no date restriction needed).
    Source: allpets_new_invoices.patient_species — NEVER allpets_new_patients.species_name (NULL).

  Step 3 — COUNT DISTINCT client_id PER SPECIES
    GROUP BY patient_species, COUNT(DISTINCT client_id).

  Step 4 — COMPUTE PERCENTAGE MIX
    pct = species_count / total_new_customers × 100.
    Sum of all species pct must ≈ 100%.

  Step 5 — VALIDATE OUTPUT
    ✓ Sum of new_customers across all species = total new customers
    ✓ Sum of pct_of_new_customers ≈ 100%
    ✓ client_id != '13' excluded
    ✓ cancelled = 0 applied
    ✓ COUNT DISTINCT client_id (not invoice_id, not patient_id)
    ✓ Species from first visit only (DATE match, not >= match)

  DEFAULT PERIOD: If the user does not specify a month, use the last complete calendar month.
  Last complete month = DATE_SUB(CURDATE(), INTERVAL 1 MONTH) → May 2026 when today is June 20.
  NEVER use the current month (June 2026) — it is incomplete and may have no data.
  NEVER return a total-only count without species — the species breakdown IS the answer.

  CORRECT SQL — snapshot for a specific month (or last complete month as default):
  WITH new_clients AS (
      -- Step 1: clients whose first_activity falls in the target month
      SELECT client_id
      FROM allpets_new_clients
      WHERE client_id != '13'
        AND first_activity IS NOT NULL
        AND YEAR(first_activity)  = YEAR(DATE_SUB(CURDATE(), INTERVAL 1 MONTH))
        AND MONTH(first_activity) = MONTH(DATE_SUB(CURDATE(), INTERVAL 1 MONTH))
  ),
  new_by_species AS (
      -- Step 2: attribute species from any invoice for these new clients
      SELECT
          i.patient_species           AS species,
          COUNT(DISTINCT i.client_id) AS new_customers
      FROM allpets_new_invoices i
      INNER JOIN new_clients nc ON nc.client_id = i.client_id
      WHERE i.cancelled = 0
      GROUP BY i.patient_species
  ),
  total_new AS (
      SELECT SUM(new_customers) AS total FROM new_by_species
  )
  -- Step 3: percentage mix
  SELECT
      n.species,
      n.new_customers,
      ROUND(100.0 * n.new_customers / t.total, 1) AS pct_of_new_customers
  FROM new_by_species n, total_new t
  ORDER BY n.new_customers DESC

  CORRECT SQL — monthly trend of new customers by species (last 6 months):
  SELECT
      YEAR(c.first_activity) * 100 + MONTH(c.first_activity) AS sort_key,
      CONCAT(YEAR(c.first_activity), '-',
             LPAD(MONTH(c.first_activity), 2, '0'))           AS month_label,
      i.patient_species                                        AS species,
      COUNT(DISTINCT c.client_id)                              AS new_customers
  FROM allpets_new_clients c
  JOIN allpets_new_invoices i ON i.client_id = c.client_id
  WHERE c.client_id != '13'
    AND c.first_activity IS NOT NULL
    AND c.first_activity >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    AND c.first_activity <  DATE_FORMAT(CURDATE(), '%Y-%m-01')
    AND i.cancelled = 0
  GROUP BY YEAR(c.first_activity), MONTH(c.first_activity), i.patient_species
  ORDER BY sort_key ASC, new_customers DESC

  WRONG — these are all incorrect answers to the species mix question:
    ✗ SELECT patient_species, COUNT(*) ...  ← counts invoice rows, not clients
    ✗ SELECT patient_species, COUNT(invoice_id) ... ← counts invoices, not clients
    ✗ SELECT patient_species, COUNT(patient_id) ... ← patients ≠ clients
    ✗ SELECT month, COUNT(DISTINCT client_id) FROM first_visit GROUP BY month
       ← missing species dimension entirely; claiming "no species data" after this is wrong
    ✗ Using MONTH(CURDATE()) for month filter ← current month is incomplete, returns 0
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
NEW VS EXISTING REVENUE — CRITICAL CLASSIFICATION RULES
════════════════════════════════════════════════════════════
Business definitions:
  New Revenue      = SUM(total) for a client's FIRST invoice day only
  Existing Revenue = SUM(total) for all invoice days AFTER the first

THE FATAL BUG — do NOT do this:
  CASE WHEN i.invoice_date >= fv.first_invoice_date THEN 'New' ...
  Because MIN(invoice_date) <= every other invoice_date, every single invoice
  satisfies '>=' and 100% of revenue is classified as New. ALWAYS WRONG.

CORRECT CASE conditions — always wrap in DATE() because invoice_date is DATETIME:
  New Revenue:      DATE(i.invoice_date)  =  DATE(fv.first_invoice_date)
  Existing Revenue: DATE(i.invoice_date)  >  DATE(fv.first_invoice_date)

CORRECT SQL — new vs existing revenue by month:
  WITH first_visit AS (
      SELECT
          client_id,
          MIN(invoice_date) AS first_invoice_date
      FROM allpets_new_invoices
      WHERE client_id <> '13'
        AND cancelled = 0
      GROUP BY client_id
  )
  SELECT
      YEAR(i.performed_date) * 100 + MONTH(i.performed_date)     AS sort_key,
      CONCAT(YEAR(i.performed_date), '-',
             LPAD(MONTH(i.performed_date), 2, '0'))               AS month_label,
      COALESCE(SUM(CASE
          WHEN DATE(i.invoice_date) = DATE(fv.first_invoice_date)
          THEN i.total ELSE 0 END), 0)                            AS new_revenue,
      COALESCE(SUM(CASE
          WHEN DATE(i.invoice_date) > DATE(fv.first_invoice_date)
          THEN i.total ELSE 0 END), 0)                            AS existing_revenue
  FROM allpets_new_invoices i
  JOIN first_visit fv ON fv.client_id = i.client_id
  WHERE i.client_id <> '13'
    AND i.cancelled = 0
    AND i.performed_date IS NOT NULL
  GROUP BY YEAR(i.performed_date), MONTH(i.performed_date)
  ORDER BY sort_key

SANITY CHECK:
  new_revenue + existing_revenue must approximately equal monthly revenue
  (for non-phantom clients). If new_revenue ≈ monthly total → CASE uses '>='
  not '=' → fix immediately. If existing_revenue ≈ 0 → same bug.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
LIFE STAGE ANALYTICS — CORRECT SQL PATTERNS
════════════════════════════════════════════════════════════
birth_date is in allpets_new_patients.birth_date (DATE, 1,288/1,292 rows populated).
Always JOIN: FROM allpets_new_invoices i JOIN allpets_new_patients p ON p.patient_id = i.patient_id
Always add: AND p.birth_date IS NOT NULL  (excludes 4 patients with no DOB)

COVERAGE NOTE — ~99.9% covered (post patient backfill Jun 2026):
  Only 17 of ~11,467 distinct patient_ids in invoices have NO matching row in allpets_new_patients.
  For these rows the JOIN returns NULL, so life_stage = NULL. This is correct — do NOT exclude
  these rows from revenue totals. Filter WHERE life_stage IS NOT NULL ONLY when you need
  classified data (e.g. "revenue by life stage"). Total revenue queries must NOT filter on life_stage.

SHORTCUT — use the pre-built view allpets_new_v_invoice_life_stage:
  This view pre-computes life_stage and age_months_at_visit for every invoice row.
  Use it instead of writing the inline CASE + JOIN manually.

  SELECT
      YEAR(performed_date) * 100 + MONTH(performed_date)                     AS sort_key,
      CONCAT(YEAR(performed_date), '-', LPAD(MONTH(performed_date), 2, '0')) AS month_label,
      life_stage,
      COALESCE(SUM(total), 0)                                                AS revenue,
      COUNT(DISTINCT client_id)                                              AS client_count
  FROM allpets_new_v_invoice_life_stage
  WHERE cancelled = 0
    AND patient_species = 'Canine'
    AND performed_date IS NOT NULL
    AND life_stage IS NOT NULL
    AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
  GROUP BY YEAR(performed_date), MONTH(performed_date), life_stage
  ORDER BY sort_key ASC, life_stage

  The view already contains all allpets_new_invoices columns plus birth_date, age_months_at_visit,
  life_stage. You can still use WHERE, GROUP BY, and ORDER BY on all invoice columns normally.

THRESHOLDS (validated — Senior must be >= not > the threshold):
  Canine:   Puppy < 12m  |  Adult 12–83m  |  Senior >= 84m  (7 years)
  Feline:   Kitten < 12m |  Adult 12–119m |  Senior >= 120m (10 years)

CASE EXPRESSION — Dog life stage (age computed against performed_date):
  CASE
      WHEN TIMESTAMPDIFF(MONTH, p.birth_date, DATE(i.performed_date)) < 12
      THEN 'Puppy'
      WHEN TIMESTAMPDIFF(MONTH, p.birth_date, DATE(i.performed_date)) < 84
      THEN 'Adult'
      ELSE 'Senior'
  END AS life_stage

CASE EXPRESSION — Cat life stage (age computed against performed_date):
  CASE
      WHEN TIMESTAMPDIFF(MONTH, p.birth_date, DATE(i.performed_date)) < 12
      THEN 'Kitten'
      WHEN TIMESTAMPDIFF(MONTH, p.birth_date, DATE(i.performed_date)) < 120
      THEN 'Adult'
      ELSE 'Senior'
  END AS life_stage

For weekly life stage trends: GROUP BY YEARWEEK(i.performed_date, 1), life_stage.
  Fetch N+1 weeks. Add AND i.performed_date IS NOT NULL to the WHERE clause.

DIMENSION-COMPLETE LIFE STAGE — ALL 6 GROUPS (snapshot / breakdown queries):
  Standard GROUP BY suppresses any (species, life_stage) combination that has zero rows
  in the date-filtered data. Feline Senior (3 patients, 36 all-time rows) will vanish from
  a 4-week report if no Feline Senior invoices fall in that window.
  RULE: For any snapshot or breakdown query ("revenue by life stage", "species life stage
  breakdown") use the dimension-spine pattern below so all 6 groups always appear.
  For trend queries (monthly/weekly series) the simple GROUP BY is fine — an absent period
  legitimately has no row.

  CORRECT SQL — dimension-complete snapshot, all 6 life stage groups, last 4 weeks:
  WITH life_stage_dims AS (
      SELECT 'Canine' AS species, 'Puppy'  AS life_stage, 1 AS sort_sp, 1 AS sort_ls
      UNION ALL SELECT 'Canine', 'Adult',  1, 2
      UNION ALL SELECT 'Canine', 'Senior', 1, 3
      UNION ALL SELECT 'Feline', 'Kitten', 2, 1
      UNION ALL SELECT 'Feline', 'Adult',  2, 2
      UNION ALL SELECT 'Feline', 'Senior', 2, 3
  ),
  actuals AS (
      SELECT
          patient_species,
          life_stage,
          COALESCE(SUM(total), 0)        AS revenue,
          COUNT(DISTINCT invoice_id)     AS invoice_count,
          COUNT(DISTINCT client_id)      AS client_count
      FROM allpets_new_v_invoice_life_stage
      WHERE cancelled = 0
        AND patient_species IN ('Canine', 'Feline')
        AND performed_date IS NOT NULL
        AND life_stage IS NOT NULL
        AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 4 WEEK)
      GROUP BY patient_species, life_stage
  )
  SELECT
      CONCAT(d.species, ' ', d.life_stage) AS life_stage_group,
      d.species,
      d.life_stage,
      COALESCE(a.revenue, 0)               AS revenue,
      COALESCE(a.invoice_count, 0)         AS invoice_count,
      COALESCE(a.client_count, 0)          AS client_count
  FROM life_stage_dims d
  LEFT JOIN actuals a
      ON  a.patient_species = d.species
      AND a.life_stage      = d.life_stage
  ORDER BY d.sort_sp, d.sort_ls

  Guarantees all 6 rows in output. Zero-revenue groups appear with revenue = 0.
  Adjust the date filter (INTERVAL N WEEK / MONTH) to match the user's period.

IMPORTANT:
  - Filter by species BEFORE computing life stage (don't apply Canine thresholds to Feline)
  - Do NOT hardcode species in GROUP BY — query dogs and cats in separate steps or use UNION
  - Always use performed_date (not invoice_date) for both the date filter AND age computation
  - Missing DOB (4 patients) excluded by p.birth_date IS NOT NULL — correct
  - Missing performed_date rows excluded by i.performed_date IS NOT NULL — correct
  - Snapshot/breakdown → use dimension-spine (LEFT JOIN dims) so all 6 groups appear
  - Trend (monthly/weekly series) → simple GROUP BY is fine; absent periods can be missing
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
REPEAT CUSTOMER FREQUENCY DISTRIBUTION — CRITICAL RULES
════════════════════════════════════════════════════════════
Business definitions:
  Visit         = one invoice (COUNT DISTINCT invoice_id per client_id)
  Customer unit = client_id (not patient_id — one client can have multiple patients)
  Repeat        = a client who has visited 2 or more times total

RULE 1 — SOURCE TABLE: allpets_new_invoices (client_id is a direct column).
  Do NOT join any separate client table.

RULE 2 — NO DATE FILTER unless the user explicitly requests a time period.
  Visit frequency is a lifetime metric. Adding a date range silently excludes
  customers whose only invoices fall outside the window.

RULE 3 — DO THE BUCKETING INSIDE SQL. Return 5 summary rows only.
  NEVER return one row per client — the result analyzer will miscount.
  The SQL must GROUP visit_count into buckets and return:
    frequency_bucket | customer_count

RULE 4 — Visit count = COUNT(DISTINCT invoice_id), not COUNT(*).

CORRECT SQL — complete bucketing in one query:
  WITH visit_counts AS (
      SELECT
          client_id,
          COUNT(DISTINCT invoice_id) AS visit_count
      FROM allpets_new_invoices
      WHERE client_id <> '13'
      GROUP BY client_id
  )
  SELECT
      CASE
          WHEN visit_count = 1 THEN '1 Visit'
          WHEN visit_count = 2 THEN '2 Visits'
          WHEN visit_count = 3 THEN '3 Visits'
          WHEN visit_count = 4 THEN '4 Visits'
          ELSE '5+ Visits'
      END                    AS frequency_bucket,
      COUNT(*)               AS customer_count,
      MIN(visit_count)       AS sort_key
  FROM visit_counts
  GROUP BY frequency_bucket
  ORDER BY MIN(visit_count)
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
NTH VISIT ANALYSIS — VISIT SEQUENCE PER CLIENT
════════════════════════════════════════════════════════════
Use when: user asks for "2nd visit", "3rd visit", "repeat visit timing",
  "how many clients came back", "visit sequence", or "visit retention".

ROW_NUMBER() pattern — number each client's invoices chronologically:
  WITH visit_ranked AS (
      SELECT
          client_id,
          invoice_id,
          MIN(performed_date)              AS visit_date,
          ROW_NUMBER() OVER (
              PARTITION BY client_id
              ORDER BY MIN(performed_date)
          )                                AS visit_num
      FROM allpets_new_invoices
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND client_id != '13'
      GROUP BY client_id, invoice_id
  )

Visit sequence count (how many clients reached each visit number):
  SELECT
      visit_num,
      COUNT(*) AS client_count
  FROM visit_ranked
  WHERE visit_num <= 10
  GROUP BY visit_num
  ORDER BY visit_num

Clients who came back for their 2nd visit in a specific month:
  SELECT COUNT(DISTINCT client_id) AS returned_clients
  FROM visit_ranked
  WHERE visit_num = 2
    AND YEAR(visit_date) = <year>
    AND MONTH(visit_date) = <month>

IMPORTANT:
  - GROUP BY client_id, invoice_id then take MIN(performed_date) so multi-line invoices
    count as one visit (one invoice = one visit).
  - visit_num = 1 is the first ever visit (same as new customer).
  - visit_num = 2 is the first return visit.
  - Exclude client_id = '13' (OTC phantom) from all visit-sequence queries.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
CATEGORY ANALYTICS — INVOICE CATEGORY BREAKDOWN
════════════════════════════════════════════════════════════
Source: plan_category_name in allpets_new_invoices (direct column, no join needed).

QUERY INTENT — CRITICAL: choose the right SQL before writing anything.

  DEFAULT (use for most queries): Use plan_category_name DIRECTLY.
    Triggered by: "sales by category", "category breakdown", "category report",
    "category wise", "category-wise", "category sales", "which categories",
    "top categories", or any category question WITHOUT explicit grouping language.
    → SELECT plan_category_name AS category, SUM(total) AS revenue …
    → GROUP BY plan_category_name
    → This matches VetBuddy's "Sales by Category Report" — one row per category name.
    → Do NOT apply the CASE grouping below.

  COMBINE VARIANTS (only when explicitly asked to consolidate/group name variants):
    Triggered by: "combine Prescription", "all Prescription together", "consolidate Pet Shop",
    "group Pet Shop variants", "combine all Prescription variants into one",
    "club prescriptions", "club petfood and prescriptions", "merge prescription variants",
    "group prescriptions together", "all prescriptions as one category".
    → Use: allpets_new_v_category_normalized.logical_category
    → Example: SELECT logical_category AS category, SUM(total) AS revenue
               FROM allpets_new_v_category_normalized
               WHERE cancelled = 0 AND performed_date IS NOT NULL …
               GROUP BY logical_category ORDER BY revenue DESC
    → The view maps all 6 Prescription name variants → 'Prescription (All)'
      and all 6 Pet Shop name variants → 'Pet Shop (All)'. Everything else passes through.
    → Do NOT use this view for a simple category breakdown — it changes category labels.

  GROUPED VIEW (only when user explicitly asks): Apply the 5-bucket CASE grouping.
    Triggered by: "group by type", "category type", "category group",
    "high-level categories", "category summary by type",
    "Diagnostics vs Pharmacy vs Services", "broad category breakdown".
    → Apply the CASE expression:
      Diagnostics / Pharmacy / Clinical Services / Food & Pet Shop / Other
    → NEVER use category_bucket from the view — it contains the OLD bucket names
      (Pharmacy/Food/Pet Shop/Services/Other) and misclassifies Diagnostics.

CORRECT SQL — sales by individual category for a specific month (DEFAULT):
  SELECT
      plan_category_name                        AS category,
      COALESCE(SUM(total), 0)                   AS revenue,
      COUNT(DISTINCT invoice_id)                AS invoice_count
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND YEAR(performed_date) = <year> AND MONTH(performed_date) = <month>
  GROUP BY plan_category_name
  ORDER BY revenue DESC

CORRECT SQL — sales by individual category for a date range (DEFAULT):
  Filter by performed_date >= '<start>' AND performed_date < '<exclusive_end>'.
  GROUP BY plan_category_name, ORDER BY revenue DESC.
  Do NOT use BETWEEN on DATETIME columns (see DATE RANGE FILTER rule).

For weekly category trends: GROUP BY YEARWEEK(performed_date, 1), plan_category_name.
  Fetch 7 weeks so each display week has a prior-week WoW value (N+1 rule).

  ⚠️ LIMIT BUG — CRITICAL for multi-category weekly queries:
  When GROUP BY has two dimensions (YEARWEEK × plan_category_name), the result set has
  N rows per week (one per category). With 30+ distinct categories per week, LIMIT 7
  returns 7 (week, category) pairs — ALL from the same most recent week, never reaching
  any prior week. This breaks both "top N" ranking AND WoW calculation.

  WRONG (breaks for multi-category):
    GROUP BY YEARWEEK(performed_date, 1), plan_category_name
    ORDER BY sort_key DESC
    LIMIT 7   ← returns 7 rows from 1 week, not 7 weeks

  CORRECT — use week_spine CTE to LIMIT distinct weeks first:
    See "CORRECT SQL — top N categories, weekly trend" below.

CORRECT SQL — top N categories, weekly trend (DEFAULT path, raw plan_category_name):
  Use when: "top 5 categories by revenue per week", "top categories weekly trend",
  "weekly breakdown top categories", "which categories growing week over week",
  "top N categories last X weeks", "category wise weekly trend".
  NEVER use the grouped CASE for this — raw plan_category_name matches VetBuddy labels.

  WITH week_spine AS (
      -- LIMIT distinct YEARWEEK values, not rows — safe for multi-category queries
      SELECT DISTINCT YEARWEEK(performed_date, 1) AS wk
      FROM allpets_new_invoices
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 11 WEEK)
      ORDER BY wk DESC
      LIMIT 7  -- 6 display weeks + 1 WoW baseline (N+1 rule)
  ),
  top_cats AS (
      -- Rank by total revenue across all 7 fetched weeks (not just the most recent week)
      SELECT plan_category_name
      FROM allpets_new_invoices
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND YEARWEEK(performed_date, 1) IN (SELECT wk FROM week_spine)
      GROUP BY plan_category_name
      ORDER BY SUM(total) DESC
      LIMIT 5  -- change to match user request (top 3, top 10, etc.)
  )
  SELECT
      CONCAT(
          DATE_FORMAT(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), '%d %b %Y'),
          ' - ',
          DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY), '%d %b %Y')
      )                               AS week_range,
      YEARWEEK(performed_date, 1)     AS sort_key,
      plan_category_name              AS category,
      COALESCE(SUM(total), 0)         AS revenue,
      COUNT(DISTINCT invoice_id)      AS invoice_count
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND YEARWEEK(performed_date, 1) IN (SELECT wk FROM week_spine)
    AND plan_category_name IN (SELECT plan_category_name FROM top_cats)
  GROUP BY YEARWEEK(performed_date, 1), plan_category_name
  ORDER BY sort_key ASC, revenue DESC

  Expected result: 5 categories × 7 weeks = 35 rows.
  Result_analyzer drops oldest week (WoW baseline) -> 30 display rows (5 x 6).
  WoW % for each category = (week_n - week_n-1) / week_n-1 * 100, computed by result_analyzer.

CORRECT SQL — top N categories with COMBINE VARIANTS, weekly trend:
  Use when: "top 5 categories club prescriptions", "combine prescription variants weekly",
  "club petfood and prescriptions", "group prescription variants together weekly trend",
  "top N categories merge variants week over week".
  Source: allpets_new_v_category_normalized.logical_category (not plan_category_name).
  logical_category collapses:
    All TRIM(plan_category_name) LIKE 'Prescription%%' -> 'Prescription (All)'
    All LOWER(TRIM()) LIKE 'pet shop%%' or Non%%Prescription Items%% -> 'Pet Shop (All)'
    Everything else -> TRIM(plan_category_name) unchanged
  This is different from the 5-bucket CASE — Services sub-categories (Lab, Consultation, Surgery)
  remain as individual rows; only Prescription and Pet Shop name variants are collapsed.

  Production-validated variant counts (7-week window Apr-May 2026):
    Prescription (All) = 6 raw variants: Prescription / Prescription 5%% / Prescription 12%% /
      Prescription 18%% / Prescription  12%% (double-space) / Prescription+C2524 (corrupt entry)
    Pet Shop (All) = 8 raw variants: Pet Shop  18%% / Pet shop 18%% / Pet shop / Pet Shop  12%% /
      Non-Prescription Items / Non Prescription Items / Non-Prescription Items 12%% / Non-Prescription Items 18%%
    NOTE: Pet Shop names have INCONSISTENT CASE and DOUBLE SPACES in DB — never match with = operator.

  WITH week_spine AS (
      SELECT DISTINCT YEARWEEK(performed_date, 1) AS wk
      FROM allpets_new_v_category_normalized
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 11 WEEK)
      ORDER BY wk DESC
      LIMIT 7
  ),
  top_cats AS (
      SELECT logical_category
      FROM allpets_new_v_category_normalized
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND YEARWEEK(performed_date, 1) IN (SELECT wk FROM week_spine)
      GROUP BY logical_category
      ORDER BY SUM(total) DESC
      LIMIT 5
  )
  SELECT
      CONCAT(
          DATE_FORMAT(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), '%d %b %Y'),
          ' - ',
          DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY), '%d %b %Y')
      )                               AS week_range,
      YEARWEEK(performed_date, 1)     AS sort_key,
      logical_category                AS category,
      COALESCE(SUM(total), 0)         AS revenue,
      COUNT(DISTINCT invoice_id)      AS invoice_count
  FROM allpets_new_v_category_normalized
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND YEARWEEK(performed_date, 1) IN (SELECT wk FROM week_spine)
    AND logical_category IN (SELECT logical_category FROM top_cats)
  GROUP BY YEARWEEK(performed_date, 1), logical_category
  ORDER BY sort_key ASC, revenue DESC

  Expected result: 5 logical categories x 7 weeks = 35 rows.
  Result_analyzer drops oldest week (baseline) -> 30 display rows.

For monthly category trends: GROUP BY YEAR(performed_date), MONTH(performed_date), plan_category_name.
  Use performed_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) for last 6 months.

════════════════════════════════════════════════════════════
GROUPED VIEW — 5-bucket CASE (only when explicitly requested by the user)
════════════════════════════════════════════════════════════
DB-VALIDATED buckets (verified June 2026 against live data — 35 source categories):

  Diagnostics      = 34.8% of revenue — Pathology, Laboratory, Imaging, Cardiology
  Pharmacy         = 26.0% of revenue — Prescription drugs, Preventive Medicine,
                     Parenteral Medication/Fluids, Anaesthesia, Sedation
  Clinical Services= 22.0% of revenue — Consultation, Surgery, Hospitalization & Care,
                     Procedure, Grooming, Swimming, Boarding, Registration, Behavior,
                     Euthanasia, Certificates, Exam, Telephone, Neoplasia/Tumor, Medvet, etc.
  Food & Pet Shop  =  7.7% of revenue — Diets/Food + Pet Shop (All variants)
  Other            = <2%  of revenue — Education, Instructions, At Home Protocols

CRITICAL: Diagnostics is the LARGEST revenue bucket at 34.8%.
  NEVER lump Pathology, Laboratory, or Imaging into 'Services' — they must be 'Diagnostics'.

CRITICAL: Do NOT use invoice_type = 'Service' for any bucket classification — it
  misclassifies rows. Many clinical categories have invoice_type='Pharmacy' in VetBuddy.
  Always use plan_category_name patterns, never invoice_type.

CATEGORY CASE FOR INVOICES (use alias i for allpets_new_invoices):
  CASE
      WHEN i.plan_category_name IN (
          'Pathology','Laboratory','Imaging','Cardiology')
      THEN 'Diagnostics'
      WHEN i.plan_category_name LIKE 'Prescription%%'
        OR i.plan_category_name LIKE 'Preventive Medicine%%'
        OR i.plan_category_name IN ('Parenteral Medication','Parenteral Fluids',
           'Parenteral Medication 12%%','Anesthesia Gas','Anesthesia Parenteral','Sedation')
      THEN 'Pharmacy'
      WHEN i.plan_category_name IN (
          'Consultation','Surgery','Hospitalization & Care','Procedure',
          'Grooming','Swimming','Boarding','Registration','Behavior',
          'Euthanasia','Certificates','Exam','Telephone','Neoplasia / Tumor',
          'Medvet','New Category','Body Care','Office Supplies')
      THEN 'Clinical Services'
      WHEN i.plan_category_name = 'Diets/Food'
        OR i.plan_category_name LIKE 'Pet Shop%%'
        OR i.plan_category_name LIKE 'Pet shop%%'
        OR i.plan_category_name LIKE 'Non%%Prescription Items%%'
        OR i.plan_category_name = 'Non Prescription Items'
      THEN 'Food & Pet Shop'
      ELSE 'Other'
  END AS category_group

  NOTE: 'Diets/Food' is an EXACT value — never use 'Food' or 'Diets' alone.
  NOTE: Pet Shop names have inconsistent case in DB — use LIKE not =.
  NOTE: Diagnostics must always be checked FIRST in the CASE chain.
  NOTE: invoice_type is NEVER used for category bucketing — plan_category_name only.

CORRECT SQL — weekly 6-week category trend, grouped 5-bucket view (N+1 rule):
  ⚠️ LIMIT cannot be applied to GROUP BY YEARWEEK × category_group directly — with 5 groups
  per week, LIMIT 7 returns ~1 week only. Use week_spine CTE instead.

  WITH week_spine AS (
      SELECT DISTINCT YEARWEEK(performed_date, 1) AS wk
      FROM allpets_new_invoices
      WHERE cancelled = 0
        AND performed_date IS NOT NULL
        AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 11 WEEK)
      ORDER BY wk DESC
      LIMIT 7  -- 6 display weeks + 1 WoW baseline
  )
  SELECT
      CONCAT(
          DATE_FORMAT(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), '%d %b %Y'),
          ' - ',
          DATE_FORMAT(DATE_ADD(DATE_SUB(DATE(performed_date), INTERVAL WEEKDAY(performed_date) DAY), INTERVAL 6 DAY), '%d %b %Y')
      )                                    AS week_range,
      YEARWEEK(performed_date, 1)          AS sort_key,
      CASE
          WHEN plan_category_name IN (
              'Pathology','Laboratory','Imaging','Cardiology')
          THEN 'Diagnostics'
          WHEN plan_category_name LIKE 'Prescription%%'
            OR plan_category_name IN ('Parenteral Medication','Parenteral Fluids',
               'Preventive Medicine','Preventive Medicine 12%%',
               'Anesthesia Gas','Anesthesia Parenteral','Sedation')
          THEN 'Pharmacy'
          WHEN plan_category_name IN (
              'Consultation','Surgery','Hospitalization & Care','Procedure',
              'Grooming','Swimming','Boarding','Registration','Behavior',
              'Euthanasia','Certificates','Exam','Telephone','Neoplasia / Tumor',
              'Medvet','New Category','Body Care','Office Supplies',
              'Canine Education','General Education','Exotics Education','Birds Education',
              'At Home Protocols','Instructions')
          THEN 'Clinical Services'
          WHEN plan_category_name = 'Diets/Food'
            OR plan_category_name LIKE 'Pet Shop%%'
            OR plan_category_name LIKE 'Pet shop%%'
            OR plan_category_name LIKE 'Non%%Prescription Items%%'
            OR plan_category_name = 'Non Prescription Items'
          THEN 'Food & Pet Shop'
          ELSE 'Other'
      END                                  AS category_group,
      COALESCE(SUM(total), 0)              AS revenue
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND YEARWEEK(performed_date, 1) IN (SELECT wk FROM week_spine)
  GROUP BY YEARWEEK(performed_date, 1), category_group
  ORDER BY sort_key ASC, revenue DESC
  -- Oldest week (week 1) is baseline for WoW; result_analyzer drops it from display.
  -- Expected: 5 groups × 7 weeks = up to 35 rows.

CORRECT SQL — monthly category revenue (last 6 months):
  SELECT
      YEAR(performed_date) * 100 + MONTH(performed_date)                     AS sort_key,
      CONCAT(YEAR(performed_date), '-', LPAD(MONTH(performed_date), 2, '0')) AS month_label,
      CASE
          WHEN plan_category_name IN (
              'Pathology','Laboratory','Imaging','Cardiology')
          THEN 'Diagnostics'
          WHEN plan_category_name LIKE 'Prescription%%'
            OR plan_category_name IN ('Parenteral Medication','Parenteral Fluids',
               'Preventive Medicine','Preventive Medicine 12%%',
               'Anesthesia Gas','Anesthesia Parenteral','Sedation')
          THEN 'Pharmacy'
          WHEN plan_category_name IN (
              'Consultation','Surgery','Hospitalization & Care','Procedure',
              'Grooming','Swimming','Boarding','Registration','Behavior',
              'Euthanasia','Certificates','Exam','Telephone','Neoplasia / Tumor',
              'Medvet','New Category','Body Care','Office Supplies',
              'Canine Education','General Education','Exotics Education','Birds Education',
              'At Home Protocols','Instructions')
          THEN 'Clinical Services'
          WHEN plan_category_name = 'Diets/Food'
            OR plan_category_name LIKE 'Pet Shop%%'
            OR plan_category_name LIKE 'Pet shop%%'
            OR plan_category_name LIKE 'Non%%Prescription Items%%'
            OR plan_category_name = 'Non Prescription Items'
          THEN 'Food & Pet Shop'
          ELSE 'Other'
      END                                                                      AS category_group,
      COALESCE(SUM(total), 0)                                                 AS revenue
  FROM allpets_new_invoices
  WHERE cancelled = 0
    AND performed_date IS NOT NULL
    AND performed_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
  GROUP BY YEAR(performed_date), MONTH(performed_date), category_group
  ORDER BY sort_key ASC, revenue DESC
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
INVENTORY ANALYTICS — CRITICAL CATEGORY RULES
════════════════════════════════════════════════════════════
NEVER USE bin_name AS A CATEGORY.
  bin_name has only 3 values: 'Pharmacy', 'Inventory Bin', 'Petshop'.
  These are PHYSICAL STORAGE LOCATIONS, NOT product categories.
  Using bin_name as a category returns wrong results — it covers only ~5,092 of 9,282 SKUs.

CORRECT CATEGORY SOURCE: allpets_new_stocks.plan_category_name
  plan_category_name is already in allpets_new_stocks — NO JOIN NEEDED.
  Do NOT join allpets_stock_plan_category — that table does not exist.

CATEGORY GROUPINGS — production-validated CASE expressions:
  WHEN s.plan_category_name LIKE 'Prescription%%'
    OR s.plan_category_name IN ('Parenteral Medication', 'Parenteral Fluids',
       'Preventive Medicine', 'Preventive Medicine 12%%',
       'Anesthesia Gas', 'Anesthesia Parenteral', 'Sedation')
    THEN 'Pharmacy'

  WHEN s.plan_category_name = 'Diets/Food'
    THEN 'Food'
    -- IMPORTANT: exact value is 'Diets/Food'. NEVER use 'Food' or 'Diets' alone.

  WHEN s.plan_category_name LIKE 'Pet Shop%%'
    OR s.plan_category_name LIKE 'Pet shop%%'
    OR s.plan_category_name LIKE 'Non%%Prescription Items%%'
    THEN 'Pet Shop'
    -- NOTE: Pet Shop category names have inconsistent case in DB — use LIKE.

  ELSE 'Other'

STOCK COVERAGE CALCULATION — mandatory pattern:
  stock_value  = SUM(GREATEST(COALESCE(s.onhand_qty, 0), 0) * COALESCE(s.purchase_cost, 0))
  daily_rate   = SUM(i.total) / 31  (use a complete calendar month as baseline)
  coverage_days = stock_value / daily_rate

  MySQL does NOT support NULLS LAST. For zero-sales categories:
    ORDER BY
      CASE WHEN daily_rate IS NULL OR daily_rate = 0 THEN 1 ELSE 0 END,
      coverage_days DESC

CORRECT SQL — current closing stock by grouped category:
  SELECT
      CASE
          WHEN s.plan_category_name LIKE 'Prescription%%'
            OR s.plan_category_name IN ('Parenteral Medication','Parenteral Fluids',
               'Preventive Medicine','Preventive Medicine 12%%',
               'Anesthesia Gas','Anesthesia Parenteral','Sedation')
          THEN 'Pharmacy'
          WHEN s.plan_category_name = 'Diets/Food'
          THEN 'Food'
          WHEN s.plan_category_name LIKE 'Pet Shop%%'
            OR s.plan_category_name LIKE 'Pet shop%%'
            OR s.plan_category_name LIKE 'Non%%Prescription Items%%'
          THEN 'Pet Shop'
          ELSE 'Other'
      END                                                                       AS category_group,
      COUNT(DISTINCT s.stock_id)                                                AS distinct_skus,
      ROUND(SUM(GREATEST(COALESCE(s.onhand_qty, 0), 0)), 2)                     AS onhand_qty,
      ROUND(SUM(GREATEST(COALESCE(s.onhand_qty, 0), 0)
                * COALESCE(s.purchase_cost, 0)), 2)                             AS stock_value
  FROM allpets_new_stocks s
  GROUP BY category_group
  ORDER BY stock_value DESC

CORRECT SQL — 21-day stock coverage check (join stocks + invoices on plan_category_name):
  WITH daily_rate AS (
      SELECT
          plan_category_name,
          SUM(total) / 31 AS daily_rate
      FROM allpets_new_invoices
      WHERE performed_date IS NOT NULL
        AND YEAR(performed_date) = <year>
        AND MONTH(performed_date) = <month>
        AND cancelled = 0
      GROUP BY plan_category_name
  ),
  current_stock AS (
      SELECT
          plan_category_name,
          SUM(GREATEST(COALESCE(onhand_qty, 0), 0)
              * COALESCE(purchase_cost, 0))                                     AS stock_value
      FROM allpets_new_stocks
      GROUP BY plan_category_name
  )
  SELECT
      cs.plan_category_name,
      ROUND(cs.stock_value, 2)                          AS stock_value,
      ROUND(dr.daily_rate, 2)                           AS daily_revenue_rate,
      ROUND(cs.stock_value / dr.daily_rate, 1)          AS coverage_days
  FROM current_stock cs
  JOIN daily_rate dr ON dr.plan_category_name = cs.plan_category_name
  WHERE dr.daily_rate > 0
    AND cs.stock_value / dr.daily_rate > 21
  ORDER BY cs.stock_value / dr.daily_rate DESC

CORRECT SQL — per-SKU days cover (unit-based, uses quantity not revenue):
  WITH sku_sold AS (
      SELECT
          plan_item_id,
          SUM(quantity) / 30.0    AS daily_units_sold
      FROM allpets_new_invoices
      WHERE performed_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        AND cancelled = 0
        AND performed_date IS NOT NULL
        AND plan_item_id IS NOT NULL
      GROUP BY plan_item_id
  )
  SELECT
      s.stock_id                                                            AS sku_id,
      s.plan_item_name                                                      AS sku_name,
      s.plan_category_name                                                  AS category,
      GREATEST(COALESCE(s.onhand_qty, 0), 0)                               AS onhand_qty,
      ROUND(
          GREATEST(COALESCE(s.onhand_qty, 0), 0)
          / NULLIF(ss.daily_units_sold, 0),
          1
      )                                                                     AS days_cover
  FROM allpets_new_stocks s
  LEFT JOIN sku_sold ss ON ss.plan_item_id = s.plan_item_id
  WHERE GREATEST(COALESCE(s.onhand_qty, 0), 0) > 0
  ORDER BY days_cover DESC
  LIMIT 25

  Threshold: days_cover > 21 = potential excess stock. days_cover IS NULL = no sales in last 30 days.
  Use SUM(quantity) not SUM(total) for days cover — covers stock in units, not revenue.

TOP SKUs — use last 3 months as the default sales window unless the user specifies otherwise.
  Filter: performed_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH), cancelled = 0, plan_item_id IS NOT NULL.
  GROUP BY plan_item_id, plan_item_name. ORDER BY revenue DESC. LIMIT 25.
  For Pareto (SKUs covering 80%% of sales), return all SKUs with cumulative revenue — the result
  analyzer will apply the 80%% cutoff in Python.

SKU SALES TREND — when asked how a SKU or set of SKUs is trending:
  GROUP BY YEAR(performed_date), MONTH(performed_date), plan_item_id, plan_item_name.
  Use performed_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) for last 6 months.
  Include sort_key = YEAR*100+MONTH for chronological ordering.

VENDOR / BRAND ANALYTICS — BLOCKED:
  brand_name and brand_id are NOT loaded in the current allpets_new_invoices table.
  The VetBuddy Invoice API returns Brand data under PlanItem → Brand → {BrandID, BrandName}
  but this was never extracted into the database.
  If the user asks "who is the vendor for top SKUs" or "which brand does this SKU belong to":
    Inform them: "Vendor/brand analytics require a brand_name column which has not yet been
    loaded into the database. This is a planned ETL enhancement."
  Do NOT attempt to query brand_name — it does not exist in the schema.

CLOSING STOCK TREND — BLOCKED:
  allpets_new_stocks contains only the CURRENT snapshot of stock (latest load date).
  There is no snapshot_date column and no historical stock records.
  Week-over-week or month-over-month closing stock trend CANNOT be computed.
  If the user asks "how has closing stock changed over time" or "stock trend for last 4 weeks":
    Inform them: "Closing stock trend requires historical snapshots (snapshot_date column)
    which has not yet been added to the stocks table. Current query can show the live
    closing stock level but not the trend over time."
  Current query can still answer: closing stock by category TODAY and 21-day coverage check.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
QUERY QUALITY CHECK — VERIFY BEFORE RETURNING
════════════════════════════════════════════════════════════
Before finalising each SQL query, confirm all eight checks pass:

  1. Uses only approved tables (allpets_new_* — no banned table names)
  2. Uses correct revenue measure (SUM(total), never SUM(invoice_amount))
  3. Uses correct cancellation filter (cancelled = 0, not VARCHAR pattern)
  4. Uses correct aggregation grain (matches the reporting grain from Step 4)
  5. Uses correct KPI column (matches the measure identified in Step 5)
  6. Avoids unnecessary joins (flat schema — patient/category/client are direct columns)
  7. Produces business-ready output (column aliases are meaningful, sort_key included for time series)
  8. Date filter matches the scope the user requested:
       • Specific month named → WHERE clause has BOTH YEAR() AND MONTH() filters
       • YEAR-only filter is only valid when user asked for a full year or YTD

If any check fails, rewrite the query before returning it.
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
SUCCESS CRITERIA
════════════════════════════════════════════════════════════
The goal is not generating SQL.
The goal is generating correct business answers that remain valid as new
data is loaded into the warehouse.

Favor correctness, maintainability, and business meaning over query complexity.
════════════════════════════════════════════════════════════
"""


def _make_sql_generator(llm: ChatOpenAI, debug: bool):
    def sql_generator(state: SQLAgentState) -> SQLAgentState:
        retry  = state.get("retry_count", 0)
        failed = state.get("failed_queries") or []
        label  = f" (retry #{retry})" if retry else ""
        print(f"\n🔧  SQL Generator{label}…")

        if retry and failed:
            # ── Retry path: failure context goes FIRST so the LLM reads it
            # before processing the schema (which is ~6k tokens of distraction).
            # Each failed step is shown with its exact SQL and extracted DB error.
            header_lines = [
                f"⚠️ RETRY #{retry} — {len(failed)} step(s) failed validation or database execution.",
                "Read each error below, diagnose the cause, and write DIFFERENT SQL.\n",
                "FAILED STEPS:",
            ]
            for f in failed:
                header_lines.append(
                    f"\n  Step {f['step_id']} — {f['description']}"
                    f"\n  Previous SQL (DO NOT REPEAT):\n{f['sql']}"
                    f"\n  Database error:\n{f['error']}"
                    "\n  → Diagnose and rewrite with a completely different approach."
                )
            header_lines.append(
                "\n────────────────────────────────────────────────────────────\n"
            )

            if debug:
                print(f"   [DEBUG] Retry context injected at top of user prompt:")
                for f in failed:
                    print(f"      Step {f['step_id']} prev SQL  : {f['sql'][:200].replace(chr(10), ' ')}")
                    print(f"      Step {f['step_id']} db error  : {f['error'][:200]}")

            user = (
                "\n".join(header_lines)
                + f"Today's date: {date.today().isoformat()} "
                f"(current month: {date.today().strftime('%B %Y')}, "
                f"current year: {date.today().year})\n\n"
                f"User question: {state['user_query']}\n\n"
                f"Execution plan:\n{json.dumps(state['query_plan'], indent=2)}"
            )
        else:
            # ── First attempt: standard order (schema last so plan is in view
            # when the model starts generating).
            user = (
                f"Today's date: {date.today().isoformat()} "
                f"(current month: {date.today().strftime('%B %Y')}, "
                f"current year: {date.today().year})\n\n"
                f"User question: {state['user_query']}\n\n"
                f"Execution plan:\n{json.dumps(state['query_plan'], indent=2)}"
            )

        try:
            resp = _llm_invoke_with_retry(llm, [SystemMessage(content=_SQL_GENERATOR_SYSTEM), HumanMessage(content=user)])
            raw_content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
            if debug:
                print(f"   [DEBUG] LLM response (600 chars): {raw_content[:600].replace(chr(10),' ')}")
            queries = _parse_json(raw_content, is_array=True, debug=debug)
            print(f"   ✅  {len(queries)} quer{'y' if len(queries)==1 else 'ies'} generated")
            for q in queries:
                print(f"   📝  Step {q['step_id']}: {q['description']}")
                if debug:
                    print(f"   [DEBUG] SQL: {q['sql']}")
                    if retry:
                        prev = next((f for f in failed if f["step_id"] == q["step_id"]), None)
                        if prev:
                            changed = prev["sql"].strip() != q["sql"].strip()
                            status  = "CHANGED ✓" if changed else "UNCHANGED ✗ (same as failed!)"
                            print(f"   [DEBUG] Step {q['step_id']} SQL {status}")
            return {**state, "generated_queries": queries, "error": None}
        except Exception as exc:
            return {**state, "error": f"SQL generator failed: {exc}"}
    return sql_generator


def _make_sql_executor(engine: Engine, debug: bool):
    def sql_executor(state: SQLAgentState) -> SQLAgentState:
        print("\n⚡  SQL Executor…")
        results: List[Dict[str, Any]] = []
        failed:  List[Dict[str, Any]] = []
        result_aliases: List[str] = [
            q.get("result_alias", "")
            for q in (state.get("generated_queries") or [])
            if isinstance(q.get("result_alias"), str) and q.get("result_alias")
        ]

        for q in (state.get("generated_queries") or []):
            sid = q["step_id"]
            print(f"   🔄  Step {sid}: {q['description']}")
            err = _validate_no_result_alias_references(q["sql"], result_aliases)
            if err:
                df = pd.DataFrame()
            else:
                df, err = _execute_sql(q["sql"], engine, debug=debug)

            if err:
                print(f"   ❌  Step {sid} failed: {err}")
                failed.append({"step_id": sid, "description": q["description"],
                               "sql": q["sql"], "error": err})
                results.append({**q, "success": False, "error": err,
                                "data": [], "columns": [], "row_count": 0})
            else:
                print(f"   ✅  Step {sid}: {len(df)} row(s)")
                raw_data = df.to_dict(orient="records")
                for row in raw_data:
                    null_keys = [k for k, v in row.items() if v is None]
                    if null_keys:
                        print(f"   ⚠️  Step {sid}: NULL in {null_keys} — coercing to 0")
                        for k in null_keys:
                            row[k] = 0
                results.append({**q, "success": True, "error": None,
                                "data": raw_data,
                                "columns": list(df.columns),
                                "row_count": len(df)})

        return {**state, "query_results": results,
                "failed_queries": failed or None, "error": None}
    return sql_executor


def _make_increment_retry():
    def increment_retry(state: SQLAgentState) -> SQLAgentState:
        new_count = state.get("retry_count", 0) + 1
        failed = state.get("failed_queries") or []
        print(f"\n   🔁  Retry #{new_count}/{state['max_retries']} for step(s): {[f['step_id'] for f in failed]}")
        for f in failed:
            print(f"      Step {f['step_id']} error: {str(f['error'])[:120]}")
        return {**state, "retry_count": new_count, "generated_queries": None}
    return increment_retry


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def _route_after_executor(state: SQLAgentState) -> str:
    results     = state.get("query_results", [])
    failed      = state.get("failed_queries") or []
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    any_success = any(r["success"] for r in (results or []))

    # Retry whenever any step failed — not only when all failed.
    # temperature=0 makes re-running successful steps safe; the error context
    # in the prompt lets the LLM fix the failing step(s).
    if failed and retry_count < max_retries:
        decision = "retry"
    elif any_success:
        decision = "done"
    else:
        decision = "error"

    if state.get("debug"):
        print(f"   [DEBUG] route_after_executor → {decision} "
              f"(any_success={any_success}, failed={len(failed)}, retries={retry_count}/{max_retries})")
    return decision


def _route_after_generator_inner(state: SQLAgentState) -> str:
    return "error" if state.get("error") else "sql_executor"


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SQLDynamicAgent:
    """
    Reusable SQL generation + execution agent with automatic retry on failure.

    Usage:
        agent = SQLDynamicAgent(llm=my_llm, engine=my_engine, max_retries=3, debug=True)
        results = agent.run(schema=schema_str, plan=query_plan_dict, user_query="…")
        # results: List[Dict] — each dict has keys: step_id, sql, success, data, columns, row_count, error
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        engine: Engine,
        max_retries: int = 3,
        debug: bool = False,
    ):
        self.llm         = llm
        self.engine      = engine
        self.max_retries = max_retries
        self.debug       = debug
        self._graph      = self._build_graph()

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(SQLAgentState)

        g.add_node("sql_generator",   _make_sql_generator(self.llm, self.debug))
        g.add_node("sql_executor",    _make_sql_executor(self.engine, self.debug))
        g.add_node("increment_retry", _make_increment_retry())

        # terminal sinks — just pass state through
        g.add_node("done",  lambda state: state)
        g.add_node("error", lambda state: state)


        g.set_entry_point("sql_generator")

        g.add_conditional_edges("sql_generator", _route_after_generator_inner, {
            "sql_executor": "sql_executor",
            "error":        "error",
        })
        g.add_conditional_edges("sql_executor", _route_after_executor, {
            "done":  "done",
            "retry": "increment_retry",
            "error": "error",
        })
        g.add_edge("increment_retry", "sql_generator")
        g.add_edge("done",  END)
        g.add_edge("error", END)

        return g.compile()

    # ── Public interface ─────────────────────────────────────────────────────

    def run(
        self,
        schema:     str,
        plan:       Dict[str, Any],
        user_query: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL for every step in `plan`, execute each query, and return
        the raw result list.  Retries up to max_retries on failure.

        Returns:
            List of result dicts.  Each dict contains:
                step_id, description, sql, result_alias,
                success (bool), data (list[dict]), columns (list[str]),
                row_count (int), error (str | None)
        """
        initial: SQLAgentState = {
            "user_query":        user_query,
            "schema":            schema,
            "query_plan":        plan,
            "generated_queries": None,
            "query_results":     None,
            "failed_queries":    None,
            "retry_count":       0,
            "max_retries":       self.max_retries,
            "error":             None,
            "debug":             self.debug,
        }
        final = self._graph.invoke(initial)

        if final.get("error") and not any(
            r.get("success") for r in (final.get("query_results") or [])
        ):
            raise RuntimeError(f"SQLDynamicAgent failed: {final['error']}")

        return final.get("query_results") or []


# ═══════════════════════════════════════════════════════════════════════════════
#  STAND-ALONE SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from urllib.parse import quote_plus
    from dotenv import load_dotenv
    load_dotenv()

    _openai_key = os.getenv("OPENAI_API_KEY")
    _db_host    = os.getenv("DB_HOST")
    _db_port    = os.getenv("DB_PORT", "3306")
    _db_user    = os.getenv("DB_USER")
    _db_pass    = os.getenv("DB_PASSWORD")
    _db_name    = os.getenv("DB_NAME")

    missing = [k for k, v in {
        "OPENAI_API_KEY": _openai_key,
        "DB_HOST": _db_host, "DB_USER": _db_user,
        "DB_PASSWORD": _db_pass, "DB_NAME": _db_name,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing required .env variables: {missing}")

    DB_URL = (
        f"mysql+pymysql://{_db_user}:{quote_plus(_db_pass or '')}"
        f"@{_db_host}:{_db_port}/{_db_name}"
    )
    OPENAI_KEY  = _openai_key
    SCHEMA_FILE = os.getenv("SCHEMA_FILE", "etc/secrets/schema_all.txt")

    with open(SCHEMA_FILE) as f:
        schema = f.read()

    from pydantic import SecretStr as _SecretStr
    llm    = ChatOpenAI(model="gpt-4o", api_key=_SecretStr(OPENAI_KEY or ""), temperature=0, model_kwargs={"max_tokens": 4096})
    engine = create_engine(DB_URL)

    test_plan = {
        "complexity":  "simple",
        "reasoning":   "single aggregate query",
        "output_type": "aggregate",
        "steps": [
            {"step_id": 1, "description": "total invoices count", "depends_on": []}
        ],
    }

    agent = SQLDynamicAgent(llm=llm, engine=engine, max_retries=3, debug=True)
    results = agent.run(schema=schema, plan=test_plan, user_query="How many invoices are there?")
    print("\nResults:", json.dumps(results, indent=2, default=str))
