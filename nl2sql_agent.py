#!/usr/bin/env python3
import sys, io
if __name__ == '__main__' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
nl2sql_agent.py — Natural Language → SQL Agent
================================================
Uses LangGraph to orchestrate a multi-step pipeline:
  1. Query Planner    — decides simple vs complex, creates step-by-step plan
  2. SQL Dynamic Agent— generates + executes SQL with automatic retry
                        (imported from sql_dynamic_agent.py)
  3. Result Analyzer  — combines/transforms results, computes comparisons
  4. Output Formatter — renders a clean table to the terminal
  5. Insights Generator— LLM-generated business narrative

Install dependencies:
    pip install langchain-openai langgraph sqlalchemy pandas tabulate pymysql python-dotenv

Run:
    python nl2sql_agent.py
    python nl2sql_agent.py "Compare current month sales with last 3 months"
"""

import os
import sys
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv(override=True)

# ── LangGraph / LangChain ────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Database ──────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text

# ── Data & Display ────────────────────────────────────────────────────────────
import pandas as pd
from tabulate import tabulate

# ── SQL Sub-Agent ─────────────────────────────────────────────────────────────
from sql_dynamic_agent import SQLDynamicAgent
from openai import RateLimitError as _OpenAIRateLimitError


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

_db_host     = os.getenv("DB_HOST")
_db_port     = os.getenv("DB_PORT", "3306")
_db_user     = os.getenv("DB_USER")
_db_password = os.getenv("DB_PASSWORD")
_db_name     = os.getenv("DB_NAME")

if not all([_db_host, _db_user, _db_password, _db_name]):
    raise ValueError("One or more DB_* variables missing from .env (need DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)")

DB_CONNECTION_STRING = (
    f"mysql+pymysql://{_db_user or ''}:{quote_plus(_db_password or '')}"
    f"@{_db_host or ''}:{_db_port}/{_db_name or ''}"
)

SCHEMA_FILE    = os.getenv("SCHEMA_FILE", "etc/secrets/cohort_main-schema_latest.sql")
MAX_RETRIES    = 3
DEBUG_MODE     = os.getenv("DEBUG_MODE", "false").lower() == "true"
LLM_MODEL      = "gpt-4o"
QUERY_LOG_FILE = os.getenv(
    "QUERY_LOG_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_run_log.jsonl")
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED RESOURCES  (created once, reused across graph nodes)
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import SecretStr

_llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=SecretStr(OPENAI_API_KEY),
    temperature=0,
    max_completion_tokens=4096,
)

_engine = create_engine(DB_CONNECTION_STRING)


def _ensure_log_table() -> None:
    with _engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS allpets_query_log (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                ts         DATETIME     NOT NULL,
                request_id VARCHAR(30),
                query      TEXT,
                plan       JSON,
                sql_steps  JSON,
                analysis   JSON,
                insights   TEXT,
                error      TEXT
            )
        """))
        # Add request_id column to existing tables that predate this column
        try:
            conn.execute(text(
                "ALTER TABLE allpets_query_log ADD COLUMN request_id VARCHAR(30)"
            ))
        except Exception:
            pass  # column already exists
        conn.commit()

try:
    _ensure_log_table()
except Exception as _e:
    print(f"   ⚠️  Could not create allpets_query_log table: {_e}")


_sql_agent = SQLDynamicAgent(
    llm=_llm,
    engine=_engine,
    max_retries=MAX_RETRIES,
    debug=DEBUG_MODE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_schema(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: '{path}' — check SCHEMA_FILE in .env")


_LLM_RETRY_DELAYS = [0, 5, 10, 20]

_ERROR_MAP = [
    ("rate limit",        "Analytics service is currently busy. Please try again in a few seconds."),
    ("ratelimit",         "Analytics service is currently busy. Please try again in a few seconds."),
    ("429",               "Analytics service is currently busy. Please try again in a few seconds."),
    ("timed out",         "This request is taking longer than expected. Please try again."),
    ("timeout",           "This request is taking longer than expected. Please try again."),
    ("pymysql",           "Unable to retrieve analytics data at the moment. Please try again shortly."),
    ("operationalerror",  "Unable to retrieve analytics data at the moment. Please try again shortly."),
    ("sqlalchemy",        "Unable to retrieve analytics data at the moment. Please try again shortly."),
    ("unknown column",    "The requested analysis could not be completed. Please contact support if the problem persists."),
    ("syntax error",      "The requested analysis could not be completed. Please contact support if the problem persists."),
    ("doesn't exist",     "The requested analysis could not be completed. Please contact support if the problem persists."),
    ("openai",            "Analytics service is currently busy. Please try again shortly."),
    ("gpt-",              "An unexpected error occurred. Please try again."),
    ("traceback",         "An unexpected error occurred. Please try again."),
    ("file \"",           "An unexpected error occurred. Please try again."),
    ("sk-",               "An unexpected error occurred. Please try again."),
    ("org-",              "An unexpected error occurred. Please try again."),
]


def _sanitize_error(raw: Optional[str], request_id: str) -> Optional[str]:
    if raw is None:
        return None
    lower = raw.lower()
    for pattern, msg in _ERROR_MAP:
        if pattern in lower:
            return f"{msg} (Reference ID: {request_id})"
    return f"We were unable to process this request. Please try again. (Reference ID: {request_id})"


def call_llm(system: str, user: str) -> str:
    last_exc: Exception = RuntimeError("LLM call failed after retries")
    for delay in _LLM_RETRY_DELAYS:
        if delay:
            print(f"   ⚠️  Rate limit — retrying in {delay}s…")
            time.sleep(delay)
        try:
            resp = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            if DEBUG_MODE:
                print(f"   [DEBUG] LLM response (600 chars): {content[:600].replace(chr(10), ' ')}")
            return content
        except _OpenAIRateLimitError as exc:
            last_exc = exc
    raise last_exc


def parse_json(raw: str, is_array: bool = False) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = (raw.find("["), raw.rfind("]") + 1) if is_array else (raw.find("{"), raw.rfind("}") + 1)
        if start != -1 and end > 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON: {exc}\n{raw[:400]}")
        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_query:        str
    schema:            str
    query_plan:        Optional[Dict[str, Any]]
    query_results:     Optional[List[Dict[str, Any]]]   # from SQLDynamicAgent
    analysis:          Optional[Dict[str, Any]]
    final_output:      Optional[str]
    insights:          Optional[str]
    error:             Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — QUERY PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def query_planner(state: AgentState) -> AgentState:
    print("\n📋  [1/5] Planning execution strategy…")

    system = """You are an expert SQL architect.
Given a natural-language question and a database schema, create a JSON execution plan.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "complexity": "simple" | "complex",
  "reasoning": "one-line explanation of strategy",
  "steps": [
    {
      "step_id": 1,
      "description": "what data this step retrieves",
      "depends_on": [],
      "metric_type": "gross_revenue",
      "date_column": "performed_date"
    }
  ],
  "output_type": "table" | "comparison" | "aggregate" | "summary"
}

metric_type values and when to use them:
  gross_revenue  — "total sales", "all sales", "gross revenue", "total revenue" (NO cancelled filter)
  active_revenue — "active sales", "net sales", "excluding cancelled" (cancelled = 0 required)
  invoice_count  — "number of invoices", "how many invoices" (cancelled = 0 required)
  patient_count  — "number of patients", "unique patients"
  customer_count — "number of customers", "unique clients"
  stock_value    — "stock value", "inventory value"
  other          — any metric not listed above

date_column values:
  performed_date — DEFAULT: use unless user explicitly asks for invoice_date/billing date
  invoice_date   — use when user says "invoice date", "billing date", "by invoice", or
                   the question semantically refers to when the invoice was created

Decision guide:
- simple  → one SQL query answers the question directly
- complex → needs several independent queries whose results must be merged in Python
Each step must be independently executable against the database.
"""

    user = (
        f"Today's date: {date.today().isoformat()} "
        f"(current month: {date.today().strftime('%B %Y')}, "
        f"current year: {date.today().year})\n\n"
        f"User question: {state['user_query']}\n\n"
        f"Database schema:\n{state['schema']}"
    )

    try:
        raw  = call_llm(system, user)
        plan = parse_json(raw)
        print(f"   ✅  {plan['complexity'].upper()} query | {len(plan['steps'])} step(s)")
        print(f"   📌  {plan['reasoning']}")
        if DEBUG_MODE:
            print(f"   [DEBUG] Plan:\n{json.dumps(plan, indent=4)}")
        return {**state, "query_plan": plan, "error": None}
    except Exception as exc:
        return {**state, "error": f"Planner failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — SQL DYNAMIC AGENT  (delegates to sql_dynamic_agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sql_dynamic_agent(state: AgentState) -> AgentState:
    """
    Hands the plan off to SQLDynamicAgent, which runs its own internal
    generate → execute → retry loop and returns the final result list.
    """
    print("\n🤖  [2/5] Running SQL Dynamic Agent…")
    plan = state.get("query_plan")
    if plan is None:
        return {**state, "error": "Query plan is missing — cannot run SQL agent"}
    try:
        results = _sql_agent.run(
            schema=state["schema"],
            plan=plan,
            user_query=state["user_query"],
        )
        successful = sum(1 for r in results if r["success"])
        print(f"   ✅  {successful}/{len(results)} step(s) succeeded")
        return {**state, "query_results": results, "error": None}
    except RuntimeError as exc:
        return {**state, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — RESULT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

def result_analyzer(state: AgentState) -> AgentState:
    print("\n🧠  [3/5] Analyzing & combining results…")

    if not state.get("query_results"):
        return {**state, "error": "No query results to analyze"}

    good = [r for r in (state["query_results"] or []) if r["success"]]

    if not good:
        return {**state, "error": "All SQL steps failed — no successful results to analyze"}

    llm_payload = [
        {
            "alias":       r.get("result_alias", f"step_{r.get('step_id', '?')}"),
            "description": r["description"],
            "columns":     r["columns"],
            "row_count":   r["row_count"],
            "data":        r["data"][:200],
        }
        for r in good
    ]

    system = """You are a senior data analyst producing business-ready reports.

Given the raw query results, produce a final answer to the user's question.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "summary":        "2–4 sentence narrative directly answering the question",
  "key_insights":   ["specific data-driven observation 1", "…"],
  "output_format":  "table" | "comparison_table" | "summary_only",
  "output_title":   "Short title for the results table",
  "output_columns": ["col1", "col2", "…"],
  "output_data":    [{"col1": "val", "col2": "val"}, …]
}

════════════════════════════════════════════════════════════
PRESENTATION RULES — MANDATORY
════════════════════════════════════════════════════════════

WEEK LABELS — always read from SQL data, NEVER compute from today's date:
  Week labels must come from the week_range column in the SQL result rows — not from any calendar calculation.
  NEVER calculate what "last 4 weeks" or "last N weeks" means from today's date.
  NEVER generate, invent, or override week_range values — use only what the SQL result rows contain.
  The data covers whatever dates the DB actually has — read them from the rows, do not assume.
  If data contains sort_key / week_key values like 202621, 202622:
    → Do NOT include them in output_columns.
    → Use the week_range column exactly as returned by the SQL.
    → NEVER substitute your own date calculation for the week_range value from the data.
  Use sort_key only for internal ordering — remove it from output_columns.

COLUMN NAMES — always use business labels:
  sort_key / week_key       → drop from output
  week_range                → "Week"
  species                   → "Species"
  patient_species           → "Species"
  category / plan_category  → "Category"
  sku_name / plan_item_name → "SKU"
  revenue                   → "Revenue"
  prev_revenue              → "Previous Week Revenue" or "Previous Month Revenue"
  growth_pct                → "Growth %"
  invoice_count             → "Invoices"
  customer_count            → "Customers"
  units_sold                → "Units Sold"
  stock_value               → "Stock Value"
  coverage_days             → "Coverage Days"

CURRENCY FORMAT — prefix ₹ and add commas:
  264946.50 → "₹2,64,947"   (Indian number format)

PERCENTAGE FORMAT — always include % sign:
  -24.9 → "-24.9%"
  12.3  → "+12.3%"

KPI / TREND QUERIES — when growth % is present or derivable:
  Always add a "Trend" column using these EXACT thresholds:
    growth_pct > +10%  → "Growing ↑"
    growth_pct < -10%  → "Declining ↓"
    between -10% and +10% → "Stable →"

  Always include: Current Value | Prior Value | Growth % | Trend

DO NOT INFER direction without a calculated growth_pct.
  If growth % is not in the data and cannot be computed from available columns,
  omit the Trend column entirely. Do not guess.

REPORTING WINDOW RULE — output exactly what the user asked for:
  Weekly queries fetch one extra week as a WoW baseline. You MUST drop it before output.
  Step 1: Find the oldest distinct week (lowest sort_key or earliest week_range) in the result.
  Step 2: Remove ALL rows belonging to that oldest week from output_data — every species/category row for that week.
  Step 3: Any week where previous_week_revenue would be ₹0 or absent has no valid comparison — exclude it entirely.
  Only weeks that have a real prior-week value to compare against appear in the output table.
  If fewer display weeks remain than the user asked for, show what is available — never pad with N/A rows.

  ⚠️ SUMMARY STATS — compute ONLY from the post-drop rows that end up in output_data:
  You MUST drop the baseline week FIRST. Then compute every number you quote in summary
  and key_insights — min, max, average, total — ONLY from the remaining display rows.
  NEVER compute min/max/avg from the full raw SQL result if it includes the baseline week.
  The baseline week is a partial or prior week and its values must not appear in any narrative.

  WRONG: raw data has 13 rows → compute min=6 from row 1 (baseline) → write "lowest was 6 new customers"
  CORRECT: drop row 1 → 12 display rows remain → compute min=11 → write "lowest was 11 new customers"

DATA GRANULARITY LABEL — always state in output_title and summary:
  If output_data contains a week_range column (weekly data):
    → output_title must include "(Weekly)" e.g. "Species Sales by Week (Weekly)"
    → summary must contain this sentence: "Each row represents one ISO week (Monday to Sunday) — these are not calendar month totals."
    → NEVER describe a single week's revenue as "March revenue" or "April revenue".
      Use only the exact week_range dates from the data (e.g. "23 Mar 2026 – 29 Mar 2026").
  If output_data contains a month_label column (monthly data):
    → output_title must include "(Monthly)" e.g. "Species Revenue by Month (Monthly)"
    → summary must state the calendar month range covered (e.g. "January 2026 to June 2026").

OUTPUT DATA RULES:
  • Sort chronologically for time-series; by value descending for rankings.
  • For comparisons: include absolute change AND % change columns.
  • If an aggregate is NULL or 0, show "₹0" or "0", not blank.
  • Remove internal keys (sort_key, week_key, year_num, month_num) from output_columns.
  • NEVER use "N/A" in output_data. A row showing "N/A" for growth% means it
    is the WoW baseline week — it must be dropped entirely, not shown with N/A.
    If a metric does not apply to a segment, split into separate focused tables
    instead of one wide table with N/A cells.

REPORTING PERIOD — derive from the actual data rows, never from assumptions:
  Read the date values in the query results to determine year and month.
  If invoice_date or month_label values are from 2026, write "March 2026" — not "March 2025".
  Never default to a year from training knowledge. The data is authoritative.

════════════════════════════════════════════════════════════
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"Query results:\n{json.dumps(llm_payload, indent=2, default=str)}"
    )

    if DEBUG_MODE:
        print(f"   [DEBUG] Analyzer payload (1000 chars):\n{json.dumps(llm_payload, indent=2, default=str)[:1000]}")

    try:
        raw      = call_llm(system, user)
        analysis = parse_json(raw)
        print(f"   ✅  Analysis complete ({analysis.get('output_format', '?')} output)")
        if DEBUG_MODE:
            print(f"   [DEBUG] Analysis:\n{json.dumps(analysis, indent=2, default=str)[:1000]}")
        return {**state, "analysis": analysis, "error": None}
    except Exception as exc:
        return {**state, "error": f"Analyzer failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 4 — OUTPUT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def output_formatter(state: AgentState) -> AgentState:
    print("\n📊  [4/5] Formatting output…\n")

    a    = state.get("analysis") or {}
    W    = 68
    SEP  = "─" * W
    DSEP = "═" * W

    lines = [DSEP, "  NL2SQL AGENT — RESULTS", DSEP, f"\n❓  {state['user_query']}\n"]

    data    = a.get("output_data", [])
    columns = a.get("output_columns", [])
    title   = a.get("output_title", "Results")

    if data and columns:
        lines += [SEP, f"📋  {title.upper()}", SEP]
        df = pd.DataFrame(data, columns=columns)
        lines.append(tabulate(df.values.tolist(), headers=list(df.columns), tablefmt="pretty"))
    else:
        lines += [SEP, a.get("summary", "(no data returned)")]

    # Show executed SQL
    lines += [f"\n{SEP}", "🔍  EXECUTED SQL", SEP]
    for r in (state.get("query_results") or []):
        sql_preview = r.get("sql", "").replace("\n", " ")
        if len(sql_preview) > 220:
            sql_preview = sql_preview[:220] + "…"
        lines.append(f"\nStep {r['step_id']} — {r['description']}")
        lines.append(f"  {sql_preview}")

    lines.append(f"\n{DSEP}")

    output = "\n".join(lines)
    print(output)
    return {**state, "final_output": output}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 5 — INSIGHTS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def insights_generator(state: AgentState) -> AgentState:
    print("\n💡  [5/5] Generating insights…\n")

    a            = state.get("analysis") or {}
    data         = a.get("output_data", [])
    columns      = a.get("output_columns", [])
    title        = a.get("output_title", "Results")
    summary      = a.get("summary", "")
    key_insights = a.get("key_insights", [])

    context_parts = []
    if summary:
        context_parts.append(f"Analytical summary: {summary}")
    if key_insights:
        context_parts.append(f"Key insights: {json.dumps(key_insights, default=str)}")
    if data and columns:
        context_parts.append(
            f"Data table — {title}\n"
            f"Columns: {columns}\n"
            f"Rows:\n{json.dumps(data, indent=2, default=str)}"
        )
    data_context = "\n\n".join(context_parts) if context_parts else "(no data returned)"

    if DEBUG_MODE:
        print(f"   [DEBUG] Insights context (300 chars):\n   {data_context[:300]}")

    system = """You are a senior business analyst producing insights for clinic owners and marketing teams.

Read the data carefully and produce sharp, actionable insights.

Structure your response in plain text:

INSIGHT SUMMARY
One paragraph directly answering what the data tells us.

KEY OBSERVATIONS
1. [specific observation with exact numbers from the data]
2. [trend, pattern, or anomaly — only if a growth % or comparison was actually calculated]
3. [comparison or ratio that stands out]

RECOMMENDATION
One or two sentences on what action this data suggests.

════════════════════════════════════════════════════════════
INSIGHT RULES — MANDATORY
════════════════════════════════════════════════════════════

NEVER state "growing", "declining", or "stable" unless a growth % was
actually computed in the data. If only averages are present:
  CORRECT: "Average weekly revenue over the last 4 weeks was ₹2,61,801"
  WRONG:   "Revenue is stable" (no growth % was calculated)

ALWAYS quote the actual numbers from the data using ₹ for currency.
  CORRECT: "Canine revenue was ₹2,64,947 this week, up +12.3% from last week"
  WRONG:   "Canine revenue increased this week"

Use direction words only when supported by a calculated metric.
Apply these EXACT thresholds — do not use any other values:
  growth_pct > +10% → "Growing ↑"
  growth_pct < -10% → "Declining ↓"
  between -10% and +10% → "Stable →"

RECOMMENDATION DISCIPLINE — mandatory:
  Do NOT recommend promotions, discounts, campaigns, or operational changes
  unless the data directly shows a specific gap or opportunity.

  NEVER mention data gaps, API gaps, missing data, or data recovery anywhere
  in the insights — not in INSIGHT SUMMARY, KEY OBSERVATIONS, or RECOMMENDATION.
  Treat the available data as complete and give business insights only.
  The clinic owner does not need to know about missing dates.

  PREFER investigative recommendations:
    - "Investigate why Canine revenue dropped 18% in Week 22"
    - "Compare this category across the 3 clinics to identify which is driving the decline"
    - "Check if the Feline spike is concentrated in one SKU or broad-based"
    - "Compare this week's mix to the 4-week average to identify the anomaly"

  Only escalate to business actions (offers, campaigns, stocking decisions)
  when the data unambiguously supports them.

If data is empty or zero, acknowledge it and suggest what query to run next.

Format currency in Indian style: ₹2,64,947 (not ₹264947).

REPORTING PERIOD — always read from the data rows, never assume or compute:
  Extract dates, week ranges, and month labels directly from the SQL result rows provided to you.
  NEVER compute what "last 4 weeks" or "last month" means from today's calendar.
  NEVER assume what date range the database covers — read it from the actual data.
  Whatever dates appear in the result rows are the correct dates — use them exactly.
════════════════════════════════════════════════════════════
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"{data_context}"
    )

    W    = 68
    DSEP = "═" * W

    try:
        raw = call_llm(system, user)
        lines = [f"\n{DSEP}", "  🧠  AI INSIGHTS", DSEP, "", raw.strip(), f"\n{DSEP}\n"]
        print("\n".join(lines))
        return {**state, "insights": raw.strip()}
    except Exception as exc:
        print(f"\n⚠️   Insights generation failed: {exc}")
        return {**state, "insights": None}


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR NODE
# ═══════════════════════════════════════════════════════════════════════════════

def handle_error(state: AgentState) -> AgentState:
    W = 68
    print(f"\n{'═'*W}\n  ❌  AGENT ERROR\n{'═'*W}")
    print(f"  {state.get('error', 'Unknown error')}")
    print(f"{'═'*W}\n")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def _route(state: AgentState, next_node: str) -> str:
    return "error" if state.get("error") else next_node


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("query_planner",       query_planner)
    g.add_node("sql_dynamic_agent",   run_sql_dynamic_agent)
    g.add_node("result_analyzer",     result_analyzer)
    g.add_node("output_formatter",    output_formatter)
    g.add_node("insights_generator",  insights_generator)
    g.add_node("handle_error",        handle_error)

    g.set_entry_point("query_planner")

    g.add_conditional_edges("query_planner", lambda s: _route(s, "sql_dynamic_agent"), {
        "sql_dynamic_agent": "sql_dynamic_agent",
        "error":             "handle_error",
    })
    g.add_conditional_edges("sql_dynamic_agent", lambda s: _route(s, "result_analyzer"), {
        "result_analyzer": "result_analyzer",
        "error":           "handle_error",
    })
    g.add_conditional_edges("result_analyzer", lambda s: _route(s, "output_formatter"), {
        "output_formatter": "output_formatter",
        "error":            "handle_error",
    })
    g.add_edge("output_formatter",   "insights_generator")
    g.add_edge("insights_generator", END)
    g.add_edge("handle_error",       END)

    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _append_query_log(state: Dict[str, Any], request_id: str) -> None:
    sql_steps = []
    for r in (state.get("query_results") or []):
        sql_steps.append({
            "step_id":      r.get("step_id"),
            "description":  r.get("description"),
            "sql":          r.get("sql"),
            "success":      r.get("success"),
            "row_count":    r.get("row_count", 0),
            "columns":      r.get("columns", []),
            "error":        r.get("error"),
            "data_preview": r.get("data", [])[:50],
        })

    record = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id,
        "query":      state.get("user_query"),
        "plan":       state.get("query_plan"),
        "sql_steps":  sql_steps,
        "analysis":   state.get("analysis"),
        "insights":   state.get("insights"),
        "error":      state.get("error"),
    }

    # ── Local flat file (dev) ─────────────────────────────────────────────────
    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        print(f"\n   📝  Run appended → {QUERY_LOG_FILE}")
    except OSError as exc:
        print(f"\n   ⚠️  Could not write local log: {exc}")

    # ── DB log (works on Render and locally) ─────────────────────────────────
    try:
        with _engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO allpets_query_log
                    (ts, request_id, query, plan, sql_steps, analysis, insights, error)
                VALUES
                    (:ts, :request_id, :query, :plan, :sql_steps, :analysis, :insights, :error)
            """), {
                "ts":         record["timestamp"],
                "request_id": request_id,
                "query":      record["query"],
                "plan":       json.dumps(record["plan"],      default=str),
                "sql_steps":  json.dumps(record["sql_steps"], default=str),
                "analysis":   json.dumps(record["analysis"],  default=str),
                "insights":   record["insights"],
                "error":      record["error"],
            })
            conn.commit()
        print(f"   📝  Run logged → allpets_query_log [{request_id}]")
    except Exception as exc:
        print(f"\n   ⚠️  Could not write DB log: {exc}")


def run_agent(user_query: str) -> Dict[str, Any]:
    request_id = f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    W = 68
    print(f"\n{'═'*W}")
    print(f"  NL2SQL AGENT  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{request_id}]")
    print(f"{'═'*W}")

    final: Dict[str, Any] = {
        "user_query":    user_query,
        "schema":        None,
        "query_plan":    None,
        "query_results": None,
        "analysis":      None,
        "final_output":  None,
        "insights":      None,
        "error":         None,
    }

    try:
        schema = load_schema(SCHEMA_FILE)
        final["schema"] = schema

        initial: AgentState = {
            "user_query":    user_query,
            "schema":        schema,
            "query_plan":    None,
            "query_results": None,
            "analysis":      None,
            "final_output":  None,
            "insights":      None,
            "error":         None,
        }

        agent = build_graph()
        final = agent.invoke(initial)

    except Exception as exc:
        final["error"] = str(exc)

    # Always log — even on unexpected crashes
    _append_query_log(final, request_id)

    raw_error = final.get("error")
    return {
        "user_query":    final.get("user_query"),
        "query_plan":    final.get("query_plan"),
        "query_results": final.get("query_results"),
        "analysis":      final.get("analysis"),
        "final_output":  final.get("final_output"),
        "insights":      final.get("insights"),
        "error":         _sanitize_error(raw_error, request_id),
        "request_id":    request_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\n  Example queries:")
        print('  · "What is the total sales for this month?"')
        print('  · "Compare current month sales with the last 3 months, show month-on-month % change"')
        print('  · "Top 10 customers by revenue this quarter"')
        print()
        query = input("🔹  Enter your query: ").strip()
        if not query:
            query = "Compare current month sales with the last 3 months, show month-on-month change"

    run_agent(query)
