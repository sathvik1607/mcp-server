# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

An MCP (Model Context Protocol) server that exposes AllPets Clinic & Beyond analytics to Claude Desktop. It provides four tools:

- `get_current_week_dates` — ISO week boundaries (Monday/Sunday)
- `get_weekly_dashboard` — Pre-computed KPI dashboard (sales, species, inventory, life stages)
- `ask_analytics` — Natural language → SQL pipeline for ad-hoc queries
- `generate_excel_report` — Branded Excel export of the weekly dashboard (returned as base64)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

Credentials live in `.env` (DB + OpenAI API key). `config.py` loads them and creates a shared SQLAlchemy engine singleton — it must be the first import in any entry point.

## Running

```powershell
# Dev mode — browser UI for manual tool testing
.\venv\Scripts\python.exe -m mcp dev server.py

# Production — stdio transport for Claude Desktop
.\venv\Scripts\python.exe server.py
```

Core modules have `if __name__ == "__main__"` blocks for standalone testing:

```powershell
python dashboard_queries.py          # runs weekly dashboard against live DB
python nl2sql_agent.py "your question"  # tests NL→SQL pipeline
```

There is no automated test suite.

## Architecture

The codebase has three layers:

```
tools/          ← MCP interface (FastMCP tool definitions, type hints)
adapters/       ← Thin delegation wrappers to core logic
*.py (root)     ← Core business logic
```

### Dashboard (`dashboard_queries.py`)

`DashboardService.run_weekly()` fetches raw data once (invoices, stocks, patients, client IDs) then computes all metrics in Python. SQL is used only for aggregation and filtering; all business definitions (e.g. repeat customer %, species splits, life stage buckets) are computed in Python. Output is a Pydantic `WeeklyDashboard` model with typed metric objects.

### NL→SQL Pipeline (`nl2sql_agent.py` + `sql_dynamic_agent.py`)

LangGraph orchestrator with these nodes: Query Planner → SQL Dynamic Agent (generate + execute, up to 3 retries with LLM self-correction) → Result Analyzer → Insights Generator. The schema for SQL generation is read from `etc/secrets/schema_all.txt` (6 MySQL tables: clinics, clients, patients, stocks, invoices, payments).

`sql_dynamic_agent.py` is the self-contained SQL sub-agent; it handles rate-limit backoff (5s → 10s → 20s), unwraps SQLAlchemy error boilerplate to surface clean MySQL errors to the LLM, and parses JSON from LLM responses that may be wrapped in markdown code blocks.

### Excel Export (`excel_export.py`)

Two-sheet openpyxl workbook: a print-ready branded dashboard sheet (A4 landscape, teal `#1B6B72` theme) and a flat raw-data sheet for pivot tables. The MCP tool returns the workbook as base64-encoded bytes.

## Key Files

| File | Purpose |
|---|---|
| `server.py` | Entry point — registers all tools via FastMCP |
| `config.py` | `.env` loader + SQLAlchemy engine singleton |
| `dashboard_queries.py` | Weekly KPI dashboard engine (754 lines) |
| `nl2sql_agent.py` | LangGraph NL→SQL orchestrator (750 lines) |
| `sql_dynamic_agent.py` | SQL generation + execution sub-agent (2,219 lines) |
| `excel_export.py` | openpyxl workbook generator (470 lines) |
| `etc/secrets/schema_all.txt` | Full MySQL schema for SQL generation context |

## Database

MySQL on AWS RDS (`ap-south-1`). Six tables in `cohort_main`:

- `allpets_new_invoices` — transactional (Jan 2025 – present), primary data source
- `allpets_new_payments` — payment records
- `allpets_new_clients` / `allpets_new_patients` — CRM data
- `allpets_new_stocks` — latest inventory snapshot
- `allpets_new_clinics` — 3 clinic locations
