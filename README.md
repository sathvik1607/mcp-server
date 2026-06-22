# allpets_mcp

Fully self-contained MCP server for AllPets Clinic & Beyond analytics.
All business logic lives inside this repo — no dependency on any sibling folder at runtime.

---

## Setup (one-time)

```powershell
cd C:\Users\sathv\Desktop\allpets_mcp

# Create venv
python -m venv venv

# Install dependencies
.\venv\Scripts\pip.exe install -r requirements.txt

# Fill in credentials (already populated for local dev)
notepad .env
```

---

## Local test

```powershell
# Dev inspector — opens browser UI to call tools manually
.\venv\Scripts\python.exe -m mcp dev server.py

# Or run directly (exits immediately in stdio mode — expected)
.\venv\Scripts\python.exe server.py
```

---

## Claude Desktop setup

1. Open `%APPDATA%\Claude\claude_desktop_config.json`
2. Add the `allpets` entry inside `"mcpServers"`:

```json
{
  "mcpServers": {
    "allpets": {
      "command": "C:\\Users\\sathv\\Desktop\\allpets_mcp\\venv\\Scripts\\python.exe",
      "args":    ["C:\\Users\\sathv\\Desktop\\allpets_mcp\\server.py"]
    }
  }
}
```

3. Restart Claude Desktop completely (quit and reopen).
4. The tool icon appears in the chat input — click it to see the 4 AllPets tools.

---

## Phase 1 tools

| Tool | What it does |
|---|---|
| `get_current_week_dates` | Returns this ISO week's Monday/Sunday |
| `get_weekly_dashboard` | Full KPI dashboard for a date range |
| `ask_analytics` | Freeform NL2SQL question |
| `generate_excel_report` | Weekly dashboard as .xlsx (base64) |

---

## Project structure

```
allpets_mcp/
├── server.py              ← Entry point (stdio MCP server)
├── config.py              ← DB engine + env loading (self-contained)
├── dashboard_queries.py   ← Pre-computed weekly KPI engine
├── nl2sql_agent.py        ← 5-step LangGraph NL2SQL pipeline
├── sql_dynamic_agent.py   ← SQL generation + retry (~1970 lines)
├── excel_export.py        ← openpyxl report generator
├── etc/secrets/
│   └── schema_all.txt     ← Full DB schema DDL (used by SQL agent)
├── adapters/
│   ├── dashboard.py       ← Wraps DashboardService.run_weekly()
│   ├── analytics.py       ← Wraps nl2sql_agent.run_agent()
│   └── excel.py           ← Wraps generate_excel() → base64
├── tools/
│   ├── dashboard.py       ← get_weekly_dashboard, get_current_week_dates
│   ├── analytics.py       ← ask_analytics
│   └── reports.py         ← generate_excel_report
├── .env                   ← All credentials (DB + OpenAI)
└── requirements.txt
```

---

## Updating business logic

When `dashboard_queries.py`, `nl2sql_agent.py`, `sql_dynamic_agent.py`, or `excel_export.py`
change in `allpets_new_schema`, copy the updated file here:

```powershell
Copy-Item ..\allpets_new_schema\dashboard_queries.py .\dashboard_queries.py -Force
Copy-Item ..\allpets_new_schema\nl2sql_agent.py .\nl2sql_agent.py -Force
Copy-Item ..\allpets_new_schema\sql_dynamic_agent.py .\sql_dynamic_agent.py -Force
Copy-Item ..\allpets_new_schema\excel_export.py .\excel_export.py -Force
Copy-Item ..\allpets_new_schema\etc\secrets\schema_all.txt .\etc\secrets\schema_all.txt -Force
```
