"""
allpets_mcp — MCP server for AllPets Clinic & Beyond.

Exposes existing AllPets analytics capabilities to Claude Desktop and other MCP clients.
All business logic lives in allpets_new_schema — this layer only exposes it.

Phase 1 tools:
    get_current_week_dates   → ISO week helper (no DB call)
    get_weekly_dashboard     → Full KPI dashboard via DashboardService.run_weekly()
    ask_analytics            → Freeform NL2SQL via nl2sql_agent.run_agent()
    generate_excel_report    → Excel workbook via excel_export.generate_excel()

Run:
    python server.py                        (stdio — for Claude Desktop)
    python -m mcp dev server.py             (dev inspector — for local testing)
"""
import builtins
import sys
import os

# Redirect all print() calls to stderr so they never corrupt the MCP stdio transport.
# The transport reads/writes JSON-RPC over stdout; any stray text breaks the framing.
_real_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)
builtins.print = _stderr_print

import config  # MUST be the very first import — injects sys.path + loads .env

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, PlainTextResponse

from tools import dashboard as dashboard_tools
from tools import analytics as analytics_tools
from tools import reports   as report_tools

mcp = FastMCP(
    name="allpets",
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp-server-sjse.onrender.com"],
        allowed_origins=["https://mcp-server-sjse.onrender.com"],
    ),
    instructions=(
        "You are connected to AllPets Clinic & Beyond's analytics platform. "
        "When the user says 'this week' without specifying dates, call "
        "get_current_week_dates first to get week_start and week_end. "
        "Use get_weekly_dashboard for structured KPI data (species, categories, "
        "customers, inventory, life stage). "
        "Use ask_analytics for ad-hoc comparisons, trends, or questions outside "
        "the pre-computed dashboard. "
        "Use generate_excel_report to produce the weekly downloadable report. "
        "IMPORTANT — server cold start: if a tool call times out or returns a "
        "connection error on the first attempt, the server is warming up (takes "
        "up to 50 seconds). Immediately tell the user: 'The analytics server is "
        "starting up — this takes about 50 seconds on the first request. Please "
        "hold on...' then retry the same tool call once after a brief wait. "
        "Do not report it as a failure on the first timeout."
    ),
)

dashboard_tools.register(mcp)
analytics_tools.register(mcp)
report_tools.register(mcp)

if __name__ == "__main__":
    # ── Render (HTTP + bearer token auth) ─────────────────────────────────────
    if os.getenv("RENDER"):
        import uvicorn

        SECRET_TOKEN = os.getenv("MCP_SECRET_TOKEN", "")

        class BearerAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path == "/health":
                    return PlainTextResponse("OK")
                if SECRET_TOKEN:
                    auth = request.headers.get("Authorization", "")
                    if not auth.startswith("Bearer ") or auth[7:] != SECRET_TOKEN:
                        return Response("Unauthorized", status_code=401)
                return await call_next(request)

        app = mcp.streamable_http_app()
        app.add_middleware(BearerAuthMiddleware)

        port = int(os.getenv("PORT", 8000))
        print(f"Starting AllPets MCP on port {port} (HTTP + bearer auth)")
        uvicorn.run(app, host="0.0.0.0", port=port)

    # ── Local dev (stdio) ──────────────────────────────────────────────────────
    else:
        mcp.run(transport="stdio")
