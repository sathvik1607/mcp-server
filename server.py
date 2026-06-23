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
from starlette.responses import PlainTextResponse

from tools import dashboard as dashboard_tools
from tools import analytics as analytics_tools
from tools import reports   as report_tools

_INSTRUCTIONS = (
    "You are connected to AllPets Clinic & Beyond's analytics platform. "
    "When the user says 'this week' without specifying dates, call "
    "get_current_week_dates first to get week_start and week_end. "
    "Use get_weekly_dashboard for structured KPI data (species, categories, "
    "customers, inventory, life stage). "
    "Use ask_analytics for ad-hoc comparisons, trends, or questions outside "
    "the pre-computed dashboard. "
    "Use generate_excel_report to produce the weekly downloadable report — "
    "when it returns a download_url, present it to the user as a markdown "
    "clickable link and do NOT attempt to fetch, curl, or decode the file. "
    "IMPORTANT — server cold start: if a tool call times out or returns a "
    "connection error on the first attempt, the server is warming up (takes "
    "up to 50 seconds). Immediately tell the user: 'The analytics server is "
    "starting up — this takes about 50 seconds on the first request. Please "
    "hold on...' then retry the same tool call once after a brief wait. "
    "Do not report it as a failure on the first timeout."
)

_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["mcp-server-sjse.onrender.com"],
    allowed_origins=["https://mcp-server-sjse.onrender.com"],
)

# Module-level mcp — used for stdio (local dev) and mcp dev inspector.
# No OAuth here; HTTP+OAuth lives in the __main__ Render branch below.
mcp = FastMCP(
    name="allpets",
    json_response=True,
    transport_security=_TRANSPORT_SECURITY,
    instructions=_INSTRUCTIONS,
)

dashboard_tools.register(mcp)
analytics_tools.register(mcp)
report_tools.register(mcp)


if __name__ == "__main__":
    # ── Render (HTTP + full OAuth 2.0 / Dynamic Client Registration) ──────────
    if os.getenv("RENDER"):
        import secrets
        import time
        import uvicorn

        from mcp.server.auth.provider import (
            AuthorizationCode,
            AuthorizationParams,
            AccessToken,
            RefreshToken,
            TokenError,
        )
        from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
        from mcp.shared.auth import OAuthToken
        from pydantic import AnyHttpUrl as _AnyHttpUrl, TypeAdapter as _TypeAdapter
        _url = _TypeAdapter(_AnyHttpUrl).validate_python
        import re as _re
        from starlette.routing import Route
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, RedirectResponse, Response, FileResponse

        SECRET_TOKEN = os.getenv("MCP_SECRET_TOKEN", "")
        BASE_URL = "https://mcp-server-sjse.onrender.com"
        port = int(os.getenv("PORT", 8000))

        # ── In-memory OAuth 2.0 provider ──────────────────────────────────────
        # Implements OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken].
        # State is ephemeral (lost on redeploy), which is fine — clients re-auth automatically.
        class AllPetsOAuthProvider:
            def __init__(self) -> None:
                self._clients: dict = {}   # client_id → OAuthClientInformationFull
                self._sessions: dict = {}  # session_id → (client, AuthorizationParams)
                self._codes: dict = {}     # code → AuthorizationCode
                self._tokens: dict = {}    # token → AccessToken

            async def get_client(self, client_id: str):
                return self._clients.get(client_id)

            async def register_client(self, client_info) -> None:
                self._clients[client_info.client_id] = client_info

            async def authorize(self, client, params: AuthorizationParams) -> str:
                sid = secrets.token_urlsafe(16)
                self._sessions[sid] = (client, params)
                return f"{BASE_URL}/auth/consent?session={sid}"

            async def load_authorization_code(self, client, authorization_code: str):
                return self._codes.get(authorization_code)

            async def exchange_authorization_code(self, client, authorization_code: AuthorizationCode) -> OAuthToken:
                token = secrets.token_urlsafe(32)
                expires = int(time.time()) + 86400 * 30  # 30-day token
                self._tokens[token] = AccessToken(
                    token=token,
                    client_id=client.client_id,
                    scopes=authorization_code.scopes or [],
                    expires_at=expires,
                )
                self._codes.pop(authorization_code.code, None)
                return OAuthToken(
                    access_token=token,
                    token_type="Bearer",
                    expires_in=86400 * 30,
                    scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
                )

            async def load_refresh_token(self, client, refresh_token: str):
                return None

            async def exchange_refresh_token(self, client, refresh_token, scopes) -> OAuthToken:
                raise TokenError(error="unsupported_grant_type")

            async def load_access_token(self, token: str):
                at = self._tokens.get(token)
                if at is None:
                    return None
                if at.expires_at and time.time() > at.expires_at:
                    self._tokens.pop(token, None)
                    return None
                return at

            async def revoke_token(self, token) -> None:
                t = getattr(token, "token", None)
                if t:
                    self._tokens.pop(t, None)

        # ── Consent page (browser PIN entry) ──────────────────────────────────
        _CONSENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AllPets Analytics — Authorize</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:#f0f4f8;display:flex;align-items:center;
         justify-content:center;min-height:100vh}}
    .card{{background:#fff;border-radius:12px;padding:40px;
           box-shadow:0 4px 24px rgba(0,0,0,.10);max-width:420px;width:90%}}
    h1{{color:#1B6B72;font-size:1.4rem;margin-bottom:12px}}
    p{{color:#555;line-height:1.5;margin-bottom:20px}}
    label{{font-weight:600;display:block;margin-bottom:6px;color:#333}}
    input{{width:100%;padding:10px 12px;border:1px solid #ccc;border-radius:6px;
           font-size:1rem;margin-bottom:16px}}
    button{{width:100%;padding:12px;background:#1B6B72;color:#fff;border:none;
            border-radius:6px;font-size:1rem;cursor:pointer}}
    button:hover{{background:#155a60}}
    .error{{color:#c0392b;font-size:.9rem;margin-bottom:12px}}
  </style>
</head>
<body>
  <div class="card">
    <h1>AllPets Clinic &amp; Beyond</h1>
    <p>Claude is requesting access to your analytics tools.<br>
       Enter the access PIN to authorize.</p>
    <form method="POST">
      <label for="pin">Access PIN</label>
      <input type="password" name="pin" id="pin" placeholder="Enter PIN" required autofocus>
      {error}
      <button type="submit">Authorize Access</button>
    </form>
  </div>
</body>
</html>"""

        def _make_consent_handler(provider: AllPetsOAuthProvider):
            async def handler(request: Request) -> Response:
                sid = request.query_params.get("session", "")
                if sid not in provider._sessions:
                    return HTMLResponse(
                        "<h1>Invalid or expired session. Please retry the connection.</h1>",
                        status_code=400,
                    )

                if request.method == "GET":
                    return HTMLResponse(_CONSENT_HTML.format(error=""))

                # POST — validate PIN
                form = await request.form()
                pin = str(form.get("pin", ""))
                if not SECRET_TOKEN or pin != SECRET_TOKEN:
                    err = '<p class="error">Incorrect PIN. Please try again.</p>'
                    return HTMLResponse(_CONSENT_HTML.format(error=err), status_code=401)

                client, params = provider._sessions.pop(sid)
                code = secrets.token_urlsafe(32)
                provider._codes[code] = AuthorizationCode(
                    code=code,
                    scopes=params.scopes or [],
                    expires_at=time.time() + 600,  # 10-minute auth code
                    client_id=client.client_id,
                    code_challenge=params.code_challenge,
                    redirect_uri=params.redirect_uri,
                    redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                )

                redirect = str(params.redirect_uri)
                sep = "&" if "?" in redirect else "?"
                redirect += f"{sep}code={code}"
                if params.state:
                    redirect += f"&state={params.state}"

                return RedirectResponse(url=redirect, status_code=302,
                                        headers={"Cache-Control": "no-store"})

            return handler

        # ── Diagnostic middleware (request/response logging) ──────────────────
        class DiagnosticMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                import logging
                log = logging.getLogger("mcp.diagnostic")
                log.warning(
                    "REQUEST %s %s | Accept: %s | Content-Type: %s | Origin: %s | Host: %s",
                    request.method, request.url.path,
                    request.headers.get("accept", "-"),
                    request.headers.get("content-type", "-"),
                    request.headers.get("origin", "-"),
                    request.headers.get("host", "-"),
                )
                response = await call_next(request)
                log.warning("RESPONSE %s %s → %s",
                            request.method, request.url.path, response.status_code)
                return response

        # ── Build the full app ─────────────────────────────────────────────────
        oauth_provider = AllPetsOAuthProvider()

        # Separate FastMCP instance for Render; auth requires a fresh instance.
        mcp_render = FastMCP(
            name="allpets",
            json_response=True,
            auth_server_provider=oauth_provider,
            auth=AuthSettings(
                issuer_url=_url(BASE_URL),
                client_registration_options=ClientRegistrationOptions(
                    enabled=True,
                    valid_scopes=["mcp:tools"],
                    default_scopes=["mcp:tools"],
                ),
                # resource_server_url drives /.well-known/oauth-protected-resource path.
                # Using base URL (no /mcp suffix) so the metadata lives at the root path
                # that claude.ai Connector actually requests.
                resource_server_url=_url(BASE_URL),
            ),
            transport_security=_TRANSPORT_SECURITY,
            instructions=_INSTRUCTIONS,
        )
        dashboard_tools.register(mcp_render)
        analytics_tools.register(mcp_render)
        report_tools.register(mcp_render)

        # base_app has a lifespan that starts the StreamableHTTPSessionManager.
        # We must NOT wrap it in an outer Starlette app — Mount() doesn't propagate
        # the inner app's lifespan, which breaks /mcp entirely.
        # Instead, inject the extra routes directly into base_app's router.
        base_app = mcp_render.streamable_http_app()

        async def health(request: Request) -> Response:
            return PlainTextResponse("OK")

        async def download_file(request: Request) -> Response:
            file_id = request.path_params.get("file_id", "")
            if not _re.fullmatch(r"[a-f0-9]{32}", file_id):
                return Response("Not found", status_code=404)
            tmp_path = f"/tmp/{file_id}.xlsx"
            if not os.path.exists(tmp_path):
                return Response("File not found or expired", status_code=404)
            return FileResponse(
                tmp_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"allpets_report.xlsx",
            )

        # Prepend so they are checked before FastMCP's own routes
        base_app.router.routes.insert(0, Route("/health", endpoint=health, methods=["GET"]))
        base_app.router.routes.insert(0, Route("/download/{file_id}", endpoint=download_file, methods=["GET"]))
        base_app.router.routes.insert(0, Route("/auth/consent",
                                               endpoint=_make_consent_handler(oauth_provider),
                                               methods=["GET", "POST"]))
        base_app.add_middleware(DiagnosticMiddleware)

        print(f"Starting AllPets MCP on port {port} (HTTP + OAuth 2.0)")
        uvicorn.run(base_app, host="0.0.0.0", port=port)

    # ── Local dev (stdio) ──────────────────────────────────────────────────────
    else:
        mcp.run(transport="stdio")
