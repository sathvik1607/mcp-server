"""
Adapter: excel_export.generate_excel

Local mode  (no RENDER env): saves .xlsx to the user's machine, returns file path.
Remote mode (RENDER env set): returns base64-encoded bytes — Claude tells the user
to save the content or download it, since the server filesystem is ephemeral.
"""
import os
import base64
import config  # MUST be first — sets sys.path + loads env

from dashboard_queries import DashboardService
from excel_export import generate_excel as _generate_excel

_service    = DashboardService(config.engine)
_IS_REMOTE  = bool(os.getenv("RENDER"))
_OUTPUT_DIR = os.getenv(
    "EXCEL_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "Desktop")
)


def generate_excel_report(week_start: str, week_end: str, prior_weeks: int = 3, output_dir: str = None) -> dict:
    dashboard  = _service.run_weekly(week_start, week_end, prior_weeks)
    xlsx_bytes = _generate_excel(dashboard)
    filename   = f"allpets_dashboard_{week_start}_to_{week_end}.xlsx"

    # ── Remote (Render) — return base64, server filesystem is ephemeral ───────
    if _IS_REMOTE:
        return {
            "filename":   filename,
            "size_bytes": len(xlsx_bytes),
            "week":       dashboard.week,
            "base64":     base64.b64encode(xlsx_bytes).decode("utf-8"),
            "note":       "Save the base64 content as a .xlsx file on your machine.",
        }

    # ── Local — save to disk and return the file path ─────────────────────────
    save_dir = output_dir or _OUTPUT_DIR
    save_dir = save_dir.replace("Desktop",   os.path.join(os.path.expanduser("~"), "Desktop"))
    save_dir = save_dir.replace("Downloads", os.path.join(os.path.expanduser("~"), "Downloads"))
    filepath = os.path.join(save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(xlsx_bytes)

    return {
        "saved_to":   filepath,
        "filename":   filename,
        "size_bytes": len(xlsx_bytes),
        "week":       dashboard.week,
    }
