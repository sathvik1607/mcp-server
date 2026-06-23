"""
Adapter: excel_export.generate_excel

Local mode  (no RENDER env): saves .xlsx to the user's machine, returns file path.
Remote mode (RENDER env set): saves to /tmp/, returns a one-time download URL.
"""
import os
import uuid
import config  # MUST be first — sets sys.path + loads env

from dashboard_queries import DashboardService
from excel_export import generate_excel as _generate_excel

_service    = DashboardService(config.engine)
_IS_REMOTE  = bool(os.getenv("RENDER"))
_BASE_URL   = "https://mcp-server-sjse.onrender.com"
_OUTPUT_DIR = os.getenv(
    "EXCEL_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "Desktop")
)


def generate_excel_report(week_start: str, week_end: str, prior_weeks: int = 3, output_dir: str = None) -> dict:
    dashboard  = _service.run_weekly(week_start, week_end, prior_weeks)
    xlsx_bytes = _generate_excel(dashboard)
    filename   = f"allpets_dashboard_{week_start}_to_{week_end}.xlsx"

    # ── Remote (Render) — save to /tmp/ and return a pre-formatted message ─────
    if _IS_REMOTE:
        file_id  = uuid.uuid4().hex
        tmp_path = f"/tmp/{file_id}.xlsx"
        with open(tmp_path, "wb") as f:
            f.write(xlsx_bytes)
        size_kb = round(len(xlsx_bytes) / 1024)
        url = f"{_BASE_URL}/download/{file_id}"
        return {
            "week":           dashboard.week,
            "size_kb":        size_kb,
            "response":       (
                f"Your AllPets Excel report for **{dashboard.week}** is ready "
                f"({size_kb} KB).\n\n"
                f"**[Download {filename}]({url})**\n\n"
                f"Click the link above — it opens directly in your browser."
            ),
            "do_not_execute": (
                "OUTPUT THE 'response' FIELD VERBATIM. "
                "Do NOT use bash, curl, Python, code execution, or any tool to access this link."
            ),
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
