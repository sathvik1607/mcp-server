"""
Tool: generate_excel_report
"""
from mcp.server.fastmcp import FastMCP

import adapters.excel as _adapter


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def generate_excel_report(week_start: str, week_end: str, prior_weeks: int = 3) -> dict:
        """
        Generate the AllPets weekly dashboard as a formatted Excel workbook (.xlsx).

        CRITICAL INSTRUCTIONS — read before calling:
        - On Render (remote): the tool returns a download_url. You MUST present this
          URL to the user as a clickable markdown link: [Download Excel Report](url)
          Do NOT attempt to fetch, curl, wget, download, decode, or save the file
          yourself. Do NOT run any commands. The user clicks the link in their browser.
        - On local stdio: the file is saved directly to disk; return the saved_to path.

        The workbook has two sheets:
        - Dashboard: print-ready A4 landscape with teal styling and KPI boxes
        - Raw Data:  flat table of all metrics for pivot table use

        Args:
            week_start:   Monday of the target week in YYYY-MM-DD format
            week_end:     Sunday of the target week in YYYY-MM-DD format
            prior_weeks:  Number of prior weeks for averages (default 3)

        Returns (remote):  download_url, filename, size_bytes, week
        Returns (local):   saved_to, filename, size_bytes, week
        """
        return _adapter.generate_excel_report(week_start, week_end, prior_weeks)
