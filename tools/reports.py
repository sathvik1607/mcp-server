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

        Remote mode (Render): saves the file server-side and returns a download_url.
        Tell the user to click the download_url to save the file — do NOT attempt to
        decode or save it yourself.

        Local mode: saves directly to the user's machine and returns saved_to path.

        The workbook has two sheets:
        - Dashboard: print-ready A4 landscape with teal styling, KPI boxes, two-column tables
        - Raw Data:  flat table of all metrics for pivot table use

        Args:
            week_start:   Monday of the target week in YYYY-MM-DD format
            week_end:     Sunday of the target week in YYYY-MM-DD format
            prior_weeks:  Prior weeks for averages (default 3)

        Returns:
            Remote: download_url, filename, size_bytes, week, note
            Local:  saved_to, filename, size_bytes, week
        """
        return _adapter.generate_excel_report(week_start, week_end, prior_weeks)
