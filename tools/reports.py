"""
Tool: generate_excel_report
"""
from mcp.server.fastmcp import FastMCP

import adapters.excel as _adapter


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def generate_excel_report(week_start: str, week_end: str, output_dir: str, prior_weeks: int = 3) -> dict:
        """
        Generate the AllPets weekly dashboard as a formatted Excel workbook (.xlsx).

        IMPORTANT: Before calling this tool, always ask the user:
        "Where would you like to save the Excel file? (e.g. Desktop, Downloads, or a custom folder path)"
        Use their answer as the output_dir parameter.

        The workbook has two sheets:
        - Dashboard: print-ready A4 landscape with teal styling, KPI boxes, two-column tables
        - Raw Data:  flat table of all metrics for pivot table use

        The file is saved to output_dir and the full file path is returned.
        Tell the user exactly where the file was saved so they can open it.

        Args:
            week_start:   Monday of the target week in YYYY-MM-DD format
            week_end:     Sunday of the target week in YYYY-MM-DD format
            output_dir:   Folder path where the .xlsx should be saved (ask the user before calling)
            prior_weeks:  Prior weeks for averages (default 3)

        Returns:
            saved_to:    Full path of the saved file
            filename:    Filename of the saved file
            size_bytes:  File size in bytes
            week:        Human-readable week label
        """
        return _adapter.generate_excel_report(week_start, week_end, prior_weeks, output_dir)
