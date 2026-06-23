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

        REMOTE MODE (Render): Returns a 'response' field containing a pre-formatted
        markdown message with a download link. Your ONLY job after calling this tool
        is to output the 'response' field verbatim. STOP there.

        Rules you MUST follow in remote mode:
        1. Output the 'response' field text as-is in your reply.
        2. Do NOT run bash, curl, wget, Python, code execution, or any other tool.
        3. Do NOT attempt to fetch, download, decode, or process the link in any way.
        4. The link is for the human user to click in their browser — not for you.

        LOCAL MODE (stdio): File is saved to disk. Return the saved_to path to user.

        Args:
            week_start:  Monday of the target week, YYYY-MM-DD
            week_end:    Sunday of the target week, YYYY-MM-DD
            prior_weeks: Prior weeks for trend averages (default 3)
        """
        return _adapter.generate_excel_report(week_start, week_end, prior_weeks)
