"""
Tool: ask_analytics
"""
from mcp.server.fastmcp import FastMCP

import adapters.analytics as _adapter


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def ask_analytics(question: str) -> dict:
        """
        Ask any natural language question about AllPets clinic data.

        Runs the full NL2SQL pipeline (LangGraph + GPT-4o). Use this for:
        - Ad-hoc questions not answered by the pre-computed dashboard
        - Comparisons across custom date ranges
        - Top-N lists (top SKUs, top clients, etc.)
        - Trend analysis over multiple months
        - Explanations of anomalies ("why did revenue drop on June 18?")

        The response includes:
        - analysis:          structured data (table rows + columns)
        - insights:          narrative summary
        - generated_queries: the SQL that was executed
        - error:             null on success, message on failure

        Args:
            question: A plain-language analytics question about the clinic
        """
        return _adapter.run_agent(question)
