"""
Tools: get_weekly_dashboard, get_current_week_dates
"""
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

import adapters.dashboard as _adapter


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_current_week_dates() -> dict:
        """
        Get the current ISO week's start (Monday) and end (Sunday) dates as YYYY-MM-DD strings.

        Call this first whenever the user asks about 'this week' without specifying dates.
        Returns week_start, week_end, and a human-readable week_label.
        """
        today  = date.today()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return {
            "week_start": monday.isoformat(),
            "week_end":   sunday.isoformat(),
            "week_label": f"{monday.strftime('%b %d')} – {sunday.strftime('%b %d, %Y')}",
        }

    @mcp.tool()
    def get_weekly_dashboard(week_start: str, week_end: str, prior_weeks: int = 3) -> dict:
        """
        Get the full AllPets Clinic weekly performance dashboard for a given week.

        Returns all pre-computed KPIs including:
        - total_sales: revenue value, prior 3-week average, variance %
        - repeat_customer_pct: repeat %, new count, total count
        - species_split: Canine / Feline / Others revenue with growth %
        - category_top5: top 5 categories by revenue with growth %
        - invoice_count_by_species: bill counts with growth %
        - day_night_split: Day vs Night revenue
        - new_vs_existing_customers: breakdown by species and customer type
        - new_vs_existing_revenue: new vs existing revenue share
        - life_stage: Puppy/Kitten/Adult/Senior revenue by species
        - inventory_by_type: Pharmacy / Non-Food / Food stock values

        Args:
            week_start:   Monday of the target week in YYYY-MM-DD format
            week_end:     Sunday of the target week in YYYY-MM-DD format
            prior_weeks:  Number of prior weeks used for averages (default 3, max 12)
        """
        return _adapter.get_weekly_dashboard(week_start, week_end, prior_weeks)
