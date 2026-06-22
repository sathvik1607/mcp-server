"""
Adapter: dashboard_queries.DashboardService

Wraps DashboardService.run_weekly() and returns a plain dict.
No business logic here — delegation only.
"""
import config  # MUST be first — sets sys.path + loads env

from dashboard_queries import DashboardService

_service = DashboardService(config.engine)


def get_weekly_dashboard(week_start: str, week_end: str, prior_weeks: int = 3) -> dict:
    result = _service.run_weekly(week_start, week_end, prior_weeks)
    return result.model_dump()
