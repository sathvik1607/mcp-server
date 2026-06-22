"""
Adapter: nl2sql_agent.run_agent

Wraps run_agent() from allpets_new_schema.
env vars (OPENAI_API_KEY, DB_*) are loaded by config before this module is imported.
"""
import config  # MUST be first — sets sys.path + loads env

from nl2sql_agent import run_agent as _run_agent


def run_agent(question: str) -> dict:
    try:
        result = _run_agent(question)
        result.pop("query_results", None)
        result.pop("final_output",  None)
        result.pop("query_plan",    None)
        return result
    except Exception as exc:
        # Catch anything the pipeline didn't handle internally.
        # Return a safe message — never expose tracebacks or internal details.
        return {
            "analysis": None,
            "insights": None,
            "error":    "An unexpected error occurred. Please try again later.",
        }
