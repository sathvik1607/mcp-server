"""
config.py — Central bootstrap for allpets_mcp.

Self-contained: all business logic lives inside this repo.
No dependency on allpets_new_schema folder at runtime.

Must be the first import in server.py and every adapter so that:
  1. Environment variables (DB creds, OpenAI key) are loaded before any module
  2. SCHEMA_FILE points to the local etc/secrets/schema_all.txt
  3. A single SQLAlchemy engine is shared across all adapters
"""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# ── 1. Locate this package's root ─────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ── 2. Environment variables ───────────────────────────────────────────────────
load_dotenv(ROOT / ".env")

DB_HOST     = os.environ["DB_HOST"]
DB_USER     = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME     = os.environ.get("DB_NAME", "cohort_main")
DB_PORT     = os.environ.get("DB_PORT", "3306")

# Point sql_dynamic_agent at the local schema file
os.environ.setdefault("SCHEMA_FILE", str(ROOT / "etc" / "secrets" / "schema_all.txt"))

# ── 3. Database engine singleton ───────────────────────────────────────────────
from sqlalchemy import create_engine

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_recycle=3600,
)
