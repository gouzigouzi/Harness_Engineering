"""
Harness configuration.
Uses OpenAI-compatible API so it works with any provider.

Setup:
  cp .env.template .env   # then fill in your real values
"""
import os
from pathlib import Path


def _load_dotenv():
    """Load .env file if it exists. No third-party dependency needed."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # .env takes priority over shell env vars
        if key:
            os.environ[key] = value


_load_dotenv()

# --- API ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("HARNESS_MODEL", "gpt-4o")

# --- Token budgets ---
COMPRESS_THRESHOLD = int(os.environ.get("COMPRESS_THRESHOLD", "80000"))
RESET_THRESHOLD = int(os.environ.get("RESET_THRESHOLD", "150000"))

# --- Harness loop ---
MAX_HARNESS_ROUNDS = int(os.environ.get("MAX_HARNESS_ROUNDS", "5"))
PASS_THRESHOLD = float(os.environ.get("PASS_THRESHOLD", "7.0"))

# --- Agent limits ---
# NOTE: Do NOT use iteration count as the primary stop condition.
# With ~8-9s per iteration, 80 iterations = ~700s, which silently
# truncates 900s+ tasks. Use a high ceiling here; TimeBudgetMiddleware
# handles the real time-based stop.
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "500"))
MAX_TOOL_ERRORS = 5           # consecutive tool errors before abort

# --- Paths ---
WORKSPACE = os.path.abspath(os.environ.get("HARNESS_WORKSPACE", "./workspace"))
SPEC_FILE = "spec.md"
FEEDBACK_FILE = "feedback.md"
CONTRACT_FILE = "contract.md"
PROGRESS_FILE = "progress.md"
