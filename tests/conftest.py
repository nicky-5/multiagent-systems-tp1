"""pytest loads this file before any test module."""

import os

# Tests must not send traces to LangSmith. services/config.py loads .env with
# load_dotenv, which doesn't override variables that are already set.
os.environ["LANGSMITH_TRACING"] = "false"
