"""Settings read from the environment, backed by the repo's .env file."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parent.parent

# Put .env into os.environ (without overriding variables already set). Besides feeding
# Settings below, this is what turns LangSmith tracing on: LangChain reads the
# LANGSMITH_* variables straight from the environment.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    """Each field is filled from the environment variable of the same name
    (case-insensitive). Fields without a default are required."""

    litellm_base_url: str
    litellm_api_key: SecretStr  # SecretStr prints as '**********', so it can't leak into logs
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # vector size of embedding_model

    pinecone_api_key: SecretStr
    pinecone_index_name: str = "mas-2026s2-tables"
    # Where a new index is created (the free plan supports aws / us-east-1)
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # How many times the ingestion agent asks the LLM for a table's metadata before giving up
    ingest_max_attempts: int = 3
    # How many tables the ingestion agent processes in parallel
    ingest_max_concurrency: int = 4

    # Analyst: results per Pinecone search, and how many tables the LLM may choose from
    analyst_top_k: int = 5
    analyst_max_candidates: int = 8

    # Sandbox for the analyst's generated code: wall-clock limit and memory (address space)
    sandbox_timeout_s: float = 30
    sandbox_memory_mb: int = 1024

    data_dir: Path = ROOT / "data"

    @property
    def openai_base_url(self) -> str:
        """The proxy's OpenAI-compatible URL, with exactly one /v1 suffix."""
        root = self.litellm_base_url.rstrip("/")
        return root if root.endswith("/v1") else f"{root}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  (values come from the environment)
