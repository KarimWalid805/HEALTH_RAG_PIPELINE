"""Shared Neon and embedding settings for ingestion and RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
DOCS_DIR = PIPELINE_DIR / "docs"

# Load the Neon-linked variables first; preserve any provider keys in the
# pipeline-local .env without allowing it to replace the Neon connection URL.
load_dotenv(PROJECT_ROOT / ".env.local")
load_dotenv(PIPELINE_DIR / ".env", override=False)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
COLLECTION_NAME = os.getenv("NEON_VECTOR_COLLECTION", "ai_health_documents_v3")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "1000")))
NEON_WRITE_BATCH_SIZE = max(1, int(os.getenv("NEON_WRITE_BATCH_SIZE", "500")))

def get_embeddings() -> OpenAIEmbeddings:
    """Return configured OpenAIEmbeddings with batching enabled."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )
def get_neon_connection_url() -> str:
    """Return the direct Neon URL in psycopg 3 SQLAlchemy URL format."""
    # This is a long-running local process, and PGVector initializes schema
    # objects on connect; prefer Neon’s direct URL for that session-sensitive work.
    url = os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is missing. Link the Neon project and pull its variables "
            "into the project-root .env.local file."
        )

    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + url[len("postgresql+psycopg2://"):]
    return url

