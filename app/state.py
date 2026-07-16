"""
app/state.py — Shared application state initialised at startup.

`state` is a module-level singleton populated by the lifespan in main.py.
Import it wherever you need the embedding model or Supabase client:

  from app.state import state
  state.embed_model.encode(...)
  state.supabase.table("patent_documents").select("*").execute()

Fields:
  embed_model  — Loaded SentenceTransformer (jinaai/jina-embeddings-v3, trust_remote_code=True).
                 Lazy-loaded on first access so the app can start and pass health checks before
                 the model download/load completes (important on memory-constrained hosts).
  supabase     — Supabase client (sync; wrap with asyncio.to_thread in async routes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AppState:
    _embed_model: Optional[object] = field(default=None, repr=False)
    supabase: Optional[object] = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            from app.config import EMBEDDING_MODEL
            from sentence_transformers import SentenceTransformer
            log.info("Lazy-loading embedding model %s…", EMBEDDING_MODEL)
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
            log.info("Embedding model loaded.")
        return self._embed_model

    @embed_model.setter
    def embed_model(self, value):
        self._embed_model = value


state = AppState()