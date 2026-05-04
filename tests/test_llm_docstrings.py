"""Tests for LLM-augmented docstring persistence and retrieval."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.index import Indexer
from docgraph.retrieve import Retriever
from docgraph.embed import Embedder

@pytest.fixture
def repo_with_functions(tmp_path: Path) -> Path:
    repo = tmp_path / "llm_repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "def hello_world():\n"
        "    return 'hello'\n"
        "\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n",
        encoding="utf-8"
    )
    return repo

def test_llm_docstring_persistence(repo_with_functions: Path):
    """Verify that llm_doc is stored in the DB and returned by retrieval."""
    cfg = load_config(repo_with_functions, llm_docstrings=True, llm_model="test-model")
    
    # Mock LLMClient.summarize to return a specific string
    with patch("docgraph.llm.LLMClient.summarize") as mock_summarize:
        mock_summarize.return_value = "This is an AI summary."
        
        db = GraphDB(cfg.db_path, embedding_dim=384)
        db.init_schema()
        embedder = Embedder(cfg.embedding_model)
        indexer = Indexer(cfg, db, embedder=embedder)
        
        indexer.index_all(incremental=False)
        
        # Use indexer.db as the original db object was closed and replaced during index_all(incremental=False)
        db = indexer.db
        
        # Verify it's in the DB
        rows = db.fetch_all("MATCH (n:Function) WHERE n.name = 'hello_world' RETURN n.llm_doc AS doc")
        assert rows
        assert rows[0]["doc"] == "This is an AI summary."
        
        rows = db.fetch_all("MATCH (n:Class) WHERE n.name = 'MyClass' RETURN n.llm_doc AS doc")
        assert rows
        assert rows[0]["doc"] == "This is an AI summary."

        # Verify Retriever returns it
        retriever = Retriever(db, embedder, cfg)
        
        # 1. via search()
        results = retriever.search("hello")
        match = next(r for r in results if r["name"] == "hello_world")
        assert match["llm_doc"] == "This is an AI summary."
        
        # 2. via definition()
        defn = retriever.definition("hello_world")
        assert defn[0]["llm_doc"] == "This is an AI summary."
        
        db.close()

