"""Tests for the document + asset indexing pass.

Covers:
- Text-tier docs (.md/.txt/.rst) become Doc nodes with file-shaped sources.
- Asset-tier files become Asset nodes with size + ext + mime.
- REFERENCES_ edges resolve from code string literals AND markdown link
  syntax to matching Asset nodes.
- CSV size gate: small CSVs land in tier 2; oversize CSVs in tier 3.
- Idempotent re-run: a second index pass leaves the same row counts.
- Off by default: cfg.index_documents=False produces no Doc/Asset/edge.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer


def _materialize(root: Path) -> None:
    """Build a tiny repo with one Python file, one markdown doc, one PDF
    (placeholder bytes) referenced from both, plus a CSV and an image."""
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / "src" / "loader.py").write_text(
        '"""Loader module."""\n'
        "from pathlib import Path\n\n"
        "DATA = Path('data/sales.xlsx')\n"
        "REPORT = 'docs/report.pdf'\n\n"
        "def load():\n"
        "    \"\"\"Read the report.\"\"\"\n"
        "    return open(REPORT, 'rb').read()\n",
        encoding="utf-8",
    )
    (root / "docs" / "intro.md").write_text(
        "# Intro\n\n"
        "See the [report](report.pdf) for details. "
        "Architecture diagram: ![arch](diagrams/arch.png).\n",
        encoding="utf-8",
    )
    (root / "docs" / "diagrams").mkdir()
    (root / "docs" / "diagrams" / "arch.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 32)
    (root / "docs" / "report.pdf").write_bytes(b"%PDF-1.4\n" + b"\x00" * 64)
    (root / "data" / "sales.xlsx").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    (root / "data" / "tiny.csv").write_text("name,age\nalice,30\nbob,25\n", encoding="utf-8")


@pytest.fixture(scope="module")
def doc_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("docgraph_documents")
    _materialize(root)
    return root


def _index(cfg, *, documents: bool):
    cfg.index_documents = documents
    writer = GraphDB(cfg.db_path, embedding_dim=384)
    writer.init_schema()
    embedder = Embedder(cfg.embedding_model)
    indexer = Indexer(cfg, writer, embedder=embedder)
    stats = indexer.index_all(incremental=False)
    indexer.db.close()
    writer.close()
    del writer, indexer
    gc.collect()
    reader = GraphDB(cfg.db_path, read_only=True)
    return reader, stats


def test_off_by_default(doc_repo: Path):
    """No documents flag → no Doc / Asset rows. Code pass should still work."""
    cfg = load_config(doc_repo)
    reader, stats = _index(cfg, documents=False)
    assert stats["errors"] == 0
    assert reader.fetch_all("MATCH (a:Asset) RETURN count(a) AS c")[0]["c"] == 0
    # URL Doc rows could exist from prior tests; this fixture has none.
    docs = reader.fetch_all(
        "MATCH (d:Doc) WHERE NOT d.source STARTS WITH 'http' RETURN count(d) AS c"
    )
    assert docs[0]["c"] == 0
    reader.close()


def test_documents_pass_creates_assets(doc_repo: Path):
    cfg = load_config(doc_repo)
    reader, stats = _index(cfg, documents=True)
    assert stats["errors"] == 0
    rows = reader.fetch_all("MATCH (a:Asset) RETURN a.path AS path, a.ext AS ext, a.size AS size")
    paths = {r["path"]: r for r in rows}
    # Three asset-tier files in the fixture
    assert "data/sales.xlsx" in paths
    assert "docs/report.pdf" in paths
    assert "docs/diagrams/arch.png" in paths
    assert paths["data/sales.xlsx"]["ext"] == "xlsx"
    assert paths["docs/report.pdf"]["size"] > 0
    reader.close()


def test_documents_pass_creates_text_docs(doc_repo: Path):
    cfg = load_config(doc_repo)
    reader, stats = _index(cfg, documents=True)
    assert stats["errors"] == 0
    docs = reader.fetch_all(
        "MATCH (d:Doc) WHERE NOT d.source STARTS WITH 'http' "
        "RETURN d.source AS source, d.title AS title"
    )
    sources = {d["source"] for d in docs}
    assert "docs/intro.md" in sources
    assert "data/tiny.csv" in sources
    titles = {d["source"]: d["title"] for d in docs}
    assert titles["docs/intro.md"] == "Intro"
    reader.close()


def test_references_resolve_from_code_and_markdown(doc_repo: Path):
    cfg = load_config(doc_repo)
    reader, stats = _index(cfg, documents=True)
    assert stats["errors"] == 0
    # Code → Asset edges (loader.py mentions sales.xlsx and report.pdf)
    code_refs = reader.fetch_all(
        "MATCH (f:File)-[:REFERENCES_]->(a:Asset) "
        "WHERE f.path = 'src/loader.py' "
        "RETURN a.path AS path"
    )
    code_paths = {r["path"] for r in code_refs}
    assert "data/sales.xlsx" in code_paths
    assert "docs/report.pdf" in code_paths

    # Markdown → Asset edges (intro.md mentions report.pdf and diagrams/arch.png)
    doc_refs = reader.fetch_all(
        "MATCH (d:Doc)-[:REFERENCES_]->(a:Asset) "
        "WHERE d.source = 'docs/intro.md' "
        "RETURN DISTINCT a.path AS path"
    )
    doc_paths = {r["path"] for r in doc_refs}
    assert "docs/report.pdf" in doc_paths
    # Relative path 'diagrams/arch.png' resolves via basename match
    assert "docs/diagrams/arch.png" in doc_paths
    reader.close()


def test_idempotent_rerun(doc_repo: Path):
    """A second pass should leave the same row counts (drop+rebuild)."""
    cfg = load_config(doc_repo)
    cfg.index_documents = True
    writer = GraphDB(cfg.db_path, embedding_dim=384)
    writer.init_schema()
    embedder = Embedder(cfg.embedding_model)
    Indexer(cfg, writer, embedder=embedder).index_all(incremental=False)
    writer.close()
    gc.collect()

    writer2 = GraphDB(cfg.db_path, embedding_dim=384)
    Indexer(cfg, writer2, embedder=embedder).index_all(incremental=False)
    writer2.close()
    gc.collect()

    reader = GraphDB(cfg.db_path, read_only=True)
    a = reader.fetch_all("MATCH (a:Asset) RETURN count(a) AS c")[0]["c"]
    d = reader.fetch_all(
        "MATCH (d:Doc) WHERE NOT d.source STARTS WITH 'http' RETURN count(d) AS c"
    )[0]["c"]
    assert a == 3  # three asset files
    assert d >= 2  # md + csv (intro.md may chunk into multiple)
    reader.close()


def test_csv_size_gate(tmp_path: Path):
    """Big CSVs route to the asset tier instead of being embedded as text."""
    root = tmp_path / "csv_repo"
    root.mkdir()
    (root / "small.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    # Build a CSV >1 MiB so it crosses the size gate
    big_lines = ["col1,col2"] + [f"row{i},{i}" for i in range(80_000)]
    (root / "big.csv").write_text("\n".join(big_lines), encoding="utf-8")

    cfg = load_config(root)
    cfg.index_documents = True
    # csv has to be in the asset list for the gate's overflow to land somewhere
    cfg.asset_extensions = ("csv", *cfg.asset_extensions)

    writer = GraphDB(cfg.db_path, embedding_dim=384)
    writer.init_schema()
    embedder = Embedder(cfg.embedding_model)
    Indexer(cfg, writer, embedder=embedder).index_all(incremental=False)
    writer.close()
    gc.collect()

    reader = GraphDB(cfg.db_path, read_only=True)
    docs = reader.fetch_all(
        "MATCH (d:Doc) WHERE NOT d.source STARTS WITH 'http' "
        "RETURN d.source AS source"
    )
    assets = reader.fetch_all("MATCH (a:Asset) RETURN a.path AS path")
    doc_sources = {d["source"] for d in docs}
    asset_paths = {a["path"] for a in assets}
    assert "small.csv" in doc_sources
    assert "big.csv" in asset_paths
    assert "big.csv" not in doc_sources
    reader.close()
