# Chroma DB UI (Streamlit)

A production-grade Streamlit UI to manage ChromaDB collections used by DocGraph. Features include:

- Browse documents with pagination and metadata filters
- Add documents with metadata (embeddings auto-generated)
- Edit existing documents and re-embed
- Delete documents
- Semantic search by text (via Sentence-Transformers)
- Collection administration: clear (delete & recreate)
 - Import JSONL with optional auto-embedding
 - Export selection to JSONL via download
- **Collection Management**: List, create, delete, and clear collections

- Python 3.10+
- Dependencies installed from the project `requirements.txt`
- Environment variables (optional):
  - `CHROMADB_PERSIST_DIR` (default: `%USERPROFILE%/.docgraph/chromadb`)
  - `CHROMADB_COLLECTION` (default: `code_entities`)
  - `DOCGRAPH_CODEBASE` (default: `default`)

## Quick Start

```bash
# using a venv
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

# run the app
streamlit run chroma/app.py
```

## Usage Notes

- The UI uses the base collection name + codebase ID to resolve the actual collection, matching DocGraph's `VectorDB` behavior.
- Embeddings are generated using the configured Sentence-Transformer in `src/embeddings/models.py`.
- For large datasets, use the `Limit` and `Offset` fields to paginate through documents.
- Metadata fields accept JSON.
 - Import expects JSON Lines (one JSON object per line). Each object may include `id`, `document`, and `metadata`. Missing `id` will be auto-generated.

## Admin Panel Features

**Collection Statistics:**
- View document count in the current collection
- List all collections in the persist directory with their document counts

**Create Collections:**
- Create a new named collection (with cosine distance metric)

**Delete Collections:**
- Safely delete any collection with confirmation

**Clear Current Collection:**
- Delete all documents and recreate the collection
