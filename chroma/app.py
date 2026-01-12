import os
import sys
import json
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import streamlit as st

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports from the project
from src.storage.vector_db import VectorDB
from src.embeddings.models import EmbeddingModel
from src.utils.config import config

# -------- Utility & State -------- #

def get_client_config() -> Dict[str, Any]:
    chroma_cfg = config.get_chromadb_config()
    return {
        "persist_directory": chroma_cfg["persist_directory"],
        "default_collection": chroma_cfg["collection_name"],
    }

@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()

@st.cache_resource
def get_vector_db(collection_base: str, codebase_id: str) -> VectorDB:
    return VectorDB(collection_name=collection_base, codebase_id=codebase_id,
                    persist_directory=get_client_config()["persist_directory"]) 

def parse_metadata(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.error("Invalid metadata JSON")
        return {}

# -------- Sidebar / Navigation -------- #

def sidebar() -> Dict[str, Any]:
    cfg = get_client_config()

    if "collection_base" not in st.session_state:
        st.session_state.collection_base = cfg["default_collection"]
    if "codebase_id" not in st.session_state:
        st.session_state.codebase_id = os.getenv("DOCGRAPH_CODEBASE", "default")

    st.sidebar.title("Chroma DB Manager")
    st.sidebar.caption(f"Persist dir: {cfg['persist_directory']}")

    # Fetch all collections to extract bases and codebase IDs
    try:
        vdb_temp = VectorDB(collection_name=cfg["default_collection"], codebase_id="default")
        all_collections = [c.name for c in vdb_temp.client.list_collections()]
        
        # Extract unique collection bases and codebase IDs from collection names
        # Pattern: {collection_base}_{codebase_id}
        collection_bases = set()
        codebase_ids = set()
        for col_name in all_collections:
            parts = col_name.rsplit("_", 1)  # Split from the right on last underscore
            if len(parts) == 2:
                collection_bases.add(parts[0])
                codebase_ids.add(parts[1])
        
        collection_bases = sorted(collection_bases) or [cfg["default_collection"]]
        codebase_ids = sorted(codebase_ids) or ["default"]
    except:
        collection_bases = [cfg["default_collection"]]
        codebase_ids = ["default"]

    # Use selectbox (dropdown) instead of text_input
    selected_base = st.sidebar.selectbox(
        "Collection base",
        collection_bases,
        index=collection_bases.index(st.session_state.collection_base) if st.session_state.collection_base in collection_bases else 0,
        key="collection_base"
    )
    
    selected_codebase = st.sidebar.selectbox(
        "Codebase ID",
        codebase_ids,
        index=codebase_ids.index(st.session_state.codebase_id) if st.session_state.codebase_id in codebase_ids else 0,
        key="codebase_id"
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Browse Documents",
            "Add Document",
            "Edit Document",
            "Delete Document",
            "Search",
            "Import / Export",
            "Admin: Collection",
        ],
        index=0,
    )
    return {
        "page": page,
        "collection_base": selected_base,
        "codebase_id": selected_codebase,
    }

# -------- Pages -------- #

def page_overview(collection_base: str, codebase_id: str):
    st.header("Overview")
    vdb = get_vector_db(collection_base, codebase_id)
    st.metric("Embeddings in collection", vdb.count())

    st.subheader("Active Collection")
    st.write({
        "base": collection_base,
        "codebase_id": codebase_id,
        "resolved_name": vdb.collection_name,
        "persist_directory": vdb.persist_directory,
    })

    st.info("Use the sidebar to switch base name and codebase ID.")


def page_browse(collection_base: str, codebase_id: str):
    st.header("Browse Documents")
    vdb = get_vector_db(collection_base, codebase_id)

    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=50, step=1)
    with col2:
        offset = st.number_input("Offset", min_value=0, value=0, step=1)

    where_text = st.text_area("Metadata filter (JSON)", value="")
    where = parse_metadata(where_text)

    try:
        res = vdb.get(where=where or None, limit=limit, offset=offset)
        ids = res.get("ids", [])
        docs = res.get("documents", []) or [None] * len(ids)
        metas = res.get("metadatas", []) or [{}] * len(ids)

        rows = []
        for i, _id in enumerate(ids):
            rows.append({
                "id": _id,
                "document": (docs[i] if i < len(docs) else None),
                "metadata": json.dumps(metas[i] if i < len(metas) else {}),
            })
        st.dataframe(rows, use_container_width=True)
        st.caption(f"Total shown: {len(rows)}")
    except Exception as e:
        st.error(f"Error fetching documents: {e}")


def page_add_document(collection_base: str, codebase_id: str):
    st.header("Add Document")
    vdb = get_vector_db(collection_base, codebase_id)
    model = get_embedding_model()

    with st.form("add_doc_form"):
        doc_id = st.text_input("ID (leave blank for auto)")
        document_text = st.text_area("Document text", height=200)
        metadata_text = st.text_area("Metadata (JSON)", value="{}", height=120)
        submitted = st.form_submit_button("Add")

    if submitted:
        if not document_text.strip():
            st.warning("Document text is required")
            return
        if not doc_id.strip():
            doc_id = str(uuid.uuid4())
        metadata = parse_metadata(metadata_text)
        try:
            embedding = model.encode([document_text])[0]
            vdb.add_embeddings(ids=[doc_id], embeddings=[embedding], metadatas=[metadata], documents=[document_text])
            st.success(f"Added document {doc_id}")
        except Exception as e:
            st.error(f"Error adding document: {e}")


def page_edit_document(collection_base: str, codebase_id: str):
    st.header("Edit Document")
    vdb = get_vector_db(collection_base, codebase_id)
    model = get_embedding_model()

    target_id = st.text_input("Document ID to edit")

    if target_id:
        try:
            res = vdb.get(ids=[target_id])
            ids = res.get("ids", [])
            if not ids:
                st.warning("No document found for that ID")
                return
            doc = (res.get("documents") or [None])[0]
            meta = (res.get("metadatas") or [{}])[0]

            with st.form("edit_doc_form"):
                new_text = st.text_area("Document text", value=doc or "", height=200)
                new_meta_text = st.text_area("Metadata (JSON)", value=json.dumps(meta), height=120)
                submitted = st.form_submit_button("Update")

            if submitted:
                new_meta = parse_metadata(new_meta_text)
                try:
                    new_embedding = model.encode([new_text])[0]
                    vdb.update(ids=[target_id], embeddings=[new_embedding], metadatas=[new_meta], documents=[new_text])
                    st.success("Document updated")
                except Exception as e:
                    st.error(f"Error updating document: {e}")
        except Exception as e:
            st.error(f"Error loading document: {e}")


def page_delete_document(collection_base: str, codebase_id: str):
    st.header("Delete Document")
    vdb = get_vector_db(collection_base, codebase_id)

    target_id = st.text_input("Document ID to delete")
    confirm = st.checkbox("Confirm deletion")
    if st.button("Delete", disabled=not confirm):
        try:
            vdb.delete(ids=[target_id])
            st.success("Document deleted")
        except Exception as e:
            st.error(f"Error deleting document: {e}")


def page_search(collection_base: str, codebase_id: str):
    st.header("Search")
    vdb = get_vector_db(collection_base, codebase_id)
    model = get_embedding_model()

    query_text = st.text_input("Query text")
    n_results = st.number_input("Top K", min_value=1, max_value=50, value=10)

    if st.button("Search"):
        if not query_text.strip():
            st.warning("Enter text to search")
            return
        try:
            q_emb = model.encode([query_text])
            res = vdb.query(query_embeddings=q_emb, n_results=int(n_results))
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            rows = []
            for i in range(len(ids)):
                rows.append({
                    "id": ids[i],
                    "distance": dists[i] if i < len(dists) else None,
                    "document": docs[i] if i < len(docs) else None,
                    "metadata": json.dumps(metas[i] if i < len(metas) else {}),
                })
            st.dataframe(rows, use_container_width=True)
            st.caption(f"Results: {len(rows)}")
        except Exception as e:
            st.error(f"Error performing search: {e}")


def page_import_export(collection_base: str, codebase_id: str):
    st.header("Import / Export")
    vdb = get_vector_db(collection_base, codebase_id)
    model = get_embedding_model()

    st.subheader("Import JSONL")
    st.caption("Each line: {id, document, metadata}. If id missing, auto-generated.")
    up = st.file_uploader("Upload JSONL file", type=["jsonl", "txt", "json"])
    auto_embed = st.checkbox("Generate embeddings on import", value=True)
    batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=64)
    if st.button("Import"):
        if up is None:
            st.warning("Please upload a file")
        else:
            try:
                content = up.read().decode("utf-8")
                lines = [l for l in content.splitlines() if l.strip()]
                rows: List[Dict[str, Any]] = []
                for ln in lines:
                    obj = json.loads(ln)
                    rid = obj.get("id") or str(uuid.uuid4())
                    doc = obj.get("document", "")
                    meta = obj.get("metadata", {})
                    rows.append({"id": rid, "document": doc, "metadata": meta})

                # Embed if requested
                if auto_embed:
                    texts = [r["document"] for r in rows]
                    embeddings: List[List[float]] = []
                    for i in range(0, len(texts), int(batch_size)):
                        chunk = texts[i:i+int(batch_size)]
                        embeddings.extend(model.encode(chunk))
                else:
                    embeddings = [[0.0]] * len(rows)

                vdb.add_embeddings(
                    ids=[r["id"] for r in rows],
                    embeddings=embeddings,
                    metadatas=[r["metadata"] for r in rows],
                    documents=[r["document"] for r in rows],
                )
                st.success(f"Imported {len(rows)} documents")
            except Exception as e:
                st.error(f"Import failed: {e}")

    st.subheader("Export JSONL")
    limit = st.number_input("Limit", min_value=1, max_value=5000, value=1000)
    where_text = st.text_area("Metadata filter (JSON)", value="")
    where = parse_metadata(where_text)
    if st.button("Export"):
        try:
            res = vdb.get(where=where or None, limit=limit)
            ids = res.get("ids", [])
            docs = res.get("documents", []) or [None] * len(ids)
            metas = res.get("metadatas", []) or [{}] * len(ids)
            out_lines = []
            for i, _id in enumerate(ids):
                out_lines.append(json.dumps({
                    "id": _id,
                    "document": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                }))
            blob = "\n".join(out_lines)
            st.download_button(
                label="Download export.jsonl",
                data=blob,
                file_name="export.jsonl",
                mime="application/json",
            )
            st.success(f"Prepared {len(out_lines)} records for download")
        except Exception as e:
            st.error(f"Export failed: {e}")


def page_admin_collection(collection_base: str, codebase_id: str):
    st.header("Admin: Collection")
    vdb = get_vector_db(collection_base, codebase_id)

    st.write("**Resolved collection name:**", vdb.collection_name)
    st.write("**Persist directory:**", vdb.persist_directory)

    st.subheader("Collection Statistics")
    st.metric("Documents in current collection", vdb.count())

    st.subheader("All Collections")
    try:
        all_cols = vdb.client.list_collections()
        if all_cols:
            cols_data = []
            for c in all_cols:
                try:
                    cols_data.append({"name": c.name, "count": c.count()})
                except:
                    cols_data.append({"name": c.name, "count": "N/A"})
            st.dataframe(cols_data, use_container_width=True)
        else:
            st.info("No collections found")
    except Exception as e:
        st.error(f"Error listing collections: {e}")

    st.subheader("Create New Collection")
    new_col_name = st.text_input("New collection name")
    if st.button("Create"):
        if not new_col_name.strip():
            st.warning("Enter a collection name")
        else:
            try:
                vdb.client.create_collection(name=new_col_name, metadata={"hnsw:space": "cosine"})
                st.success(f"Created collection: {new_col_name}")
            except Exception as e:
                st.error(f"Error creating collection: {e}")

    st.subheader("Delete Collection")
    col_to_delete = st.text_input("Collection name to delete")
    confirm_delete = st.checkbox("Confirm deletion")
    if st.button("Delete Collection", disabled=not confirm_delete):
        if not col_to_delete.strip():
            st.warning("Enter a collection name")
        else:
            try:
                vdb.client.delete_collection(name=col_to_delete)
                st.success(f"Deleted collection: {col_to_delete}")
            except Exception as e:
                st.error(f"Error deleting collection: {e}")

    st.subheader("Danger Zone")
    if st.button("Clear current collection (delete & recreate)"):
        try:
            vdb.clear()
            st.success("Collection cleared and recreated.")
        except Exception as e:
            st.error(f"Error clearing collection: {e}")

# -------- Main -------- #

def main():
    st.set_page_config(page_title="DocGraph Chroma UI", layout="wide")
    nav = sidebar()
    page = nav["page"]
    base = nav["collection_base"]
    codebase = nav["codebase_id"]

    if page == "Overview":
        page_overview(base, codebase)
    elif page == "Browse Documents":
        page_browse(base, codebase)
    elif page == "Add Document":
        page_add_document(base, codebase)
    elif page == "Edit Document":
        page_edit_document(base, codebase)
    elif page == "Delete Document":
        page_delete_document(base, codebase)
    elif page == "Search":
        page_search(base, codebase)
    elif page == "Import / Export":
        page_import_export(base, codebase)
    elif page == "Admin: Collection":
        page_admin_collection(base, codebase)


if __name__ == "__main__":
    main()
