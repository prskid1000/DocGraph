if __name__ == "__main__":
    # Imported lazily: multiprocessing's spawn re-runs this module in every
    # parse worker (as __mp_main__), and the CLI pulls in uvicorn, kuzu,
    # pyarrow ... that a tree-sitter worker never needs.
    from docgraph.cli import app

    app()
