"""Task handler registration for the task management system."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from _mcp.logger import app_logger as logger
from _mcp.service.task.task_manager import get_task_queue
from _mcp.service.task.task_utils import update_progress, is_cancelled


def index_codebase_handler(codebase_path: str, languages: list = None, 
                          codebase_id: str = None, clear_existing: bool = False) -> Dict[str, Any]:
    """Handle index_codebase background task.
    
    Args:
        codebase_path: Path to codebase root
        languages: List of languages to index
        codebase_id: Unique identifier for codebase
        clear_existing: Whether to clear existing data
        
    Returns:
        Task result dictionary
    """
    from index_codebase import get_code_files
    from src.utils.config import config
    from src.parsers import ParserFactory
    from src.extractors.entity_extractor import EntityExtractor
    from src.extractors.reference_resolver import ReferenceResolver
    from src.graph.builder import GraphBuilder
    from src.embeddings.generator import EmbeddingGenerator
    
    logger.info(f"Starting codebase indexing: {codebase_path}")
    update_progress(0.1, "Initializing...")
    
    codebase_dir = Path(codebase_path)
    if not codebase_dir.exists():
        raise ValueError(f"Codebase directory does not exist: {codebase_path}")
    
    # Generate codebase_id from directory name if not provided
    if codebase_id is None:
        codebase_id = codebase_dir.name
    
    if languages is None:
        languages = list(config.get_enabled_languages().keys())
    
    update_progress(0.2, f"Scanning codebase for {len(languages)} languages...")
    
    # Get code files
    code_files = get_code_files(codebase_dir, languages)
    total_files = len(code_files)
    logger.info(f"Found {total_files} code files to process")
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    update_progress(0.3, f"Processing {total_files} files...")
    
    # Initialize components
    entity_extractor = EntityExtractor()
    graph_builder = GraphBuilder(codebase_id=codebase_id)
    embedding_generator = EmbeddingGenerator(codebase_id=codebase_id)
    
    # Process files
    entities_by_language = {}
    references_by_language = {}
    processed_files = 0
    
    for file_path in code_files:
        if is_cancelled():
            return {"success": False, "error": "Task was cancelled"}
        
        # Determine language from extension
        ext = file_path.suffix
        language = None
        for lang, lang_config in config.get_enabled_languages().items():
            if ext in lang_config.get('extensions', []):
                language = lang
                break
        
        if not language:
            continue
        
        try:
            parser = ParserFactory.create_parser(language)
            entities, references = entity_extractor.extract_from_file(parser, file_path)
            
            if language not in entities_by_language:
                entities_by_language[language] = []
                references_by_language[language] = []
            
            entities_by_language[language].extend(entities)
            references_by_language[language].extend(references)
            
            processed_files += 1
            
            # Update progress (0.3 to 0.6 for file processing)
            progress = 0.3 + (0.3 * (processed_files / total_files))
            if processed_files % 10 == 0 or processed_files == total_files:
                update_progress(progress, f"Processed {processed_files}/{total_files} files...")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    # Resolve references
    update_progress(0.65, "Resolving references...")
    reference_resolver = ReferenceResolver(entity_extractor)
    reference_resolver.resolve_references()
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    # Build graph
    update_progress(0.75, "Building knowledge graph...")
    all_entities = entity_extractor.get_all_entities()
    all_references = entity_extractor.get_all_references()
    
    graph_builder.add_entities(all_entities)
    graph_builder.add_references(all_references)
    graph_builder.build(clear_existing=clear_existing)
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    # Generate embeddings
    update_progress(0.9, "Generating embeddings...")
    embedding_generator.generate_embeddings(all_entities)
    
    update_progress(1.0, "Indexing complete!")
    
    # Return summary
    return {
        "success": True,
        "codebase_id": codebase_id,
        "files_indexed": total_files,
        "entities_found": len(all_entities),
        "references_found": len(all_references),
        "languages": list(entities_by_language.keys()),
    }


def register_task_handlers():
    """Register all task handlers with the task queue."""
    queue = get_task_queue()
    
    # Register index_codebase task
    queue.register_handler(
        "INDEX_CODEBASE",
        index_codebase_handler,
        description="Index a codebase into the knowledge graph with entity extraction and embedding generation",
        params_schema={
            "codebase_path": {"type": "string", "description": "Path to the codebase root directory"},
            "languages": {"type": "array", "items": {"type": "string"}, "description": "List of languages to index"},
            "codebase_id": {"type": "string", "description": "Unique identifier for this codebase"},
            "clear_existing": {"type": "boolean", "description": "Clear existing data before indexing"}
        }
    )
    
    logger.info("✅ Task handlers registered successfully")
