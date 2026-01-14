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
    """Handle index_codebase background task using the new processor system.
    
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
    from src.processors import LanguageProcessorFactory
    from src.processors.base import ScopedReference
    from src.graph.builder import GraphBuilder
    from src.embeddings.generator import EmbeddingGenerator
    from collections import defaultdict
    
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
    graph_builder = GraphBuilder(codebase_id=codebase_id)
    embedding_generator = EmbeddingGenerator(codebase_id=codebase_id)
    
    # Collect all entities and references
    all_entities = []
    all_references: list[ScopedReference] = []
    processors_by_lang = {}
    processed_files = 0
    
    # Process files using language-specific processors
    for file_path in code_files:
        if is_cancelled():
            return {"success": False, "error": "Task was cancelled"}
        
        # Get processor for this file
        processor = LanguageProcessorFactory.get_processor_for_file(file_path)
        if not processor:
            continue
        
        language = processor.language
        if language not in processors_by_lang:
            processors_by_lang[language] = processor
        
        try:
            # Process file
            entities, references = processor.process_file(file_path)
            
            all_entities.extend(entities)
            all_references.extend(references)
            
            processed_files += 1
            
            # Update progress (0.3 to 0.6 for file processing)
            progress = 0.3 + (0.3 * (processed_files / total_files))
            if processed_files % 10 == 0 or processed_files == total_files:
                update_progress(progress, f"Processed {processed_files}/{total_files} files...")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    logger.info(f"Extracted {len(all_entities)} entities and {len(all_references)} references")
    
    # Resolve references using language-specific resolvers
    update_progress(0.65, "Resolving references...")
    resolved_references_by_type = defaultdict(list)
    
    # Group references by language and resolve
    references_by_lang = defaultdict(list)
    for ref in all_references:
        file_path = Path(ref.file_path)
        processor = LanguageProcessorFactory.get_processor_for_file(file_path)
        if processor:
            references_by_lang[processor.language].append(ref)
    
    # Resolve references for each language
    for language, lang_references in references_by_lang.items():
        processor = processors_by_lang.get(language)
        if processor:
            # Create entity container for this language's entities
            from src.processors.entity_container import EntityContainer
            lang_entity_container = EntityContainer()
            # Filter entities by language
            lang_extensions = {
                'python': ['.py'],
                'javascript': ['.js', '.jsx'],
                'typescript': ['.ts', '.tsx'],
                'java': ['.java'],
                'kotlin': ['.kt', '.kts'],
                'html': ['.html', '.htm'],
                'scss': ['.scss', '.sass', '.css'],
            }
            lang_exts = lang_extensions.get(language, [])
            lang_entities = [e for e in all_entities if Path(e.file_path).suffix in lang_exts]
            
            lang_entity_container.add_entities(lang_entities)
            
            # Create resolver and resolve
            resolver = processor.create_reference_resolver(lang_entity_container)
            resolved = resolver.resolve_references(lang_references)
            
            # Merge resolved references
            for ref_type, refs in resolved.items():
                resolved_references_by_type[ref_type].extend(refs)
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    # Build graph
    update_progress(0.75, "Building knowledge graph...")
    graph_builder.add_entities(all_entities)
    
    # Convert ScopedReference to Reference format for graph builder
    from src.parsers.base import Reference
    resolved_references = []
    for ref_type, refs in resolved_references_by_type.items():
        for scoped_ref, target_entity in refs:
            if target_entity:
                ref = Reference(
                    from_entity=scoped_ref.from_entity,
                    to_entity=scoped_ref.to_entity,
                    reference_type=scoped_ref.reference_type,
                    file_path=scoped_ref.file_path,
                    line_number=scoped_ref.line_number,
                    metadata=scoped_ref.metadata
                )
                resolved_references.append((ref, target_entity))
    
    graph_builder.add_references_from_resolved(resolved_references)
    graph_builder.build(clear_existing=clear_existing)
    
    if is_cancelled():
        return {"success": False, "error": "Task was cancelled"}
    
    # Generate embeddings
    update_progress(0.9, "Generating embeddings...")
    embedding_generator.generate_embeddings(all_entities)
    
    update_progress(1.0, "Indexing complete!")
    
    total_resolved = sum(len(v) for v in resolved_references_by_type.values())
    
    # Return summary
    return {
        "success": True,
        "codebase_id": codebase_id,
        "files_indexed": total_files,
        "entities_found": len(all_entities),
        "references_found": len(all_references),
        "references_resolved": total_resolved,
        "languages": list(processors_by_lang.keys()),
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
