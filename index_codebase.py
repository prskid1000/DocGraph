"""Main script for indexing a codebase using the new processor system."""
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

from src.processors import LanguageProcessorFactory
from src.processors.base import ScopedReference
from src.graph.builder import GraphBuilder
from src.embeddings.generator import EmbeddingGenerator
from src.utils.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_code_files(directory: Path, languages: List[str] = None) -> List[Path]:
    """Get all code files in directory.
    
    Args:
        directory: Root directory to search.
        languages: List of language names to include. If None, uses all supported.
        
    Returns:
        List of file paths.
    """
    code_files = []
    
    if languages is None:
        languages = LanguageProcessorFactory.get_supported_languages()
    
    # Get extensions from language config or use factory mapping
    language_configs = config.get_enabled_languages()
    extensions = set()
    
    for lang in languages:
        if lang in language_configs:
            lang_config = language_configs[lang]
            for ext in lang_config.get('extensions', []):
                extensions.add(ext)
        else:
            # Fallback: use factory's extension mapping
            ext_to_lang = {
                'python': ['.py'],
                'javascript': ['.js', '.jsx'],
                'typescript': ['.ts', '.tsx'],
                'java': ['.java'],
                'kotlin': ['.kt', '.kts'],
                'html': ['.html', '.htm'],
                'scss': ['.scss', '.sass', '.css'],
            }
            if lang in ext_to_lang:
                extensions.update(ext_to_lang[lang])
    
    for ext in extensions:
        code_files.extend(directory.rglob(f"*{ext}"))
    
    # Filter out common ignore patterns and ensure we only have actual files
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.idea', '.vscode'}
    filtered_files = []
    for file_path in code_files:
        # Skip if in ignored directory
        if any(ignore in file_path.parts for ignore in ignore_dirs):
            continue
        
        # CRITICAL: Only process actual files, not directories
        # Some systems may have directories with file extensions in their names
        if not file_path.is_file():
            logger.debug(f"Skipping non-file path: {file_path} (is_dir={file_path.is_dir()})")
            continue
        
        filtered_files.append(file_path)
    
    return filtered_files


def index_codebase(codebase_path: str, languages: List[str] = None,
                  codebase_id: Optional[str] = None, clear_existing: bool = False):
    """Index a codebase into the knowledge graph using the new processor system.
    
    Args:
        codebase_path: Path to the codebase root.
        languages: List of languages to index. Defaults to all enabled.
        codebase_id: Unique identifier for this codebase. Defaults to directory name.
        clear_existing: If True, clear existing data for this codebase before indexing.
    """
    codebase_dir = Path(codebase_path)
    if not codebase_dir.exists():
        logger.error(f"Codebase directory does not exist: {codebase_path}")
        return
    
    # Generate codebase_id from directory name if not provided
    if codebase_id is None:
        codebase_id = codebase_dir.name
    
    if languages is None:
        languages = list(config.get_enabled_languages().keys())
    
    logger.info(f"Indexing codebase: {codebase_path}")
    logger.info(f"Codebase ID: {codebase_id}")
    logger.info(f"Languages: {', '.join(languages)}")
    
    # Get code files
    code_files = get_code_files(codebase_dir, languages)
    logger.info(f"Found {len(code_files)} code files")
    
    # Initialize graph builder and embedding generator
    graph_builder = GraphBuilder(codebase_id=codebase_id)
    embedding_generator = EmbeddingGenerator(codebase_id=codebase_id)
    
    # Collect all entities and references by language
    all_entities = []
    all_references: List[ScopedReference] = []
    processors_by_lang = {}
    
    # Process files using language-specific processors
    logger.info("Processing files...")
    for file_path in code_files:
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
            
            if len(entities) > 0:
                logger.debug(f"Extracted {len(entities)} entities, {len(references)} references from {file_path}")
        
        except (IOError, PermissionError, OSError) as e:
            # File access errors - log but don't show full traceback
            logger.warning(f"Skipping file due to access error: {file_path} - {e}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    
    logger.info(f"Total entities extracted: {len(all_entities)}")
    logger.info(f"Total references extracted: {len(all_references)}")
    
    # Resolve references using language-specific resolvers
    logger.info("Resolving references...")
    resolved_references_by_type = defaultdict(list)
    
    # Group references by language and resolve
    references_by_lang = defaultdict(list)
    for ref in all_references:
        # Determine language from file path
        file_path = Path(ref.file_path)
        processor = LanguageProcessorFactory.get_processor_for_file(file_path)
        if processor:
            references_by_lang[processor.language].append(ref)
    
    # Resolve references for each language
    for language, lang_references in references_by_lang.items():
        processor = processors_by_lang.get(language)
        if processor:
            # Verify processor language matches
            if processor.language != language:
                logger.warning(f"Processor language mismatch: processor.language={processor.language}, expected={language}")
                continue
            
            # Create entity container for this language's entities
            from src.processors.entity_container import EntityContainer
            lang_entity_container = EntityContainer()
            
            # Filter entities by language based on file extension
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
            
            # Verify all references are for this language
            ref_file_exts = {Path(ref.file_path).suffix for ref in lang_references}
            invalid_refs = [ref for ref in lang_references if Path(ref.file_path).suffix not in lang_exts]
            if invalid_refs:
                logger.warning(f"Language {language} has {len(invalid_refs)} references from wrong language files: {ref_file_exts}")
            
            lang_entity_container.add_entities(lang_entities)
            
            # Create resolver and resolve
            resolver = processor.create_reference_resolver(lang_entity_container)
            logger.debug(f"Resolving {len(lang_references)} references for {language} using {resolver.__class__.__name__}")
            resolved = resolver.resolve_references(lang_references)
            
            # Merge resolved references
            for ref_type, refs in resolved.items():
                resolved_references_by_type[ref_type].extend(refs)
    
    logger.info(f"Resolved references: {sum(len(v) for v in resolved_references_by_type.values())}")
    
    # Build graph
    logger.info("Building knowledge graph...")
    graph_builder.add_entities(all_entities)
    
    # Convert ScopedReference to Reference format for graph builder
    from src.parsers.base import Reference
    resolved_references = []
    
    # Add resolved references (with target entities, or without for IMPORTS/INHERITS)
    for ref_type, refs in resolved_references_by_type.items():
        for scoped_ref, target_entity in refs:
            # Add all references, including those with None target_entity for IMPORTS/INHERITS
            # (they're handled specially by the graph builder)
            ref = Reference(
                from_entity=scoped_ref.from_entity,
                to_entity=scoped_ref.to_entity,
                reference_type=scoped_ref.reference_type,
                file_path=scoped_ref.file_path,
                line_number=scoped_ref.line_number,
                metadata=scoped_ref.metadata if hasattr(scoped_ref, 'metadata') else {}
            )
            resolved_references.append((ref, target_entity))
    
    graph_builder.add_references_from_resolved(resolved_references)
    graph_builder.build(clear_existing=clear_existing)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_generator.generate_embeddings(all_entities)
    
    logger.info("Indexing complete!")
    logger.info(f"Total entities: {len(all_entities)}")
    logger.info(f"Total references: {len(all_references)}")
    logger.info(f"Resolved references: {sum(len(v) for v in resolved_references_by_type.values())}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index a codebase into knowledge graph")
    parser.add_argument("codebase_path", help="Path to codebase root directory")
    parser.add_argument("--languages", nargs="+", help="Languages to index (default: all enabled)")
    parser.add_argument("--codebase-id", help="Unique identifier for this codebase (default: directory name)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data for this codebase before indexing")
    
    args = parser.parse_args()
    
    index_codebase(args.codebase_path, args.languages, args.codebase_id, args.clear)


if __name__ == "__main__":
    main()
