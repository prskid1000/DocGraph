"""Main script for indexing a codebase."""
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from src.parsers import ParserFactory
from src.extractors.entity_extractor import EntityExtractor
from src.extractors.reference_resolver import ReferenceResolver
from src.graph.builder import GraphBuilder
from src.embeddings.generator import EmbeddingGenerator
from src.utils.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_code_files(directory: Path, languages: List[str]) -> List[Path]:
    """Get all code files in directory.
    
    Args:
        directory: Root directory to search.
        languages: List of language names to include.
        
    Returns:
        List of file paths.
    """
    code_files = []
    language_configs = config.get_enabled_languages()
    
    extensions = set()
    for lang in languages:
        if lang in language_configs:
            lang_config = language_configs[lang]
            for ext in lang_config.get('extensions', []):
                extensions.add(ext)
    
    for ext in extensions:
        code_files.extend(directory.rglob(f"*{ext}"))
    
    # Filter out common ignore patterns
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}
    filtered_files = []
    for file_path in code_files:
        if not any(ignore in file_path.parts for ignore in ignore_dirs):
            filtered_files.append(file_path)
    
    return filtered_files


def index_codebase(codebase_path: str, languages: List[str] = None,
                  codebase_id: Optional[str] = None, clear_existing: bool = False):
    """Index a codebase into the knowledge graph.
    
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
    
    # Initialize components with codebase_id
    entity_extractor = EntityExtractor()
    graph_builder = GraphBuilder(codebase_id=codebase_id)
    embedding_generator = EmbeddingGenerator(codebase_id=codebase_id)
    
    # Process files
    entities_by_language = {}
    references_by_language = {}
    
    for file_path in code_files:
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
            # Create parser
            parser = ParserFactory.create_parser(language)
            
            # Extract entities and references
            entities, references = entity_extractor.extract_from_file(parser, file_path)
            
            if language not in entities_by_language:
                entities_by_language[language] = []
                references_by_language[language] = []
            
            entities_by_language[language].extend(entities)
            references_by_language[language].extend(references)
            
            if len(entities) > 0:
                logger.debug(f"Extracted {len(entities)} entities from {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Build graph (this will also resolve references)
    logger.info("Building knowledge graph...")
    all_entities = entity_extractor.get_all_entities()
    all_references = entity_extractor.get_all_references()
    
    graph_builder.add_entities(all_entities)
    graph_builder.add_references(all_references)
    graph_builder.build(clear_existing=clear_existing)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_generator.generate_embeddings(all_entities)
    
    logger.info("Indexing complete!")
    logger.info(f"Total entities: {len(all_entities)}")
    logger.info(f"Total references: {len(all_references)}")


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

