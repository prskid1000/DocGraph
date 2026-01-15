"""Graph builder for constructing Neo4j knowledge graph."""
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import hashlib
import logging

from ..parsers.base import CodeEntity, Reference
from .schema import GraphSchema
from .queries import GraphQueries

logger = logging.getLogger(__name__)

# Neo4j index property size limit (conservative to avoid indexing errors)
MAX_INDEXED_PROPERTY_SIZE = 5000


def _truncate_signature(signature: Optional[str]) -> Optional[str]:
    """Truncate signature if it exceeds Neo4j index limits.
    
    Args:
        signature: Function/method signature.
        
    Returns:
        Truncated signature or original if within limits.
    """
    if not signature or len(signature) <= MAX_INDEXED_PROPERTY_SIZE:
        return signature
    return signature[:MAX_INDEXED_PROPERTY_SIZE] + "..."


class GraphBuilder:
    """Builds Neo4j knowledge graph from extracted entities and references."""
    
    def __init__(self, neo4j_client=None, codebase_id: Optional[str] = None):
        """Initialize graph builder.
        
        Args:
            neo4j_client: Neo4j client instance (will be created if None).
            codebase_id: Unique identifier for the codebase being indexed.
        """
        self.neo4j_client = neo4j_client
        self.codebase_id = codebase_id or "default"
        self.entities: List[CodeEntity] = []
        self.references: List[Reference] = []
        self.node_ids: Dict[str, str] = {}  # Map entity to node ID
        self.pending_nodes: List[Dict[str, Any]] = []
        self.pending_relationships: List[Dict[str, Any]] = []
    
    def add_entities(self, entities: List[CodeEntity]):
        """Add entities to be inserted into the graph.
        
        Args:
            entities: List of code entities.
        """
        self.entities.extend(entities)
    
    def add_references(self, references: List[Reference]):
        """Add references to be inserted into the graph.
        
        Args:
            references: List of references.
        """
        self.references.extend(references)
    
    def add_references_from_resolved(self, resolved_references: List[tuple]):
        """Add resolved references (reference, target_entity tuples).
        
        Args:
            resolved_references: List of (Reference, CodeEntity) tuples.
        """
        self.resolved_references = resolved_references
    
    def _generate_node_id(self, entity: CodeEntity) -> str:
        """Generate unique node ID for an entity.
        
        Args:
            entity: Code entity.
            
        Returns:
            Unique node ID string.
        """
        # Create unique ID based on entity properties, including column to handle multiple entities per line
        key = f"{entity.entity_type}:{entity.file_path}:{entity.name}:{entity.start_line}:{entity.start_column}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _entity_to_node(self, entity: CodeEntity) -> Dict[str, Any]:
        """Convert entity to Neo4j node properties.
        
        Args:
            entity: Code entity.
            
        Returns:
            Dictionary of node properties.
        """
        node_id = self._generate_node_id(entity)
        self.node_ids[f"{entity.entity_type}:{entity.name}:{entity.file_path}"] = node_id
        
        # Map entity type to Neo4j label
        label_map = {
            'class': GraphSchema.NODE_CLASS,
            'function': GraphSchema.NODE_FUNCTION,
            'variable': GraphSchema.NODE_VARIABLE,
            'module': GraphSchema.NODE_MODULE,
            'parameter': GraphSchema.NODE_PARAMETER,
            'type': GraphSchema.NODE_TYPE,
        }
        
        label = label_map.get(entity.entity_type, 'Entity')
        
        properties = {
            'id': node_id,
            'name': entity.name,
            'file_path': entity.file_path,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'start_column': entity.start_column,
            'end_column': entity.end_column,
            'codebase_id': self.codebase_id,
        }
        
        if entity.signature:
            properties['signature'] = _truncate_signature(entity.signature)
        if entity.docstring:
            properties['docstring'] = entity.docstring
        if entity.parent:
            properties['parent'] = entity.parent
        if entity.metadata:
            # Flatten metadata, but exclude complex structures that can't be stored in Neo4j
            for key, value in entity.metadata.items():
                if key in ['parameters', 'return_type']:
                    # Skip these - they're handled separately for relationships
                    continue
                elif isinstance(value, (str, int, float, bool, list)):
                    # Only store primitive types and lists of primitives
                    if isinstance(value, list) and value and not all(isinstance(item, (str, int, float, bool)) for item in value):
                        continue  # Skip lists with complex objects
                    properties[key] = value
                elif value is None:
                    continue
                # Skip dicts and other complex types
        
        return {
            'id': node_id,
            'label': label,
            'properties': properties
        }
    
    def _create_file_node(self, file_path: str, language: str) -> Dict[str, Any]:
        """Create a file node.
        
        Args:
            file_path: Path to the file.
            language: Programming language.
            
        Returns:
            File node dictionary.
        """
        path_obj = Path(file_path)
        file_id = hashlib.md5(file_path.encode()).hexdigest()
        
        return {
            'id': file_id,
            'label': GraphSchema.NODE_FILE,
            'properties': {
                'id': file_id,
                'path': file_path,
                'name': path_obj.name,
                'language': language,
                'size': path_obj.stat().st_size if path_obj.exists() else 0,
                'codebase_id': self.codebase_id,
            }
        }
    
    def _create_module_node(self, module_name: str) -> Dict[str, Any]:
        """Create a module node from import statement.
        
        Args:
            module_name: Module name/path.
            
        Returns:
            Module node dictionary.
        """
        module_id = hashlib.md5(f"module:{module_name}".encode()).hexdigest()
        
        return {
            'id': module_id,
            'label': GraphSchema.NODE_MODULE,
            'properties': {
                'id': module_id,
                'name': module_name.split('.')[-1],  # Last part of module path
                'path': module_name,
                'codebase_id': self.codebase_id,
            }
        }
    
    def _reference_to_relationship(self, ref: Reference, 
                                    resolved_target: Optional[CodeEntity] = None) -> Optional[Dict[str, Any]]:
        """Convert reference to Neo4j relationship.
        
        Args:
            ref: Reference object.
            resolved_target: Optional resolved target entity.
            
        Returns:
            Relationship dictionary or None if cannot be created.
        """
        # Map reference type to relationship type
        rel_map = {
            'calls': GraphSchema.REL_CALLS,
            'references': GraphSchema.REL_REFERENCES,
            'imports': GraphSchema.REL_IMPORTS,
            'inherits': GraphSchema.REL_INHERITS,
            'has_parameter': GraphSchema.REL_HAS_PARAMETER,
            'returns': GraphSchema.REL_RETURNS,
            'contains': GraphSchema.REL_CONTAINS,
        }
        
        rel_type = rel_map.get(ref.reference_type)
        if not rel_type:
            return None
        
        # Find source and target node IDs
        source_id = None
        target_id = None
        
        # Handle IMPORTS relationships (File -> Module/File)
        if ref.reference_type == 'imports':
            # Source is the file - find file node ID
            file_id = hashlib.md5(ref.file_path.encode()).hexdigest()
            # Normalize path for comparison (handle Windows/Unix differences)
            normalized_path = ref.file_path.replace('\\', '/')
            # Try different key formats
            file_key_variants = [
                f"file:{Path(ref.file_path).name}:{ref.file_path}",
                f"file:::{ref.file_path}",
                f"file:{Path(ref.file_path).name}:{normalized_path}",
                f"file:::{normalized_path}",
            ]
            source_id = None
            for key in file_key_variants:
                source_id = self.node_ids.get(key)
                if source_id:
                    break
            if not source_id:
                # Try to find by file path in any format (with normalization)
                # Also try matching just the filename
                file_name = Path(ref.file_path).name
                for key, node_id in self.node_ids.items():
                    if key.startswith('file:'):
                        # Normalize both paths for comparison
                        key_normalized = key.replace('\\', '/')
                        key_lower = key_normalized.lower()
                        search_path_lower = normalized_path.lower()
                        # Try multiple matching strategies
                        if (normalized_path in key_normalized or 
                            ref.file_path in key or
                            search_path_lower in key_lower or
                            (file_name in key and (normalized_path in key_normalized or ref.file_path in key))):
                            source_id = node_id
                            break
            if not source_id:
                source_id = file_id  # Use hash as fallback
            
            # Target is a Module node or File node (for relative imports)
            module_name = ref.to_entity
            
            # Handle relative imports (./base, ../module, etc.)
            if module_name.startswith('./') or module_name.startswith('../'):
                # For relative imports, try to resolve to a file
                source_file = Path(ref.file_path)
                try:
                    target_file = (source_file.parent / module_name).resolve()
                    target_file_str = str(target_file)
                    target_id = hashlib.md5(target_file_str.encode()).hexdigest()
                    
                    # Try to find existing file node (try multiple key formats and path normalizations)
                    target_file_key_variants = [
                        f"file:{target_file.name}:{target_file_str}",
                        f"file:::{target_file_str}",
                        f"file:{target_file.name}:{target_file_str.replace('\\', '/')}",
                        f"file:::{target_file_str.replace('\\', '/')}",
                    ]
                    found_target = False
                    for key in target_file_key_variants:
                        existing_id = self.node_ids.get(key)
                        if existing_id:
                            target_id = existing_id
                            found_target = True
                            break
                    
                    # If not found by exact match, try searching by path
                    if not found_target:
                        normalized_target = target_file_str.replace('\\', '/')
                        for key, node_id in self.node_ids.items():
                            if key.startswith('file:') and (normalized_target in key.replace('\\', '/') or target_file_str in key):
                                target_id = node_id
                                found_target = True
                                break
                    
                    # If still not found, use the hash (will be used to create/link the file node)
                    # The relationship will be created with this ID
                except (ValueError, OSError):
                    # If path resolution fails, fall back to module node
                    target_id = hashlib.md5(f"module:{module_name}".encode()).hexdigest()
            else:
                # Absolute import - create Module node
                module_id = hashlib.md5(f"module:{module_name}".encode()).hexdigest()
                module_key = f"module:{module_name.split('.')[-1]}::{module_name}"
                target_id = self.node_ids.get(module_key)
                # If not found, use the module_id (module node should be created)
                if not target_id:
                    target_id = module_id
        
        # Handle INHERITS relationships (Class -> Class)
        elif ref.reference_type == 'inherits':
            # Source is the class making the inheritance
            normalized_file_path = ref.file_path.replace('\\', '/')
            source_key = f"class:{ref.from_entity}:{ref.file_path}"
            source_id = self.node_ids.get(source_key)
            
            # If not found by exact match, try with normalized path
            if not source_id:
                source_key_normalized = f"class:{ref.from_entity}:{normalized_file_path}"
                source_id = self.node_ids.get(source_key_normalized)
            
            # If not found by exact match, try partial match
            if not source_id:
                for key, node_id in self.node_ids.items():
                    if key.startswith(f"class:{ref.from_entity}:"):
                        key_normalized = key.replace('\\', '/')
                        if normalized_file_path in key_normalized or ref.file_path in key:
                            source_id = node_id
                            break
            
            # Target is the base class (try resolved first, then lookup)
            if resolved_target:
                target_key = f"class:{resolved_target.name}:{resolved_target.file_path}"
                target_id = self.node_ids.get(target_key)
                # If not found, try partial match
                if not target_id:
                    for key, node_id in self.node_ids.items():
                        if key.startswith(f"class:{resolved_target.name}:") and resolved_target.file_path in key:
                            target_id = node_id
                            break
            else:
                # Try to find base class by name (search across all files)
                base_name = ref.to_entity
                # First try in same file
                target_key = f"class:{base_name}:{ref.file_path}"
                target_id = self.node_ids.get(target_key)
                # If not found, try with normalized path
                if not target_id:
                    normalized_path = ref.file_path.replace('\\', '/')
                    for key, node_id in self.node_ids.items():
                        if key.startswith(f"class:{base_name}:") and normalized_path in key.replace('\\', '/'):
                            target_id = node_id
                            break
                # If still not found, search across all files (any file)
                if not target_id:
                    for key, node_id in self.node_ids.items():
                        if key.startswith(f"class:{base_name}:"):
                            target_id = node_id
                            break
        
        # Handle CONTAINS relationships (File -> File)
        elif ref.reference_type == 'contains':
            # Source is the file containing the reference
            # Try multiple key formats to find the file node
            normalized_path = ref.file_path.replace('\\', '/')
            file_key_variants = [
                f"file:{Path(ref.file_path).name}:{ref.file_path}",
                f"file:::{ref.file_path}",
                f"file:{Path(ref.file_path).name}:{normalized_path}",
                f"file:::{normalized_path}",
            ]
            source_id = None
            for key in file_key_variants:
                source_id = self.node_ids.get(key)
                if source_id:
                    break
            
            # If not found by exact match, try searching by path
            if not source_id:
                file_name = Path(ref.file_path).name
                for key, node_id in self.node_ids.items():
                    if key.startswith('file:'):
                        key_normalized = key.replace('\\', '/')
                        if (normalized_path in key_normalized or 
                            ref.file_path in key or
                            (file_name in key and (normalized_path in key_normalized or ref.file_path in key))):
                            source_id = node_id
                            break
            
            if not source_id:
                # Fallback to hash (file node should exist)
                source_id = hashlib.md5(ref.file_path.encode()).hexdigest()
            
            # Target is the referenced file (resolve relative path)
            source_file = Path(ref.file_path)
            target_file = (source_file.parent / ref.to_entity).resolve()
            target_file_str = str(target_file)
            target_id = hashlib.md5(target_file_str.encode()).hexdigest()
            
            # Try multiple key formats to find existing file node
            target_key_variants = [
                f"file:{target_file.name}:{target_file_str}",
                f"file:{target_file.name}:{ref.to_entity}",  # Try relative path
            ]
            
            # Check if target file node exists in node_ids
            found_target = False
            for key in target_key_variants:
                if key in self.node_ids:
                    target_id = self.node_ids[key]
                    found_target = True
                    break
            
            # Also search by file name in case path format differs
            if not found_target:
                for key, node_id in self.node_ids.items():
                    if key.startswith(f"file:{target_file.name}:") and target_file_str in key:
                        target_id = node_id
                        found_target = True
                        break
            
            # Check if it's in pending nodes
            if not found_target:
                target_exists = any(n.get('id') == target_id for n in self.pending_nodes)
                if not target_exists:
                    # Create a File node for the referenced file (even if it doesn't exist)
                    # This allows CONTAINS relationships to work for external file references
                    target_file_node = self._create_file_node(target_file_str, target_file.suffix.lstrip('.'))
                    target_file_node['id'] = target_id  # Use the computed ID
                    target_file_node['properties']['id'] = target_id
                    # Make sure codebase_id is set
                    target_file_node['properties']['codebase_id'] = self.codebase_id
                    self.pending_nodes.append(target_file_node)
                    # Add to node_ids with all key variants
                    for key in target_key_variants:
                        self.node_ids[key] = target_id
                else:
                    # Find the existing pending node and add to node_ids
                    for node in self.pending_nodes:
                        if node.get('id') == target_id:
                            for key in target_key_variants:
                                self.node_ids[key] = target_id
                            break
        
        # Handle CALLS and REFERENCES relationships
        else:
            # Source: find entity making the reference
            source_id = None
            if ref.from_entity:
                # For function calls/references, try to find the enclosing function/class
                # Try multiple entity types (including mixin for SCSS)
                for entity_type in ['function', 'class', 'variable', 'mixin']:
                    source_key = f"{entity_type}:{ref.from_entity}:{ref.file_path}"
                    source_id = self.node_ids.get(source_key)
                    if source_id:
                        break
                
                # If not found by exact match, try partial match (in case of name variations)
                if not source_id:
                    normalized_file_path = ref.file_path.replace('\\', '/')
                    for key, node_id in self.node_ids.items():
                        if (key.startswith(f"function:{ref.from_entity}:") or 
                            key.startswith(f"class:{ref.from_entity}:") or
                            key.startswith(f"variable:{ref.from_entity}:") or
                            key.startswith(f"mixin:{ref.from_entity}:")):
                            key_normalized = key.replace('\\', '/')
                            if (normalized_file_path in key_normalized or 
                                ref.file_path in key or
                                (ref.file_path in key_normalized)):
                                source_id = node_id
                                break
                
                # If source is a file path (for top-level references), use file as source
                if not source_id and ref.from_entity == ref.file_path:
                    source_id = hashlib.md5(ref.file_path.encode()).hexdigest()
            
            # Target: use resolved entity
            target_id = None
            if resolved_target:
                target_key = f"{resolved_target.entity_type}:{resolved_target.name}:{resolved_target.file_path}"
                target_id = self.node_ids.get(target_key)
            else:
                # For unresolved references, try to find by name
                target_name = ref.to_entity
                
                if ref.reference_type == 'calls':
                    # For CALLS, search for functions across all files
                    for key, node_id in self.node_ids.items():
                        if key.startswith(f"function:{target_name}:"):
                            target_id = node_id
                            break
                elif ref.reference_type == 'references':
                    # For REFERENCES, try to find variables or classes in same file first, then across files
                    for key, node_id in self.node_ids.items():
                        if (key.startswith(f"variable:{target_name}:") or 
                            key.startswith(f"class:{target_name}:")):
                            # Prefer same file
                            if ref.file_path in key:
                                target_id = node_id
                                break
                            # But also accept from other files if not found
                            elif not target_id:
                                target_id = node_id
        
        # Both source and target must be found to create a relationship
        if not source_id or not target_id:
            return None
        
        properties = {
            'line_number': ref.line_number,
            'file_path': ref.file_path,
        }
        if ref.metadata:
            properties.update(ref.metadata)
        
        return {
            'from_id': source_id,
            'to_id': target_id,
            'type': rel_type,
            'properties': properties
        }
    
    def build(self, batch_size: int = 1000, clear_existing: bool = False):
        """Build the graph by inserting nodes and relationships.
        
        Args:
            batch_size: Number of nodes/relationships to insert per batch (default: 1000).
            clear_existing: If True, delete existing nodes for this codebase first.
        
        Note:
            - Indexes are created once (the INFO logs you see are just notifications)
            - Nodes are inserted in batches of `batch_size` using UNWIND
            - Relationships are inserted in batches of `batch_size` using UNWIND
            - This prevents memory issues and improves performance vs. single inserts
        """
        if not self.neo4j_client:
            from ..storage.neo4j_client import Neo4jClient
            self.neo4j_client = Neo4jClient()
        
        # Clear existing data for this codebase if requested
        if clear_existing:
            self._clear_codebase_data()
        
        # Create indexes (only once - these are the INFO logs you're seeing)
        # They're just notifications that indexes already exist, not actual data creation
        self._create_indexes()
        
        # Group entities by file
        entities_by_file = defaultdict(list)
        for entity in self.entities:
            entities_by_file[entity.file_path].append(entity)
        
        # Create file nodes and entity nodes (collected in memory first)
        logger.info(f"Preparing {len(self.entities)} entities for batch insertion...")
        for file_path, file_entities in entities_by_file.items():
            # Determine language from file extension
            language = Path(file_path).suffix[1:] if Path(file_path).suffix else 'unknown'
            
            # Create file node
            file_node = self._create_file_node(file_path, language)
            self.pending_nodes.append(file_node)
            # Store file node ID in node_ids for CONTAINS relationships
            file_key = f"file:{Path(file_path).name}:{file_path}"
            self.node_ids[file_key] = file_node['id']
            
            # Create entity nodes
            for entity in file_entities:
                node = self._entity_to_node(entity)
                self.pending_nodes.append(node)
                
                # Create DEFINES relationship (File -> Entity)
                self.pending_relationships.append({
                    'from_id': file_node['id'],
                    'to_id': node['id'],
                    'type': GraphSchema.REL_DEFINES,
                    'properties': {}
                })
                
                # Create HAS_PARAMETER relationships (Function -> Parameter)
                if entity.entity_type == 'function' and entity.metadata and 'parameters' in entity.metadata:
                    for param_data in entity.metadata['parameters']:
                        if isinstance(param_data, dict) and 'name' in param_data:
                            param_entity = CodeEntity(
                                name=param_data['name'],
                                entity_type='parameter',
                                file_path=entity.file_path,
                                start_line=entity.start_line,
                                end_line=entity.end_line,
                                start_column=entity.start_column,
                                end_column=entity.end_column,
                                metadata={'type': param_data.get('type'), 'default_value': param_data.get('default')}
                            )
                            param_node = self._entity_to_node(param_entity)
                            self.pending_nodes.append(param_node)
                            self.pending_relationships.append({
                                'from_id': node['id'],
                                'to_id': param_node['id'],
                                'type': GraphSchema.REL_HAS_PARAMETER,
                                'properties': {'position': param_data.get('position', 0)}
                            })
                
                # Create RETURNS relationship (Function -> Type)
                if entity.entity_type == 'function' and entity.metadata and 'return_type' in entity.metadata:
                    return_type = entity.metadata['return_type']
                    if return_type:
                        type_entity = CodeEntity(
                            name=return_type.split('.')[-1] if '.' in return_type else return_type,
                            entity_type='type',
                            file_path=entity.file_path,
                            start_line=entity.start_line,
                            end_line=entity.end_line,
                            start_column=entity.start_column,
                            end_column=entity.end_column,
                            metadata={'full_name': return_type}
                        )
                        type_node = self._entity_to_node(type_entity)
                        self.pending_nodes.append(type_node)
                        self.pending_relationships.append({
                            'from_id': node['id'],
                            'to_id': type_node['id'],
                            'type': GraphSchema.REL_RETURNS,
                            'properties': {}
                        })
        
        # First, process references to create Module nodes for IMPORTS
        # This must happen before node insertion so modules are included
        if hasattr(self, 'resolved_references') and self.resolved_references:
            # Collect all import references to create Module nodes
            import_modules = set()
            for ref, target in self.resolved_references:
                if ref.reference_type == 'imports':
                    import_modules.add(ref.to_entity)
            
            # Create Module nodes for imports
            for module_name in import_modules:
                module_id = hashlib.md5(f"module:{module_name}".encode()).hexdigest()
                module_key = f"module:{module_name.split('.')[-1]}::{module_name}"
                # Check if module node already exists
                if module_key not in self.node_ids:
                    module_node = self._create_module_node(module_name)
                    self.pending_nodes.append(module_node)
                    self.node_ids[module_key] = module_id
        
        # Batch insert nodes (processes in chunks of batch_size)
        logger.info(f"Inserting {len(self.pending_nodes)} nodes in batches of {batch_size}...")
        self._batch_insert_nodes(batch_size)
        
        # IMPORTANT: After nodes are inserted, we need to rebuild node_ids mapping
        # because node IDs are now in Neo4j and we need to query them
        self._rebuild_node_ids_after_insert()
        
        # Create relationships from resolved references
        if hasattr(self, 'resolved_references') and self.resolved_references:
            logger.info(f"Creating relationships from {len(self.resolved_references)} resolved references...")
            total_resolved = 0
            imports_count = 0
            inherits_count = 0
            calls_count = 0
            refs_count = 0
            has_param_count = 0
            returns_count = 0
            contains_count = 0
            failed_imports = 0
            failed_inherits = 0
            
            for ref, target in self.resolved_references:
                rel = self._reference_to_relationship(ref, target)
                if rel:
                    self.pending_relationships.append(rel)
                    total_resolved += 1
                    # Count by type for logging
                    if rel['type'] == GraphSchema.REL_IMPORTS:
                        imports_count += 1
                    elif rel['type'] == GraphSchema.REL_INHERITS:
                        inherits_count += 1
                    elif rel['type'] == GraphSchema.REL_CALLS:
                        calls_count += 1
                    elif rel['type'] == GraphSchema.REL_REFERENCES:
                        refs_count += 1
                    elif rel['type'] == GraphSchema.REL_HAS_PARAMETER:
                        has_param_count += 1
                    elif rel['type'] == GraphSchema.REL_RETURNS:
                        returns_count += 1
                    elif rel['type'] == GraphSchema.REL_CONTAINS:
                        contains_count += 1
                else:
                    # Log failed relationship creation for debugging
                    if ref.reference_type == 'imports':
                        failed_imports += 1
                        logger.debug(f"Failed to create IMPORTS relationship: {ref.from_entity} -> {ref.to_entity} (file: {ref.file_path})")
                    elif ref.reference_type == 'inherits':
                        failed_inherits += 1
                        logger.debug(f"Failed to create INHERITS relationship: {ref.from_entity} -> {ref.to_entity} (file: {ref.file_path})")
            
            if failed_imports > 0:
                logger.warning(f"Failed to create {failed_imports} IMPORTS relationships")
            if failed_inherits > 0:
                logger.warning(f"Failed to create {failed_inherits} INHERITS relationships")
            
            # Count HAS_PARAMETER and RETURNS from pending_relationships (created during entity processing)
            for rel in self.pending_relationships:
                if rel['type'] == GraphSchema.REL_HAS_PARAMETER:
                    has_param_count += 1
                elif rel['type'] == GraphSchema.REL_RETURNS:
                    returns_count += 1
                elif rel['type'] == GraphSchema.REL_CONTAINS:
                    contains_count += 1
            
            logger.info(f"Created {len(self.pending_relationships)} relationships: {calls_count} CALLS, {refs_count} REFERENCES, {imports_count} IMPORTS, {inherits_count} INHERITS, {has_param_count} HAS_PARAMETER, {returns_count} RETURNS, {contains_count} CONTAINS")
        else:
            logger.warning("No resolved references provided. Relationships will not be created.")
        
        # Batch insert relationships (processes in chunks of batch_size)
        logger.info(f"Inserting {len(self.pending_relationships)} relationships in batches of {batch_size}...")
        self._batch_insert_relationships(batch_size)
    
    def _create_indexes(self):
        """Create indexes in Neo4j."""
        queries = GraphSchema.get_create_index_queries()
        for query in queries:
            try:
                self.neo4j_client.execute_query(query)
            except Exception as e:
                # Index might already exist
                pass
    
    def _batch_insert_nodes(self, batch_size: int):
        """Insert nodes in batches.
        
        Args:
            batch_size: Number of nodes per batch.
        """
        total_nodes = len(self.pending_nodes)
        if total_nodes == 0:
            return
        
        try:
            for i in range(0, total_nodes, batch_size):
                batch = self.pending_nodes[i:i + batch_size]
                self._insert_node_batch(batch)
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Inserted {min(i + batch_size, total_nodes)}/{total_nodes} nodes...")
        except KeyboardInterrupt:
            logger.warning(f"Node insertion interrupted. Processed {i}/{total_nodes} nodes.")
            raise
    
    def _insert_node_batch(self, nodes: List[Dict[str, Any]]):
        """Insert a batch of nodes.
        
        Args:
            nodes: List of node dictionaries.
        """
        if not nodes:
            return
        
        query = """
        UNWIND $nodes AS node
        MERGE (n {id: node.id})
        SET n = node
        SET n:`%s`
        """
        
        # Group by label
        nodes_by_label = defaultdict(list)
        for node in nodes:
            nodes_by_label[node['label']].append(node)
        
        for label, label_nodes in nodes_by_label.items():
            formatted_query = query % label
            try:
                # Use write transaction for better performance and error handling
                self.neo4j_client.execute_write(
                    formatted_query,
                    parameters={'nodes': [n['properties'] for n in label_nodes]}
                )
            except KeyboardInterrupt:
                logger.warning(f"Node batch insert interrupted. Processed {len(nodes) - len(label_nodes)} nodes so far.")
                raise
            except Exception as e:
                logger.error(f"Error inserting {len(label_nodes)} nodes with label {label}: {e}")
                raise
    
    def _batch_insert_relationships(self, batch_size: int):
        """Insert relationships in batches.
        
        Args:
            batch_size: Number of relationships per batch.
        """
        total_rels = len(self.pending_relationships)
        if total_rels == 0:
            return
        
        try:
            for i in range(0, total_rels, batch_size):
                batch = self.pending_relationships[i:i + batch_size]
                self._insert_relationship_batch(batch)
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Inserted {min(i + batch_size, total_rels)}/{total_rels} relationships...")
        except KeyboardInterrupt:
            logger.warning(f"Relationship insertion interrupted. Processed {i}/{total_rels} relationships.")
            raise
    
    def _insert_relationship_batch(self, relationships: List[Dict[str, Any]]):
        """Insert a batch of relationships.
        
        Args:
            relationships: List of relationship dictionaries.
        """
        if not relationships:
            return
        
        query = """
        UNWIND $rels AS rel
        MATCH (from {id: rel.from_id})
        MATCH (to {id: rel.to_id})
        MERGE (from)-[r:`%s`]->(to)
        SET r = rel.properties
        """
        
        # Group by relationship type
        rels_by_type = defaultdict(list)
        for rel in relationships:
            rels_by_type[rel['type']].append(rel)
        
        for rel_type, type_rels in rels_by_type.items():
            formatted_query = query % rel_type
            try:
                # Use write transaction for better performance and error handling
                self.neo4j_client.execute_write(
                    formatted_query,
                    parameters={'rels': [
                        {
                            'from_id': r['from_id'],
                            'to_id': r['to_id'],
                            'properties': r['properties']
                        }
                        for r in type_rels
                    ]}
                )
            except KeyboardInterrupt:
                logger.warning(f"Relationship batch insert interrupted. Processed {len(relationships) - len(type_rels)} relationships so far.")
                raise
            except Exception as e:
                logger.error(f"Error inserting {len(type_rels)} relationships of type {rel_type}: {e}")
                raise
    
    def _clear_codebase_data(self):
        """Clear all data for the current codebase."""
        query = """
        MATCH (n {codebase_id: $codebase_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        result = self.neo4j_client.execute_query(query, {'codebase_id': self.codebase_id})
        if result:
            logger.info(f"Cleared {result[0]['deleted']} nodes for codebase: {self.codebase_id}")
    
    def _rebuild_node_ids_after_insert(self):
        """Rebuild node_ids mapping after nodes are inserted into Neo4j."""
        # Query all nodes for this codebase to rebuild the mapping
        query = """
        MATCH (n {codebase_id: $codebase_id})
        RETURN n.id as id, labels(n)[0] as label, 
               COALESCE(n.name, '') as name, 
               COALESCE(n.file_path, n.path, '') as file_path,
               COALESCE(n.path, n.file_path, '') as path
        """
        results = self.neo4j_client.execute_query(query, {'codebase_id': self.codebase_id})
        
        for record in results:
            node_id = record['id']
            label = record['label']
            name = record.get('name', '')
            file_path = record.get('file_path', '') or record.get('path', '')
            path = record.get('path', '') or record.get('file_path', '')
            
            # Map entity type from label
            entity_type_map = {
                'Class': 'class',
                'Function': 'function',
                'Variable': 'variable',
                'File': 'file',
                'Module': 'module',
                'Parameter': 'parameter',
                'Type': 'type',
            }
            entity_type = entity_type_map.get(label, label.lower())
            
            # Rebuild the key - File nodes use path, others use file_path
            if label == 'File':
                # File nodes: use path and name
                key = f"file:{Path(path).name if path else name}:{path}"
            else:
                # Entity nodes: use entity_type, name, file_path
                key = f"{entity_type}:{name}:{file_path}"
            
            self.node_ids[key] = node_id
    
    def clear(self):
        """Clear pending nodes and relationships."""
        self.pending_nodes.clear()
        self.pending_relationships.clear()
        self.node_ids.clear()

