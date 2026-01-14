"""Java reference resolver with package resolution."""
from ..javascript.reference_resolver import JavaScriptReferenceResolver
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ..base import ScopedReference
from ...parsers.base import CodeEntity


class JavaReferenceResolver(JavaScriptReferenceResolver):
    """Java-specific reference resolver with package support."""
    
    def _build_lookup_structures(self):
        """Build lookup structures including package paths."""
        super()._build_lookup_structures()
        entities = self.entity_container.get_all_entities()
        
        self.by_package: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        for entity in entities:
            # Extract package from file path
            package = self._extract_package_from_path(entity.file_path)
            if package:
                self.by_package[package].append(entity)
                # Also index by package.class
                if entity.entity_type == 'class':
                    self.by_qualified_name[f"{package}.{entity.name}"].append(entity)
    
    def _extract_package_from_path(self, file_path: str) -> Optional[str]:
        """Extract Java package from file path."""
        # Simple heuristic - look for java/ or src/ directory
        parts = file_path.replace('\\', '/').split('/')
        if 'java' in parts:
            idx = parts.index('java')
            package_parts = parts[idx+1:-1]  # Exclude filename
            return '.'.join(package_parts) if package_parts else None
        elif 'src' in parts:
            idx = parts.index('src')
            package_parts = parts[idx+1:-1]
            return '.'.join(package_parts) if package_parts else None
        return None
    
    def _resolve_reference(self, ref: ScopedReference) -> Optional[CodeEntity]:
        """Resolve Java reference with package support."""
        # Try package-qualified name first
        if '.' in ref.to_entity and ref.reference_type == 'imports':
            # This is likely a package import
            candidates = self.by_package.get(ref.to_entity, [])
            if candidates:
                return candidates[0]
        
        return super()._resolve_reference(ref)
