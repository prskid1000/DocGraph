"""HTML reference resolver."""
from ..base import BaseReferenceResolver, ScopedReference
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from ...parsers.base import CodeEntity

logger = logging.getLogger(__name__)


class HTMLReferenceResolver(BaseReferenceResolver):
    """HTML reference resolver - resolves file paths."""
    
    def __init__(self, entity_container):
        """Initialize HTML reference resolver."""
        super().__init__(entity_container)
    
    def resolve_references(
        self, 
        references: List[ScopedReference]
    ) -> Dict[str, List[Tuple[ScopedReference, Optional[CodeEntity]]]]:
        """Resolve HTML references (mostly file paths)."""
        resolved = defaultdict(list)
        
        # HTML references are mostly file paths - we can resolve them by checking if files exist
        for ref in references:
            # For HTML, references are typically file paths
            # We could resolve them by checking if the referenced file exists
            # For now, we'll mark them as resolved if they're valid paths
            resolved[ref.reference_type].append((ref, None))  # No entity for file references
        
        self.resolved_references = resolved
        return resolved
