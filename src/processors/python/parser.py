"""Python parser using libcst."""
import libcst as cst
from libcst.metadata import PositionProvider, ParentNodeProvider

from ...parsers.base import CodeEntity


class PythonParser:
    """Python parser using libcst."""
    
    def parse(self, source_code: str) -> cst.Module:
        """Parse Python source code.
        
        Args:
            source_code: Python source code string.
            
        Returns:
            libcst Module object.
        """
        # Create metadata wrapper for position and parent information
        self.metadata_wrapper = cst.metadata.MetadataWrapper(
            cst.parse_module(source_code),
            cache={
                PositionProvider: PositionProvider(),
                ParentNodeProvider: ParentNodeProvider()
            }
        )
        
        return self.metadata_wrapper.module
    
    def get_node_code(self, node: cst.CSTNode) -> str:
        """Get source code for a node."""
        return self.metadata_wrapper.module.code_for_node(node)
    
    def get_position(self, node: cst.CSTNode):
        """Get position information for a node."""
        return self.metadata_wrapper.resolve(PositionProvider)[node]
    
    def get_parent(self, node: cst.CSTNode):
        """Get parent node for a node."""
        parent_metadata = self.metadata_wrapper.resolve(ParentNodeProvider)
        return parent_metadata.get(node)
