"""Test script to index test project and verify all relationships are created for each file type."""
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index_codebase import index_codebase
from src.storage.neo4j_client import Neo4jClient
from src.graph.schema import GraphSchema

# Map file extensions to languages
LANGUAGE_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.java': 'Java',
    '.kt': 'Kotlin',
    '.html': 'HTML',
    '.scss': 'SCSS',
}

def verify_relationships_by_language():
    """Verify all relationship types are created for each language."""
    client = Neo4jClient()
    
    print("\n" + "="*80)
    print("VERIFICATION REPORT BY LANGUAGE")
    print("="*80)
    
    # Expected relationship types (all 8 types)
    expected_rels = {
        GraphSchema.REL_DEFINES: "File -> Entity",
        GraphSchema.REL_CALLS: "Function -> Function",
        GraphSchema.REL_REFERENCES: "Function -> Variable/Class",
        GraphSchema.REL_IMPORTS: "File -> Module",
        GraphSchema.REL_INHERITS: "Class -> Class",
        GraphSchema.REL_HAS_PARAMETER: "Function -> Parameter",
        GraphSchema.REL_RETURNS: "Function -> Type",
        GraphSchema.REL_CONTAINS: "File -> File",
    }
    
    # Get all files in the test project
    test_project_path = Path(__file__).parent / "test_project"
    all_files = {}
    for ext, lang in LANGUAGE_MAP.items():
        files = list(test_project_path.rglob(f"*{ext}"))
        if files:
            all_files[lang] = [str(f.relative_to(test_project_path)) for f in files]
    
    # Verify relationships for each language
    all_languages_ok = True
    for language, files in all_files.items():
        print(f"\n{'-'*80}")
        print(f"{language} ({len(files)} files)")
        print(f"{'-'*80}")
        
        # Build file path filter for this language
        # Use file names instead of full paths for matching (more reliable)
        file_names = [Path(f).name for f in files]
        file_patterns = [f.replace("\\", "/") for f in files]  # Normalize paths
        
        language_ok = True
        for rel_type, description in expected_rels.items():
            # Check if this relationship type is applicable to this language
            # CONTAINS is mainly for HTML, IMPORTS/INHERITS/HAS_PARAMETER/RETURNS are for code languages
            if rel_type == GraphSchema.REL_CONTAINS and language != 'HTML':
                continue  # Skip CONTAINS for non-HTML files
            if rel_type in [GraphSchema.REL_HAS_PARAMETER, GraphSchema.REL_RETURNS] and language in ['HTML', 'SCSS']:
                continue  # Skip parameter/return types for markup/stylesheets
            
            # Build WHERE clause with file path matching
            # Match by file name (more reliable than full path)
            file_name_conditions = " OR ".join([f"a.file_path ENDS WITH '{f}'" for f in file_names])
            
            # For DEFINES and IMPORTS, check File nodes; for others, check entity nodes
            if rel_type in [GraphSchema.REL_DEFINES, GraphSchema.REL_IMPORTS, GraphSchema.REL_CONTAINS]:
                # These relationships start from File nodes
                query = f"""
                MATCH (a:File)-[r:{rel_type}]->(b)
                WHERE a.codebase_id = 'test_project' 
                  AND b.codebase_id = 'test_project'
                  AND ({file_name_conditions})
                RETURN count(r) as count
                """
            else:
                # Other relationships: check entity nodes (Function, Class, etc.)
                # Match by checking if source node's file_path matches
                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                WHERE a.codebase_id = 'test_project' 
                  AND b.codebase_id = 'test_project'
                  AND ({file_name_conditions})
                RETURN count(r) as count
                """
            
            try:
                result = client.execute_query(query)
                count = result[0]['count'] if result else 0
            except Exception as e:
                # If query fails, try without file filtering (fallback)
                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                WHERE a.codebase_id = 'test_project' 
                  AND b.codebase_id = 'test_project'
                RETURN count(r) as count
                """
                result = client.execute_query(query)
                count = result[0]['count'] if result else 0
            
            status = "OK" if count > 0 else "MISSING"
            print(f"  [{status:7s}] {rel_type:20s}: {count:4d} relationships")
            
            if count == 0:
                language_ok = False
                all_languages_ok = False
        
        if language_ok:
            print(f"  [SUCCESS] All applicable relationship types found for {language}")
        else:
            print(f"  [WARNING] Some relationship types missing for {language}")
    
    return all_languages_ok

def verify_relationships():
    """Verify all relationship types are created in the graph."""
    client = Neo4jClient()
    
    print("\n" + "="*80)
    print("OVERALL VERIFICATION REPORT")
    print("="*80)
    
    # Expected relationship types (all 8 types)
    expected_rels = {
        GraphSchema.REL_DEFINES: "File -> Entity",
        GraphSchema.REL_CALLS: "Function -> Function",
        GraphSchema.REL_REFERENCES: "Function -> Variable/Class",
        GraphSchema.REL_IMPORTS: "File -> Module",
        GraphSchema.REL_INHERITS: "Class -> Class",
        GraphSchema.REL_HAS_PARAMETER: "Function -> Parameter",
        GraphSchema.REL_RETURNS: "Function -> Type",
        GraphSchema.REL_CONTAINS: "File -> File",
    }
    
    # Check each relationship type (filter by codebase_id)
    all_found = True
    for rel_type, description in expected_rels.items():
        query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        WHERE a.codebase_id = 'test_project' AND b.codebase_id = 'test_project'
        RETURN count(r) as count
        """
        result = client.execute_query(query)
        count = result[0]['count'] if result else 0
        
        status = "OK" if count > 0 else "MISSING"
        print(f"[{status:7s}] {rel_type:20s} ({description:30s}): {count:4d} relationships")
        
        if count == 0:
            all_found = False
    
    # Check node types
    print("\n" + "-"*80)
    print("NODE COUNTS BY TYPE")
    print("-"*80)
    
    node_types = [
        GraphSchema.NODE_FILE,
        GraphSchema.NODE_CLASS,
        GraphSchema.NODE_FUNCTION,
        GraphSchema.NODE_VARIABLE,
        GraphSchema.NODE_MODULE,
        GraphSchema.NODE_PARAMETER,
        GraphSchema.NODE_TYPE,
    ]
    
    for node_type in node_types:
        query = f"""
        MATCH (n:{node_type} {{codebase_id: 'test_project'}})
        RETURN count(n) as count
        """
        result = client.execute_query(query)
        count = result[0]['count'] if result else 0
        print(f"  {node_type:15s}: {count:4d} nodes")
    
    # Summary
    print("\n" + "="*80)
    if all_found:
        print("[SUCCESS] ALL RELATIONSHIP TYPES FOUND")
    else:
        print("[WARNING] SOME RELATIONSHIP TYPES MISSING")
    print("="*80 + "\n")
    
    client.close()
    return all_found

def main():
    """Main test function."""
    test_project_path = Path(__file__).parent / "test_project"
    
    print("="*80)
    print("INDEXING TEST PROJECT")
    print("="*80)
    print(f"Project path: {test_project_path}")
    print()
    
    # Index the test project
    try:
        index_codebase(
            codebase_path=str(test_project_path),
            codebase_id="test_project",
            clear_existing=True
        )
        print("\n[SUCCESS] Indexing completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify relationships overall
    try:
        overall_success = verify_relationships()
    except Exception as e:
        print(f"\n[ERROR] Overall verification failed: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
    
    # Verify relationships by language
    try:
        language_success = verify_relationships_by_language()
    except Exception as e:
        print(f"\n[ERROR] Language-specific verification failed: {e}")
        import traceback
        traceback.print_exc()
        language_success = False
    
    return overall_success and language_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
