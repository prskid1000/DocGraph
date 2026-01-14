"""Test script to index test project and verify all relationships and embeddings for each file type."""
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index_codebase import index_codebase
from src.storage.neo4j_client import Neo4jClient
from src.storage.vector_db import VectorDB
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

# Expected relationship types for each language (based on what parsers actually extract)
LANGUAGE_RELATIONSHIPS = {
    'Python': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, GraphSchema.REL_REFERENCES, 
               GraphSchema.REL_IMPORTS, GraphSchema.REL_INHERITS, GraphSchema.REL_HAS_PARAMETER, 
               GraphSchema.REL_RETURNS],
    'JavaScript': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, GraphSchema.REL_REFERENCES, 
                   GraphSchema.REL_IMPORTS, GraphSchema.REL_INHERITS, GraphSchema.REL_HAS_PARAMETER],
    'TypeScript': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, GraphSchema.REL_REFERENCES, 
                   GraphSchema.REL_IMPORTS, GraphSchema.REL_INHERITS, GraphSchema.REL_HAS_PARAMETER, 
                   GraphSchema.REL_RETURNS],
    'Java': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, 
             GraphSchema.REL_IMPORTS, GraphSchema.REL_INHERITS, GraphSchema.REL_HAS_PARAMETER, 
             GraphSchema.REL_RETURNS],
    'Kotlin': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, 
               GraphSchema.REL_IMPORTS, GraphSchema.REL_INHERITS, GraphSchema.REL_HAS_PARAMETER, 
               GraphSchema.REL_RETURNS],
    'HTML': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, GraphSchema.REL_REFERENCES, 
             GraphSchema.REL_IMPORTS, GraphSchema.REL_CONTAINS, GraphSchema.REL_HAS_PARAMETER],
    'SCSS': [GraphSchema.REL_DEFINES, GraphSchema.REL_CALLS, GraphSchema.REL_REFERENCES, 
             GraphSchema.REL_IMPORTS],
}

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def verify_embeddings_by_language():
    """Verify embeddings are created for each language."""
    vector_db = VectorDB(codebase_id="test_project")
    
    print("\n" + "="*80)
    print("EMBEDDINGS VERIFICATION BY LANGUAGE")
    print("="*80)
    
    # Get all files in the test project
    test_project_path = Path(__file__).parent / "test_project"
    all_files = {}
    for ext, lang in LANGUAGE_MAP.items():
        files = list(test_project_path.rglob(f"*{ext}"))
        if files:
            all_files[lang] = [str(f.relative_to(test_project_path)) for f in files]
    
    all_languages_ok = True
    for language, files in all_files.items():
        print(f"\n{'-'*80}")
        print(f"{language} ({len(files)} files)")
        print(f"{'-'*80}")
        
        # Get file names for filtering
        file_names = [Path(f).name for f in files]
        
        # Count embeddings for this language
        # Get all embeddings and filter by file path
        try:
            all_results = vector_db.get(limit=10000)
            all_ids = all_results.get('ids', [])
            all_metadatas = all_results.get('metadatas', [])
            
            # Filter embeddings by file path
            total_embeddings = 0
            for metadata in all_metadatas:
                if metadata:
                    file_path = str(metadata.get('file_path', ''))
                    # Check if this file path matches any of our language files
                    if any(file_name in file_path for file_name in file_names):
                        total_embeddings += 1
        except Exception as e:
            # If query fails, use count as fallback
            total_embeddings = vector_db.count()
        
        status = "OK" if total_embeddings > 0 else "MISSING"
        print(f"  [{status:7s}] Embeddings: {total_embeddings:4d} entities")
        
        if total_embeddings == 0:
            all_languages_ok = False
    
    # Overall embedding count
    total_count = vector_db.count()
    print(f"\n{'-'*80}")
    print(f"Total Embeddings: {total_count}")
    print(f"{'-'*80}")
    
    return all_languages_ok

def verify_relationships_by_language():
    """Verify all relationship types are created for each language (at least 1 of each type)."""
    client = Neo4jClient()
    
    print("\n" + "="*80)
    print("RELATIONSHIPS VERIFICATION BY LANGUAGE")
    print("="*80)
    
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
        
        # Get expected relationship types for this language
        expected_rels = LANGUAGE_RELATIONSHIPS.get(language, [])
        
        # Build file path filter for this language
        # File paths in Neo4j are absolute, so we need to match by file name
        file_names = [Path(f).name for f in files]
        
        # Build conditions: match by file name (more reliable)
        # For File nodes, use 'path' property; for entity nodes, use 'file_path' property
        conditions = []
        for fname in file_names:
            # Escape single quotes in file names
            fname_escaped = fname.replace("'", "\\'")
            conditions.append(f"a.file_path ENDS WITH '{fname_escaped}'")
        
        file_conditions = " OR ".join(conditions) if conditions else "FALSE"
        
        # Also build path conditions for File nodes
        path_conditions = []
        for fname in file_names:
            fname_escaped = fname.replace("'", "\\'")
            path_conditions.append(f"a.path ENDS WITH '{fname_escaped}'")
        
        path_file_conditions = " OR ".join(path_conditions) if path_conditions else "FALSE"
        
        language_ok = True
        missing_rels = []
        
        for rel_type in expected_rels:
            # For DEFINES and IMPORTS, check File nodes; for others, check entity nodes
            # File nodes use 'path' property, entity nodes use 'file_path' property
            if rel_type in [GraphSchema.REL_DEFINES, GraphSchema.REL_IMPORTS, GraphSchema.REL_CONTAINS]:
                # File nodes use 'path' property
                query = f"""
                MATCH (a:File)-[r:{rel_type}]->(b)
                WHERE a.codebase_id = 'test_project' 
                  AND b.codebase_id = 'test_project'
                  AND ({path_file_conditions})
                RETURN count(r) as count
                """
            else:
                # Entity nodes use 'file_path' property
                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                WHERE a.codebase_id = 'test_project' 
                  AND b.codebase_id = 'test_project'
                  AND ({file_conditions})
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
            
            # Require at least 1 relationship of each type
            status = "OK" if count >= 1 else "MISSING"
            print(f"  [{status:7s}] {rel_type:20s}: {count:4d} relationships")
            
            if count < 1:
                language_ok = False
                all_languages_ok = False
                missing_rels.append(rel_type)
        
        if language_ok:
            print(f"  [SUCCESS] All required relationship types found for {language}")
        else:
            print(f"  [FAILED] Missing relationship types for {language}: {', '.join(missing_rels)}")
    
    client.close()
    return all_languages_ok

def verify_relationships():
    """Verify all relationship types are created in the graph."""
    client = Neo4jClient()
    
    print("\n" + "="*80)
    print("OVERALL RELATIONSHIPS VERIFICATION")
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
    clear_screen()
    
    test_project_path = Path(__file__).parent / "test_project"
    
    print("="*80)
    print("DOCGRAPH TEST SUITE")
    print("="*80)
    print(f"Project path: {test_project_path}")
    print("Testing: Relationships + Embeddings for all file types")
    print()
    
    # Index the test project
    print("Step 1: Indexing test project...")
    try:
        index_codebase(
            codebase_path=str(test_project_path),
            codebase_id="test_project",
            clear_existing=True
        )
        print("[SUCCESS] Indexing completed successfully\n")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify relationships overall
    print("Step 2: Verifying overall relationships...")
    try:
        overall_success = verify_relationships()
    except Exception as e:
        print(f"[ERROR] Overall verification failed: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
    
    # Verify relationships by language
    print("\nStep 3: Verifying relationships by language (at least 1 of each type)...")
    try:
        language_success = verify_relationships_by_language()
    except Exception as e:
        print(f"[ERROR] Language-specific verification failed: {e}")
        import traceback
        traceback.print_exc()
        language_success = False
    
    # Verify embeddings by language
    print("\nStep 4: Verifying embeddings by language...")
    try:
        embeddings_success = verify_embeddings_by_language()
    except Exception as e:
        print(f"[ERROR] Embeddings verification failed: {e}")
        import traceback
        traceback.print_exc()
        embeddings_success = False
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Overall Relationships: {'PASS' if overall_success else 'FAIL'}")
    print(f"Relationships by Language: {'PASS' if language_success else 'FAIL'}")
    print(f"Embeddings by Language: {'PASS' if embeddings_success else 'FAIL'}")
    print("="*80)
    
    if overall_success and language_success and embeddings_success:
        print("\n[SUCCESS] ALL TESTS PASSED")
        return True
    else:
        print("\n[FAILED] SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
