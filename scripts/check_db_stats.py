
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.neo4j_client import Neo4jClient

def check_stats():
    print("Checking Neo4j Database Statistics...")
    client = Neo4jClient()
    
    # Check Node Counts
    print("\n--- Node Counts ---")
    query_nodes = "MATCH (n) RETURN labels(n) as labels, count(n) as count"
    results_nodes = client.execute_query(query_nodes)
    if not results_nodes:
        print("No nodes found.")
    for record in results_nodes:
        print(f"{record['labels']}: {record['count']}")
        
    # Check Relationship Counts
    print("\n--- Relationship Counts ---")
    query_rels = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
    results_rels = client.execute_query(query_rels)
    if not results_rels:
        print("No relationships found.")
    
    total_rels = 0
    for record in results_rels:
        print(f"{record['type']}: {record['count']}")
        total_rels += record['count']
        
    print(f"\nTotal Relationships: {total_rels}")
    client.close()

if __name__ == "__main__":
    check_stats()
