"""
Simple test script for the patched model
Run from project root
"""

import json
import os
import sys

# Change to src directory for imports
os.chdir('src')
sys.path.insert(0, os.getcwd())

from inference import BusNERInference

def test_queries():
    print("\n" + "=" * 70)
    print("TESTING PATCHED MODEL ON PREVIOUSLY FAILING QUERIES")
    print("=" * 70)
    
    model_path = "../models/bus_ner_minilm_v7_patched"
    print(f"\nLoading model: {model_path}")
    
    predictor = BusNERInference(model_path)
    print("✓ Model loaded\n")
    
    test_cases = [
        "show me buses from mumbai to delhi tomorrow morning",
        "show me buses from mumbai to delhi on tomorrow morning",
        "Can an unmarried couple book sharing seats on the Journey with IntrCity SmartBus from Nagpur to Pune?",
        "find buses from bangalore to chennai tomorrow evening",
        "book IntrCity SmartBus from pune to mumbai",
    ]
    
    print("=" * 70)
    
    for i, query in enumerate(test_cases):
        print(f"\n[Test {i+1}] {query}")
        entities = predictor.extract(query)
        print("\nEntities:")
        for entity_type, values in sorted(entities.items()):
            if values:
                for value in values:
                    print(f"  - {entity_type}: '{value}'")
        
        print("-" * 70)
    
    print("\n✓ Testing complete!")

if __name__ == "__main__":
    test_queries()
