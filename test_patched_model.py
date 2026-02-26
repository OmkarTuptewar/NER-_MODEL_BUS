"""
Test the patched model on previously failing queries
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.inference import BusNERInference


def test_failing_queries():
    """Test the patched model on queries that previously failed"""
    
    # Initialize inference with the patched model
    print("\n" + "=" * 70)
    print("TESTING PATCHED MODEL ON PREVIOUSLY FAILING QUERIES")
    print("=" * 70)
    
    model_path = "models/bus_ner_minilm_v7_patched"
    print(f"\nLoading model from: {model_path}")
    
    try:
        predictor = BusNERInference(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test queries that previously failed
    test_cases = [
        {
            "query": "show me buses from mumbai to delhi tomorrow morning",
            "expected_entities": ["mumbai", "delhi", "tomorrow", "morning"],
            "expected_types": ["SOURCE_NAME", "DESTINATION_NAME", "DEPARTURE_DATE", "DEPARTURE_TIME"]
        },
        {
            "query": "show me buses from mumbai to delhi on tomorrow morning",
            "expected_entities": ["mumbai", "delhi", "tomorrow", "morning"],
            "expected_types": ["SOURCE_NAME", "DESTINATION_NAME", "DEPARTURE_DATE", "DEPARTURE_TIME"]
        },
        {
            "query": "Can an unmarried couple book sharing seats on the Journey with IntrCity SmartBus from Nagpur to Pune?",
            "expected_entities": ["unmarried couple", "sharing seats", "IntrCity", "SmartBus", "Nagpur", "Pune"],
            "expected_types": ["TRAVELER", "SEAT_TYPE", "OPERATOR", "BUS_TYPE", "SOURCE_NAME", "DESTINATION_NAME"]
        },
        {
            "query": "find buses from bangalore to chennai tomorrow evening",
            "expected_entities": ["bangalore", "chennai", "tomorrow", "evening"],
            "expected_types": ["SOURCE_NAME", "DESTINATION_NAME", "DEPARTURE_DATE", "DEPARTURE_TIME"]
        },
        {
            "query": "book IntrCity SmartBus from pune to mumbai",
            "expected_entities": ["IntrCity", "SmartBus", "pune", "mumbai"],
            "expected_types": ["OPERATOR", "BUS_TYPE", "SOURCE_NAME", "DESTINATION_NAME"]
        },
    ]
    
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases):
        print(f"\n[Test {i+1}] Query: \"{test['query']}\"")
        
        # Get prediction
        result = predictor.predict(test['query'])
        entities = result.get('entities', {})
        
        # Print detected entities
        print("\nDetected Entities:")
        has_entities = False
        for entity_type, values in entities.items():
            if values:
                has_entities = True
                for value in values:
                    print(f"  ✓ {entity_type}: '{value}'")
        
        if not has_entities:
            print("  (No entities detected)")
        
        # Check if expected entities are found
        print("\nExpected Entities:")
        all_found = True
        for entity, entity_type in zip(test['expected_entities'], test['expected_types']):
            found = entity_type in entities and any(entity.lower() in str(v).lower() for v in entities[entity_type])
            status = "✓" if found else "✗"
            print(f"  {status} {entity_type}: '{entity}'")
            if not found:
                all_found = False
        
        if all_found:
            print("\n  Result: ✅ PASSED")
            passed += 1
        else:
            print("\n  Result: ❌ FAILED")
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    test_failing_queries()
