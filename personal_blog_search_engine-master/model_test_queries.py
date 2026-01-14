from src.search_engine.search_engine import SearchEngine
from src.classifier.classifier import BlogClassifier
import logging

def test_model_accuracy():
    # Initialize
    classifier = BlogClassifier()
    classifier.load_model()
    search_engine = SearchEngine(classifier=classifier)
    search_engine.load_index()

    # Test cases with expected content types
    test_cases = [
        {
            "query": "I learned programming",
            "expected": "personal",
            "description": "Should return personal learning experiences"
        },
        {
            "query": "our company values",
            "expected": "corporate",
            "description": "Should filter out corporate content"
        },
        {
            "query": "my first Python project",
            "expected": "personal",
            "description": "Should return personal project stories"
        },
        {
            "query": "privacy policy terms",
            "expected": "corporate",
            "description": "Should filter out legal pages"
        }
    ]

    print("Model Evaluation Results:")
    print("=" * 50)

    for case in test_cases:
        print(f"\nTesting: {case['description']}")
        print(f"Query: {case['query']}")
        print(f"Expected content type: {case['expected']}")
        
        results = search_engine.search(case['query'], num_results=2)
        
        if results:
            print(f"Got {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Relevance: {result['relevance_score']:.4f}")
                print(f"   Snippet: {result['snippet'][:100]}...")
        else:
            print("No results found (filtered out)")
        
        print("-" * 50)

if __name__ == "__main__":
    test_model_accuracy() 