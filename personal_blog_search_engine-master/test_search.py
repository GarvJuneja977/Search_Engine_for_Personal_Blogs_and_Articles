from src.search_engine.search_engine import SearchEngine
from src.classifier.classifier import BlogClassifier
import logging

def test_search_engine():
    # Initialize classifier and search engine
    classifier = BlogClassifier()
    classifier.load_model()
    search_engine = SearchEngine(classifier=classifier)
    search_engine.load_index()

    # Test queries designed to evaluate the model
    test_queries = [
        'personal experience in tech',      # Should return personal blog posts
        'my journey as a developer',        # Should return personal blog posts
        'company products and services',    # Should be filtered out or low-ranked
        'learning to code blog',            # Should return personal blog posts
        'team announcement'                 # Should be filtered out or low-ranked
    ]

    for query in test_queries:
        print(f'\nQuery: {query}')
        results = search_engine.search(query, num_results=3)
        print(f'Found {len(results)} results')
        for i, result in enumerate(results, 1):
            print(f'{i}. {result["title"]}')
            print(f'   URL: {result["url"]}')
            print(f'   Relevance: {result["relevance_score"]:.4f}')
            print(f'   PageRank: {result["pagerank_score"]:.4f}')
            print(f'   Snippet: {result["snippet"][:150]}...')
            print('-' * 80)

if __name__ == "__main__":
    test_search_engine() 