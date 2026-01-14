import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Failed to download NLTK resources. Some features may not work correctly.")

class SearchEngine:
    def __init__(self, data_dir='data', index_path='index', classifier=None):
        """
        Initialize the search engine.
        
        Args:
            data_dir (str): Directory containing crawled data
            index_path (str): Directory to store the search index
            classifier: Optional classifier to filter results
        """
        self.data_dir = data_dir
        self.index_path = index_path
        self.classifier = classifier
        self.documents = []
        self.urls = []
        self.titles = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.pagerank_scores = {}
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            logger.warning("Stopwords not available. Using empty stopword list.")
    
    def preprocess_text(self, text):
        """Preprocess text for indexing and searching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def build_index(self):
        """Build the search index from crawled data."""
        logger.info("Building search index...")
        
        # Reset all data structures
        self.documents = []
        self.urls = []
        self.titles = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.pagerank_scores = {}
        
        # Process each crawled page
        processed_documents = []
        for filename in tqdm(os.listdir(self.data_dir), desc="Indexing documents"):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Filter using classifier if available
                        if self.classifier:
                            label, confidence = self.classifier.hierarchical_classify(data['content'])
                            if label != "personal" or confidence < 0.6:
                                continue
                        
                        # Preprocess the content
                        processed_content = self.preprocess_text(data['content'])
                        
                        # Only add document if it has content
                        if processed_content.strip():
                            processed_documents.append(processed_content)
                            self.documents.append(data['content'])  # Store original content
                            self.urls.append(data['url'])
                            self.titles.append(data['title'])
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
        
        if not processed_documents:
            logger.error("No valid documents found to index")
            return 0
        
        # Build TF-IDF matrix
        try:
            self.vectorizer = TfidfVectorizer(max_features=10000)
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_documents)
            
            logger.info(f"Indexed {len(self.documents)} documents")
            
            # Build PageRank scores
            self._build_pagerank()
            
            # Save the index
            self._save_index()
            
            return len(self.documents)
        except Exception as e:
            logger.error(f"Error building TF-IDF matrix: {str(e)}")
            return 0
    
    def _build_pagerank(self):
        """Build PageRank scores for indexed documents."""
        logger.info("Building PageRank scores...")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (documents)
        for i, url in enumerate(self.urls):
            G.add_node(i, url=url)
        
        # Add edges (links between documents)
        url_to_index = {url: i for i, url in enumerate(self.urls)}
        
        # Process each crawled page to extract links
        for filename in tqdm(os.listdir(self.data_dir), desc="Building graph"):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        source_url = data['url']
                        
                        # Skip if source URL is not in our index
                        if source_url not in url_to_index:
                            continue
                        
                        source_idx = url_to_index[source_url]
                        
                        # Add edges for each link
                        for link in data['links']:
                            if link in url_to_index:
                                target_idx = url_to_index[link]
                                G.add_edge(source_idx, target_idx)
                except Exception as e:
                    logger.error(f"Error processing links in {filename}: {str(e)}")
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            self.pagerank_scores = pagerank
            logger.info(f"PageRank calculated for {len(pagerank)} nodes")
        except Exception as e:
            logger.error(f"Error calculating PageRank: {str(e)}")
            self.pagerank_scores = {i: 1.0 for i in range(len(self.urls))}
    
    def _save_index(self):
        """Save the search index to disk."""
        index_data = {
            'urls': self.urls,
            'titles': self.titles,
            'pagerank': self.pagerank_scores,
            'documents': self.documents  # Add documents to saved data
        }
        
        # Save index data
        with open(os.path.join(self.index_path, 'index_data.json'), 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False)
        
        # Save TF-IDF vectorizer
        import pickle
        with open(os.path.join(self.index_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save TF-IDF matrix
        import scipy.sparse
        scipy.sparse.save_npz(os.path.join(self.index_path, 'tfidf_matrix.npz'), self.tfidf_matrix)
        
        logger.info(f"Search index saved to {self.index_path}")
    
    def load_index(self):
        """Load the search index from disk."""
        try:
            # Load index data
            with open(os.path.join(self.index_path, 'index_data.json'), 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                self.urls = index_data['urls']
                self.titles = index_data['titles']
                self.documents = index_data['documents']  # Load documents from saved data
                self.pagerank_scores = {int(k): v for k, v in index_data['pagerank'].items()}
            
            # Load TF-IDF vectorizer
            import pickle
            with open(os.path.join(self.index_path, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load TF-IDF matrix
            import scipy.sparse
            self.tfidf_matrix = scipy.sparse.load_npz(os.path.join(self.index_path, 'tfidf_matrix.npz'))
            
            logger.info(f"Search index loaded with {len(self.urls)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading search index: {str(e)}")
            return False
    
    def search(self, query, num_results=10):
        """
        Search for documents matching the query.
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return
        
        Returns:
            list: List of search results
        """
        if not self.vectorizer or self.tfidf_matrix is None:
            logger.error("Search index not built or loaded")
            return []
        
        try:
            # Preprocess the query
            processed_query = self.preprocess_text(query)
            
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Combine with PageRank (weighted sum)
            pagerank_array = np.array([self.pagerank_scores.get(i, 0.0) for i in range(len(similarities))])
            combined_scores = 0.7 * similarities + 0.3 * pagerank_array
            
            # Get top results
            top_indices = combined_scores.argsort()[-num_results:][::-1]
            
            # Create results
            results = []
            for idx in top_indices:
                # Ensure index is valid
                if idx >= len(self.urls) or idx >= len(self.titles):
                    logger.warning(f"Invalid index {idx} encountered during search")
                    continue
                    
                try:
                    # Get a snippet from the document
                    snippet = self._get_snippet(self.documents[idx], processed_query)
                    
                    results.append({
                        'url': self.urls[idx],
                        'title': self.titles[idx],
                        'snippet': snippet,
                        'relevance_score': float(similarities[idx]),
                        'pagerank_score': float(self.pagerank_scores.get(idx, 0.0))
                    })
                except Exception as e:
                    logger.warning(f"Error processing search result at index {idx}: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def _get_snippet(self, document, query, max_length=200):
        """Extract a relevant snippet from the document."""
        try:
            if not document or not isinstance(document, str):
                return "No preview available"
                
            # Simple approach: find first occurrence of any query term
            query_terms = query.split()
            
            for term in query_terms:
                if term and term in document:
                    # Find position of the term
                    pos = document.find(term)
                    
                    # Get context around the term
                    start = max(0, pos - max_length // 2)
                    end = min(len(document), pos + max_length // 2)
                    
                    # Adjust to avoid cutting words
                    while start > 0 and document[start] != ' ':
                        start -= 1
                    while end < len(document) and document[end] != ' ':
                        end += 1
                    
                    return document[start:end].strip() + "..."
            
            # If no term found, return the beginning of the document
            if len(document) > max_length:
                # Find a good breaking point
                end = max_length
                while end < len(document) and document[end] != ' ':
                    end -= 1
                return document[:end].strip() + "..."
            return document.strip()
            
        except Exception as e:
            logger.warning(f"Error generating snippet: {str(e)}")
            return "No preview available"

if __name__ == "__main__":
    # Example usage
    from src.classifier.classifier import BlogClassifier
    
    # Initialize classifier
    classifier = BlogClassifier()
    classifier.load_model()
    
    # Initialize search engine
    search_engine = SearchEngine(classifier=classifier)
    
    # Build or load index
    if os.path.exists(os.path.join(search_engine.index_path, 'index_data.json')):
        search_engine.load_index()
    else:
        search_engine.build_index()
    
    # Test search
    results = search_engine.search("product management career advice")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet']}")
        print(f"   Relevance: {result['relevance_score']:.4f}, PageRank: {result['pagerank_score']:.4f}")
        print("-" * 50) 