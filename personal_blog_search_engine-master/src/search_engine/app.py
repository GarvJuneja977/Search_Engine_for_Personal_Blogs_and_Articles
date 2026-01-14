from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine import SearchEngine
from classifier.classifier import BlogClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
CORS(app)

# Initialize classifier
classifier = BlogClassifier()
classifier.load_model()

# Initialize search engine
search_engine = SearchEngine(classifier=classifier)

# Check if index exists, otherwise build it
if os.path.exists(os.path.join(search_engine.index_path, 'index_data.json')):
    search_engine.load_index()
else:
    # If data directory exists, build index
    if os.path.exists(search_engine.data_dir) and os.listdir(search_engine.data_dir):
        search_engine.build_index()
    else:
        logger.warning(f"No data found in {search_engine.data_dir}. Search functionality will be limited.")

@app.route('/')
def index():
    """Render the search page."""
    return render_template('index.html')

@app.route('/api/search')
def search_get():
    """Legacy API endpoint for searching (GET method)."""
    query = request.args.get('q', '')
    num_results = int(request.args.get('n', 10))
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        results = search_engine.search(query, num_results=num_results)
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return jsonify({'error': 'Search error'}), 500

@app.route('/search', methods=['POST'])
def search():
    """API endpoint for searching with advanced filters."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        filters = data.get('filters', {})
        
        # Get filter values with defaults
        date_range = filters.get('dateRange', 'any')
        sort_by = filters.get('sortBy', 'relevance')
        content_type = filters.get('contentType', 'all')
        num_results = filters.get('numResults', 10)
        
        # Perform the search
        results = search_engine.search(query, num_results=num_results)
        
        # Apply additional filtering based on date_range if needed
        if date_range != 'any':
            # TODO: Implement date filtering
            pass
        
        # Apply sorting if needed
        if sort_by == 'date':
            # Sort by date (newest first)
            results.sort(key=lambda x: x.get('date', ''), reverse=True)
        elif sort_by == 'pagerank':
            # Sort by pagerank score
            results.sort(key=lambda x: x.get('pagerank_score', 0), reverse=True)
        # Default 'relevance' sorting is already handled by search_engine.search()
        
        # Add additional metadata to results
        for result in results:
            # Add estimated read time (rough estimate based on content length)
            if 'snippet' in result:
                words = len(result['snippet'].split())
                result['readTime'] = max(1, round(words / 200))  # Assume 200 words per minute
            
            # Add formatted date
            result['date'] = '2024-01-01'  # TODO: Add actual date from document metadata
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """API endpoint for checking the status of the search engine."""
    return jsonify({
        'status': 'ok',
        'documents_indexed': len(search_engine.urls) if search_engine.urls else 0,
        'index_built': search_engine.tfidf_matrix is not None
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 