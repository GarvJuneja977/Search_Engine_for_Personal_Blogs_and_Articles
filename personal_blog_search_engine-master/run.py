import os
import argparse
import logging
import sys
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_crawler(max_pages=1000, data_dir='data'):
    """Run the web crawler to collect data."""
    logger.info(f"Starting web crawler to collect up to {max_pages} pages...")
    
    try:
        cmd = [sys.executable, 'src/crawler/run_crawler.py', 
               '--max_pages', str(max_pages),
               '--data_dir', data_dir]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Crawler failed with error: {stderr.decode('utf-8')}")
            return False
        
        logger.info(f"Crawler completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error running crawler: {str(e)}")
        return False

def train_classifier(data_dir='data', model_path='models/blog_classifier.joblib'):
    """Train the blog classifier on collected data."""
    logger.info("Training blog classifier...")
    
    try:
        cmd = [sys.executable, 'src/classifier/train_classifier.py',
               '--data_dir', data_dir,
               '--model_path', model_path]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Classifier training failed with error: {stderr.decode('utf-8')}")
            return False
        
        logger.info(f"Classifier training completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error training classifier: {str(e)}")
        return False

def run_search_engine(host='0.0.0.0', port=5000):
    """Run the Flask search engine API."""
    logger.info(f"Starting search engine on {host}:{port}...")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['FLASK_APP'] = 'src/search_engine/app.py'
        env['PORT'] = str(port)
        
        cmd = [sys.executable, 'src/search_engine/app.py']
        
        # Run the Flask app
        process = subprocess.Popen(cmd, env=env)
        
        # Wait a moment to ensure the server starts
        time.sleep(2)
        
        logger.info(f"Search engine running at http://{host}:{port}")
        
        # Return the process so it can be terminated later if needed
        return process
    except Exception as e:
        logger.error(f"Error starting search engine: {str(e)}")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the personal blog search engine')
    parser.add_argument('--crawl', action='store_true', help='Run the web crawler')
    parser.add_argument('--train', action='store_true', help='Train the classifier')
    parser.add_argument('--max_pages', type=int, default=1000, help='Maximum number of pages to crawl')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store crawled data')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the search engine on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the search engine on')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the crawler if requested
    if args.crawl:
        success = run_crawler(max_pages=args.max_pages, data_dir=args.data_dir)
        if not success:
            logger.warning("Crawler encountered issues. Proceeding with existing data if available.")
    
    # Train the classifier if requested
    if args.train:
        success = train_classifier(data_dir=args.data_dir)
        if not success:
            logger.warning("Classifier training encountered issues. Proceeding with existing model if available.")
    
    # Run the search engine
    process = run_search_engine(host=args.host, port=args.port)
    
    if process:
        try:
            # Keep the script running while the search engine is running
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping search engine...")
            process.terminate()
            process.wait()
            logger.info("Search engine stopped.")

if __name__ == "__main__":
    main() 