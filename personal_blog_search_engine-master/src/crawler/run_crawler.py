import os
import argparse
import json
from crawler import BlogCrawler
import logging

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the blog crawler')
    parser.add_argument('--max_pages', type=int, default=1000, help='Maximum number of pages to crawl')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store crawled data')
    parser.add_argument('--seed_file', type=str, default=None, help='JSON file with seed URLs')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Load seed URLs if provided
    seed_urls = None
    if args.seed_file and os.path.exists(args.seed_file):
        try:
            with open(args.seed_file, 'r') as f:
                seed_urls = json.load(f)
            logging.info(f"Loaded {len(seed_urls)} seed URLs from {args.seed_file}")
        except Exception as e:
            logging.error(f"Error loading seed URLs: {str(e)}")
    
    # Initialize and run the crawler
    crawler = BlogCrawler(
        seed_urls=seed_urls,
        max_pages=args.max_pages,
        delay=args.delay,
        data_dir=args.data_dir
    )
    
    # Start crawling
    pages_crawled = crawler.crawl()
    
    print(f"Crawling complete. Crawled {pages_crawled} pages.")
    print(f"Data saved to {os.path.abspath(args.data_dir)}")

if __name__ == "__main__":
    main() 