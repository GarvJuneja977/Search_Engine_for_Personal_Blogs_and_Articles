import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BlogCrawler:
    def __init__(self, seed_urls=None, max_pages=1000, delay=1.0, data_dir='data'):
        """
        Initialize the crawler with seed URLs and parameters.
        
        Args:
            seed_urls (list): List of URLs to start crawling from
            max_pages (int): Maximum number of pages to crawl
            delay (float): Delay between requests to be polite
            data_dir (str): Directory to store crawled data
        """
        self.seed_urls = seed_urls or [
            # Existing Tech and Programming
            "https://manassaloi.com/",
            "https://waitbutwhy.com/",
            "https://www.kalzumeus.com/",
            "https://www.joelonsoftware.com/",
            "https://paulgraham.com/articles.html",
            
            # Additional Tech and Programming
            "https://techsavvymama.com/",
            "https://www.neilpatel.com/blog/",
            "https://www.jeffgeerling.com/blog",
            "https://www.taniarascia.com/blog",
            "https://www.smashingmagazine.com/articles/",
            
            # Existing Science and Philosophy
            "https://slatestarcodex.com/",
            "https://www.scottaaronson.com/blog/",
            "https://www.lesswrong.com/",
            "https://www.brainpickings.org/",
            "https://aeon.co/essays/",
            
            # Additional Science and Philosophy
            "https://nautil.us/",
            "https://blogs.scientificamerican.com/",
            "https://www.3quarksdaily.com/",
            "https://philosophynow.org/articles",
            
            # Existing Personal Development
            "https://markmanson.net/",
            "https://jamesclear.com/articles",
            "https://zenhabits.net/",
            "https://www.raptitude.com/",
            "https://www.stevepavlina.com/blog/",
            
            # Additional Personal Development
            "https://bemorewithless.com/",
            "https://www.becomingminimalist.com/",
            "https://tinybuddha.com/blog/",
            "https://www.positivityblog.com/",
            
            # Existing Travel and Lifestyle
            "https://www.nomadicmatt.com/travel-blog/",
            "https://www.legalnomads.com/",
            "https://www.alexinwanderland.com/",
            "https://www.youngadventuress.com/",
            "https://www.adventurouskate.com/",
            
            # Additional Travel and Lifestyle
            "https://www.theblondeabroad.com/blog/",
            "https://expertvagabond.com/blog/",
            "https://www.danflyingsolo.com/",
            "https://www.earthtrekkers.com/blog/",
            
            # Existing Writing and Creativity
            "https://austinkleon.com/",
            "https://www.brainpickings.org/",
            "https://www.themarginalian.org/",
            "https://www.stevenpressfield.com/blog/",
            "https://writerunboxed.com/",
            
            # Additional Writing and Creativity
            "https://goinswriter.com/blog/",
            "https://www.writersdigest.com/write-better-fiction",
            "https://thewritepractice.com/blog/",
            
            # Existing Health and Wellness
            "https://zenhabits.net/",
            "https://www.marksdailyapple.com/",
            "https://www.nerdfitness.com/blog/",
            "https://www.precisionnutrition.com/blog",
            "https://www.healthline.com/nutrition/blog",
            
            # Additional Health and Wellness
            "https://www.mindbodygreen.com/articles",
            "https://wellnessmama.com/blog/",
            "https://whole30.com/blog/",
            "https://www.thehealthyhomeeconomist.com/",
            
            # Existing Finance and Investment
            "https://www.mrmoneymustache.com/",
            "https://jlcollinsnh.com/",
            "https://www.financialsamurai.com/",
            "https://affordanything.com/blog/",
            "https://www.getrichslowly.org/blog/",
            
            # Additional Finance and Investment
            "https://www.whitecoatinvestor.com/blog/",
            "https://www.physicianonfire.com/blog/",
            "https://www.budgetsaresexy.com/",
            "https://www.madfientist.com/blog/",
            
            # Existing Art and Design
            "https://www.brainpickings.org/",
            "https://www.thisiscolossal.com/",
            "https://www.swiss-miss.com/",
            "https://designobserver.com/",
            "https://www.creativebloq.com/",
            
            # Additional Art and Design
            "https://www.artistsnetwork.com/art-blogs/",
            "https://www.booooooom.com/blog/",
            "https://www.juxtapoz.com/news/",
            "https://www.creativeboom.com/",
            
            # Existing Books and Literature
            "https://www.austinkleon.com/",
            "https://www.brainpickings.org/",
            "https://www.themarginalian.org/",
            "https://www.openculture.com/",
            "https://lithub.com/",
            
            # Additional Books and Literature
            "https://bookriot.com/",
            "https://www.thebooksmugglers.com/",
            "https://electricliterature.com/",
            "https://www.bookforum.com/blog"
        ]
        
        # Increase max pages to gather more content
        self.max_pages = max_pages
        # Adjust delay to be more polite to servers
        self.delay = delay
        self.data_dir = data_dir
        self.visited_urls = set()
        self.queue = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def is_valid_url(self, url):
        """Check if URL is valid and should be crawled."""
        try:
            parsed = urlparse(url)
            # Avoid crawling non-http(s) URLs
            if parsed.scheme not in ('http', 'https'):
                return False
            
            # Skip social media and large platforms to focus on personal blogs
            domains_to_skip = [
                'facebook.com', 'twitter.com', 'instagram.com', 
                'youtube.com', 'linkedin.com', 'medium.com',
                'reddit.com', 'github.com', 'amazon.com'
            ]
            
            if any(domain in parsed.netloc.lower() for domain in domains_to_skip):
                return False
                
            # Avoid media files, PDFs, etc.
            extensions_to_avoid = [
                '.jpg', '.jpeg', '.png', '.gif', '.pdf', 
                '.zip', '.mp3', '.mp4', '.css', '.js'
            ]
            if any(url.lower().endswith(ext) for ext in extensions_to_avoid):
                return False
                
            return True
        except:
            return False
    
    def extract_content(self, soup, url):
        """Extract relevant content from a page."""
        # Get title
        title = ""
        if soup.title:
            title = soup.title.string
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
            
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
            element.decompose()
            
        # Try to find the main content using common blog layouts
        content = ""
        main_content = None
        
        # Try different common content containers
        content_selectors = [
            'article',
            'main',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.blog-post',
            '.content',
            '#content',
            '.post',
            '.blog-entry'
        ]
        
        for selector in content_selectors:
            try:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 200:  # Ensure it has substantial content
                    main_content = element
                    break
            except:
                continue
        
        # If no specific content container found, try to find the largest text block
        if not main_content:
            text_blocks = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 100:  # Only consider paragraphs with substantial text
                    text_blocks.append(text)
            
            if text_blocks:
                content = ' '.join(text_blocks)
        else:
            # Clean up the main content
            for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
                element.decompose()
            content = main_content.get_text(separator=' ', strip=True)
        
        # Extract all links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(url, href)
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
        
        # Extract metadata
        meta_tags = {}
        for meta in soup.find_all('meta'):
            if meta.get('name') and meta.get('content'):
                meta_tags[meta['name']] = meta['content']
            elif meta.get('property') and meta.get('content'):  # Also get OpenGraph metadata
                meta_tags[meta['property']] = meta['content']
        
        # Add date if available
        date = None
        date_selectors = [
            'time',
            '.post-date',
            '.entry-date',
            '.published',
            '.date'
        ]
        
        for selector in date_selectors:
            try:
                date_element = soup.select_one(selector)
                if date_element and date_element.get('datetime'):
                    date = date_element['datetime']
                    break
                elif date_element:
                    date = date_element.get_text(strip=True)
                    break
            except:
                continue
        
        if date:
            meta_tags['published_date'] = date
        
        return {
            "url": url,
            "title": title,
            "content": content,
            "links": links,
            "meta": meta_tags
        }
    
    def crawl(self):
        """Start the crawling process."""
        self.queue = self.seed_urls.copy()
        page_count = 0
        
        with tqdm(total=self.max_pages, desc="Crawling") as pbar:
            while self.queue and page_count < self.max_pages:
                # Get the next URL
                url = self.queue.pop(0)
                
                # Skip if already visited
                if url in self.visited_urls:
                    continue
                
                # Mark as visited
                self.visited_urls.add(url)
                
                try:
                    # Fetch the page
                    response = requests.get(url, headers=self.headers, timeout=10)
                    
                    # Skip if not successful
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                        continue
                    
                    # Parse the page
                    soup = BeautifulSoup(response.text, 'lxml')
                    
                    # Extract content
                    data = self.extract_content(soup, url)
                    
                    # Save the data
                    filename = f"{self.data_dir}/page_{page_count}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    # Add new links to the queue
                    for link in data['links']:
                        if link not in self.visited_urls:
                            self.queue.append(link)
                    
                    # Increment page count
                    page_count += 1
                    pbar.update(1)
                    
                    # Be polite and wait between requests
                    time.sleep(self.delay + random.uniform(0, 1))
                    
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
        
        logger.info(f"Crawling complete. Crawled {page_count} pages.")
        return page_count

if __name__ == "__main__":
    # Example usage
    crawler = BlogCrawler(max_pages=100)
    crawler.crawl() 