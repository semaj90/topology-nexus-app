"""
Web scraper module for fetching and parsing webpages into JSON/JSONL format.
"""

import requests
import json
import jsonlines
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, delay: float = 1.0, headers: Optional[Dict] = None):
        self.delay = delay
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        else:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

    def fetch_webpage(self, url: str) -> Optional[Dict]:
        """Fetch and parse a single webpage into structured data."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract structured data
            data = {
                'url': url,
                'title': soup.title.string.strip() if soup.title else '',
                'content': self._extract_content(soup),
                'metadata': self._extract_metadata(soup),
                'links': self._extract_links(soup, url),
                'timestamp': time.time()
            }
            
            time.sleep(self.delay)  # Rate limiting
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the webpage."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '.main-content', '.post-content',
            'div.content', 'div.article'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text(strip=True, separator='\n')
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(strip=True, separator='\n')
        
        return soup.get_text(strip=True, separator='\n')

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from the webpage."""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Structured data
        scripts = soup.find_all('script', type='application/ld+json')
        structured_data = []
        for script in scripts:
            try:
                structured_data.append(json.loads(script.string))
            except:
                pass
        
        if structured_data:
            metadata['structured_data'] = structured_data
            
        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the webpage."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            if urlparse(absolute_url).netloc:  # Only external links
                links.append(absolute_url)
        return list(set(links))  # Remove duplicates

    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs and return structured data."""
        results = []
        for url in urls:
            logger.info(f"Scraping: {url}")
            data = self.fetch_webpage(url)
            if data:
                results.append(data)
        return results

    def save_to_json(self, data: List[Dict], filename: str) -> None:
        """Save scraped data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_to_jsonl(self, data: List[Dict], filename: str) -> None:
        """Save scraped data to JSONL file."""
        with jsonlines.open(filename, 'w') as writer:
            for item in data:
                writer.write(item)


if __name__ == "__main__":
    # Example usage
    scraper = WebScraper()
    urls = [
        "https://example.com",
        "https://httpbin.org/html"
    ]
    
    data = scraper.scrape_urls(urls)
    scraper.save_to_json(data, "scraped_data.json")
    scraper.save_to_jsonl(data, "scraped_data.jsonl")
    print(f"Scraped {len(data)} pages")