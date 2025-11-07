"""
Web scraper module for fetching and parsing webpages into JSON/JSONL format.
"""

import requests
import json
import jsonlines
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:  # pragma: no cover
    import re

    _HAS_BS4 = False

    class _SimpleTag:
        def __init__(self, html: str, tag: str, attrs: Optional[Dict[str, str]] = None) -> None:
            self._html = html
            self.name = tag
            self.attrs = attrs or {}

        def get_text(self, strip: bool = False, separator: str = "") -> str:
            text = re.sub(r"<[^>]+>", "", self._html)
            if strip:
                text = text.strip()
            return separator.join(part.strip() for part in text.splitlines() if part.strip()) if separator else text

        @property
        def string(self) -> str:
            return self.get_text(strip=True)

        def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
            return self.attrs.get(key, default)

        def decompose(self) -> None:
            self._html = ""

        def find_all(self, tag: str, attrs: Optional[Dict[str, str]] = None, **kwargs):
            soup = BeautifulSoup(self._html, "html.parser")
            return soup.find_all(tag, attrs=attrs, **kwargs)

    class _FallbackSoup:
        def __init__(self, html: str, _parser: str) -> None:
            self._html = html

        @property
        def title(self) -> Optional[_SimpleTag]:
            match = re.search(r"<title[^>]*>(.*?)</title>", self._html, re.S | re.I)
            if match:
                return _SimpleTag(match.group(1), "title")
            return None

        def get_text(self, strip: bool = False, separator: str = "") -> str:
            text = re.sub(r"<[^>]+>", "", self._html)
            if strip:
                text = text.strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return separator.join(lines) if separator else text

        def find(self, tag: str):
            matches = self.find_all(tag)
            return matches[0] if matches else None

        def find_all(self, tag: str, attrs: Optional[Dict[str, str]] = None, **kwargs):
            pattern = rf"<{tag}([^>]*)>(.*?)</{tag}>"
            matches = []
            attr_filter = dict(attrs or {})
            attr_filter.update({key: value for key, value in kwargs.items() if value is not None})
            for attr_text, body in re.findall(pattern, self._html, re.S | re.I):
                attr_map = {}
                for key, value in re.findall(r'(\w+)=\"(.*?)\"', attr_text):
                    attr_map[key] = value
                if attr_filter:
                    match_attrs = True
                    for key, expected in attr_filter.items():
                        if attr_map.get(key) != expected:
                            match_attrs = False
                            break
                    if not match_attrs:
                        continue
                matches.append(_SimpleTag(body, tag, attr_map))
            return matches

        def select_one(self, selector: str):
            selector = selector.strip()
            if selector.startswith("."):
                class_name = selector[1:]
                pattern = rf"<([a-z0-9]+)[^>]*class=\"[^\"]*{class_name}[^\"]*\"[^>]*>(.*?)</\1>"
                match = re.search(pattern, self._html, re.S | re.I)
                if match:
                    attrs = {"class": class_name}
                    return _SimpleTag(match.group(0), match.group(1), attrs)
            elif selector.startswith("[") and "role=" in selector:
                role = selector.split("role=")[-1].strip("[]\"")
                pattern = rf"<([a-z0-9]+)[^>]*role=\"{role}\"[^>]*>(.*?)</\1>"
                match = re.search(pattern, self._html, re.S | re.I)
                if match:
                    return _SimpleTag(match.group(0), match.group(1), {"role": role})
            else:
                tag = selector.split(".")[0]
                found = self.find(tag)
                if found:
                    return found
            return None

        def __call__(self, tags):
            if isinstance(tags, (list, tuple, set)):
                results = []
                for tag in tags:
                    results.extend(self.find_all(tag))
                return results
            return self.find_all(tags)

    def BeautifulSoup(html: str, parser: str):  # type: ignore
        return _FallbackSoup(html, parser)
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
            raw_html, final_url = self._get_page_content(url)
            if raw_html is None:
                return None

            soup = BeautifulSoup(raw_html, 'html.parser')
            
            # Extract structured data
            data = {
                'url': url,
                'title': soup.title.string.strip() if soup.title else '',
                'content': self._extract_content(soup),
                'metadata': self._extract_metadata(soup),
                'links': self._extract_links(soup, url),
                'timestamp': time.time(),
                'raw_html': raw_html,
                'resolved_url': final_url,
            }
            
            time.sleep(self.delay)  # Rate limiting
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def _get_page_content(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch raw HTML for a URL or local file path."""
        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            path = Path(parsed.path if parsed.scheme else url)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            try:
                html = path.read_text(encoding='utf-8')
                return html, str(path)
            except Exception as exc:
                logger.error(f"Error reading local file {path}: {exc}")
                return None, str(path)

        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text, response.url

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