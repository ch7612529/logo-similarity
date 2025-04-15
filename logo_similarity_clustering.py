import os
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import hashlib
import imagehash
from PIL import Image
import io
import pandas as pd
import numpy as np
from collections import defaultdict
import concurrent.futures
import tqdm
import logging
import time
from urllib3.exceptions import InsecureRequestWarning
from requests.exceptions import RequestException

# Suppress only the specific InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LogoExtractor:
    def __init__(self, output_dir="logos"):
        """Initialize the logo extractor."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Common logo file names and paths
        self.common_logo_filenames = ['logo', 'brand', 'site-logo', 'header-logo', 'main-logo']
        self.common_logo_paths = ['images/logo', 'assets/logo', 'img/logo', 'static/logo', 'media/logo']
        
    def extract_domain(self, url):
        """Extract the domain from a URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed_url = urlparse(url)
        return parsed_url.netloc

    def download_image(self, img_url, domain):
        """Download an image and return it as a PIL Image object."""
        try:
            response = requests.get(img_url, headers=self.headers, timeout=10, verify=False)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Failed to download image from {img_url} for {domain}: {str(e)}")
            return None

    def is_valid_logo(self, img):
        """Check if an image is likely to be a logo."""
        if img is None:
            return False
            
        # Check dimensions (logos are typically not too large or too small)
        width, height = img.size
        if width < 16 or height < 16:
            return False
        if width > 800 or height > 800:
            return False
            
        # Check if image is not too complex (logos are usually simpler)
        try:
            # Convert to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Check if the image has a reasonable number of colors
            colors = img.getcolors(maxcolors=1024)
            if colors is None:  # More than 1024 colors
                return False
        except Exception:
            return False
            
        return True

    def find_logo_in_website(self, domain):
        """Extract the logo from a website."""
        if not domain:
            return None
            
        # Ensure domain has a scheme
        if not domain.startswith(('http://', 'https://')):
            url = 'https://' + domain
        else:
            url = domain
            
        try:
            # Try to fetch the website
            response = requests.get(url, headers=self.headers, timeout=15, verify=False)
            if response.status_code != 200:
                # Try with www. prefix if the original URL failed
                if not urlparse(url).netloc.startswith('www.'):
                    parsed = urlparse(url)
                    new_netloc = 'www.' + parsed.netloc
                    url = parsed._replace(netloc=new_netloc).geturl()
                    response = requests.get(url, headers=self.headers, timeout=15, verify=False)
                    
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for logo in various elements
            potential_logos = []
            
            # 1. Check meta tags for logo info
            meta_logos = soup.find_all('meta', property=['og:image', 'twitter:image'])
            for meta in meta_logos:
                if 'content' in meta.attrs:
                    potential_logos.append((meta['content'], 10))  # Higher score for meta tags
            
            # 2. Look for images with 'logo' in their attributes
            logo_imgs = []
            for img in soup.find_all('img'):
                img_src = img.get('src', '')
                img_alt = img.get('alt', '').lower()
                img_class = ' '.join(img.get('class', [])).lower()
                img_id = img.get('id', '').lower()
                
                score = 0
                # Check if it looks like a logo based on attributes
                if any(name in img_src.lower() for name in self.common_logo_filenames):
                    score += 5
                if any(name in img_alt for name in ['logo', 'brand']):
                    score += 3
                if any(name in img_class for name in ['logo', 'brand', 'header']):
                    score += 3
                if any(name in img_id for name in ['logo', 'brand', 'header']):
                    score += 3
                
                # Check position (logos are often in header)
                header = img.find_parent(['header', 'div'], class_=['header', 'navbar', 'nav', 'top'])
                if header:
                    score += 2
                
                if score > 0:
                    logo_imgs.append((img, score))
            
            # Sort by score and process
            logo_imgs.sort(key=lambda x: x[1], reverse=True)
            for img, score in logo_imgs:
                img_src = img.get('src', '')
                if not img_src:
                    continue
                
                # Make the URL absolute
                if not img_src.startswith(('http://', 'https://')):
                    img_src = urljoin(url, img_src)
                
                potential_logos.append((img_src, score))
            
            # 3. Look for favicons as a fallback
            favicon_links = soup.find_all('link', rel=['icon', 'shortcut icon', 'apple-touch-icon'])
            for link in favicon_links:
                if 'href' in link.attrs:
                    favicon_url = urljoin(url, link['href'])
                    potential_logos.append((favicon_url, 1))  # Lower score for favicons
            
            # Sort by score and try to download each potential logo
            potential_logos.sort(key=lambda x: x[1], reverse=True)
            
            for img_url, _ in potential_logos:
                logo_img = self.download_image(img_url, domain)
                if self.is_valid_logo(logo_img):
                    # Save the logo
                    clean_domain = self.extract_domain(domain)
                    logo_path = os.path.join(self.output_dir, f"{clean_domain}.png")
                    logo_img.save(logo_path)
                    logger.info(f"Saved logo for {clean_domain}")
                    return logo_path
            
            logger.warning(f"No valid logo found for {domain}")
            return None
            
        except RequestException as e:
            logger.error(f"Request error for {domain}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing {domain}: {str(e)}")
            return None

    def extract_logos_from_csv(self, csv_file):
        """Extract logos from a list of domains in a CSV file."""
        df = pd.read_csv(csv_file)
        domains = df['domain'].tolist()
        return self.extract_logos_from_list(domains)
        
    def extract_logos_from_list(self, domains):
        """Extract logos from a list of domains."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_domain = {executor.submit(self.find_logo_in_website, domain): domain for domain in domains}
            
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_domain), total=len(domains)):
                domain = future_to_domain[future]
                try:
                    logo_path = future.result()
                    results[domain] = logo_path
                except Exception as e:
                    logger.error(f"Error processing {domain}: {str(e)}")
                    results[domain] = None
        
        # Calculate success rate
        success_count = sum(1 for path in results.values() if path is not None)
        success_rate = success_count / len(domains) * 100
        logger.info(f"Logo extraction success rate: {success_rate:.2f}% ({success_count}/{len(domains)})")
        
        return results

class LogoSimilarityMatcher:
    def __init__(self, hash_method='phash', hash_size=32):
        """Initialize the logo similarity matcher."""
        self.hash_method = hash_method
        self.hash_size = hash_size
        self.hash_func = self._get_hash_function()
        
    def _get_hash_function(self):
        """Get the appropriate hash function."""
        if self.hash_method == 'phash':
            return lambda img: imagehash.phash(img, hash_size=self.hash_size)
        elif self.hash_method == 'dhash':
            return lambda img: imagehash.dhash(img, hash_size=self.hash_size)
        elif self.hash_method == 'ahash':
            return lambda img: imagehash.average_hash(img, hash_size=self.hash_size)
        elif self.hash_method == 'whash':
            return lambda img: imagehash.whash(img, hash_size=self.hash_size)
        else:
            # Default to phash
            return lambda img: imagehash.phash(img, hash_size=self.hash_size)
    
    def compute_hash(self, image_path):
        """Compute the hash for an image."""
        try:
            img = Image.open(image_path)
            return self.hash_func(img)
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {str(e)}")
            return None
            
    def compute_hashes(self, logo_paths):
        """Compute hashes for all logos."""
        hashes = {}
        for domain, path in logo_paths.items():
            if path is not None:
                hash_value = self.compute_hash(path)
                if hash_value is not None:
                    hashes[domain] = hash_value
        return hashes
        
    def calculate_similarity(self, hash1, hash2):
        """Calculate similarity between two hashes."""
        # Lower distance means more similar
        distance = hash1 - hash2
        # Convert to similarity score (0-100)
        max_distance = self.hash_size * self.hash_size
        similarity = 100 - (distance / max_distance) * 100
        return similarity
        
    def find_similar_logos(self, logo_hashes, threshold=90):
        """Find similar logos based on hash similarity."""
        domains = list(logo_hashes.keys())
        n = len(domains)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        # Compute pairwise similarities
        for i in range(n):
            for j in range(i+1, n):
                domain1, domain2 = domains[i], domains[j]
                hash1, hash2 = logo_hashes[domain1], logo_hashes[domain2]
                similarity = self.calculate_similarity(hash1, hash2)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Group similar logos
        groups = []
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
                
            domain = domains[i]
            group = [domain]
            visited.add(i)
            
            for j in range(n):
                if j == i or j in visited:
                    continue
                
                if similarity_matrix[i, j] >= threshold:
                    group.append(domains[j])
                    visited.add(j)
            
            groups.append(group)
        
        return groups

def main():
    # Create test CSV with the domains provided
    domains = [
        "stanbicbank.co.zw",
        "astrazeneca.ua",
        "autosecuritas-ct-seysses.fr",
        "ovb.ro",
        "mazda-autohaus-hellwig-hoyerswerda.de",
        "toyota-buchreiter-eisenstadt.at",
        "ebay.cn",
        "greatplacetowork.com.bo",
        "wurth-international.com",
        "plameco-hannover.de",
        "kia-moeller-wunstorf.de",
        "ccusa.co.nz",
        "tupperware.at",
        "zalando.cz",
        "crocs.com.uy",
        "ymcasteuben.org",
        "engie.co.uk",
        "ibc-solar.jp"
    ]
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({'domain': domains})
    csv_file = 'domains.csv'
    df.to_csv(csv_file, index=False)
    
    # Extract logos
    extractor = LogoExtractor(output_dir="extracted_logos")
    logo_paths = extractor.extract_logos_from_csv(csv_file)
    
    # Match similar logos
    matcher = LogoSimilarityMatcher(hash_method='phash', hash_size=32)
    logo_hashes = matcher.compute_hashes(logo_paths)
    
    # Try different thresholds to find optimal grouping
    thresholds = [90, 85, 80, 75, 70]
    best_groups = []
    best_threshold = 0
    
    for threshold in thresholds:
        groups = matcher.find_similar_logos(logo_hashes, threshold=threshold)
        logger.info(f"Threshold {threshold}: Found {len(groups)} groups")
        
        # Simple heuristic: prefer more groups with reasonable sizes
        group_sizes = [len(g) for g in groups]
        avg_size = sum(group_sizes) / len(groups)
        score = len(groups) * (1 / (avg_size if avg_size > 1 else 1))
        
        if not best_groups or score > best_score:
            best_groups = groups
            best_threshold = threshold
            best_score = score
    
    logger.info(f"Best threshold: {best_threshold} with {len(best_groups)} groups")
    
    # Save results
    results = []
    for i, group in enumerate(best_groups):
        group_dict = {
            'group_id': i + 1,
            'websites': group,
            'size': len(group)
        }
        results.append(group_dict)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('logo_groups.csv', index=False)
    
    # Print results
    for i, group in enumerate(best_groups):
        print(f"Group {i+1} ({len(group)} websites):")
        for domain in group:
            print(f"  - {domain}")
        print()

if __name__ == "__main__":
    main()
