import os
from dotenv import load_load
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time
import hashlib

# Load environment variables
load_dotenv()

class GoogleImageDownloader:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_output_dir = Path('downloaded_images')
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Validate credentials
        if not self.api_key or not self.search_engine_id:
            raise ValueError("Missing required environment variables. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID")
    
    def search_images(self, query, num_images=5):
        """
        Search for images using Google Custom Search API
        
        Args:
            query (str): Search query
            num_images (int): Number of images to retrieve (max 10 per request)
            
        Returns:
            list: List of image URLs
        """
        search_url = "https://www.googleapis.com/customsearch/v1"
        image_urls = []
        
        # Google CSE only allows 10 results per request, so we need to paginate
        for start_index in range(1, num_images + 1, 10):
            params = {
                'q': query,
                'cx': self.search_engine_id,
                'key': self.api_key,
                'searchType': 'image',
                'num': min(10, num_images - start_index + 1),
                'start': start_index,
                'safe': 'high',
                'imgSize': 'large'  # Prefer large images
            }
            
            try:
                response = requests.get(search_url, params=params)
                response.raise_for_status()
                results = response.json()
                
                if 'items' in results:
                    image_urls.extend(item['link'] for item in results['items'])
                
                # Respect rate limits
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error searching for '{query}': {e}")
                break
                
        return image_urls[:num_images]
    
    def download_image(self, url, output_path):
        """
        Download and save an image from a URL
        
        Args:
            url (str): Image URL
            output_path (Path): Path to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary (e.g., for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save the image
            img.save(output_path, 'JPEG', quality=85)
            return True
            
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return False
    
    def download_images_for_keywords(self, keywords, images_per_keyword=5):
        """
        Download images for multiple keywords
        
        Args:
            keywords (list): List of search keywords
            images_per_keyword (int): Number of images to download per keyword
        """
        for keyword in keywords:
            # Create sanitized folder name
            folder_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in keyword)
            folder_name = folder_name.strip().replace(' ', '_')
            
            # Create folder for this keyword
            keyword_dir = self.base_output_dir / folder_name
            keyword_dir.mkdir(exist_ok=True)
            
            print(f"\nSearching images for: {keyword}")
            image_urls = self.search_images(keyword, images_per_keyword)
            
            for index, url in enumerate(image_urls, 1):
                # Create unique filename using hash of URL
                url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
                image_path = keyword_dir / f"image_{index}_{url_hash}.jpg"
                
                print(f"Downloading image {index}/{len(image_urls)} for '{keyword}'")
                if self.download_image(url, image_path):
                    print(f"Successfully downloaded: {image_path}")
                
                # Add delay between downloads
                time.sleep(0.5)

def main():
    # Example usage
    keywords = [
        "Mountain landscape",
        "Ocean sunset",
        "Forest wildlife",
        "Urban architecture",
        "Desert dunes"
    ]
    
    try:
        downloader = GoogleImageDownloader()
        downloader.download_images_for_keywords(keywords, images_per_keyword=3)
        print("\nImage download process completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()