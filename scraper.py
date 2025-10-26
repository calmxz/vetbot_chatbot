# scrape_aspca.py
# Install: pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

def scrape_aspca_articles(article_urls, output_folder='./documents/aspca_scraped/cats'):
    """
    Scrape ASPCA articles directly from provided URLs
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 70)
    print("ASPCA CAT CARE ARTICLES SCRAPER")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(article_urls, 1):
        print(f"\n[{i}/{len(article_urls)}] Scraping: {url}")
        
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()
            
            # Extract title
            title = None
            title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
            else:
                title = "Untitled"
            
            print(f"   Title: {title}")
            
            # Extract main content - ASPCA specific structure
            # Remove navigation, headers, footers before extracting
            for unwanted in soup.find_all(['nav', 'header', 'footer', 'aside', 'form', 'button']):
                unwanted.decompose()
            
            # ASPCA content is in specific divs - try multiple selectors
            main_content = (
                soup.find('div', class_='field-item') or
                soup.find('div', class_='content') or
                soup.find('article') or 
                soup.find('main')
            )
            
            if main_content:
                # Remove any remaining unwanted sections
                for unwanted_class in ['donate-banner', 'related-content', 'sidebar', 'menu']:
                    for elem in main_content.find_all(class_=lambda x: x and unwanted_class in str(x).lower()):
                        elem.decompose()
                
                # Get all paragraphs, headings, and list items (not ul/ol containers)
                content_elements = main_content.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                
                if content_elements:
                    text_parts = []
                    for elem in content_elements:
                        # Add spaces around inline formatting tags before extracting text
                        for inline_tag in elem.find_all(['strong', 'b', 'em', 'i', 'a', 'span']):
                            if inline_tag.string:
                                inline_tag.string.replace_with(f" {inline_tag.string} ")
                        
                        text = elem.get_text(strip=True)
                        # Clean up multiple spaces
                        text = ' '.join(text.split())
                        
                        # Keep headings even if short
                        is_heading = elem.name in ['h2', 'h3', 'h4']
                        
                        # Filter criteria
                        is_long_enough = len(text) > 20 if not is_heading else len(text) > 0
                        is_not_promo = not any(skip in text.lower() for skip in ['donate', 'help the aspca', 'sign up for', 'shop now'])
                        
                        if text and is_long_enough and is_not_promo:
                            # Add visual separation for headings
                            if is_heading:
                                text_parts.append(f"\n{text}\n")
                            else:
                                text_parts.append(text)
                    
                    content = '\n\n'.join(text_parts)
                else:
                    # Fallback to full text
                    content = main_content.get_text(separator='\n\n', strip=True)
                
                print(f"   Extracted {len(content)} characters")
                
                # Only save if we got substantial content
                if len(content) > 200:
                    # Create filename from URL
                    filename = url.split('/')[-1]
                    if not filename.endswith('.txt'):
                        filename = filename + '.txt'
                    
                    filepath = os.path.join(output_folder, filename)
                    
                    # Save article with metadata
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {title}\n")
                        f.write(f"Source: {url}\n")
                        f.write(f"Category: ASPCA Dog Care\n")
                        f.write(f"{'=' * 70}\n\n")
                        f.write(content)
                    
                    print(f"   ✓ Saved: {filename}")
                    successful += 1
                else:
                    print(f"   ✗ Insufficient content (only {len(content)} characters)")
                    failed += 1
            else:
                print(f"   ✗ Could not find main content")
                failed += 1
        
        except requests.exceptions.RequestException as e:
            print(f"   ✗ Request failed: {e}")
            failed += 1
        except Exception as e:
            print(f"   ✗ Error: {e}")
            failed += 1
        
        # Be polite - wait between requests
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📁 Files saved to: {os.path.abspath(output_folder)}")
    print("=" * 70)
    
    if successful > 0:
        print("\nNext steps:")
        print("1. Check the scraped files in", output_folder)
        print("2. Copy files to your main documents folder:")
        print(f"   cp {output_folder}/* ./documents/")
        print("   (Windows: copy {output_folder}\\* documents\\)")
        print("3. Delete 'chroma_db' folder if it exists")
        print("4. Run your chatbot to index the new articles!")

# ============================================================================
# ARTICLE URLs - Add any ASPCA dog care articles you want
# ============================================================================

if __name__ == "__main__":
    # List of ASPCA dog care article URLs to scrape
    article_urls = [
        # Grooming
        "https://www.aspca.org/pet-care/cat-care/cat-grooming-tips",
        
        # General Care
        "https://www.aspca.org/pet-care/cat-care/general-cat-care",
        "https://www.aspca.org/pet-care/general-pet-care/vaccinations-your-pet",
        
        # Nutrition
        "https://www.aspca.org/pet-care/cat-care/cat-nutrition-tips",
        
        # Diseases
        "https://www.aspca.org/pet-care/cat-care/common-cat-diseases",
        "https://www.aspca.org/pet-care/general-pet-care/fleas-and-ticks",
        
        # Behavior
        "https://www.aspca.org/pet-care/cat-care/common-cat-behavior-issues",
        
        # Safety
        "https://www.aspca.org/pet-care/dog-care/dog-bite-prevention",
        "https://www.aspca.org/pet-care/general-pet-care/spayneuter-your-pet",
        
        # Add more URLs here as needed
    ]
    
    print(f"\nPreparing to scrape {len(article_urls)} ASPCA articles...\n")
    scrape_aspca_articles(article_urls)