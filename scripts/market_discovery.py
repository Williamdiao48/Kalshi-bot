import requests
import logging
import base64
import time
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FEDERAL_REGISTER_API_BASE = "https://www.federalregister.gov/api/v1"
KALSHI_ENVIRONMENT = "demo" # Change to "production" if your key is for the live API
KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2" if KALSHI_ENVIRONMENT == "production" else "https://demo-api.kalshi.co/trade-api/v2"
EPA_AGENCY_SLUG = "environmental-protection-agency"
TARGET_TOPICS = ["Biomass", "Carbon", "Methane"]

# Load environment variables from .env file
load_dotenv()

# Kalshi V2 Authentication (Loaded from .env)
# 1. Provide your API Key ID
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")

# 2. Provide your private key string (from the .key file you downloaded)
KALSHI_PRIVATE_KEY_STR = os.environ.get("KALSHI_PRIVATE_KEY_STR", "")
# Fix python-dotenv multiline string issue if the user didn't wrap the .env var in quotes
if "\\n" in KALSHI_PRIVATE_KEY_STR:
    KALSHI_PRIVATE_KEY_STR = KALSHI_PRIVATE_KEY_STR.replace("\\n", "\n")

def generate_kalshi_headers(method: str, path: str) -> dict:
    """Generates the RSA-PSS signature headers required for Kalshi V2 API requests."""
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY_STR:
        logging.debug("Kalshi credentials not set. Proceeding without authentication headers.")
        return {}

    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + method + path
    
    try:
        private_key = serialization.load_pem_private_key(
            KALSHI_PRIVATE_KEY_STR.encode('utf-8'),
            password=None,
        )
        
        signature = private_key.sign(
            msg_string.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Kalshi expects base64 encoded signature
        encoded_signature = base64.b64encode(signature).decode('utf-8')
        
        return {
            "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": encoded_signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }
    except Exception as e:
        logging.error(f"Failed to generate Kalshi auth signature. Check your Private Key structure. Error: {e}")
        return {}

def fetch_epa_documents(agency_slug: str = EPA_AGENCY_SLUG) -> List[Dict[str, Any]]:
    """
    Fetches the latest documents from the Federal Register API for a specific agency.
    """
    url = f"{FEDERAL_REGISTER_API_BASE}/documents.json"
    params = {
        "conditions[agencies][]": agency_slug,
        "order": "newest",
        "per_page": 50
    }
    
    try:
        logging.info(f"Fetching recent documents for agency: {agency_slug}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching EPA documents: {e}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error fetching EPA documents: {e}")
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout error fetching EPA documents: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"An unexpected error occurred while fetching EPA documents: {e}")
    
    return []

def fetch_kalshi_markets() -> List[Dict[str, Any]]:
    """
    Fetches the list of current active markets from the Kalshi API.
    Note: Depending on the environment (demo/prod), authentication may be required.
    For public market discovery, standard GET requests often suffice for Kalshi.
    """
    url = f"{KALSHI_API_BASE}/markets"
    params = {
        "status": "open", # Kalshi v2 uses 'open', 'closed', 'settled'
        "limit": 100
    }
    
    try:
        logging.info("Fetching active markets from Kalshi API...")
        
        # Generate the RSA-PSS auth signature
        headers = generate_kalshi_headers(method="GET", path="/trade-api/v2/markets")
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('markets', [])
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching Kalshi markets: {e}")
        if e.response is not None:
            logging.error(f"Kalshi API error response: {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error fetching Kalshi markets: {e}")
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout error fetching Kalshi markets: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"An unexpected error occurred while fetching Kalshi markets: {e}")
        
    return []

def find_trading_opportunities(epa_docs: List[Dict[str, Any]], kalshi_markets: List[Dict[str, Any]], topics: List[str]) -> List[Dict[str, Any]]:
    """
    Compares EPA documents with Kalshi markets to find overlapping topics.
    Returns a list of identified opportunities.
    """
    opportunities = []
    
    # Lowercase topics for case-insensitive matching
    lower_topics = [t.lower() for t in topics]
    
    for doc in epa_docs:
        doc_title = doc.get("title", "")
        doc_abstract = doc.get("abstract", "")
        # Combined text to search through
        doc_text = f"{doc_title} {doc_abstract}".lower()
        
        # Check if the document mentions any of our target topics
        matched_topics = [topic for topic in lower_topics if topic in doc_text]
        
        if not matched_topics:
            continue
            
        # If topics matched, search for Kalshi markets related to these topics
        for market in kalshi_markets:
            market_title = market.get("title", "").lower()
            market_ticker = market.get("ticker", "").lower()
            
            # Note: Kalshi v2 may return price in cents directly or might require orderbook endpoint
            # We assume 'last_price' is present. Adapt based on live payload.
            market_price = market.get("last_price", "N/A") 
            
            # Simplified matching logic: check if the matched topic is in the market title/ticker
            for topic in matched_topics:
                if topic in market_title or topic in market_ticker:
                    opportunities.append({
                        "topic": topic.capitalize(),
                        "market_id": market.get("ticker"),
                        "market_title": market.get("title"),
                        "current_price": market_price,
                        "epa_document_title": doc_title,
                        "epa_document_url": doc.get("html_url", "No URL provided"),
                    })
                    break # Stop checking other topics for this specific market-doc pair once matched

    return opportunities

def main():
    logging.info("Starting Market Discovery Tool")
    
    # 1. Fetch EPA Documents
    epa_docs = fetch_epa_documents(EPA_AGENCY_SLUG)
    if not epa_docs:
        logging.warning("No EPA documents found or an error occurred. Exiting.")
        return
        
    logging.info(f"Successfully retrieved {len(epa_docs)} EPA documents.")
    
    # Optional Output: Print the EPA documents
    print("\n" + "="*60)
    print("📰 RECENT EPA DOCUMENTS 📰")
    print("="*60)
    for idx, doc in enumerate(epa_docs[:10], 1): # Print top 10 to avoid huge spam
        title = doc.get("title", "No Title")
        url = doc.get("html_url", "No URL")
        print(f"[{idx}] {title}")
        print(f"    Link: {url}")
    print("="*60 + "\n")

    # 2. Fetch Kalshi Markets
    kalshi_markets = fetch_kalshi_markets()
    if not kalshi_markets:
        logging.warning("No Kalshi markets found or an error occurred. Exiting.")
        return
        
    logging.info(f"Successfully retrieved {len(kalshi_markets)} active markets from Kalshi.")

    # 3. Compare and Identify Opportunities
    logging.info(f"Comparing data for topics: {', '.join(TARGET_TOPICS)}")
    opportunities = find_trading_opportunities(epa_docs, kalshi_markets, TARGET_TOPICS)
    
    # 4. Output Results
    if not opportunities:
        logging.info("No trading opportunities identified based on the current data.")
    else:
        logging.info(f"Found {len(opportunities)} potential trading opportunity(ies)!\n")
        print("="*60)
        print("💡 MARKET DISCOVERY REPORT 💡")
        print("="*60)
        
        for idx, opt in enumerate(opportunities, 1):
            print(f"Opportunity {idx}:")
            print(f"  - Topic:              {opt['topic']}")
            print(f"  - Market ID (Ticker): {opt['market_id']}")
            print(f"  - Market Title:       {opt['market_title']}")
            print(f"  - Current Price:      {opt['current_price']}¢")
            print(f"  - EPA Document:       {opt['epa_document_title']}")
            print(f"  - Document URL:       {opt['epa_document_url']}")
            print("-" * 60)

if __name__ == "__main__":
    main()
