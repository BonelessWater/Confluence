"""
Convert CSV tweet data to parquet format with embeddings.

This script converts the CSV file (tweet_analysis/trump_tweets_20251026.txt) 
to the required parquet format. It will:
1. Parse the CSV file
2. Clean HTML tags from tweet content
3. Generate embeddings (using sentence-transformers or OpenAI)
4. Save as parquet file in the expected location
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.settings import TWEET_PARQUET, DATA_DIR

def clean_html(text):
    """Remove HTML tags and entities from text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Clean up whitespace
    text = ' '.join(text.split())
    return text.strip()

def generate_embeddings(texts, method='sentence-transformers'):
    """
    Generate embeddings for tweet texts.
    
    Args:
        texts: List of tweet texts
        method: 'sentence-transformers' or 'openai'
    
    Returns:
        Array of embeddings
    """
    print(f"\nGenerating embeddings using {method}...")
    
    if method == 'sentence-transformers':
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
            print("  Loading sentence-transformers model...")
            embeddings = model.encode(texts, show_progress_bar=True)
            print(f"  Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            return embeddings
        except ImportError:
            print("  sentence-transformers not installed. Install with: pip install sentence-transformers")
            print("  Falling back to random embeddings (for testing only)")
            return np.random.randn(len(texts), 384).astype(np.float32)
    
    elif method == 'openai':
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                print("  OPENAI_API_KEY not set. Falling back to random embeddings")
                return np.random.randn(len(texts), 1536).astype(np.float32)
            
            print("  Using OpenAI embeddings...")
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
            
            return np.array(embeddings)
        except Exception as e:
            print(f"  Error with OpenAI: {e}")
            print("  Falling back to random embeddings")
            return np.random.randn(len(texts), 1536).astype(np.float32)
    
    else:
        # Fallback: random embeddings (for testing)
        print("  Using random embeddings (for testing only)")
        return np.random.randn(len(texts), 384).astype(np.float32)

def convert_csv_to_parquet(csv_path, output_path=None):
    """
    Convert CSV tweet file to parquet format.
    
    Args:
        csv_path: Path to CSV file
        output_path: Output parquet path (uses config default if None)
    """
    if output_path is None:
        output_path = TWEET_PARQUET
    
    print("="*80)
    print("CONVERTING CSV TO PARQUET FORMAT")
    print("="*80)
    
    # Read CSV
    print(f"\nReading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Parse and clean data
    print("\nCleaning and parsing data...")
    tweets_data = []
    
    for idx, row in df.iterrows():
        # Extract tweet content
        headline = row.get('Headline', '')
        content = clean_html(headline)
        
        # Skip empty tweets
        if not content or len(content) < 5:
            continue
        
        # Parse timestamp
        insert_time = row.get('InsertTime', '')
        headline_date = row.get('HeadlineDate', '')
        headline_time = row.get('HeadlineTime', '')
        
        # Try to parse timestamp
        try:
            if insert_time:
                created_at = pd.to_datetime(insert_time)
            elif headline_date and headline_time:
                datetime_str = f"{headline_date} {headline_time}"
                created_at = pd.to_datetime(datetime_str)
            else:
                created_at = pd.Timestamp.now()
        except:
            created_at = pd.Timestamp.now()
        
        tweets_data.append({
            'id': f"tweet_{idx}",
            'content': content,
            'created_at': created_at,
            'replies_count': 0,  # Not in CSV, set defaults
            'reblogs_count': 0,
            'favourites_count': 0,
            'url': ''
        })
    
    tweets_df = pd.DataFrame(tweets_data)
    print(f"  Processed {len(tweets_df)} valid tweets")
    
    # Generate embeddings
    texts = tweets_df['content'].tolist()
    embeddings = generate_embeddings(texts, method='sentence-transformers')
    
    # Add embeddings to dataframe
    tweets_df['embedding'] = embeddings.tolist()
    
    # Ensure datetime is timezone-naive
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at']).dt.tz_localize(None)
    
    # Sort by timestamp
    tweets_df = tweets_df.sort_values('created_at').reset_index(drop=True)
    
    print(f"\nFinal dataset:")
    print(f"  Tweets: {len(tweets_df)}")
    print(f"  Date range: {tweets_df['created_at'].min()} to {tweets_df['created_at'].max()}")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    
    # Save to parquet
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tweets_df.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved! File size: {file_size_mb:.2f} MB")
    
    return tweets_df

def main():
    """Main function."""
    csv_path = Path(__file__).parent.parent / 'tweet_analysis' / 'trump_tweets_20251026.txt'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("\nPlease ensure the tweet CSV file exists.")
        return
    
    try:
        tweets_df = convert_csv_to_parquet(csv_path)
        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print("="*80)
        print(f"\nYou can now run:")
        print("  python scripts/precompute_features.py")
        print("\nto generate the features file.")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
