import csv
import re
from datetime import datetime
from collections import Counter, defaultdict
import statistics

# Read the CSV file
tweets = []
dates = []
tweet_lengths = []
tweets_per_day = defaultdict(int)

file_path = r'C:\Users\domdd\Documents\GitHub\confluence\trump_tweets_20251026.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        headline = row.get('Headline', '')

        # Skip empty tweets or system messages
        if not headline or headline.startswith('STARTING TRUTH_TRUMP_API'):
            continue

        # Clean HTML tags to get actual text
        text_only = re.sub(r'<[^>]+>', '', headline)
        text_only = re.sub(r'&[a-z]+;', ' ', text_only)  # Remove HTML entities
        text_only = text_only.strip()

        # Skip if no actual content after cleaning
        if not text_only or len(text_only) < 2:
            continue

        tweets.append({
            'timestamp': row.get('InsertTime', ''),
            'date': row.get('HeadlineDate', ''),
            'time': row.get('HeadlineTime', ''),
            'text': text_only,
            'length': len(headline)  # Original length with HTML
        })

        # Parse date
        date_str = row.get('HeadlineDate', '')
        if date_str:
            dates.append(date_str)
            tweets_per_day[date_str] += 1

        # Track lengths
        tweet_lengths.append(len(headline))

# Calculate statistics
total_tweets = len(tweets)
print(f"=" * 80)
print(f"TRUMP TWEETS ANALYSIS - File: trump_tweets_20251026.txt")
print(f"=" * 80)
print()

# 1. Total number of tweets
print(f"1. TOTAL NUMBER OF TWEETS: {total_tweets:,}")
print()

# 2. Date range
if dates:
    dates_sorted = sorted(dates)
    earliest_date = dates_sorted[0]
    latest_date = dates_sorted[-1]

    print(f"3. EARLIEST TWEET DATE: {earliest_date}")
    print(f"4. LATEST TWEET DATE: {latest_date}")
    print()

    # Calculate date range
    start = datetime.strptime(earliest_date, '%Y-%m-%d')
    end = datetime.strptime(latest_date, '%Y-%m-%d')
    days_span = (end - start).days + 1
    print(f"   DATE RANGE: {days_span} days")
    print()

# 2. Tweets per day statistics
if tweets_per_day:
    daily_counts = list(tweets_per_day.values())
    avg_per_day = statistics.mean(daily_counts)
    median_per_day = statistics.median(daily_counts)
    max_per_day = max(daily_counts)
    min_per_day = min(daily_counts)

    print(f"2. TWEETS PER DAY:")
    print(f"   - Average: {avg_per_day:.2f} tweets/day")
    print(f"   - Median: {median_per_day:.0f} tweets/day")
    print(f"   - Maximum: {max_per_day} tweets/day")
    print(f"   - Minimum: {min_per_day} tweets/day")
    print(f"   - Total days with tweets: {len(tweets_per_day)}")

    # Find busiest days
    top_days = sorted(tweets_per_day.items(), key=lambda x: x[1], reverse=True)[:10]
    print()
    print("   TOP 10 BUSIEST DAYS:")
    for date, count in top_days:
        print(f"      {date}: {count} tweets")
    print()

# 5 & 6. Tweet length statistics
if tweet_lengths:
    avg_length = statistics.mean(tweet_lengths)
    median_length = statistics.median(tweet_lengths)
    max_length = max(tweet_lengths)
    min_length = min(tweet_lengths)

    print(f"5. AVERAGE TWEET LENGTH: {avg_length:.2f} characters")
    print()

    print(f"6. TWEET LENGTH DISTRIBUTION:")
    print(f"   - Minimum: {min_length} characters")
    print(f"   - Maximum: {max_length} characters")
    print(f"   - Median: {median_length:.0f} characters")
    print(f"   - Standard Deviation: {statistics.stdev(tweet_lengths):.2f} characters")

    # Length buckets
    buckets = {
        '0-100': 0,
        '101-200': 0,
        '201-300': 0,
        '301-500': 0,
        '501-1000': 0,
        '1000+': 0
    }

    for length in tweet_lengths:
        if length <= 100:
            buckets['0-100'] += 1
        elif length <= 200:
            buckets['101-200'] += 1
        elif length <= 300:
            buckets['201-300'] += 1
        elif length <= 500:
            buckets['301-500'] += 1
        elif length <= 1000:
            buckets['501-1000'] += 1
        else:
            buckets['1000+'] += 1

    print()
    print("   LENGTH DISTRIBUTION BY BUCKET:")
    for bucket, count in buckets.items():
        pct = (count / total_tweets) * 100
        print(f"      {bucket:>12} chars: {count:>6} tweets ({pct:5.2f}%)")
    print()

# 7. Additional interesting statistics
print(f"7. ADDITIONAL STATISTICS:")
print()

# Retweets
retweets = sum(1 for t in tweets if 'RT ' in t['text'] or 'RT:' in t['text'])
original_tweets = total_tweets - retweets
print(f"   - Retweets (RT): {retweets:,} ({(retweets/total_tweets)*100:.2f}%)")
print(f"   - Original tweets: {original_tweets:,} ({(original_tweets/total_tweets)*100:.2f}%)")
print()

# Common words (excluding RT, common articles, etc)
all_text = ' '.join([t['text'] for t in tweets])
words = re.findall(r'\b[A-Z]{2,}\b', all_text)  # All caps words
word_counts = Counter(words)
stop_words = {'RT', 'THE', 'TO', 'AND', 'OF', 'IN', 'FOR', 'IS', 'ON', 'AT', 'BY', 'A', 'AN'}
filtered_words = {w: c for w, c in word_counts.items() if w not in stop_words and len(w) > 2}

print("   TOP 20 ALL-CAPS WORDS (often used for emphasis):")
for word, count in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"      {word}: {count}")
print()

# URLs
url_count = sum(1 for t in tweets if 'http' in t['text'] or 'www.' in t['text'])
print(f"   - Tweets with URLs: {url_count:,} ({(url_count/total_tweets)*100:.2f}%)")
print()

# Hashtags
hashtag_pattern = re.compile(r'#\w+')
all_hashtags = []
for t in tweets:
    hashtags = hashtag_pattern.findall(t['text'])
    all_hashtags.extend(hashtags)

if all_hashtags:
    hashtag_counts = Counter(all_hashtags)
    print(f"   - Total hashtags used: {len(all_hashtags):,}")
    print(f"   - Unique hashtags: {len(hashtag_counts):,}")
    print()
    print("   TOP 10 HASHTAGS:")
    for tag, count in hashtag_counts.most_common(10):
        print(f"      {tag}: {count}")
    print()

# Mentions
mention_pattern = re.compile(r'@\w+')
all_mentions = []
for t in tweets:
    mentions = mention_pattern.findall(t['text'])
    all_mentions.extend(mentions)

if all_mentions:
    mention_counts = Counter(all_mentions)
    print(f"   - Total mentions: {len(all_mentions):,}")
    print(f"   - Unique accounts mentioned: {len(mention_counts):,}")
    print()
    print("   TOP 10 MENTIONED ACCOUNTS:")
    for mention, count in mention_counts.most_common(10):
        print(f"      {mention}: {count}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
