import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Read the file and parse timestamps
timestamps = []
with open(r'C:\Users\domdd\Documents\GitHub\confluence\trump_tweets_20251026.txt', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header

    for row in reader:
        if row and row[0]:  # Check if row exists and has timestamp
            try:
                timestamp_str = row[0]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(timestamp)
            except (ValueError, IndexError):
                continue

# Sort timestamps chronologically
timestamps.sort()

print(f"Total tweets analyzed: {len(timestamps)}")
print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
print("\n" + "="*80 + "\n")

# Question 1: How many tweets were posted within the same 5-minute period as the previous tweet?
tweets_within_5min_of_previous = 0
for i in range(1, len(timestamps)):
    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
    if time_diff <= 5:
        tweets_within_5min_of_previous += 1

print(f"QUESTION 1: Tweets posted within 5 minutes of the previous tweet")
print(f"Count: {tweets_within_5min_of_previous} out of {len(timestamps)-1} tweet pairs")
print(f"Percentage: {tweets_within_5min_of_previous/(len(timestamps)-1)*100:.2f}%")
print("\n" + "="*80 + "\n")

# Question 2 & 3: Group tweets into 5-minute windows and analyze distribution
# Round each timestamp down to its 5-minute window
def round_to_5min_window(dt):
    """Round datetime down to the nearest 5-minute window"""
    minutes = (dt.minute // 5) * 5
    return dt.replace(minute=minutes, second=0, microsecond=0)

window_counts = Counter()
for ts in timestamps:
    window = round_to_5min_window(ts)
    window_counts[window] += 1

# Calculate statistics
total_windows = len(window_counts)
total_tweets = len(timestamps)
avg_tweets_per_window = total_tweets / total_windows if total_windows > 0 else 0

print(f"QUESTION 2: Distribution of tweets by 5-minute intervals")
print(f"Total 5-minute windows with tweets: {total_windows}")
print(f"Average tweets per 5-minute window: {avg_tweets_per_window:.2f}")
print(f"\nDistribution breakdown:")

# Show distribution histogram
distribution = Counter(window_counts.values())
for count in sorted(distribution.keys()):
    num_windows = distribution[count]
    print(f"  {num_windows} windows with {count} tweet(s) ({num_windows/total_windows*100:.1f}%)")

print("\n" + "="*80 + "\n")

# Question 3: Find the 5-minute period with the most tweets
print(f"QUESTION 3: 5-minute period with the most tweets")
max_count = max(window_counts.values())
max_windows = [window for window, count in window_counts.items() if count == max_count]

print(f"Maximum tweets in a single 5-minute window: {max_count}")
print(f"\nTop 10 busiest 5-minute windows:")

for i, (window, count) in enumerate(window_counts.most_common(10), 1):
    window_end = window + timedelta(minutes=5)
    print(f"  {i}. {window.strftime('%Y-%m-%d %H:%M')} - {window_end.strftime('%H:%M')}: {count} tweets")

print("\n" + "="*80 + "\n")

# Additional insights
print("ADDITIONAL INSIGHTS:")
print(f"\nTime gaps between consecutive tweets:")
gaps = []
for i in range(1, len(timestamps)):
    gap_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
    gaps.append(gap_seconds)

gaps.sort()
print(f"  Minimum gap: {min(gaps):.2f} seconds")
print(f"  Maximum gap: {max(gaps)/3600:.2f} hours")
print(f"  Median gap: {gaps[len(gaps)//2]/60:.2f} minutes")
print(f"  Average gap: {sum(gaps)/len(gaps)/60:.2f} minutes")

# Count rapid-fire tweeting (within 1 second)
rapid_fire = sum(1 for gap in gaps if gap <= 1)
print(f"\nTweets posted within 1 second of previous: {rapid_fire} ({rapid_fire/len(gaps)*100:.2f}%)")

# Count burst periods (multiple tweets in same minute)
def round_to_minute(dt):
    return dt.replace(second=0, microsecond=0)

minute_counts = Counter()
for ts in timestamps:
    minute = round_to_minute(ts)
    minute_counts[minute] += 1

minutes_with_bursts = sum(1 for count in minute_counts.values() if count > 1)
print(f"Minutes with multiple tweets: {minutes_with_bursts}")
print(f"Maximum tweets in a single minute: {max(minute_counts.values())}")
