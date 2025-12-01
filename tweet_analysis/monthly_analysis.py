import csv
import re
from collections import defaultdict
from datetime import datetime

# Read the CSV file
tweets_per_month = defaultdict(int)
tweets_per_weekday = defaultdict(int)
tweets_per_hour = defaultdict(int)

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
        text_only = text_only.strip()

        # Skip if no actual content after cleaning
        if not text_only or len(text_only) < 2:
            continue

        # Parse timestamp
        timestamp_str = row.get('InsertTime', '')
        if timestamp_str:
            try:
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                month_key = dt.strftime('%Y-%m')
                tweets_per_month[month_key] += 1

                weekday = dt.strftime('%A')
                tweets_per_weekday[weekday] += 1

                hour = dt.hour
                tweets_per_hour[hour] += 1
            except:
                pass

print("=" * 80)
print("MONTHLY AND TEMPORAL DISTRIBUTION")
print("=" * 80)
print()

# Monthly distribution
print("TWEETS PER MONTH:")
for month in sorted(tweets_per_month.keys()):
    count = tweets_per_month[month]
    bar = '#' * (count // 50)
    print(f"   {month}: {count:>5} tweets {bar}")
print()

# Weekday distribution
print("TWEETS PER WEEKDAY:")
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
max_weekday = max(tweets_per_weekday.values())
for day in weekdays:
    count = tweets_per_weekday.get(day, 0)
    bar = '#' * int((count / max_weekday) * 50)
    print(f"   {day:>10}: {count:>5} tweets {bar}")
print()

# Hourly distribution
print("TWEETS PER HOUR (24-hour format):")
max_hour = max(tweets_per_hour.values())
for hour in range(24):
    count = tweets_per_hour.get(hour, 0)
    bar = '#' * int((count / max_hour) * 50)
    time_label = f"{hour:02d}:00"
    print(f"   {time_label}: {count:>5} tweets {bar}")
print()

print("=" * 80)
