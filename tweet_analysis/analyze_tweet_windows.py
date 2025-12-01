import csv
from datetime import datetime, timedelta
from collections import defaultdict

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object"""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

def merge_intervals(intervals):
    """Merge overlapping time intervals"""
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        # If current interval overlaps with last merged interval
        if current[0] <= last[1]:
            # Merge by extending the end time if necessary
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add as new interval
            merged.append(current)

    return merged

def calculate_total_minutes(merged_intervals):
    """Calculate total minutes covered by merged intervals"""
    total_seconds = sum((end - start).total_seconds() for start, end in merged_intervals)
    return total_seconds / 60

def main():
    file_path = r"C:\Users\domdd\Documents\GitHub\confluence\trump_tweets_20251026.txt"

    # Read all tweets and create 5-minute windows
    intervals = []
    tweets_by_day = defaultdict(list)

    print("Reading tweets and creating 5-minute windows...")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        tweet_count = 0

        for row in reader:
            try:
                insert_time_str = row['InsertTime']
                tweet_time = parse_timestamp(insert_time_str)
                end_time = tweet_time + timedelta(minutes=5)

                intervals.append((tweet_time, end_time))

                # Group by day for daily analysis
                day = tweet_time.date()
                tweets_by_day[day].append((tweet_time, end_time))

                tweet_count += 1

                if tweet_count % 1000 == 0:
                    print(f"  Processed {tweet_count} tweets...")

            except (ValueError, KeyError) as e:
                # Skip malformed rows
                continue

    print(f"\nTotal tweets processed: {tweet_count}")
    print(f"Total individual 5-minute windows: {len(intervals)}")

    # Merge overlapping intervals
    print("\nMerging overlapping time windows...")
    merged_intervals = merge_intervals(intervals)

    # Calculate statistics
    total_minutes = calculate_total_minutes(merged_intervals)
    naive_minutes = len(intervals) * 5
    minutes_saved = naive_minutes - total_minutes
    overlap_percentage = (minutes_saved / naive_minutes) * 100 if naive_minutes > 0 else 0

    print("\n" + "="*70)
    print("OVERALL ANALYSIS")
    print("="*70)
    print(f"Total tweets: {tweet_count:,}")
    print(f"Total 5-minute windows (before merging): {len(intervals):,}")
    print(f"Total consolidated windows (after merging): {len(merged_intervals):,}")
    print(f"\nNaive approach (no consolidation): {naive_minutes:,.2f} minutes ({naive_minutes/60:,.2f} hours)")
    print(f"Consolidated approach: {total_minutes:,.2f} minutes ({total_minutes/60:,.2f} hours)")
    print(f"Minutes saved by consolidation: {minutes_saved:,.2f} minutes ({minutes_saved/60:,.2f} hours)")
    print(f"Overlap percentage: {overlap_percentage:.2f}%")
    print(f"Efficiency gain: {(len(intervals) - len(merged_intervals)) / len(intervals) * 100:.2f}%")

    # Daily breakdown
    print("\n" + "="*70)
    print("DAILY BREAKDOWN")
    print("="*70)

    daily_stats = []
    for day in sorted(tweets_by_day.keys()):
        day_intervals = tweets_by_day[day]
        day_merged = merge_intervals(day_intervals)

        day_naive_minutes = len(day_intervals) * 5
        day_total_minutes = calculate_total_minutes(day_merged)
        day_saved = day_naive_minutes - day_total_minutes
        day_overlap_pct = (day_saved / day_naive_minutes) * 100 if day_naive_minutes > 0 else 0

        daily_stats.append({
            'day': day,
            'tweets': len(day_intervals),
            'windows': len(day_merged),
            'naive_minutes': day_naive_minutes,
            'actual_minutes': day_total_minutes,
            'saved_minutes': day_saved,
            'overlap_pct': day_overlap_pct
        })

    for stat in daily_stats:
        print(f"\n{stat['day']}:")
        print(f"  Tweets: {stat['tweets']:,}")
        print(f"  Consolidated windows: {stat['windows']:,}")
        print(f"  Naive minutes: {stat['naive_minutes']:,.2f}")
        print(f"  Actual minutes needed: {stat['actual_minutes']:,.2f}")
        print(f"  Minutes saved: {stat['saved_minutes']:,.2f}")
        print(f"  Overlap: {stat['overlap_pct']:.2f}%")

    # Additional insights
    print("\n" + "="*70)
    print("ADDITIONAL INSIGHTS")
    print("="*70)

    if merged_intervals:
        # Find the longest consolidated window
        longest_window = max(merged_intervals, key=lambda x: (x[1] - x[0]).total_seconds())
        longest_duration = (longest_window[1] - longest_window[0]).total_seconds() / 60

        print(f"\nLongest consolidated window:")
        print(f"  Start: {longest_window[0]}")
        print(f"  End: {longest_window[1]}")
        print(f"  Duration: {longest_duration:.2f} minutes ({longest_duration/60:.2f} hours)")

        # Calculate average window size
        avg_window_minutes = total_minutes / len(merged_intervals)
        print(f"\nAverage consolidated window size: {avg_window_minutes:.2f} minutes")

        # Calculate gaps between windows
        gaps = []
        for i in range(len(merged_intervals) - 1):
            gap = (merged_intervals[i+1][0] - merged_intervals[i][1]).total_seconds() / 60
            gaps.append(gap)

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average gap between windows: {avg_gap:.2f} minutes")
            print(f"Number of gaps: {len(gaps):,}")

if __name__ == "__main__":
    main()
