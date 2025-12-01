# Tweet 5-Minute Window Consolidation Analysis

## Executive Summary

This analysis examines 16,408 tweets to determine the efficiency gained by consolidating overlapping 5-minute data windows following each tweet.

## Key Findings

### Overall Statistics
- **Total Tweets**: 16,408
- **Total Individual 5-Minute Windows**: 16,408
- **Consolidated Windows After Merging**: 1,634
- **Efficiency Gain**: 90.04% reduction in number of windows

### Data Requirements
- **Naive Approach** (no consolidation): 82,040.00 minutes (1,367.33 hours / 57 days)
- **Consolidated Approach**: 9,775.81 minutes (162.93 hours / 6.8 days)
- **Minutes Saved**: 72,264.19 minutes (1,204.40 hours / 50.2 days)
- **Overlap Percentage**: 88.08%

### What This Means
By consolidating overlapping time windows, you only need to fetch **11.92%** of the data you would need with a naive approach. This represents an **88.08% reduction** in data fetching requirements.

## Daily Breakdown Highlights

### Best Consolidation Days (Highest Overlap %)
1. **2025-06-17**: 98.32% overlap (60 tweets → 1 window, 5.04 minutes needed)
2. **2024-12-09**: 98.29% overlap (60 tweets → 1 window, 5.14 minutes needed)
3. **2025-06-18**: 98.09% overlap (57 tweets → 1 window, 5.45 minutes needed)
4. **2025-06-13**: 97.06% overlap (42 tweets → 1 window, 6.18 minutes needed)
5. **2024-12-26**: 97.21% overlap (96 tweets → 2 windows, 13.37 minutes needed)

These days show tweets clustered very tightly in time, resulting in nearly complete overlap.

### Days with Most Tweets
1. **2025-03-10**: 245 tweets → 6 windows (95.15% overlap)
2. **2025-07-02**: 240 tweets → 12 windows (93.24% overlap)
3. **2025-06-24**: 229 tweets → 23 windows (88.04% overlap)
4. **2025-07-09**: 210 tweets → 13 windows (91.96% overlap)
5. **2025-06-25**: 207 tweets → 17 windows (89.54% overlap)

### Days with Lower Consolidation (50-70% overlap)
These days had more spread-out tweets:
- **2025-03-13**: 50.97% overlap (22 tweets → 10 windows)
- **2025-03-05**: 52.89% overlap (10 tweets → 4 windows)
- **2025-03-04**: 53.90% overlap (14 tweets → 6 windows)

## Additional Insights

### Longest Consolidated Window
- **Start**: 2025-03-10 14:51:48.672994
- **End**: 2025-03-10 15:50:51.322389
- **Duration**: 59.04 minutes (0.98 hours)

This represents a period of intense tweeting activity where 245 tweets were posted within roughly 1 hour, requiring only 59 minutes of consolidated data.

### Window Statistics
- **Average Consolidated Window Size**: 5.98 minutes
- **Average Gap Between Windows**: 8.11 minutes
- **Number of Gaps**: 1,633

The average consolidated window is just slightly larger than the base 5-minute window, indicating that many tweets occur in tight clusters. The average gap of 8.11 minutes shows moderate spacing between tweet bursts.

## Practical Implications

### API/Data Fetching Strategy
1. **Consolidation is Essential**: Without consolidation, you'd need to fetch 1,367 hours of data. With consolidation, only 163 hours.
2. **Batch Processing**: Since tweets cluster together, batch fetching strategies will be highly effective.
3. **Cache Strategy**: The high overlap percentage suggests that caching 5-minute data windows would eliminate most redundant fetches.

### Cost Savings
If data fetching has any cost (API calls, bandwidth, processing time):
- You'll make **90% fewer data requests**
- You'll process **88% less data**
- Your application will be significantly faster and more efficient

## Recommendations

1. **Implement Window Merging**: Always merge overlapping time windows before fetching ticker data
2. **Use Batching**: Fetch data for consolidated windows in batches rather than individual requests
3. **Consider Caching**: With an average gap of only 8 minutes between windows, a short-lived cache could further reduce fetches
4. **Monitor High-Activity Days**: Days with 100+ tweets show the most benefit from consolidation (often 90%+ overlap)
