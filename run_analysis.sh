#!/bin/bash
# Script to run the full L1 consolidation analysis with proper output handling

cd "$(dirname "$0")/src"

echo "Starting L1 Market Data Consolidation Analysis"
echo "=============================================="
echo "Current directory: $(pwd)"
echo "Output will be saved to: ../output/"
echo ""

# Run the analysis with proper error handling
python analyze.py \
    --parquet data/market_data.parquet \
    --output ../output

echo ""
echo "Analysis complete. Checking output files..."
echo ""

# Verify output files exist
OUTPUT_DIR="../output"
if [ -f "$OUTPUT_DIR/l1_multi_stock_summary.csv" ]; then
    echo "CSV file found:"
    echo "  - Location: $OUTPUT_DIR/l1_multi_stock_summary.csv"
    echo "  - Size: $(wc -c < "$OUTPUT_DIR/l1_multi_stock_summary.csv") bytes"
    echo "  - Rows: $(wc -l < "$OUTPUT_DIR/l1_multi_stock_summary.csv")"
    echo "  - Unique stocks: $(tail -n +2 "$OUTPUT_DIR/l1_multi_stock_summary.csv" | cut -d',' -f1 | sort -u | wc -l)"
else
    echo "ERROR: CSV file not found!"
fi

if [ -f "$OUTPUT_DIR/l1_multi_stock_comparison.png" ]; then
    echo ""
    echo "PNG file found:"
    echo "  - Location: $OUTPUT_DIR/l1_multi_stock_comparison.png"
    echo "  - Size: $(wc -c < "$OUTPUT_DIR/l1_multi_stock_comparison.png") bytes"
else
    echo "ERROR: PNG file not found!"
fi

echo ""
echo "Done!"
