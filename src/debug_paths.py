#!/usr/bin/env python3
"""Debug script to test path resolution."""

from pathlib import Path

print("Current working directory:", Path.cwd())
print("Script location:", Path(__file__))
print("Script parent:", Path(__file__).parent)
print("Script parent.parent:", Path(__file__).parent.parent)

# Test different path arguments
test_paths = [
    "../data/market_data.parquet",
    "data/market_data.parquet",
    "./data/market_data.parquet",
]

base_dir = Path(__file__).parent.parent
print(f"\nBase directory: {base_dir}")
print(f"Base directory exists: {base_dir.exists()}")

for test_path in test_paths:
    p = Path(test_path)
    print(f"\nTesting: {test_path}")
    print(f"  Is absolute: {p.is_absolute()}")
    resolved = (base_dir / p).resolve()
    print(f"  Resolved: {resolved}")
    print(f"  Exists: {resolved.exists()}")

# Check what files are in data directory
data_dir = base_dir / "data"
print(f"\nFiles in {data_dir}:")
if data_dir.exists():
    for f in data_dir.iterdir():
        print(f"  - {f.name}")
else:
    print(f"  Directory does not exist!")
