
#!/usr/bin/env python3
"""
Simple script to generate retrieval chunks for the entire NQ table database.
Usage: python generate_all_chunks.py
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("🚀 Generating Retrieval Chunks for Entire Database")
    print("=" * 55)
    
    # Check if input file exists
    input_file = Path("nq_table_full_extraction.jsonl")
    if not input_file.exists():
        print("❌ Error: nq_table_full_extraction.jsonl not found")
        print("Please ensure you're running this from the WeightedRAG root directory")
        return 1
    
    # Count entries
    with open(input_file, 'r') as f:
        total_entries = sum(1 for _ in f)
    
    print(f"📊 Found {total_entries:,} entries to process")
    print(f"⏱️  Estimated time: ~{total_entries/100:.0f} minutes at 100 entries/sec")
    print(f"💾 Estimated output: ~{total_entries * 17 / 1000:.0f} MB")
    
    # Confirm
    response = input(f"\n❓ Process all {total_entries:,} entries? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return 0
    
    # Clear previous output
    output_dir = Path("retrieval_tables")
    if output_dir.exists():
        print("\n🧹 Clearing previous output...")
        for file in output_dir.glob("*.jsonl"):
            file.unlink()
    
    # Run the processing
    print(f"\n⏰ Starting at {time.strftime('%H:%M:%S')}")
    
    cmd = [
        sys.executable, 
        "rag_pipeline/create_retrieval_tables.py", 
        "--all"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n🎉 SUCCESS! Processing completed.")
        print(f"📁 Check the 'retrieval_tables/' directory for output files.")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during processing: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())