#!/usr/bin/env python3
"""
Example usage of the WeightedRAG chunking module

This script demonstrates how to use the organized chunking functionality.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    Example demonstrating chunking module usage
    """
    print("🚀 WeightedRAG Chunking Module Example")
    print("=" * 50)
    
    try:
        # Import from the organized structure
        from chunking.core.create_retrieval_tables import main as create_tables
        
        print("✅ Successfully imported chunking modules!")
        print(f"📁 Chunking module location: {os.path.dirname(__file__)}")
        
        # Show available functionality
        print("\n📋 Available Chunking Functions:")
        print("   • create_retrieval_tables.py - Main chunk generation")
        print("   • enhanced_chunk_generator.py - Advanced chunking")
        print("   • chunk_generator.py - Core chunking logic")
        print("   • table_processor.py - Table processing")
        
        print("\n🔧 Module Structure:")
        chunking_dir = os.path.dirname(__file__)
        for root, dirs, files in os.walk(chunking_dir):
            level = root.replace(chunking_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    print(f'{subindent}{file}')
        
        print("\n🎯 To run chunk generation:")
        print("   python chunking/core/create_retrieval_tables.py --all")
        print("\n🎯 To use in your code:")
        print("   from chunking.core.create_retrieval_tables import main")
        print("   main()")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the WeightedRAG root directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()