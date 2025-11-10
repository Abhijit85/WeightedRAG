#!/usr/bin/env python3
"""
Simple test to verify chunking module organization
"""

import sys
import os
from pathlib import Path

# Add the WeightedRAG root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing Chunking Module Imports")
    print("=" * 40)
    
    try:
        # Test table processor import
        from chunking.processors.table_processor import TableProcessor
        print("✅ TableProcessor imported successfully")
        
        # Test enhanced chunk generator
        from chunking.core.enhanced_chunk_generator import EnhancedChunkGenerator
        print("✅ EnhancedChunkGenerator imported successfully")
        
        # Test basic functionality
        processor = TableProcessor()
        print("✅ TableProcessor instantiated successfully")
        
        generator = EnhancedChunkGenerator()
        print("✅ EnhancedChunkGenerator instantiated successfully")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def show_structure():
    """Show the organized directory structure"""
    print("\n📁 Chunking Module Structure:")
    print("=" * 40)
    
    chunking_dir = Path(__file__).parent
    
    def print_tree(directory, prefix=""):
        """Print directory tree structure"""
        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            if item.name.startswith('.'):
                continue
                
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('__pycache__'):
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix)
    
    print_tree(chunking_dir)

def main():
    """Main test function"""
    print("🏗️ WeightedRAG Chunking Module Test")
    print("=" * 50)
    
    # Show structure
    show_structure()
    
    # Test imports
    success = test_imports()
    
    if success:
        print("\n🚀 Chunking module is properly organized and ready to use!")
        print("\n📋 Usage Examples:")
        print("   # Import table processor")
        print("   from chunking.processors.table_processor import TableProcessor")
        print("   ")
        print("   # Import chunk generator")
        print("   from chunking.core.enhanced_chunk_generator import EnhancedChunkGenerator")
        print("   ")
        print("   # Run main chunking script")
        print("   python chunking/core/create_retrieval_tables.py")
    else:
        print("\n❌ Some issues found. Check the import paths.")

if __name__ == "__main__":
    main()