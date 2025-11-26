#!/usr/bin/env python3
"""
Ultra-Simple Retrieval Tables Creator
No complex chunking - just basic table processing that definitely won't get stuck
"""

import json
import sys
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add paths for relative imports
sys.path.append(str(Path(__file__).parent.parent))  # Add chunking directory
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add WeightedRAG root

from chunking.processors.table_processor import TableProcessor

def create_enhanced_retrieval_tables(max_entries=None):
    """
    Create retrieval tables with enhanced multi-granular chunking
    Store different chunk types in separate JSONL files for better organization
    If max_entries is None, process the entire dataset
    """
    print("="*60)
    print("ENHANCED RETRIEVAL TABLES CREATOR - SEPARATE CHUNK FILES")
    print("="*60)
    
    # Initialize only table processor (skip chunk generator)
    table_processor = TableProcessor(timeout_seconds=5)  # Increased timeout for full dataset
    
    # Setup paths
    input_file = Path("datasets/nq-table/nq_table_full_extraction.jsonl")
    output_dir = Path("retrieval_tables")
    output_dir.mkdir(exist_ok=True)
    
    # Separate files for each chunk type
    chunk_files = {
        'pure_table': output_dir / "chunks_pure_table.jsonl", 
        'table_row': output_dir / "chunks_table_row.jsonl",
        'table_column': output_dir / "chunks_table_column.jsonl",
        'sliding_window': output_dir / "chunks_sliding_window.jsonl",
        'table_structure': output_dir / "chunks_table_structure.jsonl"
    }
    
    # Combined file for backward compatibility
    all_chunks_file = output_dir / "retrieval_chunks.jsonl"
    tables_file = output_dir / "processed_tables.jsonl"
    
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Chunk type files:")
    for chunk_type, file_path in chunk_files.items():
        print(f"  - {chunk_type}: {file_path.name}")
    
    # Count total lines for progress bar
    print("Counting entries...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if max_entries is None:
        actual_max = total_lines
        print(f"Will process ALL {total_lines} entries")
    else:
        actual_max = min(max_entries, total_lines)
        print(f"Will process {actual_max} entries out of {total_lines} total")
    
    # Statistics
    # Initialize statistics with chunk type breakdown
    stats = {
        'start_time': datetime.now().isoformat(),
        'processed': 0,
        'failed': 0,
        'total_chunks': 0,
        'chunk_type_counts': {chunk_type: 0 for chunk_type in chunk_files.keys()},
        'processing_times': []
    }
    
    # Start processing
    start_time = time.time()
    
    try:
        # Open all chunk type files + combined file + tables file
        chunk_file_handles = {}
        for chunk_type, file_path in chunk_files.items():
            chunk_file_handles[chunk_type] = open(file_path, 'w', encoding='utf-8')
        
        with open(all_chunks_file, 'w', encoding='utf-8') as all_chunks_out, \
             open(tables_file, 'w', encoding='utf-8') as tables_out, \
             open(input_file, 'r', encoding='utf-8') as f:
            
            # Create progress bar
            pbar = tqdm(total=actual_max, 
                       desc="Processing tables",
                       unit="entries",
                       ncols=100,
                       mininterval=0.5)  # Update every 0.5 seconds
            
            for line_num, line in enumerate(f):
                if max_entries is not None and line_num >= max_entries:
                    break
                
                entry_start = time.time()
                
                try:
                    # Parse entry
                    entry = json.loads(line.strip())
                    entry_id = entry.get('example_id', 'unknown')
                    
                    # Process table (this is the only complex step)
                    processed_table = table_processor.process_table_entry(entry)
                    
                    # Create ENHANCED chunks manually - multiple granularities
                    chunks = create_enhanced_chunks(processed_table)
                    
                    # Write immediately to files
                    tables_out.write(json.dumps(processed_table, ensure_ascii=False) + '\n')
                    
                    # Write chunks to both individual type files and combined file
                    for chunk in chunks:
                        chunk_type = chunk.get('chunk_type', 'unknown')
                        
                        # Write to combined file
                        all_chunks_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        
                        # Write to specific chunk type file
                        if chunk_type in chunk_file_handles:
                            chunk_file_handles[chunk_type].write(json.dumps(chunk, ensure_ascii=False) + '\n')
                            stats['chunk_type_counts'][chunk_type] += 1
                    
                    # Update stats
                    processing_time = time.time() - entry_start
                    stats['processed'] += 1
                    stats['total_chunks'] += len(chunks)
                    stats['processing_times'].append(processing_time)
                    
                    # Update progress bar postfix with chunk type breakdown
                    chunk_summary = ", ".join([f"{k[:4]}:{v}" for k, v in stats['chunk_type_counts'].items() if v > 0])
                    pbar.set_postfix({
                        'chunks': stats['total_chunks'],
                        'failed': stats['failed'],
                        'time': f"{processing_time:.2f}s",
                        'types': chunk_summary[:30] + "..." if len(chunk_summary) > 30 else chunk_summary
                    })
                    pbar.update(1)
                    
                    # Flush files periodically
                    if line_num % 20 == 0:
                        all_chunks_out.flush()
                        tables_out.flush()
                        for handle in chunk_file_handles.values():
                            handle.flush()
                        tables_out.flush()
                    
                    # Show detailed info for first few and slow entries
                    if line_num < 3 or processing_time > 2.0:
                        tqdm.write(f"Entry {line_num + 1} ({entry_id}): {processing_time:.2f}s, {len(chunks)} chunks")
                    
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = str(e)[:40]
                    tqdm.write(f"⚠️  Failed entry {line_num + 1}: {error_msg}")
                    pbar.set_postfix({
                        'chunks': stats['total_chunks'],
                        'failed': stats['failed'],
                        'last_error': 'failed'
                    })
                    pbar.update(1)
                    continue
            
            pbar.close()
        
        # Close individual chunk type files
        for chunk_type, handle in chunk_file_handles.items():
            handle.close()
        
        # Final statistics
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETED!")
        print("="*60)
        print(f"Processed: {stats['processed']} entries")
        print(f"Failed: {stats['failed']} entries")
        print(f"Success rate: {stats['processed']/(stats['processed']+stats['failed'])*100:.1f}%")
        print(f"Total chunks: {stats['total_chunks']}")
        
        if stats['processed'] > 0:
            print(f"Avg chunks per table: {stats['total_chunks']/stats['processed']:.1f}")
        else:
            print("No entries were successfully processed")
        print(f"Processing time: {total_time:.1f}s")
        print(f"Processing rate: {stats['processed']/total_time:.1f} entries/sec")
        
        if stats['processing_times']:
            avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
            max_time = max(stats['processing_times'])
            print(f"Avg time per entry: {avg_time:.3f}s (max: {max_time:.3f}s)")
        
        print(f"\nChunk Type Breakdown:")
        for chunk_type, count in stats['chunk_type_counts'].items():
            if count > 0:
                percentage = (count / stats['total_chunks']) * 100
                print(f"  📄 {chunk_type}: {count:,} chunks ({percentage:.1f}%)")
        
        print(f"\nOutput files:")
        print(f"  � {tables_file.name} ({stats['processed']} tables)")
        print(f"  �📄 {all_chunks_file.name} ({stats['total_chunks']} total chunks)")
        print(f"  � Individual chunk type files:")
        for chunk_type, file_path in chunk_files.items():
            count = stats['chunk_type_counts'][chunk_type]
            if count > 0:
                print(f"    - {file_path.name}: {count:,} chunks")
        
        return stats
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        # Close chunk file handles if they exist
        try:
            for handle in chunk_file_handles.values():
                handle.close()
        except:
            pass
        return stats
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Close chunk file handles if they exist
        try:
            for handle in chunk_file_handles.values():
                handle.close()
        except:
            pass
        return None

def create_enhanced_chunks(processed_table):
    """
    Create enhanced chunks with multiple granularities - no hard limits
    """
    import json
    chunks = []
    table_id = processed_table['table_id']
    question = processed_table['original_question']
    linearized = processed_table['linearized_text']
    structured_data = processed_table.get('structured_data', {})
    
    # Chunk 1: Full table content (question-agnostic)
    if linearized and linearized.strip():
        chunks.append({
            'id': f"{table_id}_full_table",
            'content': linearized,
            'chunk_type': 'full_table',
            'table_id': table_id,
            'metadata': {
                'answers': processed_table['gold_answers'],
                'example_id': processed_table['example_id']
            }
        })
    
    # Chunk 2: Pure table structure (JSON format)
    if structured_data.get('headers') and structured_data.get('rows'):
        # Create pure table JSON format without question context
        pure_table_json = {
            "columns": structured_data.get('headers', []),
            "rows": structured_data.get('rows', [])
        }
        
        # Store the pure table structure
        chunks.append({
            'id': f"{table_id}_pure_table",
            'table': pure_table_json,
            'chunk_type': 'pure_table',
            'table_id': table_id,
            'metadata': {
                'table_id': table_id,
                'num_rows': len(structured_data.get('rows', [])),
                'num_columns': len(structured_data.get('headers', [])),
                'format': 'pure_table_json'
            }
        })
    
    # Chunk 3: Just the table content (same as full_table, kept for backward compatibility)
    if linearized and len(linearized.strip()) > 0:
        chunks.append({
            'id': f"{table_id}_table_only",
            'content': linearized,
            'chunk_type': 'table_only',
            'table_id': table_id,
            'metadata': {
                'example_id': processed_table['example_id']
            }
        })
    
    # NEW: Row-level chunks (question-agnostic) for tables with structured data
    if structured_data.get('rows') and len(structured_data['rows']) > 0:
        headers = structured_data.get('headers', [])
        rows = structured_data['rows']
        
        for i, row in enumerate(rows):
            # Skip empty rows
            if not row or not any(cell.strip() for cell in row if cell):
                continue
            
            # Create row content with headers if available (no question context)
            if headers and len(headers) >= len(row):
                row_pairs = []
                for j, cell in enumerate(row):
                    if j < len(headers) and cell and cell.strip():
                        row_pairs.append(f"{headers[j]}: {cell}")
                if row_pairs:
                    row_content = ' | '.join(row_pairs)
                else:
                    continue  # Skip empty row
            else:
                # No headers or mismatched headers
                non_empty_cells = [cell for cell in row if cell and cell.strip()]
                if non_empty_cells:
                    row_content = ' | '.join(non_empty_cells)
                else:
                    continue  # Skip empty row
            
            chunks.append({
                'id': f"{table_id}_row_{i}",
                'content': row_content,
                'chunk_type': 'table_row',
                'table_id': table_id,
                'metadata': {
                    'example_id': processed_table['example_id'],
                    'row_index': i
                }
            })
    
    # Column-wise chunks for tables with structured data
    if structured_data.get('headers') and structured_data.get('rows'):
        headers = structured_data['headers']
        rows = structured_data['rows']
        
        for col_idx, header in enumerate(headers):
            # Extract all values for this column
            column_values = []
            for row in rows[:100]:  # Limit to first 100 rows for performance
                if col_idx < len(row) and row[col_idx] and row[col_idx].strip():
                    column_values.append(row[col_idx].strip())
            
            if column_values:  # Only create chunk if column has values
                # Remove duplicates while preserving order
                unique_values = []
                seen = set()
                for val in column_values:
                    if val not in seen:
                        unique_values.append(val)
                        seen.add(val)
                        if len(unique_values) >= 20:  # Limit unique values shown
                            break
                
                column_content = f"Column '{header}': {' | '.join(unique_values)}"
                if len(column_values) > len(unique_values):
                    column_content += f" (showing {len(unique_values)} unique values out of {len(column_values)} total)"
                
                chunks.append({
                    'id': f"{table_id}_column_{col_idx}",
                    'content': column_content,
                    'chunk_type': 'table_column',
                    'table_id': table_id,
                    'metadata': {
                        'example_id': processed_table['example_id'],
                        'column_index': col_idx,
                        'column_name': header,
                        'unique_values': len(unique_values),
                        'total_values': len(column_values)
                    }
                })

    # NEW: Sliding window chunks for very long tables
    if linearized and len(linearized) > 3000:  # Only for long tables
        window_size = 2000
        overlap = 200
        start = 0
        window_num = 0
        
        while start < len(linearized):
            end = min(start + window_size, len(linearized))
            window_text = linearized[start:end]
            
            # Try to break at natural boundaries
            if end < len(linearized):
                for sep in [' || ', ' | ', ' ']:
                    last_sep = window_text.rfind(sep)
                    if last_sep > start + window_size * 0.7:
                        end = start + last_sep + len(sep)
                        window_text = linearized[start:end]
                        break
            
            window_content = f"Table Section {window_num + 1}: {window_text}"
            
            chunks.append({
                'id': f"{table_id}_window_{window_num}",
                'content': window_content,
                'chunk_type': 'sliding_window',
                'table_id': table_id,
                'metadata': {
                    'example_id': processed_table['example_id'],
                    'window_number': window_num,
                    'start_position': start,
                    'end_position': end
                }
            })
            
            # Move start with overlap
            start = end - overlap
            if start >= len(linearized) - overlap:
                break
            
            window_num += 1
    
    # NEW: Sample chunk for large tables (headers + first few rows)
    if (structured_data.get('num_rows', 0) > 10 and 
        structured_data.get('headers')):
        
        headers = structured_data['headers']
        rows = structured_data.get('rows', [])
        
        # Create JSON structure for table sample
        table_json = {
            "headers": headers[:10],  # Limit headers shown
            "sample_rows": [],
            "total_rows": len(rows),
            "showing_rows": 0
        }
        
        # Add first few non-empty rows as structured data
        for i, row in enumerate(rows[:5]):  # First 5 rows
            if row and any(cell.strip() for cell in row if cell):
                # Create row object with header-value pairs
                row_obj = {}
                for j, cell in enumerate(row[:len(headers)]):  # Limit to available headers
                    if j < len(headers) and cell and cell.strip():
                        row_obj[headers[j]] = cell.strip()
                
                if row_obj:  # Only add non-empty rows
                    table_json["sample_rows"].append({
                        "row_index": i + 1,
                        "data": row_obj
                    })
        
        table_json["showing_rows"] = len(table_json["sample_rows"])
        
        # Convert to JSON string for content
        import json
        sample_content = json.dumps(table_json, ensure_ascii=False, separators=(',', ':'))
        
        # Add table sample chunk
        chunks.append({
            'id': f"{table_id}_sample",
            'content': sample_content,
            'chunk_type': 'table_sample',
            'table_id': table_id,
            'metadata': {
                'example_id': processed_table['example_id'],
                'total_rows': len(rows),
                'sample_rows': table_json["showing_rows"],
                'format': 'json',
                'structure': 'headers_with_sample_rows'
            }
        })
        
    # NEW: Table Structure chunk for schema understanding (works for all tables with headers)
    if structured_data.get('headers'):
        headers = structured_data['headers']
        rows = structured_data.get('rows', [])
        
        # Analyze column types and patterns
        columns_info = {}
        for i, header in enumerate(headers[:10]):  # Limit to 10 columns for conciseness
            column_values = []
            for row in rows[:20]:  # Sample from first 20 rows
                if i < len(row) and row[i] and row[i].strip():
                    column_values.append(row[i].strip())
            
            # Determine data type and patterns
            col_info = {
                'samples': column_values[:3]  # First 3 non-empty values
            }
            
            # Simple type detection
            if column_values:
                numeric_count = sum(1 for v in column_values if v.replace('.', '').replace('-', '').isdigit())
                if numeric_count > len(column_values) * 0.7:
                    col_info['type'] = 'numeric'
                elif any(len(v) > 50 for v in column_values):
                    col_info['type'] = 'text'
                else:
                    col_info['type'] = 'categorical'
            
            columns_info[header] = col_info
        
        # Create structure information
        structure_data = {
            'columns': columns_info,
            'rows': len(rows),
            'patterns': []
        }
        
        # Add pattern tags based on analysis
        if len(headers) > 5:
            structure_data['patterns'].append('wide')
        if len(rows) > 50:
            structure_data['patterns'].append('large')
        if any('id' in h.lower() for h in headers):
            structure_data['patterns'].append('keyed')
        
        structure_content = json.dumps(structure_data, ensure_ascii=False, separators=(',', ':'))
        
        chunks.append({
            'id': f"{processed_table['example_id']}_structure",
            'content': structure_content,
            'chunk_type': 'table_structure',
            'table_id': table_id,
            'metadata': {
                'example_id': processed_table['example_id'],
                'num_columns': len(columns_info),
                'total_columns': len(headers),
                'patterns': structure_data['patterns']
            }
        })
    
    return chunks

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create enhanced retrieval tables from NQ dataset")
    parser.add_argument("--max_entries", type=int, default=100,
                       help="Maximum entries to process (default: 100, use 0 for all entries)")
    parser.add_argument("--all", action="store_true",
                       help="Process all entries in the database")
    
    args = parser.parse_args()
    
    # Handle --all flag or max_entries=0 to process all entries
    if args.all or args.max_entries == 0:
        max_entries = None
        print(f"Creating enhanced retrieval tables for ALL entries...")
    else:
        max_entries = args.max_entries
        print(f"Creating enhanced retrieval tables for {max_entries} entries...")
    
    print("This version creates multiple granularities: table-level, row-level, and sliding window chunks.")
    
    stats = create_enhanced_retrieval_tables(max_entries)
    
    if stats and stats['processed'] > 0:
        print(f"\n✅ Successfully created retrieval tables!")
        print(f"📁 Check the 'retrieval_tables/' directory for output files.")
        
        # Show sample content from different chunk types
        output_dir = Path("retrieval_tables")
        chunk_files = {
            'full_table': "chunks_full_table.jsonl",
            'pure_table': "chunks_pure_table.jsonl", 
            'table_only': "chunks_table_only.jsonl",
            'table_row': "chunks_table_row.jsonl",
            'table_column': "chunks_table_column.jsonl",
            'table_structure': "chunks_table_structure.jsonl"
        }
        
        print(f"\nSample chunk content by type:")
        for chunk_type, filename in chunk_files.items():
            file_path = output_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if first_line:
                        chunk = json.loads(first_line.strip())
                        # Get appropriate content based on chunk type
                        if 'content' in chunk:
                            content = chunk['content']
                            if isinstance(content, dict):
                                if 'columns' in content and 'rows' in content:
                                    # Pure table format
                                    cols = content['columns'][:3]  # First 3 columns
                                    content_preview = f"Table: {' | '.join(cols)}{'...' if len(content['columns']) > 3 else ''}"
                                else:
                                    content_preview = f"{{dict with {len(content)} keys}}"
                            else:
                                content_preview = f"{str(content)[:80]}..."
                        elif 'text' in chunk:
                            content_preview = f"{chunk['text'][:80]}..."
                        else:
                            content_preview = f"{{chunk with keys: {', '.join(chunk.keys())}}}"
                        print(f"  📄 [{chunk_type}] {content_preview}")
            else:
                print(f"  📄 [{chunk_type}] (no chunks generated)")
    else:
        print(f"\n❌ Failed to create retrieval tables.")
        sys.exit(1)

if __name__ == "__main__":
    main()