#!/usr/bin/env python3
"""
Enhanced Chunk Generator for NQ Table Dataset
Creates flexible, multi-granular chunks without hard limits
"""

from typing import Dict, List, Any, Optional
import re

class EnhancedChunkGenerator:
    """Enhanced chunk generator with flexible strategies and no hard limits"""
    
    def __init__(self, 
                 max_content_length=1500,
                 enable_row_chunks=True,
                 enable_cell_chunks=False,
                 enable_sliding_window=True,
                 sliding_window_size=800,
                 sliding_overlap=100):
        """
        Initialize enhanced chunk generator
        
        Args:
            max_content_length: Maximum length for chunk content (soft limit)
            enable_row_chunks: Whether to create individual row chunks
            enable_cell_chunks: Whether to create cell-level chunks
            enable_sliding_window: Whether to create sliding window chunks for long tables
            sliding_window_size: Size of sliding window chunks
            sliding_overlap: Overlap between sliding windows
        """
        self.max_content_length = max_content_length
        self.enable_row_chunks = enable_row_chunks
        self.enable_cell_chunks = enable_cell_chunks
        self.enable_sliding_window = enable_sliding_window
        self.sliding_window_size = sliding_window_size
        self.sliding_overlap = sliding_overlap
    
    def create_enhanced_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create enhanced chunks with flexible strategies
        
        Args:
            processed_table: Output from TableProcessor.process_table_entry()
            
        Returns:
            List of chunks with different granularities and strategies
        """
        chunks = []
        table_id = processed_table['table_id']
        question = processed_table['original_question']
        linearized = processed_table['linearized_text']
        description = processed_table.get('natural_description', '')
        structured_data = processed_table.get('structured_data', {})
        
        # Strategy 1: Full table with question context (always create)
        full_chunk = self._create_full_table_chunk(processed_table)
        chunks.append(full_chunk)
        
        # Strategy 2: Question + description context (if description exists)
        if description and description.strip():
            context_chunk = self._create_context_chunk(processed_table)
            chunks.append(context_chunk)
        
        # Strategy 3: Table-only chunk (if table has content)
        if linearized and linearized.strip():
            table_only_chunk = self._create_table_only_chunk(processed_table)
            chunks.append(table_only_chunk)
        
        # Strategy 4: Row-level chunks (if enabled and table has rows)
        if self.enable_row_chunks and structured_data.get('rows'):
            row_chunks = self._create_row_chunks(processed_table)
            chunks.extend(row_chunks)
        
        # Strategy 5: Cell-level chunks (if enabled and needed)
        if self.enable_cell_chunks and structured_data.get('rows'):
            cell_chunks = self._create_cell_chunks(processed_table)
            chunks.extend(cell_chunks)
        
        # Strategy 6: Sliding window chunks (for very long tables)
        if (self.enable_sliding_window and linearized and 
            len(linearized) > self.sliding_window_size):
            sliding_chunks = self._create_sliding_window_chunks(processed_table)
            chunks.extend(sliding_chunks)
        
        # Strategy 7: Header + sample rows (for large tables)
        if (structured_data.get('num_rows', 0) > 10 and 
            structured_data.get('headers')):
            sample_chunk = self._create_sample_chunk(processed_table)
            chunks.append(sample_chunk)
        
        return chunks
    
    def _create_full_table_chunk(self, processed_table: Dict[str, Any]) -> Dict[str, Any]:
        """Create full table chunk with question context"""
        table_id = processed_table['table_id']
        question = processed_table['original_question']
        linearized = processed_table['linearized_text']
        
        # Build content with question context
        if linearized and linearized.strip():
            content = f"Question: {question} | Table: {linearized}"
        else:
            content = f"Question: {question} | Table: [Empty or malformed table]"
        
        # Truncate if too long (soft limit)
        if len(content) > self.max_content_length:
            truncate_pos = self.max_content_length - 50
            content = content[:truncate_pos] + "... [truncated]"
        
        return {
            'id': f"{table_id}_full",
            'content': content,
            'chunk_type': 'full_table',
            'table_id': table_id,
            'metadata': {
                'question': question,
                'answers': processed_table.get('gold_answers', []),
                'example_id': processed_table.get('example_id'),
                'strategy': 'full_context',
                'content_length': len(content),
                'has_table_content': bool(linearized and linearized.strip())
            }
        }
    
    def _create_context_chunk(self, processed_table: Dict[str, Any]) -> Dict[str, Any]:
        """Create question + structured table context chunk in JSON format"""
        table_id = processed_table['table_id']
        question = processed_table['original_question']
        structured_data = processed_table.get('structured_data', {})
        
        # Create structured JSON format for question context
        question_context_json = {
            "question": question,
            "table": {
                "columns": structured_data.get('headers', []),
                "rows": structured_data.get('rows', [])
            },
            "answer_coordinates": [],  # Will be populated if answer coordinates are available
            "answer_text": processed_table.get('gold_answers', []),
            "aggregation_label": "NONE"  # Default value, can be enhanced later
        }
        
        # Check if content needs to be reduced for size limits
        import json
        if self.max_content_length:
            # If JSON would be too long, reduce the number of rows
            temp_content = json.dumps(question_context_json, ensure_ascii=False, separators=(',', ':'))
            if len(temp_content) > self.max_content_length:
                rows = question_context_json["table"]["rows"]
                while len(rows) > 1 and len(temp_content) > self.max_content_length:
                    rows.pop()
                    temp_content = json.dumps(question_context_json, ensure_ascii=False, separators=(',', ':'))
        
        return {
            'id': f"{table_id}_context",
            'content': question_context_json,  # Store JSON object directly
            'chunk_type': 'question_context',
            'table_id': table_id,
            'metadata': {
                'question': question,
                'example_id': processed_table.get('example_id'),
                'strategy': 'structured_json',
                'content_length': len(json.dumps(question_context_json, ensure_ascii=False)),
                'format': 'structured_json',
                'has_answer_coordinates': False,
                'answer_count': len(processed_table.get('gold_answers', []))
            }
        }
    
    def _create_table_only_chunk(self, processed_table: Dict[str, Any]) -> Dict[str, Any]:
        """Create table-only chunk without question context"""
        table_id = processed_table['table_id']
        linearized = processed_table['linearized_text']
        
        content = linearized
        if len(content) > self.max_content_length:
            truncate_pos = self.max_content_length - 50
            content = content[:truncate_pos] + "... [truncated]"
        
        return {
            'id': f"{table_id}_table_only",
            'content': content,
            'chunk_type': 'table_only',
            'table_id': table_id,
            'metadata': {
                'example_id': processed_table.get('example_id'),
                'strategy': 'table_content',
                'content_length': len(content)
            }
        }
    
    def _create_row_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create individual row chunks"""
        chunks = []
        table_id = processed_table['table_id']
        structured_data = processed_table['structured_data']
        question = processed_table['original_question']
        
        headers = structured_data.get('headers', [])
        rows = structured_data.get('rows', [])
        
        for i, row in enumerate(rows):
            if not row or not any(cell.strip() for cell in row if cell):
                continue  # Skip empty rows
            
            # Create row content with headers if available
            if headers and len(headers) >= len(row):
                row_pairs = []
                for j, cell in enumerate(row):
                    if j < len(headers) and cell and cell.strip():
                        row_pairs.append(f"{headers[j]}: {cell}")
                content = f"Question: {question} | Row: {' | '.join(row_pairs)}"
            else:
                non_empty_cells = [cell for cell in row if cell and cell.strip()]
                content = f"Question: {question} | Row: {' | '.join(non_empty_cells)}"
            
            if len(content) > self.max_content_length:
                truncate_pos = self.max_content_length - 50
                content = content[:truncate_pos] + "... [truncated]"
            
            chunk = {
                'id': f"{table_id}_row_{i}",
                'content': content,
                'chunk_type': 'table_row',
                'table_id': table_id,
                'metadata': {
                    'question': question,
                    'example_id': processed_table.get('example_id'),
                    'row_index': i,
                    'strategy': 'row_level',
                    'content_length': len(content),
                    'has_headers': bool(headers)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_cell_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create cell-level chunks for important cells"""
        chunks = []
        table_id = processed_table['table_id']
        structured_data = processed_table['structured_data']
        question = processed_table['original_question']
        answers = processed_table.get('gold_answers', [])
        
        headers = structured_data.get('headers', [])
        rows = structured_data.get('rows', [])
        
        # Only create cell chunks if we have answers to look for
        if not answers:
            return chunks
        
        # Find cells that might contain answer content
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if not cell or not cell.strip():
                    continue
                
                # Check if cell content is related to answers
                cell_lower = cell.lower()
                for answer in answers:
                    if answer.lower() in cell_lower or cell_lower in answer.lower():
                        header = headers[j] if j < len(headers) else f"Column_{j}"
                        content = f"Question: {question} | {header}: {cell}"
                        
                        chunk = {
                            'id': f"{table_id}_cell_{i}_{j}",
                            'content': content,
                            'chunk_type': 'table_cell',
                            'table_id': table_id,
                            'metadata': {
                                'question': question,
                                'example_id': processed_table.get('example_id'),
                                'row_index': i,
                                'col_index': j,
                                'header': header,
                                'strategy': 'cell_level',
                                'content_length': len(content),
                                'answer_match': True
                            }
                        }
                        chunks.append(chunk)
                        break  # Only create one chunk per cell
        
        return chunks
    
    def _create_sliding_window_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create sliding window chunks for long tables"""
        chunks = []
        table_id = processed_table['table_id']
        linearized = processed_table['linearized_text']
        question = processed_table['original_question']
        
        if len(linearized) <= self.sliding_window_size:
            return chunks
        
        start = 0
        chunk_num = 0
        
        while start < len(linearized):
            end = min(start + self.sliding_window_size, len(linearized))
            window_text = linearized[start:end]
            
            # Try to break at natural boundaries
            if end < len(linearized):
                # Look for table separators or spaces
                for sep in [' || ', ' | ', ' ']:
                    last_sep = window_text.rfind(sep)
                    if last_sep > start + self.sliding_window_size * 0.7:
                        end = start + last_sep + len(sep)
                        window_text = linearized[start:end]
                        break
            
            content = f"Question: {question} | Table Section {chunk_num + 1}: {window_text}"
            
            chunk = {
                'id': f"{table_id}_window_{chunk_num}",
                'content': content,
                'chunk_type': 'sliding_window',
                'table_id': table_id,
                'metadata': {
                    'question': question,
                    'example_id': processed_table.get('example_id'),
                    'window_number': chunk_num,
                    'start_position': start,
                    'end_position': end,
                    'strategy': 'sliding_window',
                    'content_length': len(content)
                }
            }
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.sliding_overlap
            if start >= len(linearized) - self.sliding_overlap:
                break
            
            chunk_num += 1
        
        return chunks
    
    def _create_sample_chunk(self, processed_table: Dict[str, Any]) -> Dict[str, Any]:
        """Create header + sample rows chunk for large tables in JSON format"""
        table_id = processed_table['table_id']
        structured_data = processed_table['structured_data']
        question = processed_table['original_question']
        
        headers = structured_data.get('headers', [])
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
        content = json.dumps(table_json, ensure_ascii=False, separators=(',', ':'))
        
        if len(content) > self.max_content_length:
            # If JSON is too long, reduce the number of sample rows
            while len(table_json["sample_rows"]) > 1 and len(content) > self.max_content_length:
                table_json["sample_rows"].pop()
                table_json["showing_rows"] = len(table_json["sample_rows"])
                content = json.dumps(table_json, ensure_ascii=False, separators=(',', ':'))
        
        return {
            'id': f"{table_id}_sample",
            'content': content,
            'chunk_type': 'table_sample',
            'table_id': table_id,
            'metadata': {
                'question': question,
                'example_id': processed_table.get('example_id'),
                'total_rows': len(rows),
                'sample_rows': table_json["showing_rows"],
                'strategy': 'sample_rows_json',
                'content_length': len(content),
                'format': 'json',
                'structure': 'headers_with_sample_rows'
            }
        }
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive statistics about generated chunks"""
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_types = {}
        strategies = {}
        total_length = 0
        lengths = []
        
        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Count strategies
            strategy = chunk['metadata'].get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
            
            # Collect length statistics
            length = chunk['metadata'].get('content_length', len(chunk['content']))
            total_length += length
            lengths.append(length)
        
        return {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types,
            'strategies': strategies,
            'total_content_length': total_length,
            'avg_content_length': total_length / len(chunks),
            'min_content_length': min(lengths),
            'max_content_length': max(lengths),
            'length_distribution': {
                'short (<200)': len([l for l in lengths if l < 200]),
                'medium (200-800)': len([l for l in lengths if 200 <= l < 800]),
                'long (800-1500)': len([l for l in lengths if 800 <= l < 1500]),
                'very_long (>1500)': len([l for l in lengths if l >= 1500])
            }
        }