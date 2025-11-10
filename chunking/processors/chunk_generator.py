#!/usr/bin/env python3
"""
Chunk Generator for NQ Table Dataset
Creates multi-granular chunks for enhanced retrieval
"""

from typing import Dict, List, Any, Optional
import hashlib
import json
from bs4 import BeautifulSoup

class ChunkGenerator:
    """Generate multiple granularity chunks from processed tables"""
    
    def __init__(self, chunk_overlap=50, max_chunk_size=512):
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
    
    def create_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create multiple types of chunks from processed table data
        
        Args:
            processed_table: Output from TableProcessor.process_table_entry()
            
        Returns:
            List of chunks with different granularities
        """
        chunks = []
        table_id = processed_table['table_id']
        
        # 1. Full table chunk (highest context)
        full_chunk = self._create_full_table_chunk(processed_table)
        chunks.append(full_chunk)
        
        # 2. Row-level chunks
        row_chunks = self._create_row_chunks(processed_table)
        chunks.extend(row_chunks)
        
        # 3. Question-contextualized chunks
        context_chunks = self._create_contextualized_chunks(processed_table)
        chunks.extend(context_chunks)
        
        # 4. Sliding window chunks for long tables
        if len(processed_table['linearized_text']) > self.max_chunk_size:
            sliding_chunks = self._create_sliding_window_chunks(processed_table)
            chunks.extend(sliding_chunks)
        
        return chunks
    
    def _create_full_table_chunk(self, processed_table: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk containing the entire table"""
        table_id = processed_table['table_id']
        
        # Combine multiple representations
        content_parts = []
        
        # Add question context
        content_parts.append(f"Question: {processed_table['original_question']}")
        
        # Add natural description
        if processed_table['natural_description']:
            content_parts.append(f"Description: {processed_table['natural_description']}")
        
        # Add linearized table
        content_parts.append(f"Table: {processed_table['linearized_text']}")
        
        content = " | ".join(content_parts)
        
        return {
            'id': f"{table_id}_full",
            'content': content,
            'chunk_type': 'full_table',
            'granularity': 'table',
            'table_id': table_id,
            'metadata': {
                'question': processed_table['original_question'],
                'answers': processed_table['gold_answers'],
                'example_id': processed_table['example_id'],
                'table_metadata': processed_table['metadata'],
                'chunk_size': len(content),
                'has_context': True
            }
        }
    
    def _create_row_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create individual chunks for each table row"""
        chunks = []
        table_id = processed_table['table_id']
        structured_data = processed_table['structured_data']
        
        if not structured_data or 'rows' not in structured_data:
            return chunks
        
        headers = structured_data.get('headers', [])
        rows = structured_data['rows']
        
        for i, row in enumerate(rows):
            # Create row content with headers
            if headers and len(headers) == len(row):
                row_content = []
                for header, cell in zip(headers, row):
                    if cell.strip():  # Only include non-empty cells
                        row_content.append(f"{header}: {cell}")
                content = " | ".join(row_content)
            else:
                content = " | ".join([cell for cell in row if cell.strip()])
            
            if content.strip():  # Only create chunk if there's actual content
                chunk = {
                    'id': f"{table_id}_row_{i}",
                    'content': content,
                    'chunk_type': 'table_row',
                    'granularity': 'row',
                    'table_id': table_id,
                    'metadata': {
                        'question': processed_table['original_question'],
                        'example_id': processed_table['example_id'],
                        'row_index': i,
                        'row_data': row,
                        'chunk_size': len(content),
                        'has_headers': bool(headers)
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _create_contextualized_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks that combine question context with table parts"""
        chunks = []
        table_id = processed_table['table_id']
        question = processed_table['original_question']
        
        # Question + Natural Description
        if processed_table['natural_description']:
            content = f"Question: {question} | Context: {processed_table['natural_description']}"
            
            chunk = {
                'id': f"{table_id}_context",
                'content': content,
                'chunk_type': 'question_context',
                'granularity': 'context',
                'table_id': table_id,
                'metadata': {
                    'question': question,
                    'answers': processed_table['gold_answers'],
                    'example_id': processed_table['example_id'],
                    'chunk_size': len(content),
                    'content_type': 'contextual'
                }
            }
            chunks.append(chunk)
        
        # Question + Key Table Information
        structured_data = processed_table['structured_data']
        if structured_data and 'headers' in structured_data:
            headers = structured_data['headers']
            sample_rows = structured_data['rows'][:3] if 'rows' in structured_data else []
            
            content_parts = [f"Question: {question}"]
            content_parts.append(f"Table Headers: {' | '.join(headers)}")
            
            if sample_rows:
                for i, row in enumerate(sample_rows):
                    if row and any(cell.strip() for cell in row):
                        row_text = " | ".join([cell for cell in row if cell.strip()])
                        content_parts.append(f"Sample Row {i+1}: {row_text}")
            
            content = " | ".join(content_parts)
            
            chunk = {
                'id': f"{table_id}_question_sample",
                'content': content,
                'chunk_type': 'question_sample',
                'granularity': 'contextual',
                'table_id': table_id,
                'metadata': {
                    'question': question,
                    'answers': processed_table['gold_answers'],
                    'example_id': processed_table['example_id'],
                    'chunk_size': len(content),
                    'sample_rows_count': len(sample_rows)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_sliding_window_chunks(self, processed_table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks for very long tables"""
        chunks = []
        table_id = processed_table['table_id']
        linearized_text = processed_table['linearized_text']
        
        if len(linearized_text) <= self.max_chunk_size:
            return chunks
        
        # Split into overlapping windows
        start = 0
        chunk_num = 0
        
        while start < len(linearized_text):
            end = min(start + self.max_chunk_size, len(linearized_text))
            chunk_text = linearized_text[start:end]
            
            # Try to break at word boundaries
            if end < len(linearized_text):
                last_space = chunk_text.rfind(' ')
                if last_space > start + self.max_chunk_size * 0.8:  # At least 80% of chunk size
                    end = start + last_space
                    chunk_text = linearized_text[start:end]
            
            # Add question context to each chunk
            content = f"Question: {processed_table['original_question']} | Table Section: {chunk_text}"
            
            chunk = {
                'id': f"{table_id}_window_{chunk_num}",
                'content': content,
                'chunk_type': 'sliding_window',
                'granularity': 'section',
                'table_id': table_id,
                'metadata': {
                    'question': processed_table['original_question'],
                    'example_id': processed_table['example_id'],
                    'chunk_number': chunk_num,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(content),
                    'is_partial': True
                }
            }
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(linearized_text):
                break
            
            chunk_num += 1
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated chunks"""
        if not chunks:
            return {}
        
        chunk_types = {}
        total_size = 0
        size_distribution = []
        
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            size = chunk['metadata']['chunk_size']
            total_size += size
            size_distribution.append(size)
        
        avg_size = total_size / len(chunks) if chunks else 0
        
        return {
            'total_chunks': len(chunks),
            'chunk_types': chunk_types,
            'total_size': total_size,
            'average_size': avg_size,
            'min_size': min(size_distribution) if size_distribution else 0,
            'max_size': max(size_distribution) if size_distribution else 0,
            'size_distribution': size_distribution
        }