#!/usr/bin/env python3
"""
Table Processor for NQ Table Dataset - CLEAN FIXED VERSION
Converts HTML tables into retrievable formats for RAG pipeline
"""

import json
import re
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import hashlib
import signal
import time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class TableProcessor:
    """Process NQ table data into retrievable formats with timeout protection"""
    
    def __init__(self, timeout_seconds=10):
        self.processed_count = 0
        self.timeout_seconds = timeout_seconds
    
    def process_table_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single NQ table entry with timeout protection"""
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)
        
        try:
            table_id = self._generate_table_id(entry)
            
            processed = {
                'table_id': table_id,
                'original_question': entry['question'],
                'gold_answers': entry['answers'],
                'original_html': entry['table_text'],
                'example_id': entry['example_id'],
                
                # Multiple representation formats
                'linearized_text': self._linearize_table_safe(entry['table_text']),
                'structured_data': self._parse_table_structure_safe(entry['table_text']),
                'natural_description': self._generate_natural_description_safe(
                    entry['table_text'], entry['question']
                ),
                'metadata': self._extract_metadata_safe(entry['table_text']),
                
                # Simple chunks for now
                'chunks': []
            }
            
            self.processed_count += 1
            return processed
            
        except TimeoutError:
            print(f"⚠️  Timeout processing entry {entry.get('example_id', 'unknown')}")
            # Return minimal safe processing
            return self._create_fallback_entry(entry)
            
        except Exception as e:
            print(f"⚠️  Error processing entry {entry.get('example_id', 'unknown')}: {e}")
            return self._create_fallback_entry(entry)
            
        finally:
            signal.alarm(0)  # Cancel timeout
    
    def _create_fallback_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback entry when processing fails"""
        table_id = self._generate_table_id(entry)
        safe_text = self._safe_text_extraction(entry['table_text'])
        
        return {
            'table_id': table_id,
            'original_question': entry['question'],
            'gold_answers': entry['answers'],
            'original_html': entry['table_text'],
            'example_id': entry['example_id'],
            'linearized_text': safe_text,
            'structured_data': {'type': 'fallback', 'content': safe_text},
            'natural_description': f"Table related to: {entry['question']}",
            'metadata': {'processing': 'fallback'},
            'chunks': []
        }
    
    def _safe_text_extraction(self, html_table: str) -> str:
        """Safe text extraction without complex processing"""
        try:
            # Simple approach: remove tags and clean text
            text = re.sub(r'<[^>]*>', ' ', html_table)
            text = ' '.join(text.split())  # Normalize whitespace
            return text  # No length limit - preserve full content
        except:
            return html_table[:1000]  # Fallback with higher limit
    
    def _linearize_table_safe(self, html_table: str) -> str:
        """Safe table linearization with timeout protection"""
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            
            # Handle tables
            table = soup.find('table')
            if table:
                return self._linearize_html_table_safe(table)
            
            # Fallback to text extraction
            return self._safe_text_extraction(html_table)
            
        except Exception as e:
            return self._safe_text_extraction(html_table)
    
    def _linearize_html_table_safe(self, table) -> str:
        """Safe HTML table linearization"""
        result = []
        max_rows = 100  # Limit processing to prevent infinite loops
        
        try:
            rows = table.find_all('tr')[:max_rows]  # Limit rows
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = []
                    for cell in cells:  # Process all cells, no limit
                        cell_text = self._clean_text_safe(cell.get_text())
                        if cell_text and len(cell_text.strip()) > 0:
                            row_text.append(cell_text.strip())  # No cell length limit
                    
                    if row_text:
                        result.append(" | ".join(row_text))
            
            final_result = " || ".join(result) if result else ""
            return final_result  # No total length limit
            
        except Exception as e:
            return self._safe_text_extraction(str(table))
    
    def _clean_text_safe(self, text: str) -> str:
        """Safe text cleaning without problematic regex"""
        if not text:
            return ""
        
        try:
            # Simple cleaning without complex regex
            text = text.replace('\n', ' ').replace('\t', ' ')
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Remove obvious non-printable characters
            text = ''.join(char for char in text if ord(char) >= 32 or char.isspace())
            
            return text.strip()[:500]  # Limit length
            
        except Exception:
            return str(text)[:100]  # Ultimate fallback
    
    def _parse_table_structure_safe(self, html_table: str) -> Dict[str, Any]:
        """Safe table structure parsing"""
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            
            if table:
                rows = table.find_all('tr')[:50]  # Limit rows
                headers = []
                data_rows = []
                
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    if not cells:
                        continue
                        
                    cell_texts = []
                    for cell in cells[:10]:  # Limit cells
                        text = self._clean_text_safe(cell.get_text())
                        cell_texts.append(text)
                    
                    # Determine if header row
                    if i == 0 or (cells and cells[0].name == 'th'):
                        headers.extend([t for t in cell_texts if t])
                    else:
                        if any(cell_texts):  # Only add non-empty rows
                            data_rows.append(cell_texts)
                
                return {
                    'type': 'table',
                    'headers': headers[:10],  # Limit headers
                    'rows': data_rows[:50],   # Limit rows
                    'num_rows': len(data_rows),
                    'num_cols': max(len(row) for row in data_rows) if data_rows else 0
                }
            else:
                return {
                    'type': 'text',
                    'content': self._safe_text_extraction(html_table)
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'content': self._safe_text_extraction(html_table),
                'error': str(e)
            }
    
    def _generate_natural_description_safe(self, html_table: str, question: str) -> str:
        """Safe natural description generation"""
        try:
            structured = self._parse_table_structure_safe(html_table)
            
            if structured.get('type') == 'table':
                headers = structured.get('headers', [])[:5]  # Limit headers shown
                num_rows = structured.get('num_rows', 0)
                
                if headers:
                    header_text = ', '.join(headers)
                    desc = f"This table contains {num_rows} rows with columns: {header_text}."
                else:
                    desc = f"This table contains {num_rows} rows of data."
                    
                desc += f" It relates to the question: {question[:100]}"  # Limit question length
                
            else:
                desc = f"This content relates to the question: {question[:100]}"
            
            return desc[:500]  # Limit description length
            
        except Exception:
            return f"Table related to: {question[:100]}"
    
    def _extract_metadata_safe(self, html_table: str) -> Dict[str, Any]:
        """Safe metadata extraction"""
        try:
            return {
                'estimated_size': min(len(html_table), 10000),  # Cap the size
                'has_table_tag': '<table>' in html_table.lower(),
                'processing_method': 'safe'
            }
        except Exception:
            return {'processing_method': 'fallback'}
    
    def _generate_table_id(self, entry: Dict[str, Any]) -> str:
        """Generate unique table ID"""
        content = f"{entry['example_id']}_{entry['question']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]