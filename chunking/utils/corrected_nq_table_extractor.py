#!/usr/bin/env python3
"""
Corrected NQ Table Extractor - Fixed for actual NQ data format
Based on TAPAS methodology but adapted for NQ data structure
"""

import json
import argparse
import re
import gzip
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import html
from tqdm import tqdm

def parse_nq_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a Natural Questions example and extract table-based interactions."""
    
    try:
        # Extract basic fields - use correct field names from NQ dataset
        question_text = example['question_text']
        document_html = example['document_html']  # This is the HTML content
        document_tokens = example['document_tokens']  # Tokenized version
        annotations = example.get('annotations', [])
        long_answer_candidates = example.get('long_answer_candidates', [])
        example_id = example.get('example_id', 'unknown')
        
        # Skip if no annotations
        if not annotations:
            return None
            
        # Get table candidates (top_level=True indicates table/list elements)
        table_candidates = [c for c in long_answer_candidates if c.get('top_level', False)]
        
        if not table_candidates:
            return None
            
        # Process each annotation
        for annotation in annotations:
            long_answer = annotation.get('long_answer')
            short_answers = annotation.get('short_answers', [])
            
            # Skip if no long answer or it's null/none
            if not long_answer or long_answer.get('start_token', -1) == -1:
                continue
                
            # Get the candidate using candidate_index
            candidate_index = long_answer.get('candidate_index', -1)
            if candidate_index == -1 or candidate_index >= len(long_answer_candidates):
                continue
                
            target_candidate = long_answer_candidates[candidate_index]
            
            # Check if this candidate is a table (top_level=True)
            if not target_candidate.get('top_level', False):
                continue
                
            matching_table = target_candidate
                
            # Extract table text
            table_start = matching_table['start_token']
            table_end = matching_table['end_token']
            long_start = long_answer['start_token']
            long_end = long_answer['end_token']
            
            # Extract tokens from the token objects
            tokens = [token_dict['token'] for token_dict in document_tokens]
            
            if table_end > len(tokens):
                continue
                
            # Join tokens to get table text
            table_text = ' '.join(tokens[table_start:table_end])
            
            # Check if this looks like a table (contains multiple lines or table-like structure)
            if not is_table_like(table_text):
                continue
                
            # Extract answer text if short answers exist
            answer_texts = []
            if short_answers:
                for short_answer in short_answers:
                    start_token = short_answer['start_token']
                    end_token = short_answer['end_token']
                    if end_token <= len(tokens):
                        answer_text = ' '.join(tokens[start_token:end_token])
                        answer_texts.append(answer_text)
            
            # Return first valid table-based interaction
            return {
                'example_id': example_id,
                'question': question_text,
                'table_text': table_text,
                'answers': answer_texts,
                'table_start_token': table_start,
                'table_end_token': table_end,
                'long_answer_start': long_start,
                'long_answer_end': long_end
            }
            
        return None
        
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def is_table_like(text: str) -> bool:
    """Check if text appears to be table-like content."""
    
    # Convert to lowercase for easier checking
    text_lower = text.lower()
    
    # Strong table indicators
    strong_indicators = [
        '<table',     # Table tags
        '<th>',       # Table headers
        '<td>',       # Table data
        '<tr>',       # Table rows
        'colspan',    # Table attributes
        'rowspan',    # Table attributes
    ]
    
    # Check for strong table indicators
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True
    
    # List indicators (often structured like tables in NQ)
    list_indicators = [
        '<li>',       # List items
        '<ul>',       # Unordered lists
        '<ol>',       # Ordered lists
        '<dl>',       # Definition lists
    ]
    
    # Count list indicators
    list_count = sum(1 for indicator in list_indicators if indicator in text_lower)
    
    # If it has multiple list items, it's likely structured data
    if '<li>' in text_lower:
        li_count = text_lower.count('<li>')
        if li_count >= 3:  # At least 3 list items
            return True
    
    # Check for tabular data patterns
    lines = text.strip().split('\n')
    
    # Must have multiple lines for tabular data
    if len(lines) < 3:
        return False
    
    # Look for structured data patterns
    tab_separated = '\t' in text
    pipe_separated = '|' in text and text.count('|') >= 4
    
    if tab_separated or pipe_separated:
        return True
    
    # Check for repeated structural patterns (like multiple rows)
    if len(lines) >= 4:
        # Look for lines with similar word counts (table rows often have similar structure)
        word_counts = [len(line.split()) for line in lines[:10]]  # Check first 10 lines
        if len(word_counts) >= 4:
            # Filter out very short lines (headers, etc.)
            substantial_lines = [count for count in word_counts if count >= 3]
            if len(substantial_lines) >= 3:
                # Check if lines have similar word counts (within 50% variance)
                avg_words = sum(substantial_lines) / len(substantial_lines)
                similar_structure = sum(1 for count in substantial_lines 
                                      if abs(count - avg_words) <= avg_words * 0.5)
                
                if similar_structure >= 3:
                    return True
    
    return False

def process_nq_file(input_file: str, output_file: str, sample_size: Optional[int] = None):
    """Process NQ file and extract table interactions."""
    
    interactions = []
    total_examples = 0
    table_candidates_found = 0
    successful_extractions = 0
    
    print(f"Processing {input_file}...")
    
    # Handle both compressed and uncompressed files
    if input_file.endswith('.gz'):
        file_opener = lambda: gzip.open(input_file, 'rt', encoding='utf-8')
    else:
        file_opener = lambda: open(input_file, 'r', encoding='utf-8')
    
    # First pass: count total lines for progress bar
    print("Counting total examples...")
    total_lines = 0
    with file_opener() as f:
        for line in f:
            total_lines += 1
    
    # Limit to sample size if specified
    if sample_size:
        total_lines = min(total_lines, sample_size)
    
    print(f"Processing {total_lines:,} examples...")
    
    with file_opener() as f:
        # Create progress bar
        pbar = tqdm(total=total_lines, desc="Extracting tables", unit="examples")
        
        for line_num, line in enumerate(f):
            if sample_size and line_num >= sample_size:
                break
                
            total_examples += 1
            pbar.update(1)
            
            # Update progress bar description with current stats
            if total_examples % 100 == 0:
                success_rate = (successful_extractions / total_examples * 100) if total_examples > 0 else 0
                pbar.set_description(f"Extracted: {successful_extractions} ({success_rate:.1f}%), Candidates: {table_candidates_found}")
            
            try:
                example = json.loads(line.strip())
                
                # Quick check for table candidates
                table_candidates = [c for c in example.get('long_answer_candidates', []) 
                                  if c.get('top_level', False)]
                
                if table_candidates:
                    table_candidates_found += 1
                    
                    # Process the example
                    interaction = parse_nq_example(example)
                    
                    if interaction:
                        interactions.append(interaction)
                        successful_extractions += 1
                        
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
        
        # Final update to progress bar
        success_rate = (successful_extractions / total_examples * 100) if total_examples > 0 else 0
        pbar.set_description(f"Completed! Extracted: {successful_extractions} ({success_rate:.1f}%), Candidates: {table_candidates_found}")
        pbar.close()
    
    # Save results
    print(f"\nSaving {len(interactions)} interactions to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for interaction in interactions:
            json.dump(interaction, f, ensure_ascii=False)
            f.write('\n')
    
    # Print statistics
    print(f"\n=== Processing Results ===")
    print(f"Total examples processed: {total_examples:,}")
    print(f"Examples with table candidates: {table_candidates_found:,}")
    print(f"Successful table extractions: {successful_extractions:,}")
    print(f"Success rate: {successful_extractions/total_examples*100:.2f}%")
    print(f"Table candidate success rate: {successful_extractions/max(1,table_candidates_found)*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Extract table-based interactions from Natural Questions')
    parser.add_argument('--input', required=True, help='Input NQ JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--sample', type=int, help='Process only first N examples (for testing)')
    
    args = parser.parse_args()
    
    process_nq_file(args.input, args.output, args.sample)

if __name__ == '__main__':
    main()