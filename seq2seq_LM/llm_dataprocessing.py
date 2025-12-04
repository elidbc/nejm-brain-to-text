"""
Process HDF5 brain-to-text data into JSONL format for LLM finetuning.

Creates labeled data mapping phoneme sequences to English sentences.
Phonemes are tokenized as: HH AH L OW SEP W ER L D
"""

import os
import json
import h5py
import re
from pathlib import Path
from typing import Optional

# Phoneme mapping from seq_class_ids indices to phoneme names
# Index 0 = BLANK (omitted), Index 40 = word separator (becomes SEP)
LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # Word separator -> SEP
]


def remove_punctuation(sentence: str) -> str:
    """Remove punctuation and normalize the sentence."""
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    return sentence


def seq_class_ids_to_phoneme_tokens(seq_class_ids, seq_len: int) -> str:
    """
    Convert seq_class_ids array to tokenized phoneme string.
    
    Args:
        seq_class_ids: Array of phoneme indices
        seq_len: Number of valid phonemes (rest is padding)
    
    Returns:
        Tokenized phoneme string, e.g., "B R IH NG SEP IH T"
    """
    tokens = []
    
    # Only process up to seq_len (rest is zero-padding)
    for idx in seq_class_ids[:seq_len]:
        idx = int(idx)
        
        # Skip BLANK (index 0)
        if idx == 0:
            continue
        
        phoneme = LOGIT_TO_PHONEME[idx]
        
        # Word separator becomes SEP
        if phoneme == ' | ':
            tokens.append('SEP')
        else:
            tokens.append(f'{phoneme}')
    
    return ' '.join(tokens)


def extract_transcription(transcription_array) -> str:
    """Extract transcription string from the character array."""
    # Find the end (first zero byte)
    chars = []
    for c in transcription_array:
        if c == 0:
            break
        chars.append(chr(c))
    return ''.join(chars)


def process_hdf5_file(file_path: str) -> list[dict]:
    """
    Process a single HDF5 file and return list of {input, target} dicts.
    
    Args:
        file_path: Path to the HDF5 file
    
    Returns:
        List of dictionaries with 'input' (phoneme tokens) and 'target' (text)
    """
    examples = []
    
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            g = f[key]
            
            # Get phoneme sequence
            seq_class_ids = g['seq_class_ids'][:]
            seq_len = g.attrs['seq_len']
            
            # Get transcription
            transcription_raw = g['transcription'][:]
            transcription = extract_transcription(transcription_raw)
            
            # Convert to tokenized format
            phoneme_tokens = seq_class_ids_to_phoneme_tokens(seq_class_ids, seq_len)
            target_text = remove_punctuation(transcription)
            
            # Skip empty examples
            if not phoneme_tokens or not target_text:
                continue
            
            examples.append({
                'input': phoneme_tokens,
                'target': target_text,
            })
    
    return examples


def process_sessions(
    data_dir: str,
    output_dir: str,
    sessions: Optional[list[str]] = None,
    max_sessions: Optional[int] = None,
) -> tuple[int, int]:
    """
    Process multiple session directories and save to JSONL files.
    
    Args:
        data_dir: Path to hdf5_data_final directory
        output_dir: Path to output directory for JSONL files
        sessions: Optional list of specific session names to process
        max_sessions: Optional limit on number of sessions to process
    
    Returns:
        Tuple of (train_count, val_count) examples processed
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all session directories if not specified
    if sessions is None:
        sessions = sorted([
            d.name for d in data_path.iterdir() 
            if d.is_dir() and d.name.startswith('t15.')
        ])
    
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    
    train_examples = []
    val_examples = []
    
    for session in sessions:
        session_path = data_path / session
        
        # Process training data
        train_file = session_path / 'data_train.hdf5'
        if train_file.exists():
            examples = process_hdf5_file(str(train_file))
            train_examples.extend(examples)
            print(f"  {session}: {len(examples)} train examples")
        
        # Process validation data if it exists
        val_file = session_path / 'data_val.hdf5'
        if val_file.exists():
            examples = process_hdf5_file(str(val_file))
            val_examples.extend(examples)
            print(f"  {session}: {len(examples)} val examples")
    
    # Save to JSONL files
    train_output = output_path / 'train.jsonl'
    with open(train_output, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    print(f"\nSaved {len(train_examples)} training examples to {train_output}")
    
    if val_examples:
        val_output = output_path / 'val.jsonl'
        with open(val_output, 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(val_examples)} validation examples to {val_output}")
    
    return len(train_examples), len(val_examples)


def get_unique_phoneme_tokens(jsonl_path: str) -> set[str]:
    """Extract all unique phoneme tokens from a JSONL file."""
    tokens = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            tokens.update(example['input'].split())
    return tokens


if __name__ == '__main__':
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'hdf5_data_final'
    output_dir = project_root / 'data' / 'llm_training_data'
    
    print("Processing brain-to-text data for LLM finetuning...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process only 3 sessions for spot checking
    train_count, val_count = process_sessions(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        max_sessions=45,  # Limit for spot checking
    )
    
    print(f"\nTotal: {train_count} train, {val_count} val examples")
    
    # Show unique phoneme tokens for tokenizer
    train_jsonl = output_dir / 'train.jsonl'
    if train_jsonl.exists():
        tokens = get_unique_phoneme_tokens(str(train_jsonl))
        print(f"\nUnique phoneme tokens ({len(tokens)}): {sorted(tokens)}")
    
    # Print first 5 examples for spot checking
    print("\n" + "="*60)
    print("SAMPLE EXAMPLES FOR SPOT CHECKING:")
    print("="*60)
    with open(train_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            example = json.loads(line)
            print(f"\nExample {i+1}:")
            print(f"  Input:  {example['input']}")
            print(f"  Target: {example['target']}")
