"""
Generate synthetic training data by running the exp2 model on neural data
and saving predicted phonemes paired with ground-truth text.

This creates noisy training data where the input is the model's phoneme predictions
(which may have errors) and the target is the original sentence text.

Output format matches train.jsonl:
{"input": "B R IH NG SEP IH T SEP", "target": "bring it"}
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import Levenshtein
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_training'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_training' / 'baseline'))
from data_augmentations import gauss_smooth
from mlp_gru_model import Exp2Model

# Phoneme mapping from model output indices to phoneme names
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


def select_device(preferred: str = "auto") -> torch.device:
    """Select compute device."""
    if preferred.lower() == "cpu":
        return torch.device("cpu")
    if preferred.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if preferred.lower() == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_exp2_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load an Exp2Model from a .pt checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_days = len(config['dataset']['sessions'])

    # Create the model without loading pretrained weights via constructor
    model = Exp2Model(
        config=config,
        num_days=num_days,
        pretrained_ckpt_path=None,
        freeze_gru=False,
    )

    # Load the checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model, config


def decode_logits_to_phoneme_ids(logits_np: np.ndarray) -> list[int]:
    """
    CTC-style greedy decode: argmax → remove blanks → collapse repeats.
    Returns list of phoneme class IDs.
    """
    pred_seq = np.argmax(logits_np, axis=-1)
    # Remove blanks (index 0)
    pred_seq = [int(p) for p in pred_seq if p != 0]
    # Collapse consecutive repeats
    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i - 1]]
    return pred_seq


def phoneme_ids_to_tokens(phoneme_ids: list[int]) -> str:
    """
    Convert phoneme IDs to tokenized string format.
    Word separator (index 40) becomes 'SEP'.
    
    Returns: "B R IH NG SEP IH T SEP"
    """
    tokens = []
    for idx in phoneme_ids:
        if idx == 0:  # Skip BLANK
            continue
        phoneme = LOGIT_TO_PHONEME[idx]
        if phoneme == ' | ':
            tokens.append('SEP')
        else:
            tokens.append(phoneme)
    return ' '.join(tokens)


def remove_punctuation(sentence: str) -> str:
    """Remove punctuation and normalize the sentence."""
    import re
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    return sentence


def extract_transcription(transcription_array) -> str:
    """Extract transcription string from the character array."""
    chars = []
    for c in transcription_array:
        if c == 0:
            break
        chars.append(chr(c))
    return ''.join(chars)


def load_hdf5_trials(file_path: str) -> list[dict]:
    """
    Load trials from an HDF5 file.
    
    Returns list of dicts with keys:
        - neural_features: (T, 512) array
        - transcription: raw transcription string
        - sentence_label: sentence label attribute (if available)
    """
    trials = []
    
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            g = f[key]
            
            neural_features = g['input_features'][:]
            
            # Get transcription
            if 'transcription' in g:
                transcription = extract_transcription(g['transcription'][:])
            else:
                transcription = None
            
            # Get sentence label as fallback
            sentence_label = None
            if 'sentence_label' in g.attrs:
                sentence_label = g.attrs['sentence_label']
                if isinstance(sentence_label, (bytes, bytearray, np.ndarray)):
                    try:
                        sentence_label = bytes(sentence_label).decode().strip()
                    except Exception:
                        sentence_label = str(sentence_label)
            
            trials.append({
                'neural_features': neural_features,
                'transcription': transcription,
                'sentence_label': sentence_label,
            })
    
    return trials


def generate_synthetic_data(
    model: torch.nn.Module,
    config: dict,
    data_dir: str,
    split: str,  # 'train' or 'val'
    device: torch.device,
    show_progress: bool = True,
) -> list[dict]:
    """
    Run model on all trials from the specified split and collect predictions.
    
    Returns list of dicts with:
        - input: predicted phonemes as tokenized string
        - target: normalized ground-truth text
    """
    sessions_list = list(config['dataset']['sessions'])
    smooth_kernel_std = config['dataset']['data_transforms']['smooth_kernel_std']
    smooth_kernel_size = config['dataset']['data_transforms']['smooth_kernel_size']
    
    examples = []
    
    sessions_iter = sessions_list
    if show_progress:
        sessions_iter = tqdm(sessions_iter, desc=f"Processing {split} sessions", unit="session")
    
    for session_idx, session in enumerate(sessions_iter):
        data_file = os.path.join(data_dir, session, f"data_{split}.hdf5")
        
        if not os.path.exists(data_file):
            continue
        
        trials = load_hdf5_trials(data_file)
        
        trials_iter = trials
        if show_progress:
            trials_iter = tqdm(trials, desc=f"Trials ({session})", unit="trial", leave=False)
        
        for trial in trials_iter:
            # Get target text
            target_text = trial['transcription'] or trial['sentence_label']
            if not target_text:
                continue
            target_text = remove_punctuation(target_text)
            if not target_text:
                continue
            
            # Prepare neural features
            x_np = np.expand_dims(trial['neural_features'], axis=0)
            x = torch.tensor(x_np, device=device, dtype=torch.float32)
            
            # Apply Gaussian smoothing
            x = gauss_smooth(
                inputs=x,
                device=device,
                smooth_kernel_std=smooth_kernel_std,
                smooth_kernel_size=smooth_kernel_size,
                padding='valid',
            )
            
            # Forward pass
            with torch.no_grad():
                day_idx = torch.tensor([session_idx], device=device)
                logits = model(x, day_idx)
            
            # Decode to phonemes
            logits_np = logits.float().cpu().numpy()[0]
            pred_ids = decode_logits_to_phoneme_ids(logits_np)
            
            if not pred_ids:
                continue
            
            # Convert to tokenized string
            phoneme_tokens = phoneme_ids_to_tokens(pred_ids)
            
            if not phoneme_tokens:
                continue
            
            examples.append({
                'input': phoneme_tokens,
                'target': target_text,
            })
    
    return examples


def save_jsonl(examples: list[dict], output_path: str):
    """Save examples to JSONL file."""
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def load_jsonl(file_path: str) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_confusion_matrix(ground_truth_list: list[str], gru_prediction_list: list[str]) -> dict:
    """
    Build a phoneme confusion matrix using Levenshtein edit operations.
    
    This analyzes which ground truth phonemes the GRU most often gets incorrect,
    tracking substitutions (phoneme A predicted as phoneme B), deletions (phoneme
    was missed), and insertions (extra phonemes were predicted).
    
    Args:
        ground_truth_list: list of phoneme strings (e.g., ["W IY R SEP", "H EH L OW SEP"])
        gru_prediction_list: list of phoneme strings (e.g., ["W IY SEP", "H EH L OW SEP"])
    
    Returns:
        noise_profile dict mapping source phonemes to their error distributions
    """
    # Structure: substitutions[actual_phoneme][predicted_phoneme] = count
    substitutions = defaultdict(lambda: defaultdict(int))
    deletions = defaultdict(int)
    insertions = defaultdict(int)
    
    # Count ALL occurrences of each phoneme in ground truth (not just errors!)
    total_occurrences = defaultdict(int)
    
    for truth, pred in zip(ground_truth_list, gru_prediction_list):
        # Work with lists of phonemes (space-separated)
        t_tokens = truth.split()
        p_tokens = pred.split()
        
        # Count all ground truth phoneme occurrences
        for token in t_tokens:
            total_occurrences[token] += 1
        
        # Get alignment operations using Levenshtein
        # editops expects sequences, so we pass the token lists directly
        ops = Levenshtein.editops(t_tokens, p_tokens)
        
        # ops is a list of tuples: (operation_type, src_index, dest_index)
        # e.g., ('replace', 0, 0) means t_tokens[0] became p_tokens[0]
        
        for op, t_idx, p_idx in ops:
            if op == 'replace':
                src = t_tokens[t_idx]
                dst = p_tokens[p_idx]
                substitutions[src][dst] += 1
            elif op == 'delete':
                src = t_tokens[t_idx]
                deletions[src] += 1
            elif op == 'insert':
                # Insertions: extra phonemes predicted that weren't in ground truth
                dst = p_tokens[p_idx]
                insertions[dst] += 1
    
    # Convert counts to probabilities for noise_profile
    noise_profile = {}
    
    # Iterate over ALL phonemes that appear in ground truth
    for src in total_occurrences:
        total = total_occurrences[src]
        if total < 5:
            continue  # Skip rare phonemes to avoid noise
        
        mappings = []
        
        # Add the "error" mappings (substitutions)
        if src in substitutions:
            for dst, count in substitutions[src].items():
                prob = count / total
                mappings.append({"error": dst, "prob": prob})
        
        # Add deletion probability
        del_count = deletions.get(src, 0)
        if del_count > 0:
            mappings.append({"error": "<DELETE>", "prob": del_count / total})
        
        # Calculate error rate (remainder is probability of staying correct)
        error_sum = sum(m['prob'] for m in mappings)
        correct_prob = 1.0 - error_sum
        
        # Only add to profile if there is a significant error rate
        if error_sum > 0.01:
            # Sort mappings by probability (descending)
            mappings.sort(key=lambda x: x['prob'], reverse=True)
            noise_profile[src] = {
                "correct_prob": correct_prob,
                "error_rate": error_sum,
                "total_occurrences": total,
                "mappings": mappings
            }
    
    # Also track overall insertion statistics
    total_insertions = sum(insertions.values())
    if total_insertions > 0:
        insertion_profile = []
        for phoneme, count in sorted(insertions.items(), key=lambda x: x[1], reverse=True):
            insertion_profile.append({
                "phoneme": phoneme,
                "count": count,
                "prob": count / total_insertions
            })
        noise_profile["<INSERTIONS>"] = {
            "total_count": total_insertions,
            "distribution": insertion_profile[:20]  # Top 20 inserted phonemes
        }
    
    return noise_profile


def compute_gru_confusion_matrix(
    val_jsonl_path: str,
    val_synth_jsonl_path: str,
    output_path: str,
    show_progress: bool = True,
) -> dict:
    """
    Compute GRU confusion matrix by comparing ground truth phonemes (val.jsonl)
    with GRU predictions (val_synth.jsonl).
    
    The 'input' field in val.jsonl contains ground truth phonemes.
    The 'input' field in val_synth.jsonl contains GRU predicted phonemes.
    
    Args:
        val_jsonl_path: Path to val.jsonl (ground truth phonemes)
        val_synth_jsonl_path: Path to val_synth.jsonl (GRU predictions)
        output_path: Path to save the confusion matrix JSON
        show_progress: Whether to show progress
    
    Returns:
        The computed noise profile dictionary
    """
    print(f"Loading ground truth phonemes from: {val_jsonl_path}")
    val_examples = load_jsonl(val_jsonl_path)
    
    print(f"Loading GRU predictions from: {val_synth_jsonl_path}")
    val_synth_examples = load_jsonl(val_synth_jsonl_path)
    
    if len(val_examples) != len(val_synth_examples):
        print(f"Warning: Different number of examples - val: {len(val_examples)}, val_synth: {len(val_synth_examples)}")
        # Use the minimum
        min_len = min(len(val_examples), len(val_synth_examples))
        val_examples = val_examples[:min_len]
        val_synth_examples = val_synth_examples[:min_len]
    
    # Verify that targets match (they should be paired correctly)
    mismatches = 0
    for i, (gt, pred) in enumerate(zip(val_examples, val_synth_examples)):
        if gt['target'] != pred['target']:
            mismatches += 1
            if mismatches <= 3:
                print(f"Warning: Target mismatch at line {i}: '{gt['target']}' vs '{pred['target']}'")
    
    if mismatches > 0:
        print(f"Total target mismatches: {mismatches}")
    
    # Extract phoneme sequences
    ground_truth_list = [ex['input'] for ex in val_examples]
    gru_prediction_list = [ex['input'] for ex in val_synth_examples]
    
    print(f"\nBuilding confusion matrix from {len(ground_truth_list)} examples...")
    
    # Build the confusion matrix
    noise_profile = build_confusion_matrix(ground_truth_list, gru_prediction_list)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(noise_profile, f, indent=2)
    
    print(f"\nSaved confusion matrix to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 60)
    
    # Sort phonemes by error rate
    phoneme_errors = []
    for phoneme, data in noise_profile.items():
        if phoneme == "<INSERTIONS>":
            continue
        phoneme_errors.append((phoneme, data['error_rate'], data['total_occurrences'], data['mappings']))
    
    phoneme_errors.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 phonemes by error rate:")
    print("-" * 60)
    for i, (phoneme, error_rate, total, mappings) in enumerate(phoneme_errors[:15], 1):
        top_errors = mappings[:3]
        error_strs = [f"{m['error']}({m['prob']:.2%})" for m in top_errors]
        print(f"  {i:2d}. {phoneme:6s} | Error rate: {error_rate:.2%} | N={total:4d} | Top: {', '.join(error_strs)}")
    
    # Print insertion summary
    if "<INSERTIONS>" in noise_profile:
        ins_data = noise_profile["<INSERTIONS>"]
        print(f"\nTotal insertions: {ins_data['total_count']}")
        top_ins = ins_data['distribution'][:5]
        ins_strs = [f"{d['phoneme']}({d['count']})" for d in top_ins]
        print(f"Top inserted phonemes: {', '.join(ins_strs)}")
    
    return noise_profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using exp2 model predictions."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../model_training/trained_models/exp2/checkpoint_best.pt",
        help="Path to exp2 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../model_training/exp2_args.yaml",
        help="Path to exp2 config YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/hdf5_data_final",
        help="Path to HDF5 data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/llm_training_data",
        help="Directory to save output JSONL files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device to use",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="train_synth.jsonl",
        help="Filename for training synthetic data",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val_synth.jsonl",
        help="Filename for validation synthetic data",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--confusion_matrix",
        action="store_true",
        help="Compute GRU confusion matrix from existing val.jsonl and val_synth.jsonl files",
    )
    parser.add_argument(
        "--confusion_output",
        type=str,
        default="GRU_confusion_matrix.json",
        help="Filename for confusion matrix output",
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    output_dir = Path((script_dir / args.output_dir).resolve())
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle confusion matrix mode
    if args.confusion_matrix:
        val_jsonl_path = str(output_dir / "val.jsonl")
        val_synth_jsonl_path = str(output_dir / "val_synth.jsonl")
        confusion_output_path = str(output_dir / args.confusion_output)
        
        compute_gru_confusion_matrix(
            val_jsonl_path=val_jsonl_path,
            val_synth_jsonl_path=val_synth_jsonl_path,
            output_path=confusion_output_path,
            show_progress=not args.no_progress,
        )
        return
    
    checkpoint_path = str((script_dir / args.checkpoint).resolve())
    config_path = str((script_dir / args.config).resolve())
    data_dir = str((script_dir / args.data_dir).resolve())
    
    # Select device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using config: {config_path}")
    model, config = load_exp2_model(checkpoint_path, config_path, device)
    print("Model loaded successfully.")
    
    show_progress = not args.no_progress
    
    # Generate training data
    print("\n" + "=" * 60)
    print("Generating synthetic TRAINING data...")
    print("=" * 60)
    train_examples = generate_synthetic_data(
        model=model,
        config=config,
        data_dir=data_dir,
        split='train',
        device=device,
        show_progress=show_progress,
    )
    
    train_output_path = output_dir / args.train_output
    save_jsonl(train_examples, str(train_output_path))
    print(f"\nSaved {len(train_examples)} training examples to {train_output_path}")
    
    # Generate validation data
    print("\n" + "=" * 60)
    print("Generating synthetic VALIDATION data...")
    print("=" * 60)
    val_examples = generate_synthetic_data(
        model=model,
        config=config,
        data_dir=data_dir,
        split='val',
        device=device,
        show_progress=show_progress,
    )
    
    val_output_path = output_dir / args.val_output
    save_jsonl(val_examples, str(val_output_path))
    print(f"\nSaved {len(val_examples)} validation examples to {val_output_path}")
    
    # Print sample examples
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES (first 5 from training set):")
    print("=" * 60)
    for i, example in enumerate(train_examples[:5]):
        print(f"\nExample {i + 1}:")
        print(f"  Input:  {example['input']}")
        print(f"  Target: {example['target']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

