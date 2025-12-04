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
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_training'))
from data_augmentations import gauss_smooth
from exp2_model import Exp2Model

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
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    checkpoint_path = str((script_dir / args.checkpoint).resolve())
    config_path = str((script_dir / args.config).resolve())
    data_dir = str((script_dir / args.data_dir).resolve())
    output_dir = Path((script_dir / args.output_dir).resolve())
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

