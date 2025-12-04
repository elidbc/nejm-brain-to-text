"""Evaluate pretrained RNN or exp2 model on every validation trial across all sessions."""

import argparse
import os
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cs230_braintotext.model_training.baseline.data_augmentations import gauss_smooth
from cs230_braintotext.model_training.baseline.evaluate_model_helpers import LOGIT_TO_PHONEME, load_h5py_file
from cs230_braintotext.model_training.run_rnn_only import (
    calculate_aggregate_error_rate,
    calculate_error_rate,
    decode_logits_to_ids,
    load_model as load_baseline_model,
    select_device,
)
from cs230_braintotext.model_training.mlp_gru_model import Exp2Model


def load_exp2_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load an Exp2Model from a .pt checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_days = len(config['dataset']['sessions'])

    # Create the model (don't load pretrained weights via constructor - we'll load exp2 checkpoint directly)
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


def load_model(model_path: str, device: torch.device, config_path: str | None = None):
    """
    Load model from either:
    - A .pt checkpoint file (exp2 model format)
    - A directory with checkpoint/args.yaml and checkpoint/best_checkpoint (baseline format)
    """
    if model_path.endswith('.pt'):
        # Exp2 checkpoint format
        if config_path is None:
            # Try to find config in same directory or parent
            model_dir = os.path.dirname(model_path)
            candidate_paths = [
                os.path.join(model_dir, 'exp2_args.yaml'),
                os.path.join(os.path.dirname(model_dir), 'exp2_args.yaml'),
                os.path.join(model_dir, '..', 'exp2_args.yaml'),
            ]
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    config_path = candidate
                    break
            if config_path is None:
                # Default to model_training/exp2_args.yaml
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, 'exp2_args.yaml')
        print(f"Loading exp2 model from: {model_path}")
        print(f"Using config: {config_path}")
        return load_exp2_model(model_path, config_path, device), "exp2"
    else:
        # Baseline checkpoint format (directory)
        print(f"Loading baseline model from: {model_path}")
        return load_baseline_model(model_path, device), "baseline"


def _date_to_session(date_str: str) -> str:
    """Convert YYYY-MM-DD to dataset session name (e.g., t15.2023.08.13)."""
    cleaned = date_str.strip()
    if not cleaned:
        raise ValueError("Encountered empty date string in CSV description file.")
    return f"t15.{cleaned.replace('-', '.')}"


def evaluate_validation_split(
    model_path: str,
    data_dir: str,
    csv_path: str,
    device: torch.device,
    output_csv: str | None = None,
    per_trial_csv: str | None = None,
    show_progress: bool = True,
    config_path: str | None = None,
) -> None:
    """Run PER evaluation on the entire validation split across all sessions."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    csv_df = pd.read_csv(csv_path)
    if "Split" not in csv_df.columns:
        raise ValueError("Expected 'Split' column in dataset description CSV.")
    if "Date" not in csv_df.columns or "Block number" not in csv_df.columns:
        raise ValueError("Expected 'Date' and 'Block number' columns in dataset description CSV.")

    # Identify the validation blocks from the metadata CSV.
    val_rows = csv_df[csv_df["Split"].astype(str).str.contains("val", case=False, na=False)]
    if val_rows.empty:
        raise RuntimeError("No validation rows were found in the dataset description CSV.")

    val_blocks_by_session: dict[str, set[int]] = defaultdict(set)
    for _, row in val_rows.iterrows():
        session = _date_to_session(str(row["Date"]))
        try:
            block_num = int(row["Block number"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid block number in CSV for date {row['Date']!r}: {row['Block number']!r}") from exc
        val_blocks_by_session[session].add(block_num)

    (model, model_args), model_type = load_model(model_path, device, config_path)

    # Extract sessions list and data transforms
    sessions_list = list(model_args['dataset']['sessions'])
    smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std']
    smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size']

    dtype = torch.float32

    per_true_sequences: list[list[int]] = []
    per_pred_sequences: list[list[int]] = []

    session_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"true_len": 0.0, "edit_distance": 0.0, "count": 0.0})
    block_totals: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"true_len": 0.0, "edit_distance": 0.0, "count": 0.0}))

    per_trial_rows: list[dict[str, object]] = []

    # Track phoneme confusion statistics
    # phoneme_confusion[true_id][pred_id] = count of times true_id was predicted as pred_id
    phoneme_confusion: dict[int, Counter] = defaultdict(Counter)
    # Collect trial samples per session for random example selection
    session_samples: dict[str, list[dict]] = defaultdict(list)

    sessions_to_visit = sorted(val_blocks_by_session.keys())

    missing_sessions: list[str] = []

    iterator = sessions_to_visit
    if show_progress:
        iterator = tqdm(iterator, desc="Sessions", unit="session")

    for session in iterator:
        eval_file = os.path.join(data_dir, session, "data_val.hdf5")
        if not os.path.exists(eval_file):
            missing_sessions.append(session)
            continue

        data = load_h5py_file(eval_file, csv_df)
        try:
            input_layer = sessions_list.index(session)
        except ValueError as exc:
            raise ValueError(
                f"Session {session} from CSV is not listed in the model configuration sessions."
            ) from exc

        trial_indices = range(len(data['neural_features']))
        if show_progress:
            trial_indices = tqdm(trial_indices, desc=f"Trials ({session})", unit="trial", leave=False)

        for trial_idx in trial_indices:
            block_num = int(data['block_num'][trial_idx])
            if val_blocks_by_session[session] and block_num not in val_blocks_by_session[session]:
                # Skip trials that are not part of the validation split according to the metadata CSV.
                continue

            if data['seq_class_ids'][trial_idx] is None or data['seq_len'][trial_idx] is None:
                continue

            x_np = np.expand_dims(data['neural_features'][trial_idx], axis=0)
            x = torch.tensor(x_np, device=device, dtype=dtype)

            x = gauss_smooth(
                inputs=x,
                device=device,
                smooth_kernel_std=smooth_kernel_std,
                smooth_kernel_size=smooth_kernel_size,
                padding='valid',
            )

            with torch.no_grad():
                day_idx = torch.tensor([input_layer], device=device)
                if model_type == "exp2":
                    logits = model(x, day_idx)
                else:
                    logits = model(
                        x=x,
                        day_idx=day_idx,
                        states=None,
                        return_state=False,
                    )

            logits_np = logits.float().cpu().numpy()[0]
            pred_ids = decode_logits_to_ids(logits_np)

            true_ids_full = data['seq_class_ids'][trial_idx]
            true_len = int(data['seq_len'][trial_idx])
            true_ids = [int(p) for p in true_ids_full[:true_len]]
            if not true_ids:
                continue

            edit_distance = int(calculate_error_rate(true_ids, pred_ids))

            per_true_sequences.append(true_ids)
            per_pred_sequences.append(pred_ids)

            # Track phoneme-level confusion using alignment
            # Use simple position-based comparison with length normalization
            min_len = min(len(true_ids), len(pred_ids))
            for i in range(min_len):
                if true_ids[i] != pred_ids[i]:
                    phoneme_confusion[true_ids[i]][pred_ids[i]] += 1
            # Track deletions (extra true phonemes not covered)
            for i in range(min_len, len(true_ids)):
                phoneme_confusion[true_ids[i]][-1] += 1  # -1 represents deletion
            # Track insertions (extra predicted phonemes)
            for i in range(min_len, len(pred_ids)):
                phoneme_confusion[-2][pred_ids[i]] += 1  # -2 represents insertion source

            # Store sample for random selection
            sentence_label = data['sentence_label'][trial_idx]
            if isinstance(sentence_label, (bytes, bytearray, np.ndarray)):
                try:
                    sentence_label = bytes(sentence_label).decode().strip()
                except Exception:
                    sentence_label = str(sentence_label)
            session_samples[session].append({
                'session': session,
                'block': block_num,
                'trial': int(data['trial_num'][trial_idx]),
                'sentence_label': sentence_label,
                'true_ids': true_ids,
                'pred_ids': pred_ids,
                'edit_distance': edit_distance,
            })

            session_totals[session]['true_len'] += len(true_ids)
            session_totals[session]['edit_distance'] += edit_distance
            session_totals[session]['count'] += 1

            block_totals[session][block_num]['true_len'] += len(true_ids)
            block_totals[session][block_num]['edit_distance'] += edit_distance
            block_totals[session][block_num]['count'] += 1

            if per_trial_csv is not None:
                true_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in true_ids]
                pred_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in pred_ids]

                sentence_label = data['sentence_label'][trial_idx]
                if isinstance(sentence_label, (bytes, bytearray, np.ndarray)):
                    try:
                        sentence_label = bytes(sentence_label).decode().strip()
                    except Exception:
                        sentence_label = str(sentence_label)

                per_trial_rows.append({
                    'session': session,
                    'block': block_num,
                    'trial': int(data['trial_num'][trial_idx]),
                    'sentence_label': sentence_label,
                    'true_phonemes': ' '.join(true_phonemes),
                    'predicted_phonemes': ' '.join(pred_phonemes),
                    'per': edit_distance / len(true_ids),
                })

    if missing_sessions:
        print("Warning: the following validation sessions were listed in the CSV but missing from the data directory:")
        for session in missing_sessions:
            print(f"  - {session}")

    total_true_len = sum(t['true_len'] for t in session_totals.values())
    total_edit_distance = sum(t['edit_distance'] for t in session_totals.values())

    if total_true_len == 0:
        raise RuntimeError("No validation trials with ground-truth phonemes were evaluated.")

    aggregate_per = total_edit_distance / total_true_len

    try:
        per, per_lo, per_hi, _ = calculate_aggregate_error_rate(per_true_sequences, per_pred_sequences)
    except Exception:
        per, per_lo, per_hi = aggregate_per, float('nan'), float('nan')

    print()
    print("===== Validation PER Summary =====")
    print(f"Overall PER: {100 * aggregate_per:.2f}%")
    if not np.isnan(per_lo):
        print(f"Bootstrap 95% CI: [{100 * per_lo:.2f}%, {100 * per_hi:.2f}%]")
    print(f"Evaluated trials: {int(sum(t['count'] for t in session_totals.values()))}")
    print()

    print("Per-session PER:")
    session_rows: list[dict[str, object]] = []
    for session in sorted(session_totals.keys()):
        totals = session_totals[session]
        if totals['true_len'] == 0:
            continue
        sess_per = totals['edit_distance'] / totals['true_len']
        print(f"  {session}: {100 * sess_per:.2f}% over {int(totals['count'])} trials")
        session_rows.append({
            'session': session,
            'trials': int(totals['count']),
            'per': sess_per,
        })

    print()
    print("Per-block PER:")
    for session in sorted(block_totals.keys()):
        for block in sorted(block_totals[session].keys()):
            totals = block_totals[session][block]
            if totals['true_len'] == 0:
                continue
            block_per = totals['edit_distance'] / totals['true_len']
            print(f"  {session} | Block {block}: {100 * block_per:.2f}% over {int(totals['count'])} trials")

    # === Phoneme Confusion Analysis ===
    print()
    print("=" * 50)
    print("===== Phoneme Confusion Analysis =====")
    print("=" * 50)

    # Aggregate confusion counts per true phoneme
    phoneme_error_counts: list[tuple[int, int, list[tuple[int, int]]]] = []
    for true_id, pred_counter in phoneme_confusion.items():
        if true_id < 0:
            continue  # Skip insertion source marker
        total_errors = sum(pred_counter.values())
        # Get top confused predictions
        top_confusions = pred_counter.most_common(5)
        phoneme_error_counts.append((true_id, total_errors, top_confusions))

    # Sort by total errors descending
    phoneme_error_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nMost frequently mis-predicted phonemes (top 15):")
    print("-" * 60)
    for rank, (true_id, total_errors, top_confusions) in enumerate(phoneme_error_counts[:15], 1):
        true_phoneme = LOGIT_TO_PHONEME[true_id] if 0 <= true_id < len(LOGIT_TO_PHONEME) else f"ID={true_id}"
        confusion_strs = []
        for pred_id, count in top_confusions[:3]:
            if pred_id == -1:
                confusion_strs.append(f"DEL({count})")
            elif 0 <= pred_id < len(LOGIT_TO_PHONEME):
                confusion_strs.append(f"{LOGIT_TO_PHONEME[pred_id]}({count})")
            else:
                confusion_strs.append(f"ID={pred_id}({count})")
        print(f"  {rank:2d}. {true_phoneme:6s} -> {total_errors:4d} errors | Top confusions: {', '.join(confusion_strs)}")

    # Count insertions
    if -2 in phoneme_confusion:
        insertion_counter = phoneme_confusion[-2]
        total_insertions = sum(insertion_counter.values())
        top_inserted = insertion_counter.most_common(5)
        print(f"\n  Total insertions: {total_insertions}")
        print("  Most commonly inserted phonemes:", end=" ")
        insertion_strs = [f"{LOGIT_TO_PHONEME[pid]}({cnt})" if 0 <= pid < len(LOGIT_TO_PHONEME) else f"ID={pid}({cnt})" 
                         for pid, cnt in top_inserted]
        print(", ".join(insertion_strs))

    # === Random Sentence Examples ===
    print()
    print("=" * 50)
    print("===== Random Sentence Examples (3 per session) =====")
    print("=" * 50)

    for session in sorted(session_samples.keys()):
        samples = session_samples[session]
        if not samples:
            continue
        
        # Randomly select up to 3 samples
        num_samples = min(3, len(samples))
        selected_samples = random.sample(samples, num_samples)
        
        print(f"\n--- Session: {session} ({len(samples)} total trials) ---")
        for i, sample in enumerate(selected_samples, 1):
            true_phonemes = [LOGIT_TO_PHONEME[p] for p in sample['true_ids']]
            pred_phonemes = [LOGIT_TO_PHONEME[p] for p in sample['pred_ids']]
            per = sample['edit_distance'] / len(sample['true_ids']) if sample['true_ids'] else 0
            
            print(f"\n  Example {i} (Block {sample['block']}, Trial {sample['trial']}, PER: {100*per:.1f}%):")
            print(f"    Sentence: {sample['sentence_label']}")
            print(f"    Ground Truth: {' '.join(true_phonemes)}")
            print(f"    Predicted:    {' '.join(pred_phonemes)}")

    print()
    print("=" * 50)

    if output_csv is not None:
        output_df = pd.DataFrame(session_rows)
        output_df.sort_values('per', inplace=True)
        output_df.to_csv(output_csv, index=False)
        print(f"Session summary saved to {output_csv}")

    if per_trial_csv is not None and per_trial_rows:
        per_trial_df = pd.DataFrame(per_trial_rows)
        per_trial_df.sort_values(['session', 'block', 'trial'], inplace=True)
        per_trial_df.to_csv(per_trial_csv, index=False)
        print(f"Per-trial details saved to {per_trial_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained RNN or exp2 model on the full validation split.")
    parser.add_argument("--model_path", type=str, default="../data/t15_pretrained_rnn_baseline",
                        help="Path to pretrained model. Can be a directory (baseline) or a .pt file (exp2).")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config yaml for exp2 models. If not provided, will look for exp2_args.yaml.")
    parser.add_argument("--data_dir", type=str, default="../data/hdf5_data_final",
                        help="Path to data directory containing session subfolders.")
    parser.add_argument("--csv_path", type=str, default="../data/t15_copyTaskData_description.csv",
                        help="Path to dataset description CSV.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Select compute device. 'auto' prefers MPS on Apple Silicon, CUDA if available.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save per-session PER summary as CSV.")
    parser.add_argument("--per_trial_csv", type=str, default=None,
                        help="Optional path to save detailed per-trial predictions.")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = select_device("auto")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA device requested but CUDA is not available.")
    else:
        device = select_device(args.device)

    evaluate_validation_split(
        model_path=args.model_path,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        device=device,
        output_csv=args.output_csv,
        per_trial_csv=args.per_trial_csv,
        show_progress=not args.no_progress,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
