"""Evaluate the pretrained RNN on every validation trial across all sessions."""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_augmentations import gauss_smooth
from evaluate_model_helpers import LOGIT_TO_PHONEME, load_h5py_file
from run_rnn_only import (
    calculate_aggregate_error_rate,
    calculate_error_rate,
    decode_logits_to_ids,
    load_model,
    select_device,
)


def _date_to_session(date_str: str) -> str:
    """Convert YYYY-MM-DD to dataset session name (e.g., t15.2023.08.13)."""
    cleaned = date_str.strip()
    if not cleaned:
        raise ValueError("Encountered empty date string in CSV description file.")
    return f"t15.{cleaned.replace('-', '.')}"


def evaluate_validation_split(
    model_dir: str,
    data_dir: str,
    csv_path: str,
    device: torch.device,
    output_csv: str | None = None,
    per_trial_csv: str | None = None,
    show_progress: bool = True,
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

    model, model_args = load_model(model_dir, device)

    dtype = torch.float32

    per_true_sequences: list[list[int]] = []
    per_pred_sequences: list[list[int]] = []

    session_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"true_len": 0.0, "edit_distance": 0.0, "count": 0.0})
    block_totals: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"true_len": 0.0, "edit_distance": 0.0, "count": 0.0}))

    per_trial_rows: list[dict[str, object]] = []

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
            input_layer = model_args['dataset']['sessions'].index(session)
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
                smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )

            with torch.no_grad():
                logits = model(
                    x=x,
                    day_idx=torch.tensor([input_layer], device=device),
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
    parser = argparse.ArgumentParser(description="Evaluate the pretrained RNN on the full validation split.")
    parser.add_argument("--model_path", type=str, default="../data/t15_pretrained_rnn_baseline",
                        help="Path to pretrained model directory.")
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
        model_dir=args.model_path,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        device=device,
        output_csv=args.output_csv,
        per_trial_csv=args.per_trial_csv,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
