import os
import argparse
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf

from rnn_model import GRUDecoder
from evaluate_model_helpers import LOGIT_TO_PHONEME, load_h5py_file
from data_augmentations import gauss_smooth
try:
    from nejm_b2txt_utils.general_utils import calculate_error_rate, calculate_aggregate_error_rate
except ModuleNotFoundError:
    # Try adding project root to sys.path and retry
    import sys
    import os as _os
    _project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
    if _project_root not in sys.path:
        sys.path.append(_project_root)
    try:
        from nejm_b2txt_utils.general_utils import calculate_error_rate, calculate_aggregate_error_rate
    except Exception:
        import numpy as _np

        def calculate_error_rate(r, h):
            d = _np.zeros((len(r)+1)*(len(h)+1), dtype=_np.uint8)
            d = d.reshape((len(r)+1, len(h)+1))
            for i in range(len(r)+1):
                for j in range(len(h)+1):
                    if i == 0:
                        d[0][j] = j
                    elif j == 0:
                        d[i][0] = i
            for i in range(1, len(r)+1):
                for j in range(1, len(h)+1):
                    if r[i-1] == h[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        substitution = d[i-1][j-1] + 1
                        insertion = d[i][j-1] + 1
                        deletion = d[i-1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)
            return int(d[len(r)][len(h)])

        def calculate_aggregate_error_rate(r, h):
            err_count = []
            item_count = []
            error_rate_ind = []
            for x in range(len(h)):
                r_x = r[x]
                h_x = h[x]
                n_err = calculate_error_rate(r_x, h_x)
                item_count.append(len(r_x))
                err_count.append(n_err)
                error_rate_ind.append(n_err / len(r_x) if len(r_x) > 0 else 0.0)
            error_rate_agg = _np.sum(err_count) / _np.sum(item_count)
            item_count = _np.array(item_count)
            err_count = _np.array(err_count)
            nResamples = 10000
            resampled_error_rate = _np.zeros([nResamples,])
            for n in range(nResamples):
                if item_count.shape[0] == 0:
                    resampled_error_rate[n] = _np.nan
                    continue
                resampleIdx = _np.random.randint(0, item_count.shape[0], [item_count.shape[0]])
                resampled_error_rate[n] = _np.sum(err_count[resampleIdx]) / _np.sum(item_count[resampleIdx])
            error_rate_agg_CI = _np.percentile(resampled_error_rate[~_np.isnan(resampled_error_rate)], [2.5, 97.5]) if _np.any(~_np.isnan(resampled_error_rate)) else _np.array([_np.nan, _np.nan])
            return (error_rate_agg, error_rate_agg_CI[0], error_rate_agg_CI[1], error_rate_ind)


def select_device(preferred: str = "auto") -> torch.device:
    if preferred.lower() == "cpu":
        return torch.device("cpu")
    if preferred.lower() == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: str, device: torch.device) -> tuple[GRUDecoder, dict]:
    args_path = os.path.join(model_dir, "checkpoint", "args.yaml")
    ckpt_path = os.path.join(model_dir, "checkpoint", "best_checkpoint")

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Missing model args at: {args_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at: {ckpt_path}")

    model_args = OmegaConf.load(args_path)

    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'],
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )

    # PyTorch 2.6 defaults weights_only=True; explicitly disable to load legacy checkpoints
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get('model_state_dict', checkpoint)
    cleaned_state = {}
    for key, value in raw_state.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        cleaned_state[new_key] = value
    model.load_state_dict(cleaned_state)

    model.to(device)
    model.eval()
    return model, model_args


def decode_logits_to_phonemes(logits_np: np.ndarray) -> list[str]:
    pred_seq = np.argmax(logits_np, axis=-1)
    pred_seq = [int(p) for p in pred_seq if p != 0]  # remove blanks (0)
    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i - 1]]
    return [LOGIT_TO_PHONEME[p] for p in pred_seq]


def decode_logits_to_ids(logits_np: np.ndarray) -> list[int]:
    """CTC-style greedy decode to class ids: argmax → remove blanks → collapse repeats."""
    pred_seq = np.argmax(logits_np, axis=-1)
    pred_seq = [int(p) for p in pred_seq if p != 0]
    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i - 1]]
    return pred_seq


def run_inference(
    model_dir: str,
    data_dir: str,
    csv_path: str,
    eval_type: str,
    device: torch.device,
    session_filter: str | None,
    max_trials: int,
):
    model, model_args = load_model(model_dir, device)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    b2txt_csv_df = pd.read_csv(csv_path)

    sessions: list[str]
    if session_filter is not None:
        sessions = [session_filter]
    else:
        sessions = list(model_args['dataset']['sessions'])

    dtype = torch.float32  # CPU/MPS-friendly dtype

    printed_header = False
    total_printed = 0

    for session in sessions:
        eval_file = os.path.join(data_dir, session, f"data_{eval_type}.hdf5")
        if not os.path.exists(eval_file):
            continue

        data = load_h5py_file(eval_file, b2txt_csv_df)
        print(f"Loaded {len(data['neural_features'])} {eval_type} trials for session {session}.")

        input_layer = sessions.index(session) if session_filter is None else model_args['dataset']['sessions'].index(session)

        for trial_idx in range(min(max_trials, len(data['neural_features']))):
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
            phonemes = decode_logits_to_phonemes(logits_np)

            if not printed_header:
                device_name = device.type
                print(f"Device: {device_name} | Model dir: {model_dir}")
                printed_header = True

            block_num = data['block_num'][trial_idx]
            trial_num = data['trial_num'][trial_idx]
            print(f"Session: {session}, Block: {block_num}, Trial: {trial_num}")
            if eval_type == 'val' and data['seq_class_ids'][trial_idx] is not None and data['seq_len'][trial_idx] is not None:
                true_ids = data['seq_class_ids'][trial_idx][: data['seq_len'][trial_idx]]
                true_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in true_ids]
                sent = data['sentence_label'][trial_idx]
                if isinstance(sent, (bytes, bytearray, np.ndarray)):
                    try:
                        sent = bytes(sent).decode().strip()
                    except Exception:
                        pass
                print(f"Sentence label:      {sent}")
                print(f"True phonemes:       {' '.join(true_phonemes)}")
            print(f"Predicted phonemes:  {' '.join(phonemes)}")
            print()
            total_printed += 1

        # Only show one session by default if no filter provided
        if session_filter is None:
            break

    if total_printed == 0:
        raise RuntimeError(
            "No trials were printed. Check that your data directory contains session folders with "
            f"'data_{eval_type}.hdf5' files and that paths are correct."
        )


def run_evaluation_per(
    model_dir: str,
    data_dir: str,
    csv_path: str,
    eval_type: str,
    device: torch.device,
    session_filter: str | None,
    num_trials: int,
    randomize: bool,
    seed: int | None,
    all_sessions: bool,
):
    if eval_type != 'val':
        raise ValueError("PER evaluation requires eval_type='val' to access ground-truth phonemes.")

    model, model_args = load_model(model_dir, device)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    b2txt_csv_df = pd.read_csv(csv_path)

    if session_filter is not None:
        sessions = [session_filter]
    else:
        sessions = list(model_args['dataset']['sessions'])

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dtype = torch.float32

    total_true_len = 0
    total_edit_distance = 0

    # For optional aggregate CI
    per_true_list: list[list[int]] = []
    per_pred_list: list[list[int]] = []

    # Per-session accumulators
    session_totals: dict[str, dict[str, float]] = {}

    printed_header = False
    evaluated = 0

    for session in sessions:
        eval_file = os.path.join(data_dir, session, f"data_{eval_type}.hdf5")
        if not os.path.exists(eval_file):
            continue

        data = load_h5py_file(eval_file, b2txt_csv_df)
        if not printed_header:
            device_name = device.type
            print(f"Device: {device_name} | Model dir: {model_dir}")
            printed_header = True

        input_layer = sessions.index(session) if session_filter is None else model_args['dataset']['sessions'].index(session)

        n_trials = len(data['neural_features'])
        indices = list(range(n_trials))
        if randomize:
            rng.shuffle(indices)

        # Initialize per-session totals
        session_totals[session] = {
            'true_len': 0.0,
            'edit_distance': 0.0,
            'count': 0.0,
        }

        for trial_idx in indices:
            if evaluated >= num_trials:
                break

            # Ensure GT exists
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

            # Edit distance on id sequences
            ed = int(calculate_error_rate(true_ids, pred_ids))

            total_true_len += len(true_ids)
            total_edit_distance += ed
            per_true_list.append(true_ids)
            per_pred_list.append(pred_ids)

            session_totals[session]['true_len'] += len(true_ids)
            session_totals[session]['edit_distance'] += ed
            session_totals[session]['count'] += 1

            evaluated += 1

        if evaluated >= num_trials:
            break

    if evaluated == 0 or total_true_len == 0:
        raise RuntimeError("No validation trials with ground-truth phonemes were evaluated.")

    aggregate_per = total_edit_distance / total_true_len

    # Optional CI via bootstrap
    try:
        per, per_lo, per_hi, _ = calculate_aggregate_error_rate(per_true_list, per_pred_list)
    except Exception:
        per, per_lo, per_hi = aggregate_per, float('nan'), float('nan')

    print()
    print(f"Evaluated trials: {evaluated}")
    print(f"Aggregate PER: {100 * aggregate_per:.2f}%")
    if not np.isnan(per_lo):
        print(f"Bootstrap 95% CI: [{100 * per_lo:.2f}%, {100 * per_hi:.2f}%]")

    # Per-session stats
    print()
    print("Per-session PER:")
    for sess, totals in session_totals.items():
        if totals['true_len'] > 0:
            sess_per = totals['edit_distance'] / totals['true_len']
            print(f"  {sess}: {100 * sess_per:.2f}% over {int(totals['count'])} trials")


def main():
    parser = argparse.ArgumentParser(description="Run macOS-friendly RNN-only inference (phonemes).")
    parser.add_argument("--model_path", type=str, default="../data/t15_pretrained_rnn_baseline",
                        help="Path to pretrained model directory.")
    parser.add_argument("--data_dir", type=str, default="../data/hdf5_data_final",
                        help="Path to data directory containing session subfolders.")
    parser.add_argument("--csv_path", type=str, default="../data/t15_copyTaskData_description.csv",
                        help="Path to dataset description CSV.")
    parser.add_argument("--eval_type", type=str, choices=["val", "test"], default="val",
                        help="Use validation or test split.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"],
                        help="Select device. 'auto' prefers MPS on Apple Silicon.")
    parser.add_argument("--session", type=str, default=None,
                        help="Optional specific session (e.g., t15.2023.08.13). If omitted, uses the first available session.")
    parser.add_argument("--max_trials", type=int, default=3,
                        help="Max number of trials to print per session.")
    # Evaluation-specific flags
    parser.add_argument("--evaluate", action="store_true",
                        help="Run PER evaluation over many validation trials instead of printing samples.")
    parser.add_argument("--num_trials", type=int, default=200,
                        help="Total number of trials to evaluate (across sessions unless --all_sessions).")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize trial order before sampling.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for --randomize.")
    parser.add_argument("--all_sessions", action="store_true",
                        help="Evaluate across all sessions (ignores --max_trials break behavior).")

    args = parser.parse_args()

    device = select_device(args.device)
    if args.evaluate:
        run_evaluation_per(
            model_dir=args.model_path,
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            eval_type=args.eval_type,
            device=device,
            session_filter=args.session,
            num_trials=args.num_trials,
            randomize=args.randomize,
            seed=args.seed,
            all_sessions=args.all_sessions,
        )
    else:
        run_inference(
            model_dir=args.model_path,
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            eval_type=args.eval_type,
            device=device,
            session_filter=args.session,
            max_trials=args.max_trials,
        )


if __name__ == "__main__":
    main()


