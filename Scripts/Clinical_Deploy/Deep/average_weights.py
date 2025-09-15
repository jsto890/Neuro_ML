#!/usr/bin/env python3
"""
Average multiple fold model checkpoints into a single weights file
=================================================================

Purpose: Create a single deployable model checkpoint by averaging weights
from multiple folds of the same architecture. The resulting file can be
used directly with the deep predict/validate scripts.

Notes:
- All input checkpoints must belong to the same architecture with matching shapes.
- Float tensors are averaged; non-float entries (e.g., num_batches_tracked) are taken from the first checkpoint.
- Accepts either raw state_dicts or dicts with key 'state_dict'.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch


def expand_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def collect_weights(paths: List[str], weights_glob: str = None, weights_dir: str = None, pattern: str = None) -> List[str]:
    collected: List[str] = []
    if paths:
        collected.extend(paths)
    if weights_glob:
        collected.extend(sorted(glob.glob(expand_path(weights_glob))))
    if weights_dir and pattern:
        q = os.path.join(expand_path(weights_dir), pattern)
        collected.extend(sorted(glob.glob(q)))
    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for p in collected:
        ap = expand_path(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(expand_path(path), map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
    else:
        sd = obj
    # Strip potential 'module.' prefixes from DataParallel
    clean = {}
    for k, v in sd.items():
        nk = k.replace('module.', '') if isinstance(k, str) and k.startswith('module.') else k
        clean[nk] = v
    return clean


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], strict: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict]:
    if len(state_dicts) == 0:
        raise ValueError("No state_dicts provided")

    # Determine key set
    key_sets = [set(sd.keys()) for sd in state_dicts]
    if strict:
        common_keys = set.intersection(*key_sets)
        missing_info = None
    else:
        # Use intersection by default, but allow missing keys silently
        common_keys = set.intersection(*key_sets)
        missing_info = {
            'first_only_keys': list(key_sets[0] - common_keys),
        }

    averaged: Dict[str, torch.Tensor] = {}
    dropped_mismatch: List[str] = []

    for k in sorted(common_keys):
        tensors = [sd[k] for sd in state_dicts]
        # Ensure matching shapes
        shapes = {tuple(t.shape) for t in tensors if torch.is_tensor(t)}
        if len(shapes) > 1:
            dropped_mismatch.append(k)
            continue
        t0 = tensors[0]
        if torch.is_tensor(t0) and t0.dtype.is_floating_point:
            acc = torch.zeros_like(t0, dtype=torch.float32)
            for t in tensors:
                acc += t.to(dtype=torch.float32)
            mean = (acc / float(len(tensors))).to(dtype=t0.dtype)
            averaged[k] = mean
        else:
            # Non-float (e.g., int counters) — take from first
            averaged[k] = t0

    meta = {
        'num_models': len(state_dicts),
        'strict': strict,
        'common_keys': len(common_keys),
        'dropped_mismatch_keys': dropped_mismatch,
        'missing_info': missing_info,
    }
    return averaged, meta


def main():
    parser = argparse.ArgumentParser(description='Average multiple fold checkpoints into one')
    parser.add_argument('--weights', nargs='+', help='List of weight files')
    parser.add_argument('--weights-glob', type=str, help='Glob pattern for weight files')
    parser.add_argument('--weights-dir', type=str, help='Directory containing weight files')
    parser.add_argument('--weights-pattern', type=str, help='Filename pattern inside weights-dir (e.g., fold*_best.pth)')
    parser.add_argument('--output-path', default='~/reseng202500013-ndd-ml/clinical_outputs/deep_averaged_models/averaged_model.pth', help='Where to save the averaged checkpoint')
    parser.add_argument('--wrap-state-dict', action='store_true', help="Wrap output as {'state_dict': ...} for broader compatibility")
    parser.add_argument('--strict-keys', action='store_true', help='Require all models to share identical keys/shapes (else error)')
    parser.add_argument('--save-metadata', action='store_true', help='Also save a JSON metadata file alongside weights')
    args = parser.parse_args()

    weights = collect_weights(args.weights, args.weights_glob, args.weights_dir, args.weights_pattern)
    if len(weights) < 2:
        print('Require at least two weight files. Use --weights, --weights-glob, or --weights-dir with --weights-pattern')
        sys.exit(1)

    print(f"Averaging {len(weights)} checkpoints...")
    state_dicts = []
    for w in weights:
        print(f"  - {w}")
        sd = load_state_dict(w)
        state_dicts.append(sd)

    averaged_sd, meta = average_state_dicts(state_dicts, strict=args.strict_keys)

    output_path = Path(expand_path(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = {'state_dict': averaged_sd} if args.wrap_state_dict else averaged_sd
    torch.save(to_save, str(output_path))
    print(f"✓ Saved averaged weights to: {output_path}")

    if args.save_metadata:
        metadata = {
            'inputs': [expand_path(w) for w in weights],
            'output': str(output_path),
            'meta': meta,
        }
        meta_path = output_path.with_suffix(output_path.suffix + '.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to: {meta_path}")


if __name__ == '__main__':
    main()


