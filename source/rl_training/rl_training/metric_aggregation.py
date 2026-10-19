"""Pool conditional episode diagnostics before RSL-RL's scalar averaging.

This module has no simulator dependency. Missing observations create gaps in
TensorBoard, never artificial zero measurements or non-finite values.
"""
from __future__ import annotations

import math

import torch


def pool_episode_metrics(extras: list[dict]) -> list[dict]:
    """Resolve mean/RMS/min metadata, preserving unweighted ordinary metrics."""
    if not extras:
        return []
    kinds = {
        "weighted": ("__weighted_total__/", "__weighted_count__/"),
        "rms": ("__rms_total__/", "__rms_count__/"),
        "minimum": ("__minimum_value__/", "__minimum_count__/"),
    }
    fields = {}
    metadata = set()
    for row in extras:
        for key in row:
            for kind, (value_marker, count_marker) in kinds.items():
                for marker in (value_marker, count_marker):
                    token = "/" + marker
                    if key.startswith(marker):
                        prefix, suffix = "", key[len(marker):]
                    elif token in key:
                        prefix, suffix = key.split(token, 1)
                        prefix += "/"
                    else:
                        continue
                    target = prefix + suffix
                    fields[(kind, target)] = (prefix + value_marker + suffix,
                                              prefix + count_marker + suffix)
                    metadata.add(key)
    output = [{key: value for key, value in row.items() if key not in metadata} for row in extras]
    entries, pairs = [], []
    for (kind, target), (value_key, count_key) in fields.items():
        selected = [(row[value_key], row[count_key]) for row in extras
                    if value_key in row and count_key in row]
        entries.append((kind, target, len(pairs), len(selected)))
        pairs.extend(selected)
    if not pairs:
        return output
    reference = next((x for pair in pairs for x in pair if isinstance(x, torch.Tensor)), None)
    device = reference.device if reference is not None else "cpu"
    # One host transfer for all diagnostics, rather than synchronizing CUDA
    # twice per field. Metadata is detached; it never affects PPO gradients.
    packed = torch.stack([
        torch.stack([torch.as_tensor(pair[i], device=device).detach().reshape(()) for pair in pairs])
        for i in (0, 1)
    ]).cpu().tolist()
    targets, pooled = [], []
    for kind, target, start, size in entries:
        valid = [(v, c) for v, c in zip(packed[0][start:start + size], packed[1][start:start + size])
                 if c > 0 and math.isfinite(c) and math.isfinite(v)]
        if not valid:
            for row in output:
                row.pop(target, None)
            continue
        value = min(v for v, _ in valid) if kind == "minimum" else sum(v for v, _ in valid) / sum(c for _, c in valid)
        if kind == "rms":
            value = math.sqrt(max(0.0, value))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite pooled diagnostic: {target}")
        targets.append(target)
        pooled.append(value)
    values = torch.tensor(pooled, device=device, dtype=torch.float32)
    # RSL's logger only iterates the first dict's keys. Replicate pooled values
    # so a measurement appearing in a later reset batch is still recorded.
    for target, value in zip(targets, values):
        for row in output:
            row[target] = value
    return output
