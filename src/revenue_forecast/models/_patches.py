"""Compatibility patch for pytorch-forecasting.

``pytorch_forecasting.utils.concat_sequences`` does not always handle a mix of
``PackedSequence`` / tensor / tuple inputs across the versions pinned in
``requirements.txt``. This patched version restores correct concatenation and is
applied by importing :func:`apply_patches` before building any dataset/model.
"""
from __future__ import annotations

import torch
from torch.nn.utils import rnn


def concat_sequences_patched(sequences):
    """Concatenate a list of PackedSequences / tensors / (nested) tuples."""
    first = sequences[0]
    if isinstance(first, rnn.PackedSequence):
        return rnn.pack_sequence(sequences, enforce_sorted=False)
    if isinstance(first, torch.Tensor):
        return torch.cat(sequences, dim=0)
    if isinstance(first, (tuple, list)):
        return tuple(
            concat_sequences_patched([seq[i] for seq in sequences])
            for i in range(len(first))
        )
    raise ValueError("Unsupported sequence type")


def apply_patches() -> None:
    """Monkey-patch ``concat_sequences`` across pytorch-forecasting modules."""
    import pytorch_forecasting.models.base_model as bm
    import pytorch_forecasting.utils as pf_utils
    import pytorch_forecasting.utils._utils as _u

    _u.concat_sequences = concat_sequences_patched
    pf_utils.concat_sequences = concat_sequences_patched
    bm.concat_sequences = concat_sequences_patched
