"""Token request validation shared by generation and workload export."""

from __future__ import annotations


def token_ids(values, *, vocab_size=None):
    """Normalize one unbatched sequence; never decode and retokenize IDs."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("input_ids must be a nonempty, one-dimensional sequence of integers")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("input_ids must contain nonnegative integers (no booleans or batches)")
    if vocab_size is not None and any(value >= vocab_size for value in values):
        raise ValueError(f"input_ids must be smaller than the model vocabulary size ({vocab_size})")
    return list(values)


def generation_length(value, *, input_length=None):
    """The packed exporter has a finite session; exact CPU generation does not."""
    if type(value) is not int or value < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if input_length is not None and (value > 64 or input_length + value - 1 > 128):
        raise ValueError("prepared requests support 1..64 new tokens and at most 128 evaluations")
    return value
