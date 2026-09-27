"""Portable job execution and publication of benchmark evidence.

These utilities do not import the model or require a GPU. Raw output stays in
an ignored run directory; publication creates a separate, auditable copy.
"""
