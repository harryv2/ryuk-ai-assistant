"""Test package.

Kept as a real package so `tests.conftest` helpers can be imported by name from
anywhere in the tree, and so two test files may share a module name without
pytest's rootdir guessing getting it wrong.
"""
