"""Shared utilities for the evm_cuda project (used by both evm/ and evm_cuda/).

This package exists so the CPU baseline (``evm``) and the CUDA port
(``evm_cuda``) can share pure-Python helpers without importing each other.
"""
