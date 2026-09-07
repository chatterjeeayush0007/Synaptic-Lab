"""
Synapse Lab - Reference Models Package
Exports canonical PyTorch implementations of BDH Hebbian Fast Weights
and reference baselines (Softmax KV-Cache, Dense Linear Attention).
"""

from .bdh_toy import BDHToyModel
from .baselines import SoftmaxKVCacheBaseline, DenseLinearAttentionBaseline

__all__ = [
    "BDHToyModel",
    "SoftmaxKVCacheBaseline",
    "DenseLinearAttentionBaseline",
]