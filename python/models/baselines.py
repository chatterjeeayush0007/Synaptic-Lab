"""
Synapse Lab - Reference Comparison Baselines
Implements the Softmax KV-Cache (O(T*d)) and Dense Linear Attention (100% density).

Primary Citations:
  - Softmax KV-Cache Attention: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
  - Linear Attention & Fast Weights: Katharopoulos et al., ICML 2020; Schlag et al., ICML 2021
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxKVCacheBaseline(nn.Module):
    """
    Exact Softmax KV-Cache Attention Baseline [Vaswani et al., 2017].
    
    Stores every historical key and value vector explicitly:
        Cache memory footprint = 2 * T * d * bytes_per_elem (O(T * d))
    Retrieval evaluates scaled dot-product attention over cached tokens:
        y_hat = softmax(Q K^T / tau) V
    """

    def __init__(
        self,
        d: int = 64,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d = d
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.keys_cache: List[torch.Tensor] = []
        self.values_cache: List[torch.Tensor] = []

    def reset_memory(self) -> None:
        """Clears the historical key-value cache buffer."""
        self.keys_cache.clear()
        self.values_cache.clear()

    @property
    def memory_bytes(self) -> int:
        """Returns the current cache memory footprint in bytes."""
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        t = len(self.keys_cache)
        # 2 * T * d * elem_size
        return 2 * t * self.d * elem_size

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Appends a new key-value vector pair to the historical cache."""
        k = key.detach().to(device=self.device, dtype=self.dtype)
        v = value.detach().to(device=self.device, dtype=self.dtype)
        self.keys_cache.append(k)
        self.values_cache.append(v)

    def read(self, query: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Exact associative retrieval via scaled dot-product attention over cached tokens.
        Uses a sharp temperature default (tau = 0.1) for normalized unit vectors
        to preserve exact softmax selection fidelity across arbitrary context sizes.
        """
        if not self.keys_cache:
            return torch.zeros(self.d, device=self.device, dtype=self.dtype)

        q = query.to(device=self.device, dtype=self.dtype)
        k_stack = torch.stack(self.keys_cache, dim=0)  # (T, d)
        v_stack = torch.stack(self.values_cache, dim=0)  # (T, d)

        scale = temperature or 0.1
        scores = torch.matmul(k_stack, q) / scale  # (T,)
        attn_weights = F.softmax(scores, dim=0)  # (T,)

        y_hat = torch.matmul(attn_weights, v_stack)  # (d,)
        return y_hat

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populates the cache with sequence pairs and executes batch associative recall.
        """
        self.reset_memory()
        seq_len = keys.size(0)

        for t in range(seq_len):
            self.write(keys[t], values[t])

        num_queries = queries.size(0)
        retrieved_list = []
        for q_idx in range(num_queries):
            y_hat = self.read(queries[q_idx])
            retrieved_list.append(y_hat)

        retrieved = torch.stack(retrieved_list, dim=0)
        retrieved_norm = F.normalize(retrieved, p=2, dim=-1)
        target_norm = F.normalize(values[:num_queries], p=2, dim=-1)

        cosine_sim = torch.sum(retrieved_norm * target_norm, dim=-1)
        return retrieved, cosine_sim


class DenseLinearAttentionBaseline(nn.Module):
    """
    Standard Dense Linear Attention / Fast-Weight Recurrent Baseline [Katharopoulos et al., 2020; Schlag et al., 2021].
    
    Maintains a fixed d x d state matrix with dense, unthresholded outer-product updates:
        S_t = lambda * S_{t-1} + v_t k_t^T
    Demonstrates associative crosstalk collapse under 100% activation density.
    """

    def __init__(
        self,
        d: int = 64,
        lambda_decay: float = 1.0,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d = d
        self.lambda_decay = lambda_decay
        self.eta = eta
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.register_buffer(
            "S", torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
        )

    def reset_memory(self) -> None:
        """Zeros out the recurrent matrix state."""
        self.S.zero_()

    @property
    def memory_bytes(self) -> int:
        """Fixed d x d state memory footprint in bytes."""
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        return self.d * self.d * elem_size

    def normalize(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2 normalization along the final dimension."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / torch.clamp(norm, min=eps)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: Optional[float] = None,
    ) -> None:
        """
        Dense outer product update without sparsity thresholding (100% activation density).
        """
        lam = decay if decay is not None else self.lambda_decay
        k_norm = self.normalize(key.to(device=self.device, dtype=self.dtype))
        v_norm = self.normalize(value.to(device=self.device, dtype=self.dtype))

        outer_product = torch.outer(v_norm, k_norm)
        self.S = lam * self.S + self.eta * outer_product

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Linear readout: y_hat = S @ query."""
        q_norm = self.normalize(query.to(device=self.device, dtype=self.dtype))
        return torch.matmul(self.S, q_norm)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        decay: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes sequence writes and batch queries against the dense fast-weight state.
        """
        self.reset_memory()
        seq_len = keys.size(0)

        for t in range(seq_len):
            self.write(keys[t], values[t], decay=decay)

        num_queries = queries.size(0)
        retrieved_list = []
        for q_idx in range(num_queries):
            y_hat = self.read(queries[q_idx])
            retrieved_list.append(y_hat)

        retrieved = torch.stack(retrieved_list, dim=0)
        retrieved_norm = self.normalize(retrieved)
        target_norm = self.normalize(values[:num_queries])

        cosine_sim = torch.sum(retrieved_norm * target_norm, dim=-1)
        return retrieved, cosine_sim