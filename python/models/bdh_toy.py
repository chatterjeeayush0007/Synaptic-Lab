"""
Synapse Lab - Canonical BDH Fast-Weight Reference Model
Implements rank-1 Hebbian synaptic updates with TopK-ReLU activation sparsity.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHToyModel(nn.Module):
    """
    Dragon Hatchling (BDH) Fast-Weight Associative Memory Layer.
    
    Maintains a fixed d x d recurrent synaptic matrix updated via outer products:
        W_syn <- lambda * W_syn + eta * (v_t (x) k_t)
    Readout is computed via linear projection:
        y_hat <- W_syn @ q_t
    """

    def __init__(
        self,
        d: int = 64,
        k: int = 3,
        lambda_decay: float = 1.0,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.lambda_decay = lambda_decay
        self.eta = eta
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Synaptic fast-weight matrix: fixed O(d^2) state
        self.register_buffer(
            "W_syn", torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
        )

    def reset_memory(self) -> None:
        """Zeros out the synaptic fast-weight matrix."""
        self.W_syn.zero_()

    def topk_relu(self, x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Non-negative Top-k ReLU activation.
        Retains only the k largest positive values, setting all other coordinates to 0.
        """
        active_k = k if k is not None else self.k
        active_k = max(1, min(active_k, self.d))

        # Clamp negative entries to zero (ReLU)
        x_relu = F.relu(x)

        if active_k >= self.d:
            return x_relu

        # Find the threshold value for the top-k coordinates
        topk_vals, _ = torch.topk(x_relu, active_k, dim=-1)
        threshold = topk_vals[..., -1:]

        # Zero out values below top-k threshold
        mask = x_relu >= threshold
        return x_relu * mask.to(x.dtype)

    def normalize(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2 normalization along the final dimension."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / torch.clamp(norm, min=eps)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k: Optional[int] = None,
        decay: Optional[float] = None,
    ) -> None:
        """
        Writes a key-value associative pair into the synaptic matrix via rank-1 Hebbian update.
        
        Args:
            key: Presynaptic concept vector of shape (d,)
            value: Postsynaptic target vector of shape (d,)
            k: Active sparsity count (defaults to self.k)
            decay: Retention decay factor lambda (defaults to self.lambda_decay)
        """
        lam = decay if decay is not None else self.lambda_decay

        # Apply TopK-ReLU and L2 normalization
        k_sparse = self.normalize(self.topk_relu(key, k=k))
        v_sparse = self.normalize(self.topk_relu(value, k=k))

        # Rank-1 outer product update: W <- lambda * W + eta * (v (x) k)
        # Dimensions: (d, 1) x (1, d) -> (d, d)
        outer_product = torch.outer(v_sparse, k_sparse)
        self.W_syn = lam * self.W_syn + self.eta * outer_product

    def read(self, query: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Associative retrieval using a query key: y_hat = W_syn @ query.
        
        Args:
            query: Query cue vector of shape (d,)
            k: Active sparsity count (defaults to self.k)
            
        Returns:
            Retrieved associative vector of shape (d,)
        """
        q_sparse = self.normalize(self.topk_relu(query, k=k))
        y_hat = torch.matmul(self.W_syn, q_sparse)
        return y_hat

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        k: Optional[int] = None,
        decay: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential execution over a sequence of associative writes followed by queries.
        
        Args:
            keys: Tensor of shape (seq_len, d)
            values: Tensor of shape (seq_len, d)
            queries: Tensor of shape (num_queries, d)
            k: Active sparsity count
            decay: Retention decay factor lambda
            
        Returns:
            retrieved_values: Tensor of shape (num_queries, d)
            cosine_scores: Cosine similarities between retrieved vectors and target values
        """
        self.reset_memory()
        seq_len = keys.size(0)

        # Sequential Hebbian writes
        for t in range(seq_len):
            self.write(keys[t], values[t], k=k, decay=decay)

        # Batch queries against final synaptic state
        num_queries = queries.size(0)
        retrieved_list = []
        for q_idx in range(num_queries):
            y_hat = self.read(queries[q_idx], k=k)
            retrieved_list.append(y_hat)

        retrieved = torch.stack(retrieved_list, dim=0)
        retrieved_norm = self.normalize(retrieved)
        target_norm = self.normalize(self.topk_relu(values[:num_queries], k=k))

        cosine_sim = torch.sum(retrieved_norm * target_norm, dim=-1)
        return retrieved, cosine_sim