"""
Synapse Lab - MQAR Capacity Benchmark Runner
Sweeps sequence lengths and association counts across Softmax KV Cache,
BDH Sparse Hebbian Fast Weights, and Dense Linear Attention.
Outputs results to python/data/benchmarks.json.
"""

import json
import os
import math
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

from models.bdh_toy import BDHToyModel
from models.baselines import SoftmaxKVCacheBaseline, DenseLinearAttentionBaseline


DIMENSION = 64
DEFAULT_K = 3  # ~4.68% active units
DEVICE = torch.device("cpu")
SEEDS = [42, 1337, 2026]

# Sequence lengths T for KV-cache memory scaling sweep
CONTEXT_LENGTHS: List[int] = [10, 25, 50, 100, 200, 350, 500, 1000, 2500, 5000, 10000]

# Number of stored associative pairs P for capacity horizon sweep
FACT_COUNTS: List[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]


def generate_synthetic_batch(
    num_facts: int,
    d: int = DIMENSION,
    k: int = DEFAULT_K,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generates synthetic non-negative sparse key-value pairs and probe queries."""
    rng = np.random.default_rng(seed)

    keys = []
    values = []

    for _ in range(num_facts):
        k_idx = rng.choice(d, size=k, replace=False)
        k_vec = np.zeros(d, dtype=np.float32)
        k_vec[k_idx] = rng.uniform(0.6, 1.0, size=k)
        k_vec /= np.linalg.norm(k_vec)
        keys.append(k_vec)

        v_idx = rng.choice(d, size=k, replace=False)
        v_vec = np.zeros(d, dtype=np.float32)
        v_vec[v_idx] = rng.uniform(0.6, 1.0, size=k)
        v_vec /= np.linalg.norm(v_vec)
        values.append(v_vec)

    k_tensor = torch.tensor(np.stack(keys), dtype=torch.float32, device=DEVICE)
    v_tensor = torch.tensor(np.stack(values), dtype=torch.float32, device=DEVICE)

    return {"keys": k_tensor, "values": v_tensor}


def evaluate_retrieval(
    retrieved: torch.Tensor,
    targets: torch.Tensor,
    all_values: torch.Tensor,
) -> Dict[str, float]:
    """Computes mean cosine similarity and top-1 retrieval accuracy against candidate pool."""
    r_norm = torch.nn.functional.normalize(retrieved, p=2, dim=-1)
    t_norm = torch.nn.functional.normalize(targets, p=2, dim=-1)
    pool_norm = torch.nn.functional.normalize(all_values, p=2, dim=-1)

    # Cosine alignment with ground-truth target
    cosine_sims = torch.sum(r_norm * t_norm, dim=-1).cpu().numpy()
    mean_cos = float(np.mean(cosine_sims))

    # Top-1 accuracy against entire value candidate pool
    # (num_queries, d) @ (num_candidates, d)^T -> (num_queries, num_candidates)
    score_matrix = torch.matmul(r_norm, pool_norm.T)
    predictions = torch.argmax(score_matrix, dim=-1)
    ground_truth = torch.arange(retrieved.size(0), device=retrieved.device)

    accuracy = float((predictions == ground_truth).float().mean().item())

    return {
        "cosine_similarity": round(mean_cos, 4),
        "accuracy": round(accuracy, 4),
    }


def run_capacity_sweeps() -> Dict:
    """Executes associative capacity sweeps across models and parameter configurations."""
    bdh_model = BDHToyModel(d=DIMENSION, k=DEFAULT_K, lambda_decay=1.0, device=DEVICE)
    dense_model = DenseLinearAttentionBaseline(d=DIMENSION, lambda_decay=1.0, device=DEVICE)
    kv_model = SoftmaxKVCacheBaseline(d=DIMENSION, device=DEVICE)

    results: Dict = {
        "metadata": {
            "dimension": DIMENSION,
            "sparsity_k": DEFAULT_K,
            "device": str(DEVICE),
            "hopfield_capacity_bound": round(0.14 * DIMENSION, 2),
            "description": "Multi-Query Associative Recall (MQAR) benchmark curves",
        },
        "capacity_vs_facts": [],
        "memory_vs_context": [],
    }

    print("Running Capacity vs. Stored Facts sweep (Hopfield Bound)...")
    for p in tqdm(FACT_COUNTS, desc="Stored Facts Sweep"):
        bdh_cos_runs, bdh_acc_runs = [], []
        dense_cos_runs, dense_acc_runs = [], []
        kv_cos_runs, kv_acc_runs = [], []

        for seed in SEEDS:
            data = generate_synthetic_batch(num_facts=p, seed=seed)
            keys = data["keys"]
            values = data["values"]
            queries = keys.clone()  # Recall all stored associations

            # 1. BDH Sparse Hebbian Model (k=3)
            bdh_ret, _ = bdh_model(keys, values, queries, k=DEFAULT_K)
            bdh_metrics = evaluate_retrieval(bdh_ret, values, values)
            bdh_cos_runs.append(bdh_metrics["cosine_similarity"])
            bdh_acc_runs.append(bdh_metrics["accuracy"])

            # 2. Dense Linear Attention Baseline (100% density)
            dense_ret, _ = dense_model(keys, values, queries)
            dense_metrics = evaluate_retrieval(dense_ret, values, values)
            dense_cos_runs.append(dense_metrics["cosine_similarity"])
            dense_acc_runs.append(dense_metrics["accuracy"])

            # 3. Softmax KV-Cache (exact baseline)
            kv_ret, _ = kv_model(keys, values, queries)
            kv_metrics = evaluate_retrieval(kv_ret, values, values)
            kv_cos_runs.append(kv_metrics["cosine_similarity"])
            kv_acc_runs.append(kv_metrics["accuracy"])

        results["capacity_vs_facts"].append({
            "num_facts": p,
            "bdh_sparse": {
                "cosine_mean": round(float(np.mean(bdh_cos_runs)), 4),
                "accuracy_mean": round(float(np.mean(bdh_acc_runs)), 4),
            },
            "dense_linear": {
                "cosine_mean": round(float(np.mean(dense_cos_runs)), 4),
                "accuracy_mean": round(float(np.mean(dense_acc_runs)), 4),
            },
            "softmax_kv": {
                "cosine_mean": round(float(np.mean(kv_cos_runs)), 4),
                "accuracy_mean": round(float(np.mean(kv_acc_runs)), 4),
            },
        })

    print("Computing Memory Footprint scaling curves...")
    # Fixed recurrent state: 64 * 64 * 4 bytes (FP32) = 16,384 bytes
    bdh_bytes_fixed = DIMENSION * DIMENSION * 4
    for t in CONTEXT_LENGTHS:
        # KV Cache: 2 * L=1 * T * d * 4 bytes
        kv_bytes = 2 * 1 * t * DIMENSION * 4
        results["memory_vs_context"].append({
            "context_tokens": t,
            "bdh_bytes": bdh_bytes_fixed,
            "bdh_kb": round(bdh_bytes_fixed / 1024, 2),
            "kv_cache_bytes": kv_bytes,
            "kv_cache_kb": round(kv_bytes / 1024, 2),
            "ratio_kv_to_bdh": round(kv_bytes / bdh_bytes_fixed, 2),
        })

    return results


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmarks.json")

    results = run_capacity_sweeps()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Capacity sweep successfully completed.")
    print(f"[OK] Precomputed benchmark data saved to: {output_path}")


if __name__ == "__main__":
    main()