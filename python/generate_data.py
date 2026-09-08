"""
Synapse Lab - Synthetic Concept Vector & Probe Generator
Generates non-negative sparse concept vectors and controlled crosstalk probes.

Primary Citations:
  - BDH Hebbian fast weights: Pathway Research, arXiv:2509.26507 (2025)
  - Sparse coding: Olshausen & Field, Nature (1996)
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np

DIMENSION = 64
DEFAULT_K = 3  # ~4.68% active coordinates

FACT_PAIRS: List[Tuple[str, str]] = [
    ("Paris", "France"),
    ("Tokyo", "Japan"),
    ("Cairo", "Egypt"),
    ("Berlin", "Germany"),
    ("Rome", "Italy"),
    ("Madrid", "Spain"),
    ("Ottawa", "Canada"),
    ("Beijing", "China"),
    ("Seoul", "South Korea"),
    ("Brasilia", "Brazil"),
    ("Canberra", "Australia"),
    ("New Delhi", "India"),
    ("London", "UK"),
    ("Washington", "USA"),
    ("Bangkok", "Thailand"),
    ("Nairobi", "Kenya"),
    ("Buenos Aires", "Argentina"),
    ("Oslo", "Norway"),
    ("Athens", "Greece"),
    ("Rabat", "Morocco"),
]


def generate_sparse_unit_vector(
    d: int = DIMENSION,
    k: int = DEFAULT_K,
    rng: np.random.Generator = None,
    fixed_indices: List[int] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    vec = np.zeros(d, dtype=np.float32)
    indices = fixed_indices if fixed_indices is not None else rng.choice(d, size=k, replace=False)
    vec[indices] = rng.uniform(0.5, 1.0, size=len(indices))
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def build_dataset() -> Dict:
    rng = np.random.default_rng(seed=42)
    data = {
        "metadata": {
            "simulator_name": "Synapse Lab (BDH-Inspired Toy Simulator)",
            "dimension": DIMENSION,
            "default_k": DEFAULT_K,
            "active_percentage": f"{(DEFAULT_K / DIMENSION) * 100:.2f}%",
            "scope_disclosure": (
                "Toy simulation of rank-1 Hebbian outer-product fast weights with non-negative "
                "sparse activations. Demonstrates crosstalk reduction mechanisms, not full BDH multi-layer dynamics."
            ),
            "citations": {
                "bdh_paper": "Pathway Research (2025), arXiv:2509.26507",
                "sparse_coding": "Olshausen & Field, Nature 1996"
            }
        },
        "concepts": {},
        "facts": [],
        "crosstalk_probe": {},
    }

    for key, val in FACT_PAIRS:
        k_vec = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, rng=rng)
        v_vec = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, rng=rng)

        data["concepts"][key] = {
            "active_indices": np.where(k_vec > 0)[0].tolist(),
            "vector": [round(float(x), 5) for x in k_vec],
        }
        data["concepts"][val] = {
            "active_indices": np.where(v_vec > 0)[0].tolist(),
            "vector": [round(float(x), 5) for x in v_vec],
        }
        data["facts"].append({"key": key, "value": val})

    # Calibrated Crosstalk Probe (Shares 2 coordinates to induce controlled interference)
    shared_idx = [12, 27]
    probe_a_key = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, fixed_indices=shared_idx + [5])
    probe_b_key = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, fixed_indices=shared_idx + [44])
    overlap_sim = float(np.dot(probe_a_key, probe_b_key))

    data["crosstalk_probe"] = {
        "description": "Controlled probe pair sharing 2 active coordinates to examine associative interference",
        "cosine_similarity": round(overlap_sim, 4),
        "item_a": {
            "key": "Interference_A",
            "value": "Target_A",
            "active_indices": (shared_idx + [5]),
            "vector": [round(float(x), 5) for x in probe_a_key],
        },
        "item_b": {
            "key": "Interference_B",
            "value": "Target_B",
            "active_indices": (shared_idx + [44]),
            "vector": [round(float(x), 5) for x in probe_b_key],
        },
    }

    return data


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "associations.json")
    dataset = build_dataset()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"[OK] Generated calibrated dataset with {len(dataset['facts'])} pairs: {output_path}")


if __name__ == "__main__":
    main()