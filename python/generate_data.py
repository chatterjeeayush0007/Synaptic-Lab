"""
Synapse Lab - Synthetic Associative Data Generator
Constructs calibrated sparse concept vectors and outputs python/data/associations.json.
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np


DIMENSION = 64
DEFAULT_K = 3  # ~4.68% active units

# Canonical factual tuples (Key -> Value)
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
    """
    Generates a non-negative sparse vector in R^d with exactly k positive units, L2-normalized.
    """
    if rng is None:
        rng = np.random.default_rng()

    vec = np.zeros(d, dtype=np.float32)
    if fixed_indices is not None:
        indices = fixed_indices
    else:
        indices = rng.choice(d, size=k, replace=False)

    # Positive activations sampled from uniform distribution [0.5, 1.0]
    values = rng.uniform(0.5, 1.0, size=len(indices))
    vec[indices] = values

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def build_associations_dataset() -> Dict:
    """
    Builds the dataset containing concept vectors, fact associations, and calibrated crosstalk pairs.
    """
    rng = np.random.default_rng(seed=42)
    data: Dict = {
        "metadata": {
            "dimension": DIMENSION,
            "default_k": DEFAULT_K,
            "sparsity_ratio": f"{(DEFAULT_K / DIMENSION) * 100:.2f}%",
            "description": "Synthetic concept vectors for BDH linear Hebbian fast weights",
        },
        "concepts": {},
        "facts": [],
        "crosstalk_probe": {},
    }

    # Generate vectors for all keys and values
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

    # Calibrated Crosstalk Pair: Create two keys that share 2 overlapping coordinates
    shared_indices = [12, 27]
    unique_a = [5]
    unique_b = [44]

    crosstalk_key_a = generate_sparse_unit_vector(
        d=DIMENSION, k=DEFAULT_K, fixed_indices=shared_indices + unique_a
    )
    crosstalk_key_b = generate_sparse_unit_vector(
        d=DIMENSION, k=DEFAULT_K, fixed_indices=shared_indices + unique_b
    )
    target_val_a = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, rng=rng)
    target_val_b = generate_sparse_unit_vector(d=DIMENSION, k=DEFAULT_K, rng=rng)

    overlap_similarity = float(np.dot(crosstalk_key_a, crosstalk_key_b))

    data["crosstalk_probe"] = {
        "description": "Probe pair sharing 2 active coordinates to induce controlled synaptic crosstalk",
        "cosine_similarity": round(overlap_similarity, 4),
        "item_a": {
            "key": "Interference_A",
            "value": "Target_A",
            "active_indices": (shared_indices + unique_a),
            "vector": [round(float(x), 5) for x in crosstalk_key_a],
        },
        "item_b": {
            "key": "Interference_B",
            "value": "Target_B",
            "active_indices": (shared_indices + unique_b),
            "vector": [round(float(x), 5) for x in crosstalk_key_b],
        },
    }

    return data


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "associations.json")

    dataset = build_associations_dataset()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[OK] Generated associations dataset with {len(dataset['facts'])} pairs.")
    print(f"[OK] Output saved to: {output_path}")
    print(
        f"[OK] Crosstalk probe cosine similarity: {dataset['crosstalk_probe']['cosine_similarity']:.4f}"
    )


if __name__ == "__main__":
    main()