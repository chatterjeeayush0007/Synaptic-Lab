Synapse Lab: Attention as Dynamic Wiring

Replacing the Transformer O(T · d) KV-Cache Bottleneck with Fixed-Capacity Hebbian Fast Weights via Non-Negative Sparse Projections.

1. Executive Summary & Core Claim

Standard autoregressive Transformers suffer from an inference-time memory bottleneck: the KV Cache. Storing explicit token vectors across context length T incurs linear memory scaling:

Memory_KV = 2 · L · T · d · sizeof(dtype) = O(T · d)

At long sequences (T ≥ 32k), GPU VRAM exhaustion triggers catastrophic Out-Of-Memory (OOM) failures or requires heavy distributed KV-compression schemes.

The Core Scientific Claim

The Baby Dragon Hatchling (BDH) architecture demonstrates that recurrent, local Hebbian fast weights (M ∈ R^(d × d)) achieve associative sequence memory in strictly constant space O(d²) and constant decoding step complexity O(d²), provided neural activations are constrained to non-negative sparse subspaces (≈ 5% biological active units) to prevent crosstalk collapse.

Synapse Lab provides a dual-layer, verified research environment:

Interactive Client Visualizer: A 60 FPS in-browser simulation executing real-time Float64 Hebbian updates with a draggable telemetry HUD, dynamic crosstalk probe verification, and interactive parameter exploration.
PyTorch Research Harness: An offline, reproducible benchmarking suite (python/) providing reference implementations of BDH, Softmax KV-cache, and Dense Linear Attention models, evaluating Multi-Query Associative Recall (MQAR) and capacity scaling.
2. Mathematical Formalism
2.1 The Hebbian Outer-Product Update

Instead of appending rows to a buffer, memory writes execute a local rank-1 outer-product update directly onto the fast-weight matrix:

Mₜ = λMₜ₋₁ + η(yₜxₜᵀ)

Where:

xₜ ∈ Rᵈ: Key concept representation vector.
yₜ ∈ Rᵈ: Value concept representation vector.
λ ∈ (0, 1]: Synaptic decay factor determining retention horizon.
η > 0: Plasticity rate coefficient (set to 1.0).
2.2 Subspace Isolation via Non-Negative Sparsity

When retrieving an association using query cue q = xₐ, the linear readout expands into a target signal and a superposition of crosstalk:

ŷₐ = Mxₐ = yₐ(xₐᵀxₐ) + Σⱼ≠ₐ yⱼ(xⱼᵀxₐ)

The Dense Failure Mode: In standard dense linear attention (100% activation density), the cumulative interference Σⱼ≠ₐ yⱼ(xⱼᵀxₐ) = O(p) rapidly overpowers the target vector yₐ, causing total crosstalk collapse.
The Non-Negative Top-K Solution: BDH filters pre-synaptic activations through a non-negative thresholding function:

x = TopK-ReLU(Wₖe, k) / ‖TopK-ReLU(Wₖe, k)‖₂

y = TopK-ReLU(Wᵥe, k) / ‖TopK-ReLU(Wᵥe, k)‖₂

For d = 64, setting k ≈ 3–4 (≈ 4.7%–6.2% active coordinates) minimizes coordinate collision probabilities:

P(collision) = 1 − (1 − k/d)ᵏ < 0.14

This ensures xⱼᵀxₐ ≈ 0 for almost all j ≠ a, keeping synaptic pathways orthogonal and preserving high-fidelity retrieval.

3. Comparative Architecture Taxonomy
Architectural Dimension	Transformer (Softmax KV Cache)	Dense Linear Attention	BDH Synaptic Plasticity
State Memory Footprint	O(T · d) (Surges linearly)	O(d²) (Fixed constant)	O(d²) (Fixed constant: 32.77 KB at d = 64)
Step FLOP Complexity	O(T · d)	O(d²)	O(d²)
Activation Geometry	Unconstrained dense softmax	Dense unthresholded linear	Non-negative Top-K (~5% active units)
Failure Mode	Physical VRAM OOM crash	Instant crosstalk wash-out	Hopfield bound saturation (P > 0.14d)
Long-Context Behavior	Exact retrieval, memory wall	Signal collapse	Decaying working memory via λ < 1.0
4. Empirical Validation & Discovery Log
   Retrieval Accuracy vs. Stored Facts (d = 64, P in [2..32])
100% ─────────────────────────┐
     │ █   █   █              │  ■ Softmax KV-Cache (100% Exact)
 80% │   █   █   ▲   ▲        │  ▲ BDH Sparse Hebbian (k = 3, ~5%)
 60% │             ▲   ▲      │  ● Dense Linear Attention (k = 64)
 40% │                   ▲    │
 20% │ ●   ●                  │
  0% └───┴───┴───┴───┴───┴────┘
     2   4   8  12  16  24  32

The ~5% Biological Sparsity Invariant: In offline PyTorch MQAR sweeps across d = 64, setting k = 3 maintains >95% retrieval accuracy up to P ≈ 8 stored items. Expanding k → 32 (50% density) drops retrieval fidelity to zero under identical sequence conditions.
The Hopfield Capacity Boundary: Without synaptic decay (λ = 1.00), a 64 × 64 matrix hits theoretical capacity saturation near P ≈ 0.14d ≈ 9 items. Pushing to P = 16 causes catastrophic memory interference.
Decay-Driven Working Memory: Applying slight decay (λ = 0.94–0.98) prevents synaptic saturation over infinite contexts, converting the matrix into an adaptive rolling working memory.
5. System Architecture & Integration

The repository implements a fully coupled research loop where offline PyTorch experiments feed the interactive visualization engine:

┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PYTORCH SUITE                    │
│                                                             │
│   python/generate_data.py   ───>   python/data/associations.json
│   (Sparse vectors & Probes)        (Concept space & overlap pairs)
│                                                    │
│   python/benchmark_capacity.py ─>  python/data/benchmarks.json  │
│   (MQAR Sweeps & VRAM Scaling)     (Empirical accuracy curves)    │
└────────────────────────────────────────────────────┬────────┘
                                                     │ Static Bridge
┌────────────────────────────────────────────────────▼────────┐
│                   INTERACTIVE WEB ENGINE                    │
│                                                             │
│   src/app.js (Imports associations.json & benchmarks.json)   │
│   ├─ 60 FPS Heatmap Canvas (Viridis Colormap, Hover Tooltip)│
│   ├─ Interactive Draggable Telemetry HUD                   │
│   ├─ Dual-Slider Interference Walkthrough (k & p Sync)      │
│   └─ BDH Deep Dive (Renders empirical PyTorch MQAR data)    │
└─────────────────────────────────────────────────────────────┘

Zero-Latency Mathematical Substrate: The web visualizer runs Float64 matrix updates client-side to ensure 60 FPS slider scrubbing without HTTP request latency.
Direct Artifact Ingestion: src/app.js ingests python/data/associations.json to load the exact sparse vectors and calibrated crosstalk probes generated by PyTorch.
Empirical Dashboard: The BDH Deep Dive tab renders the precomputed MQAR accuracy and footprint scaling tables directly from python/data/benchmarks.json.
6. Repository Layout
Synaptic-Plasticity/
├── index.html                   # Production visualizer shell & draggable HUD
├── src/
│   ├── app.js                   # 60 FPS simulation engine, tour, & data bridge
│   ├── styles.css               # Viridis colormap, custom controls, dark theme
│   └── vite-env.d.ts            # TypeScript definitions for Vite
├── python/                      # Reproducible PyTorch Research Suite
│   ├── requirements.txt         # Pinned offline scientific dependencies
│   ├── generate_data.py         # Synthetic sparse concept & crosstalk generator
│   ├── benchmark_capacity.py    # Multi-Query Associative Recall (MQAR) sweep
│   ├── verify_parity.py         # Numerical parity test (PyTorch vs. Web engine)
│   ├── models/
│   │   ├── __init__.py          # Package interface
│   │   ├── bdh_toy.py           # Canonical BDH TopK-ReLU Hebbian reference
│   │   └── baselines.py         # Softmax KV-cache & Dense Linear references
│   └── data/
│       ├── associations.json    # Calibrated concept vectors & probe pairs
│       └── benchmarks.json      # Precomputed capacity and VRAM sweep curves
├── docs/
│   └── concept-summary.tex      # 1-page Academic Briefing LaTeX source
├── package.json                 # Node.js project configuration
└── vite.config.js               # Vite bundler configuration

7. Honest Failure Disclosures & Engineering Trade-Offs
Finite Associative Horizon: A fixed d × d outer-product state cannot store infinite associations without degradation. It cannot replace external persistent storage (vector databases, RAG) for static corpus retrieval.
Precision & Quantization Limits: Softmax attention is scale-invariant under temperature tuning. In linear Hebbian recurrence, unbounded rank-1 writes saturate 16-bit floats rapidly; Float32 or Float64 precision is mandatory unless weight normalization is applied.
Decay Hyperparameter Sensitivity: When λ = 1.00, the system experiences permanent retention, leading to saturation. If λ ≤ 0.90, working memory vanishes after 10–15 tokens. Production architectures require dynamic, input-gated decay (λₜ = σ(W_λxₜ)).
8. Reproduction & Execution Guide
8.1 Interactive Web Visualizer

Prerequisites: Node.js (v18+)

# Install dependencies
npm install

# Start local visualizer
npm run dev

# Verify production bundle build
npm run build

8.2 Offline PyTorch Research Suite

Prerequisites: Python (3.10+)

# 1. Initialize and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r python/requirements.txt

# 3. Generate synthetic concept spaces and calibrated crosstalk probes
python python/generate_data.py

# 4. Run Multi-Query Associative Recall (MQAR) capacity sweep
python python/benchmark_capacity.py

# 5. Assert numerical parity between PyTorch and client-side simulation
python python/verify_parity.py

9. Primary Scientific References
Baby Dragon Hatchling (BDH): Pathway Research. Dragon Hatchling: From Attention to Synapses. arXiv:2509.26507
, 2025.
Fast Weight Programmers: Schlag, I., Irie, K., & Schmidhuber, J. Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.
Test-Time Memory Plasticity: Behrouz, A., et al. Titans: Learning to Memorize at Test Time. arXiv:2412.19837
, 2024.
Associative Memory Capacity: Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. PNAS, 1982.