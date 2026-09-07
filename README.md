Synapse Lab: Attention as Dynamic Wiring

Replacing the Transformer O(T · d) KV-Cache Bottleneck with Fixed-Capacity Hebbian Fast Weights via Non-Negative Sparse Projections.

1. Executive Summary & Core Claim

Standard autoregressive Transformers suffer from an inference-time memory bottleneck: the KV Cache. Storing explicit token vectors across context length T incurs linear memory scaling:

Memory_KV = 2 · L · T · d · sizeof(dtype) = O(T · d)

At long sequences (T ≥ 32k), GPU VRAM exhaustion can trigger Out-Of-Memory (OOM) failures or require heavy distributed KV-cache compression.

The Core Scientific Claim

The Baby Dragon Hatchling (BDH) architecture demonstrates that recurrent, local Hebbian fast weights (M ∈ R^(d × d)) can provide associative sequence memory using a fixed-size state of O(d²).

The decoding step also operates at O(d²) complexity rather than growing with sequence length.

The approach relies on constraining neural activations to non-negative sparse subspaces, with approximately 5% of units active, to reduce associative crosstalk.

What Synapse Lab Provides

Synapse Lab provides two connected research components:

Interactive Client Visualizer: A 60 FPS browser-based simulation using Float64 Hebbian updates, interactive parameter controls, telemetry, and crosstalk visualization.

PyTorch Research Harness: An offline benchmarking suite providing reference implementations of BDH, Softmax KV-cache, and Dense Linear Attention, with experiments focused on Multi-Query Associative Recall (MQAR) and memory capacity.

2. Mathematical Formalism
2.1 The Hebbian Outer-Product Update

Instead of continuously appending token representations to a growing memory buffer, the model writes information directly into a fixed-size fast-weight matrix.

The update rule is:

Mₜ = λMₜ₋₁ + η(yₜxₜᵀ)

Where:

xₜ ∈ Rᵈ — Key concept representation vector.
yₜ ∈ Rᵈ — Value concept representation vector.
λ ∈ (0, 1] — Synaptic decay factor controlling the retention horizon.
η > 0 — Plasticity rate coefficient, set to 1.0 in the reference implementation.
Mₜ — Fast-weight memory state at timestep t.

Each association is therefore written through a rank-1 outer-product update, allowing the memory state to remain fixed at d × d.

2.2 Subspace Isolation via Non-Negative Sparsity

When retrieving an association using the query cue q = xₐ, the linear readout becomes:

ŷₐ = Mxₐ = yₐ(xₐᵀxₐ) + Σⱼ≠ₐ yⱼ(xⱼᵀxₐ)

The first term represents the desired association, while the second term represents crosstalk from other stored associations.

The Dense Failure Mode

With standard dense linear attention, approximately 100% of activation coordinates can participate in every representation.

As more associations are stored, accumulated interference can overwhelm the desired signal:

Σⱼ≠ₐ yⱼ(xⱼᵀxₐ) = O(p)

This eventually results in crosstalk collapse, where the retrieved representation no longer reliably identifies the requested association.

The Non-Negative Top-K Solution

BDH addresses this problem by applying TopK-ReLU to the projected representations:

x = TopK-ReLU(Wₖe, k) / ‖TopK-ReLU(Wₖe, k)‖₂

y = TopK-ReLU(Wᵥe, k) / ‖TopK-ReLU(Wᵥe, k)‖₂

For d = 64, choosing k ≈ 3–4 corresponds to approximately 4.7%–6.2% active coordinates.

The associated coordinate collision probability is:

P(collision) = 1 − (1 − k/d)ᵏ < 0.14

This reduces the probability that two independent sparse representations share active coordinates.

Consequently, xⱼᵀxₐ ≈ 0 for most unrelated associations, reducing interference and improving retrieval fidelity.

3. Comparative Architecture Taxonomy
Architectural Dimension	Transformer (Softmax KV Cache)	Dense Linear Attention	BDH Synaptic Plasticity
State Memory Footprint	O(T · d), grows with context	O(d²), fixed	O(d²), fixed
Step Complexity	O(T · d)	O(d²)	O(d²)
Activation Geometry	Dense softmax	Dense linear	Non-negative Top-K
Typical Activation Density	Dense	100%	~5%
Primary Failure Mode	VRAM / OOM	Crosstalk wash-out	Capacity saturation
Long-Context Behavior	Exact retrieval until memory limit	Signal degradation	Decaying working memory
State Size at d = 64	Grows with T	Fixed	32.77 KB
4. Empirical Validation & Discovery Log

The research harness evaluates associative retrieval capacity under different activation densities and memory configurations.

4.1 Sparse Hebbian Retrieval

The offline PyTorch MQAR experiments indicate that, for d = 64, using k = 3 maintains greater than 95% retrieval accuracy up to approximately 8 stored associations under the tested conditions.

Increasing the number of active coordinates significantly increases the probability of overlap between representations.

For example:

k = 3 → approximately 4.7% density
k = 4 → approximately 6.2% density
k = 32 → 50% density
k = 64 → 100% density

Under the tested sequence conditions, increasing the density toward 50% causes retrieval fidelity to collapse because of accumulated crosstalk.

4.2 Hopfield Capacity Boundary

Without synaptic decay:

λ = 1.00

the fast-weight matrix retains every previous association indefinitely.

For a 64 × 64 matrix, the observed theoretical capacity boundary is approximately:

P ≈ 0.14d ≈ 9 associations

Beyond this region, interference between stored patterns becomes increasingly significant.

At approximately P = 16, the tested system experiences catastrophic memory interference.

4.3 Decay-Driven Working Memory

Introducing controlled synaptic decay changes the behavior from permanent associative storage to a rolling working-memory mechanism.

The tested range:

λ = 0.94–0.98

provides a compromise between retention and saturation.

A lower decay value causes older associations to disappear more quickly, while a value closer to 1.0 provides longer retention but increases the risk of accumulated interference.

5. System Architecture & Integration

Synapse Lab connects the offline research pipeline with the interactive browser visualizer.

Offline Research Pipeline

The research process begins with python/generate_data.py.

This component:

Generates synthetic sparse concept vectors.
Creates calibrated crosstalk probe pairs.
Produces the shared association dataset.

The generated data is then used by python/benchmark_capacity.py, which:

Runs MQAR capacity experiments.
Evaluates different sparsity levels.
Measures associative retrieval performance.
Generates capacity and memory-scaling results.

Finally, python/verify_parity.py compares the numerical behavior of the PyTorch implementation with the browser-side simulation.

Shared Data

The research pipeline generates:

python/data/associations.json — Sparse concept vectors and calibrated probe configurations.
python/data/benchmarks.json — Precomputed capacity, accuracy, and scaling results.

The browser visualizer loads these artifacts directly, allowing the interactive interface to work with the same data used by the offline experiments.

Interactive Web Engine

The browser-side implementation is centered around src/app.js.

It provides:

60 FPS Float64 matrix simulation
Interactive Hebbian updates
Crosstalk visualization
Draggable telemetry HUD
Interactive sparsity controls
Parameter exploration
MQAR benchmark visualization
BDH deep-dive results

The mathematical simulation executes locally in the browser, avoiding HTTP round trips during parameter changes and allowing real-time interaction.

6. Honest Failure Disclosures & Engineering Trade-Offs
6.1 Finite Associative Horizon

A fixed d × d outer-product state cannot store an unlimited number of independent associations without degradation.

The system is therefore not a replacement for persistent external memory such as vector databases or RAG systems when retrieving information from a large static corpus.

Its intended role is closer to adaptive working memory.

6.2 Precision & Quantization Limits

Hebbian updates repeatedly accumulate rank-1 contributions into the fast-weight matrix.

With unbounded updates, low-precision representations can saturate rapidly.

In particular, 16-bit floating-point representations can become problematic without additional normalization or stabilization.

The reference implementation therefore uses:

Float32, where appropriate for performance.
Float64 for the interactive numerical simulation and parity verification.

Weight normalization or other stabilization techniques would be required for more aggressive quantization.

6.3 Decay Hyperparameter Sensitivity

When:

λ = 1.00

the system provides permanent retention but is susceptible to accumulated interference and capacity saturation.

When:

λ ≤ 0.90

working-memory retention becomes significantly shorter, with older information potentially disappearing within approximately 10–15 tokens under the tested conditions.

A production architecture would therefore benefit from dynamic, input-gated decay:

λₜ = σ(W_λxₜ)

This allows the system to adapt its retention behavior according to the incoming information.

7. Reproduction & Execution Guide
7.1 Interactive Web Visualizer

Prerequisites:

Node.js v18+
Install Dependencies
npm install

Start the Local Visualizer
npm run dev

Verify the Production Build
npm run build

7.2 Offline PyTorch Research Suite

Prerequisites:

Python 3.10+
Step 1 — Create the Virtual Environment
python -m venv .venv

Step 2 — Activate the Environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1


Linux / macOS:

source .venv/bin/activate

Step 3 — Install Dependencies
pip install -r python/requirements.txt

Step 4 — Generate Synthetic Data
python python/generate_data.py


This generates the sparse concept space and calibrated crosstalk probes used by the experiments and visualizer.

Step 5 — Run the MQAR Capacity Benchmark
python python/benchmark_capacity.py


This runs the Multi-Query Associative Recall (MQAR) capacity sweep.

Step 6 — Verify Numerical Parity
python python/verify_parity.py


This verifies numerical consistency between the PyTorch reference implementation and the browser-side simulation.

8. Primary Scientific References
Baby Dragon Hatchling (BDH)

Pathway Research. Dragon Hatchling: From Attention to Synapses. arXiv:2509.26507, 2025.

Fast Weight Programmers

Schlag, I., Irie, K., & Schmidhuber, J. Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.

Test-Time Memory Plasticity

Behrouz, A., et al. Titans: Learning to Memorize at Test Time. arXiv:2412.19837, 2024.

Associative Memory Capacity

Hopfield, J. J. Neural Networks and Physical Systems with Emergent Collective Computational Abilities. PNAS, 1982.
