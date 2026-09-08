# Synapse Lab: Attention as Dynamic Wiring

 Replacing the Transformer O(T · d) KV-Cache Bottleneck with Fixed-Capacity Hebbian Fast Weights via Non-Negative Sparse Projections.

---

 ### 1\. Executive Summary & Core Claim

 Standard autoregressive Transformers suffer from an inference-time memory bottleneck: the KV Cache. Storing explicit token vectors across context length T incurs linear memory scaling:

 Memory\_KV = 2 × L × T × d × sizeof(dtype) = O(T · d)

 At long sequences (T ≥ 32k), GPU VRAM exhaustion triggers Out-Of-Memory (OOM) failures or requires heavy distributed KV-cache compression schemes.

 #### The Core Scientific Claim

 The Baby Dragon Hatchling (BDH) architecture demonstrates that recurrent, local Hebbian fast weights (M ∈ R^(d × d)) can provide associative sequence memory within a fixed-size state of O(d²). The decoding step also operates at a strictly constant O(d²) complexity rather than growing with sequence length. The mechanism relies on constraining neural activations to non-negative sparse subspaces, with approximately 5% of units active, to substantially reduce associative crosstalk.

 #### Scope & Pedagogical Disclaimer

 Synapse Lab is a BDH-inspired toy simulator and scientific demonstration tool, not a full implementation of the complete multi-layer BDH architecture. The full BDH architecture incorporates complex scale-free neuron particle graphs, integrate-and-fire thresholds, excitatory and inhibitory circuits, and sensory steering. Synapse Lab isolates and demonstrates the fundamental linear-algebraic intuition: how rank-1 Hebbian fast weights combined with non-negative sparse projections (≈ 5% active units) significantly reduce associative crosstalk compared to dense linear recurrence within a constant O(d²) memory footprint.

 #### What Synapse Lab Provides

 - **Interactive Client Visualizer**: A 60 FPS browser-based simulation using Float64 Hebbian updates, interactive parameter controls, telemetry, and crosstalk visualization.
- **PyTorch Research Harness**: An offline benchmarking suite providing reference implementations of BDH, Softmax KV-cache, and Dense Linear Attention, with experiments focused on Multi-Query Associative Recall (MQAR) and memory capacity scaling.

---

 ### 2\. Mathematical Formalism

 #### 2.1 The Hebbian Outer-Product Update

 Instead of continuously appending token representations to a growing memory buffer, the model writes information directly into a fixed-size fast-weight matrix. The recurrent update rule is:

 M\_t = λ M\_(t−1) + η (y\_t x\_tᵀ)

 - x\_t ∈ R^d: Presynaptic key concept representation vector.
- y\_t ∈ R^d: Postsynaptic value concept representation vector.
- λ ∈ (0, 1\]: Synaptic decay factor controlling the retention horizon.
- η \> 0: Plasticity rate coefficient, set to 1.0 in the reference implementation.
- M\_t ∈ R^(d × d): Fast-weight memory state at timestep t.

 Each association is written through a rank-1 outer-product update, allowing the memory state to remain fixed at d × d.

 #### 2.2 Subspace Isolation via Non-Negative Sparsity

 When retrieving an association using the query cue q = x\_a, the linear readout becomes:

 ŷ\_a = M x\_a = y\_a (x\_aᵀ x\_a) + Σ\_(j ≠ a) y\_j (x\_jᵀ x\_a)

 The first term represents the target association signal, while the second term represents accumulated crosstalk from other stored associations.

 ##### The Dense Failure Mode

 With standard dense linear attention, 100% of activation coordinates participate in every representation. Because random dense vectors exhibit non-zero expected inner products, accumulated interference scales with the number of stored items:

 Σ\_(j ≠ a) y\_j (x\_jᵀ x\_a) = O(P)

 This results in crosstalk collapse, where cumulative interference overwhelms the target signal and retrieved representations no longer identify the requested association.

 ##### The Non-Negative Top-K Solution

 BDH addresses this problem by applying non-negative TopK-ReLU thresholding to the projected representations:

 x = TopK-ReLU(W\_K e, k) / ‖TopK-ReLU(W\_K e, k)‖₂

 y = TopK-ReLU(W\_V e, k) / ‖TopK-ReLU(W\_V e, k)‖₂

 For d = 64, choosing k ≈ 3–4 corresponds to approximately 4.7% to 6.2% active coordinates. The coordinate collision probability between two independent random representations is strictly bounded:

 P(collision) = 1 − ∏\_(i=0)^(k−1) (1 − k / (d − i)) \< 0.14

 Because over 95% of coordinates are zero, x\_jᵀ x\_a ≈ 0 holds almost surely for unrelated items, projecting representations into quasi-orthogonal subspaces and substantially reducing associative crosstalk.

---

 ### 3\. Comparative Architecture Taxonomy

 - **Primary Literature Citations**:
- Transformer (Softmax KV-Cache): Vaswani et al., NeurIPS 2017.
- Dense Linear Attention: Katharopoulos et al., ICML 2020; Schlag et al., ICML 2021.
- BDH Synaptic Plasticity: Pathway Research, arXiv:2509.26507, 2025.
- **State Memory Footprint**:
- Transformer: O(T · d), growing linearly with context length.
- Dense Linear Attention: O(d²), fixed-size recurrent state matrix.
- BDH Synaptic Plasticity: O(d²), fixed-size state matrix (32.77 KB at d = 64 in Float64).
- **Step FLOP Complexity**:
- Transformer: O(T · d) per decode step.
- Dense Linear Attention: O(d²) constant decode step.
- BDH Synaptic Plasticity: O(d²) constant decode step.
- **Activation Geometry and Density**:
- Transformer: Dense softmax attention (100% density).
- Dense Linear Attention: Dense unthresholded linear feature maps (100% density).
- BDH Synaptic Plasticity: Non-negative Top-K sparse activations (≈ 5% active units).
- **Crosstalk Behavior**:
- Transformer: Zero crosstalk (exact lookup over historical token cache).
- Dense Linear Attention: Catastrophic crosstalk collapse under accumulated associations.
- BDH Synaptic Plasticity: Substantially reduced crosstalk via quasi-orthogonal sparse subspaces.
- **Primary Failure Modes**:
- Transformer: Physical GPU VRAM Out-Of-Memory exhaustion.
- Dense Linear Attention: Crosstalk wash-out from overlapping coordinate inner products.
- BDH Synaptic Plasticity: Toy experiment single-layer capacity saturation (P \> 8–10 items).
- **Long-Context Behavior**:
- Transformer: Exact retrieval until physical memory boundaries are reached.
- Dense Linear Attention: Unbounded cross-term degradation across sequence length.
- BDH Synaptic Plasticity: Adaptive rolling working memory via decay λ \< 1.0.

---

 ### 4\. Empirical Validation & Discovery Log

 #### 4.1 Multi-Query Associative Recall (MQAR) Results

 The offline PyTorch capacity benchmark sweeps associative retrieval accuracy across dimension d = 64 over multiple random seeds:

 - **At P = 2 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention achieves 100.0%, and BDH Toy (k = 3) achieves 100.0%.
- **At P = 4 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention achieves 75.0%, and BDH Toy (k = 3) achieves 100.0%.
- **At P = 6 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention drops to 44.4% (emergence of severe interference), and BDH Toy (k = 3) achieves 100.0%.
- **At P = 8 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention drops to 25.0% (crosstalk collapse), and BDH Toy (k = 3) achieves 100.0%.
- **At P = 10 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention drops to 26.7%, and BDH Toy (k = 3) achieves 100.0%.
- **At P = 12 Stored Items**: Softmax KV achieves 100.0%, Dense Linear Attention drops to 16.7% (catastrophic collapse), and BDH Toy (k = 3) achieves 100.0%.

 Increasing the activation density directly scales coordinate overlap and accelerates associative collapse:

 - k = 3: Approximately 4.7% density, providing stable high-fidelity recall across the sequence.
- k = 4: Approximately 6.2% density.
- k = 32: 50.0% density, triggering severe interference as shared synapses overlap.
- k = 64: 100.0% density, causing complete crosstalk collapse.

 #### 4.2 Toy Experiment Capacity Horizon

 - Without synaptic decay (λ = 1.00), the fast-weight matrix retains every historical rank-1 write indefinitely.
- In this un-decayed single-layer toy model, noticeable capacity degradation begins past P ≈ 8–10 associations (≈ 0.14d).
- Beyond P = 16, the toy system experiences severe associative interference.
- This empirical transition echoes the classical Hopfield network capacity limit (C ≈ 0.138N).
- **Scope Disclosure**: This 0.14d threshold is an empirical observation specific to single-layer rank-1 outer products in this toy experiment; it is not an inherent mathematical upper bound of the full multi-layer BDH architecture, which utilizes hierarchical routing and dynamic decay.

 #### 4.3 Decay-Driven Working Memory

 - Introducing synaptic decay converts the state from an unconstrained accumulator into an adaptive rolling working-memory mechanism.
- Values of λ ∈ \[0.94, 0.98\] provide a stable operating balance between retention horizon and saturation prevention.
- Lower retention values (λ ≤ 0.90) cause working memory to fade rapidly, with earlier associations vanishing within 10 to 15 tokens.

---

 ### 5\. System Architecture & Integration

 #### Offline Research Pipeline

 - **`python/generate_data.py`**: Generates synthetic sparse concept vectors and creates calibrated crosstalk probe pairs (sharing 2 active coordinates with cos(A, B) ≈ 0.667).
- **`python/benchmark_capacity.py`**: Runs the MQAR capacity experiments across context lengths, measuring accuracy collapse in dense baselines versus sparse retention.
- **`python/verify_parity.py`**: Asserts exact numerical parity between PyTorch tensors and the browser engine (100% cosine similarity, Frobenius matrix norm parity at 1.7321).

 #### Shared Data Artifacts

 - **`python/data/associations.json`**: Pre-generated sparse concept vectors and calibrated crosstalk probe configurations.
- **`python/data/benchmarks.json`**: Precomputed empirical MQAR accuracy sweeps and VRAM footprint scaling metrics.

 #### Interactive Web Engine

 - **`src/app.js` and `index.html`**: The browser client loads the precomputed JSON artifacts directly to power the simulation.
- **In-Browser Linear Algebra**: Evaluates Float64 matrix recurrence locally at 60 FPS, eliminating network latency during interactive parameter exploration.
- **Telemetry and Tooltips**: Displays real-time matrix fill percentage, Frobenius energy (‖M‖\_F), maximum synapse weights, and candidate retrieval spectra.

---

 ### 6\. Honest Failure Disclosures & Engineering Trade-Offs

 #### 6.1 Finite Associative Horizon

 A fixed d × d outer-product state cannot store an unbounded number of independent associations without degradation. The system functions as adaptive working memory and cannot replace persistent external storage (such as vector databases or RAG) for lossless static corpus retrieval.

 #### 6.2 Precision & Quantization Limits

 Repeated rank-1 outer-product accumulation without non-linear squashing can cause matrix values to grow. Under unbounded updates, low-precision representations (such as FP16) can saturate rapidly without normalization. The reference suite utilizes Float32 for offline execution and Float64 for client simulation and parity verification.

 #### 6.3 Decay Hyperparameter Sensitivity

 - When λ = 1.00, the system provides permanent retention but is prone to interference saturation under long contexts.
- When λ ≤ 0.90, retention drops sharply, with older associations decaying within 10 to 15 tokens.
- Production architectures require dynamic, input-gated decay: λ\_t = σ(W\_λ x\_t).

 #### 6.4 Peripheral Research Note

 Subsequent architectural developments such as BDH-CQ (Pathway Research, August 2026) investigate recurrent latent reasoning loops without token generation; this operates as a distinct research direction beyond the associative fast-weight memory explored here.

---

 ### 7\. Reproduction & Execution Guide

 #### 7.1 Interactive Web Visualizer

 - **Prerequisites**: Node.js v18+.
- **Install Dependencies**: `npm install`.
- **Start Local Visualizer**: `npm run dev`.
- **Verify Production Build**: `npm run build`.

 #### 7.2 Offline PyTorch Research Suite

 - **Prerequisites**: Python 3.10+.
- **Create Environment**: `python -m venv .venv`.
- **Activate Environment**:
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`
- **Install Dependencies**: `pip install -r python/requirements.txt`.
- **Generate Synthetic Data**: `python python/generate_data.py`.
- **Run Capacity Benchmark**: `python python/benchmark_capacity.py`.
- **Verify Numerical Parity**: `python python/verify_parity.py`.

---

 ### 8\. Primary Scientific References

 - **Baby Dragon Hatchling (BDH)**: Pathway Research. _Dragon Hatchling: From Attention to Synapses_. arXiv:2509.26507, 2025.
- **Fast Weight Programmers**: Schlag, I., Irie, K., & Schmidhuber, J. _Linear Transformers Are Secretly Fast Weight Programmers_. ICML 2021.
- **Linear Attention Foundations**: Katharopoulos, A., et al. _Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention_. ICML 2020.
- **Sparse Coding in Neuroscience**: Olshausen, B. A., & Field, D. J. _Emergence of simple-cell receptive field properties by learning a sparse code for natural images_. Nature, 1996.
- **Associative Memory Capacity**: Hopfield, J. J. _Neural networks and physical systems with emergent collective computational abilities_. PNAS, 1982.
- **Transformer Attention**: Vaswani, A., et al. _Attention Is All You Need_. NeurIPS 2017.
- **Test-Time Memory Plasticity**: Behrouz, A., et al. _Titans: Learning to Memorize at Test Time_. arXiv:2412.19837, 2024.
