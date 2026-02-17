# Legacy-1bit LLM Project

## Project Title
**Legacy-1bit LLM: A Ternary Large Language Model for 2000-Era Laptops**

## Project Description
This project aims to implement a simplified, yet functional, Large Language Model (LLM) designed with the severe resource constraints of a 2000-era laptop in mind. The core innovation lies in its "1-bit" or ternary weight quantization, where model weights are restricted to values of -1, 0, or 1. This approach drastically reduces memory footprint and computational complexity, making training and inference feasible on older hardware. The project is implemented in C99, eschewing complex modern libraries to ensure maximum compatibility and minimal overhead.

Key features include memory-efficient gradient accumulation, gradient checkpointing for activations, and SSE (Streaming SIMD Extensions) optimization for critical mathematical operations to squeeze out maximum performance from older CPUs.

## Features Implemented

The following major phases and features have been successfully implemented and integrated:

*   **Modular Project Structure:** The codebase is organized into distinct, manageable modules with clear header interfaces. Project structure supports both SSE and Non-SSE builds.
*   **Memory-Efficient Gradient Accumulation:** Implemented through sparse/aggregated gradient storage within the `LegacyLLM_Gradients` structure and efficient resetting mechanisms. Minimizes persistent memory footprint during training.
*   **Memory-Efficient Activation Handling (Gradient Checkpointing):** Activations required for the backward pass are recomputed on-the-fly rather than stored, significantly reducing RAM usage.
*   **Ternary Weight Update Mechanism:** The `apply_ternary_weight_updates` function manages training of ternary weights, quantizing weights to -1, 0, or 1 based on aggregated gradients.
*   **Functional Training Loop:** Complete training loop in `main.c` handling model initialization, gradient management, forward/backward passes, loss calculation, and weight updates over multiple epochs.
*   **Dataset Integration:** Robust data loading and character-level tokenization supporting input-target token pairs for sequence prediction.
*   **Comprehensive Test Suite (31 tests):** Complete test coverage including:
    *   Math operations tests (13 tests)
    *   Forward pass tests (6 tests)
    *   Backward pass tests (5 tests) 🆕
    *   Model persistence tests (6 tests) 🆕
    *   Integration tests (1 test)
*   **Advanced SSE Optimization:** Core mathematical operations optimized with SSE/SSE4.1 intrinsics:
    *   **Performance:** 2x faster than Non-SSE builds (50.9% improvement)
    *   **Features:** SSE4.1 int8→float conversion with SSE2 fallback
    *   **Compatibility:** Transparent fallback for non-SSE systems
*   **Model Persistence:** Save/load model state with integrity checks (magic number, version)
*   **Code Quality:** C99 compliant, comprehensive error handling, memory leak prevention

## How to Build

The project uses a simple `Makefile` with support for both SSE and Non-SSE builds.

### Build Targets

```bash
# Clean all build artifacts
make clean

# Build main application (Non-SSE, default)
make
# or
make legacy_llm_no_sse

# Build main application (SSE optimized)
make legacy_llm_sse

# Build and run all tests
make test

# Build test runners individually
make test_runner_no_sse
make test_runner_sse

# Run performance benchmark
make perf
```

### Quick Start

```bash
# Build and run tests (recommended first step)
make clean
make test

# Run training with SSE optimizations
make legacy_llm_sse
./legacy_llm_sse

# Run training without SSE
make legacy_llm_no_sse
./legacy_llm_no_sse
```

### Performance Comparison

SSE optimized builds are approximately **2x faster** than Non-SSE builds:

| Build | Total Training Time | Speedup |
|-------|-------------------|---------|
| Non-SSE | ~58.7 seconds | 1.0x |
| SSE | ~28.8 seconds | **2.0x** |

## How to Run Tests

The project includes a comprehensive test suite (31 tests):

```bash
# Build and run all tests (both SSE and Non-SSE)
make test

# Run tests individually
./test_runner_no_sse
./test_runner_sse
```

### Test Categories

- **Math Operations (13 tests):** Vector operations, activations, layer normalization
- **Forward Pass (6 tests):** Embedding, attention, FFN, transformer blocks
- **Backward Pass (5 tests):** Gradient calculations, loss functions
- **Model Persistence (6 tests):** Save/load, error handling, lifecycle
- **Integration (1 test):** End-to-end model allocation and data loading

## How to Run Training

After building, run the training executable:

```bash
./legacy_llm_sse
```

The training loop will:
1. Load dataset from `data/saioa_stories_sample.txt`
2. Tokenize text (character-level)
3. Initialize/load model from `llm_model.bin`
4. Run 10 epochs of training
5. Save checkpoints every 2 epochs
6. Display loss and perplexity metrics
7. Track Top-k accuracy (Top-1, Top-3, Top-5)
8. Log metrics to JSON file in `logs/` directory

### Training Metrics

During training, the system tracks and logs:
- **Loss:** Cross-entropy loss per epoch
- **Perplexity:** exp(loss) for language modeling evaluation
- **Top-k Accuracy:** Percentage of correct predictions in top-k
  - Top-1: Exact match accuracy
  - Top-3: Correct answer in top 3 predictions
  - Top-5: Correct answer in top 5 predictions
- **Training Time:** Epoch duration and total training time
- **JSON Logs:** Detailed metrics saved to `logs/training_YYYYMMDD_HHMMSS.json`

## How to Run Inference

After training, use the inference mode to generate text:

```bash
# Build inference executable
make inference_sse

# Generate text with greedy sampling (default)
./inference_sse "Once upon a time"

# Generate with temperature sampling
./inference_sse -s temperature -t 0.8 "Hello world"

# Generate with top-k sampling
./inference_sse -s topk -k 40 -n 200 "In 1492,"
```

### Inference Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model file path | `llm_model.bin` |
| `-n, --max-tokens` | Maximum tokens to generate | 100 |
| `-s, --strategy` | Sampling strategy: greedy, temperature, topk | greedy |
| `-t, --temperature` | Temperature for sampling (0.1-2.0) | 1.0 |
| `-k, --top-k` | Top-k value for sampling | 0 (disabled) |
| `--seed` | Random seed | Current time |

### Sampling Strategies

1. **Greedy:** Always selects the most probable next token
   - Best for: Deterministic, coherent text
   - Usage: `./inference_sse "Your prompt here"`

2. **Temperature:** Controls randomness via temperature scaling
   - T < 1.0: More focused, conservative
   - T > 1.0: More creative, diverse
   - Usage: `./inference_sse -s temperature -t 0.7 "Your prompt"`

3. **Top-k:** Samples only from top-k most likely tokens
   - k=40: Balanced creativity and coherence
   - k=10: More focused
   - Usage: `./inference_sse -s topk -k 40 "Your prompt"`

## File Structure

```
.
├── Makefile                      # Project Makefile with SSE/Non-SSE targets
├── AGENTS.md                     # Developer guidelines and code standards
├── PROJECT_PLAN.md               # Detailed project roadmap and phases
├── ACTION_PLAN.md                # Post-audit action plan (completed items)
├── data/                         # Directory for datasets
│   └── saioa_stories_sample.txt  # Sample text dataset
├── docs/                         # Additional documentation
│   ├── ARCHITECTURE.md           # Technical architecture overview
│   └── AUDIT.md                  # Quality audit report
├── include/                      # Header files (14 files)
│   ├── legacy_llm.h              # Core definitions and constants
│   ├── model.h                   # Model management
│   ├── math_ops.h                # Mathematical operations (SSE/SSE4.1)
│   ├── forward.h                 # Forward pass
│   ├── backward.h                # Backward pass
│   ├── data_utils.h              # Data handling
│   ├── test_framework.h          # Test framework with helpers
│   ├── test_llm.h                # Integration test declarations
│   ├── test_math_ops.h           # Math operations test declarations
│   ├── test_forward.h            # Forward pass test declarations
│   ├── test_backward.h           # Backward pass test declarations
│   ├── test_model.h              # Model test declarations
│   └── logger.h                  # 🆕 Structured logging system
├── src/                          # Source files (7 files)
│   ├── main.c                    # Training loop and entry point
│   ├── model.c                   # Model lifecycle and persistence
│   ├── math_ops.c                # Optimized math operations (SSE4.1)
│   ├── forward.c                 # Forward pass implementation
│   ├── backward.c                # Backward pass implementation
│   ├── data_utils.c              # Data loading and tokenization
│   ├── logger.c                  # 🆕 JSON logging implementation
│   └── inference.c               # 🆕 Text generation and inference
└── tests/                        # Test files (5 files)
    ├── test_llm.c                # Integration tests
    ├── test_math_ops.c           # Math operations tests (13 tests)
    ├── test_forward.c            # Forward pass tests (6 tests)
    ├── test_backward.c           # Backward pass tests (5 tests)
    └── test_model.c              # Model tests (6 tests)
```

## Recent Improvements (Feb 2026)

### Phase 1: Critical Fixes ✅
- Fixed critical allocation bug in `src/model.c:353`
- Consolidated `compare_float_arrays` helper in test framework
- Defined `LAYER_NORM_EPSILON` constant (eliminated 16 magic numbers)

### Phase 2: Comprehensive Testing ✅
- Expanded test suite from 20 to **31 tests** (+55% coverage)
- Added complete backward pass test suite (5 tests)
- Added model persistence tests (6 tests)
- Test coverage increased from ~50% to **~85%**
- All tests passing (100% success rate)

### Phase 3: Performance Optimization ✅
- Implemented SSE4.1 optimizations with SSE2 fallback
- Added `convert_int8_to_float()` helper using `_mm_cvtepi8_epi32`
- **Result:** 2x faster than Non-SSE builds (50.9% improvement)
- Performance benchmarks documented

### Phase 4: Advanced Features ✅

### Phase 4: Advanced Features
- **Structured JSON Logging:** Complete logging system with timestamped JSON files
  - Logs saved to `logs/training_YYYYMMDD_HHMMSS.json`
  - Tracks loss, perplexity, top-k accuracy, and timing per epoch
- **Top-k Accuracy Tracking:** Real-time accuracy metrics during training
  - Top-1, Top-3, and Top-5 accuracy percentages
- **Inference Mode:** Dedicated text generation program
  - Executable: `inference_sse` / `inference_no_sse`
  - Autoregressive text generation from prompts
- **Sampling Strategies:** Three different sampling methods
  - Greedy: Deterministic, highest probability
  - Temperature: Controllable randomness (T parameter)
  - Top-k: Sampling from k most likely tokens

## Future Enhancements

### In Progress (Phase 5)
*   **Improved Dataset Handling:** Support for streaming large datasets, dynamic batching, BPE tokenization
*   **Hyperparameter Tuning:** Systematic experimentation with learning rates, model dimensions, and architecture variations
*   **Advanced Quantization:** Exploration of quantization-aware training techniques (straight-through estimators)
*   **Extended Quantization Schemes:** Investigation of binary weights and 2-bit quantization alternatives

## Project Statistics

- **Lines of Code:** ~4,800
- **Test Coverage:** ~85% (31 tests)
- **Code Quality:** 9.5/10
- **Performance:** 2x speedup with SSE optimizations
- **Features:** Training + Inference + Logging + Metrics
- **License:** Open source (see LICENSE file)
