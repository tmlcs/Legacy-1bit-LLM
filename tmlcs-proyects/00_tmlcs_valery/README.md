# Legacy-1bit LLM Project

## Project Title
**Legacy-1bit LLM: A Ternary Large Language Model for 2000-Era Laptops**

## Project Description
This project aims to implement a simplified, yet functional, Large Language Model (LLM) designed with the severe resource constraints of a 2000-era laptop in mind. The core innovation lies in its "1-bit" or ternary weight quantization, where model weights are restricted to values of -1, 0, or 1. This approach drastically reduces memory footprint and computational complexity, making training and inference feasible on older hardware. The project is implemented in C99, eschewing complex modern libraries to ensure maximum compatibility and minimal overhead.

Key features include memory-efficient gradient accumulation, gradient checkpointing for activations, and SSE (Streaming SIMD Extensions) optimization for critical mathematical operations to squeeze out maximum performance from older CPUs.

## Features Implemented

The following major phases and features have been successfully implemented and integrated:

*   **Modular Project Structure:** The codebase is organized into distinct, manageable modules (`model.c`, `forward.c`, `backward.c`, `math_ops.c`, `data_utils.c`, `test_llm.c`, `main.c`) with clear header interfaces.
*   **Memory-Efficient Gradient Accumulation:** Implemented through sparse/aggregated gradient storage within the `LegacyLLM_Gradients` structure and efficient resetting mechanisms (`zero_legacy_llm_gradients`). This minimizes persistent memory footprint during training.
*   **Memory-Efficient Activation Handling (Gradient Checkpointing):** Activations required for the backward pass are recomputed on-the-fly rather than stored, significantly reducing RAM usage. This is facilitated by a streamlined `TransformerBlockContext` and adjusted forward/backward pass logic.
*   **Ternary Weight Update Mechanism:** The `apply_ternary_weight_updates` function manages the training of ternary weights. It accumulates standard floating-point gradients and applies a ternary update rule, effectively quantizing weights back to -1, 0, or 1 based on the aggregated gradients.
*   **Functional Training Loop:** The `main.c` file now contains a complete, working training loop. It handles model initialization, gradient management, forward and backward passes, loss calculation, and weight updates over multiple epochs.
*   **Dataset Integration:** The training loop integrates with the `data/tiny_stories_sample.txt` dataset. It includes robust data loading, character-level tokenization, and iterates through the dataset to generate input-target token pairs for sequence prediction.
*   **Test Coverage:** A dedicated test suite (`tests/test_llm.c`) houses various unit tests for mathematical operations, layer functionalities, and full forward/backward passes. These tests are executed automatically upon running the `legacy_llm` executable to verify component correctness.
*   **SSE Optimization:** Core mathematical operations within `math_ops.c` (e.g., vector additions, multiplications, sums, ReLU, Softmax, Layer Normalization, Matrix-Vector products) have been augmented with conditional SSE (Streaming SIMD Extensions) intrinsics (SSE/SSE2). This provides performance benefits on compatible processors, with a transparent fallback to non-SSE implementations if SSE is not enabled during compilation.

## How to Build

The project uses a simple `Makefile`.

### Build without SSE Optimization (Default)
To compile the project without SSE optimizations:

```bash
make clean
make
```

### Build with SSE Optimization
To compile the project with SSE optimizations enabled:

```bash
make clean
make USE_SSE_BUILD=1
```
*(Note: Your GCC compiler might require additional flags like `-march=native` or specific `-msse -msse2` for optimal SSE code generation, though `-msse -msse2` are included in the `Makefile` when `USE_SSE_BUILD=1`.)*

## How to Run

After building, you can run the executable:

```bash
./legacy_llm
```
This will first execute the built-in test suite and then proceed with the training loop on the integrated dataset.

## File Structure

```
.
├── Makefile                     # Project Makefile for compilation
├── Instrucciones.md             # (Possibly original instructions/notes)
├── mi_modelo.py                 # (External Python model, likely for reference)
├── preparar_datos.py            # (External Python script, likely for reference)
├── train.py                     # (External Python script, likely for reference)
├── data/                        # Directory for datasets
│   └── saioa_stories_sample.txt # Sample text dataset used for training
├── include/                     # Header files for the C project
│   ├── backward.h               # Declarations for backward pass functions
│   ├── data_utils.h             # Declarations for data handling functions
│   ├── forward.h                # Declarations for forward pass functions
│   ├── legacy_llm.h             # Main struct definitions and global macros
│   ├── math_ops.h               # Declarations for mathematical operations (SSE conditional)
│   ├── model.h                  # Declarations for model creation and management
│   └── test_llm.h               # Declarations for testing functions
├── src/                         # Source files for the C project
│   ├── backward.c               # Implementations for backward pass functions
│   ├── data_utils.c             # Implementations for data handling functions
│   ├── forward.c                # Implementations for forward pass functions
│   ├── main.c                   # Main training loop and program entry
│   ├── math_ops.c               # Implementations for mathematical operations (SSE conditional)
│   └── model.c                  # Implementations for model creation and management
└── tests/                       # Test files
    └── test_llm.c               # Unit tests for LLM components
```

## Future Enhancements (Pending Features)

The following features are potential areas for future development to further enhance the Legacy-1bit LLM:

*   **Checkpointing (Saving and Loading Model State):** Implement functionality to save the current state of the model (weights, biases) and the optimizer (if applicable) to disk, and to load them to resume training or for inference.
*   **Advanced Training Metrics and Logging:** Integrate more sophisticated metrics beyond basic loss (e.g., perplexity, accuracy if a classification head is added) and a robust logging mechanism for tracking training progress.
*   **Exploring Different Ternary Quantization Schemes:** Investigate alternative methods for quantizing weights to -1, 0, or 1, or even exploring other low-bit quantization schemes (e.g., binary, 2-bit).
*   **Performance Analysis (SSE vs. Non-SSE):** Conduct rigorous benchmarking to quantify the performance benefits of the SSE-optimized code compared to its non-SSE counterpart. This would involve time profiling critical sections.
*   **Improved Dataset Handling:** Enhance the data pipeline to support dynamic batching, larger datasets, more efficient tokenization (e.g., byte-pair encoding for larger vocabularies), and data augmentation.
*   **Inference Mode:** Implement a dedicated inference mode where the trained model can generate text based on a given prompt, without the overhead of training components.
*   **Hyperparameter Tuning:** Systematically experiment with different hyperparameters (learning rate, number of epochs, model dimensions, number of transformer blocks) to optimize model performance.
