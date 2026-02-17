# Project Action Plan: Legacy-1bit LLM

This action plan outlines the next steps for improving the "Legacy-1bit LLM" project, based on the recent quality audit conducted on February 16, 2026.

## 1. Code Quality Improvements

*   **Critical Bug Fix in `src/model.c`:**
    *   **Description:** In `create_legacy_llm`, within the loop for transformer blocks, the allocation check for `model->transformer_blocks[i].ffn.bo` incorrectly uses `if (!model->transformer_blocks[i].attention.bo)`.
    *   **Action:** Change `if (!model->transformer_blocks[i].attention.bo)` to `if (!model->transformer_blocks[i].ffn.bo)`. This ensures that a failed allocation for the FFN output bias is correctly caught, preventing potential `NULL` dereferences.
*   **Minor Improvement in `src/model.c` (`srand` seeding):**
    *   **Description:** While `main.c` correctly seeds the random number generator, `model.c` (specifically `initialize_ternary_data` and `create_float_array`) relies on this global seeding. For stronger independence and testability of `model.c` functions in isolation, `srand(time(NULL))` could be called once within `create_legacy_llm` if it's the primary entry point for model creation. However, as `main.c` handles it, this is a very minor point.
    *   **Action:** No immediate action required, but keep in mind for future refactoring or if `model.c` functions are ever used standalone without `main.c`'s initialization.
*   **Minor Improvement in `src/math_ops.c` (SSE `int8_t` to `float` conversion):**
    *   **Description:** In SSE-optimized `ternary_matrix_vector_mul` and `matrix_transpose_vector_mul`, `int8_t` weights are converted to `float` individually. This can be a minor bottleneck.
    *   **Action:** Investigate using more advanced SSE intrinsics (e.g., `_mm_cvtepi8_epi32` followed by `_mm_cvtepi32_ps` if SSE4.1 is universally available for target systems) to vectorize this conversion for potential performance gains. This would require careful consideration of target CPU capabilities.

## 2. Testing Strategy Enhancements

The current test suite is strong for mathematical operations and forward passes but has significant gaps in other critical areas.

*   **Implement Backward Pass Tests:**
    *   **Description:** Create dedicated unit tests for all functions in `src/backward.c`. This is crucial for verifying the correctness of gradient calculations, which are fundamental to the LLM's training process.
    *   **Action:** Develop new test functions in `tests/test_backward.c` (or extend `tests/test_llm.c`) that provide specific inputs to backward pass functions and assert against manually calculated or known correct gradients. This will likely involve setting up dummy model parameters and activations similar to `tests/test_forward.c`.
*   **Add Gradient Management Tests:**
    *   **Description:** Implement unit tests for `zero_legacy_llm_gradients` and `apply_ternary_weight_updates` in `src/model.c`.
    *   **Action:** Create tests that verify `zero_legacy_llm_gradients` correctly sets all gradient arrays to zero. For `apply_ternary_weight_updates`, test with various gradient values (positive, negative, zero) and ensure the ternary weights (and float biases/LayerNorm parameters) are updated according to the specified rules.
*   **Implement Model Persistence Tests:**
    *   **Description:** Add tests for `save_model` and `load_model` functions in `src/model.c`.
    *   **Action:** Create a model, save it to a temporary file, load it back, and assert that the loaded model's parameters (weights, biases, LayerNorm params) are identical to the original saved model. Also, test error cases (e.g., corrupted file, invalid magic number).
*   **Dedicated Ternary Matrix Operation Tests:**
    *   **Description:** While indirectly covered, adding direct unit tests for `ternary_matrix_vector_mul` and `matrix_transpose_vector_mul` in `src/math_ops.c` would provide more isolated verification.
    *   **Action:** Create simple `TernaryMatrix` instances and `float` vectors with known values, perform the multiplication, and assert the output against manually calculated results. This would complement existing `math_ops` tests.
*   **Consolidate Test Framework Helper:**
    *   **Description:** The `compare_float_arrays` helper function is currently duplicated in `tests/test_math_ops.c` and `tests/test_forward.c`.
    *   **Action:** Move `compare_float_arrays` into `include/test_framework.h` to make it universally available and avoid duplication.

## 3. Future Enhancements

These are features and improvements identified in the project documentation (`README.md`, `ARCHITECTURE.md`) as potential areas for future development.

*   **Advanced Training Metrics and Logging:** Integrate more sophisticated metrics beyond basic loss (e.g., perplexity, accuracy if a classification head is added) and a robust logging mechanism for tracking training progress.
*   **Exploring Different Ternary Quantization Schemes:** Investigate alternative methods for quantizing weights to -1, 0, or 1, or even exploring other low-bit quantization schemes (e.g., binary, 2-bit).
*   **Performance Analysis (SSE vs. Non-SSE):** Conduct rigorous benchmarking to quantify the performance benefits of the SSE-optimized code compared to its non-SSE counterpart. This would involve time profiling critical sections using the existing `MEASURE_PERFORMANCE` framework.
*   **Improved Dataset Handling:** Enhance the data pipeline to support dynamic batching, larger datasets, more efficient tokenization (e.g., byte-pair encoding for larger vocabularies), and data augmentation.
*   **Inference Mode:** Implement a dedicated inference mode where the trained model can generate text based on a given prompt, without the overhead of training components.
*   **Hyperparameter Tuning:** Systematically experiment with different hyperparameters (learning rate, number of epochs, model dimensions, number of transformer blocks) to optimize model performance.
*   **Advanced Quantization (Implicit from Architecture):** Explore more sophisticated quantization-aware training techniques.
*   **Model Capacity (Implicit from Architecture):** Investigate ways to improve the model's representational power within the ternary constraint.
*   **Deployment (Implicit from Architecture):** Consider integration with specialized hardware for efficient inference.