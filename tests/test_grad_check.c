#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> // For srand

#include "config.h"
#include "legacy_llm.h"
#include "forward.h"
#include "backward.h"
#include "model.h" // For create_legacy_llm, free_legacy_llm, create_legacy_llm_gradients, free_legacy_llm_gradients, zero_legacy_llm_gradients
#include "math_ops.h" // For cross_entropy_loss, free_float_array, etc.

#define GRAD_CHECK_EPSILON 1e-4f // Adjusted for potential numerical instability
#define GRAD_CHECK_THRESHOLD 1e-3f // Allow a bit more leeway for complex model with ternary weights

// Helper function to compute loss for a given model state
float compute_loss(LegacyLLM* model, int input_token, int target_token) {
    float* probs = forward_llm(model, input_token);
    if (!probs) {
        fprintf(stderr, "DEBUG(compute_loss): forward_llm failed.\n");
        return -1.0f;
    }

    fprintf(stderr, "DEBUG(compute_loss): input_token=%d, target_token=%d\n", input_token, target_token);
    fprintf(stderr, "DEBUG(compute_loss): Predicted Probs for true_token_id (%d): %.8f\n", target_token, probs[target_token]);
    for(int k=0; k<model->vocab_size; ++k) {
        fprintf(stderr, "DEBUG(compute_loss): probs[%d] = %.8f\n", k, probs[k]);
    }

    float loss = cross_entropy_loss(probs, target_token, model->vocab_size);
    fprintf(stderr, "DEBUG(compute_loss): Computed Loss: %.8f\n", loss);
    
    free_float_array(probs);
    return loss;
}

// Function to compare analytical and numerical gradients for a float array
int check_float_array_gradients(float* weights_array, float* gradients_array, int size, const char* array_name, LegacyLLM* model, int input_token, int target_token) {
    printf("--- Checking gradients for %s (size %d) ---\n", array_name, size);
    int status = 1; // Success
    float relative_error;

    for (int i = 0; i < size; ++i) {
        float original_weight = weights_array[i];

        // Calculate loss for (weight + epsilon)
        weights_array[i] = original_weight + GRAD_CHECK_EPSILON;
        fprintf(stderr, "DEBUG(check_grad): Perturbing %s[%d] to %.8f (original %.8f)\n", array_name, i, weights_array[i], original_weight);
        float loss1 = compute_loss(model, input_token, target_token);
        fprintf(stderr, "DEBUG(check_grad): loss1 (w+eps) = %.8f\n", loss1);

        // Calculate loss for (weight - epsilon)
        weights_array[i] = original_weight - GRAD_CHECK_EPSILON;
        fprintf(stderr, "DEBUG(check_grad): Perturbing %s[%d] to %.8f (original %.8f)\n", array_name, i, weights_array[i], original_weight);
        float loss2 = compute_loss(model, input_token, target_token);
        fprintf(stderr, "DEBUG(check_grad): loss2 (w-eps) = %.8f\n", loss2);

        // Restore original weight
        weights_array[i] = original_weight;

        if (loss1 < 0 || loss2 < 0) {
            fprintf(stderr, "Error computing loss during grad check for %s[%d].\n", array_name, i);
            return 0; // Failure
        }

        // Compute numerical gradient
        float numerical_grad = (loss1 - loss2) / (2.0f * GRAD_CHECK_EPSILON);
        
        // Get analytical gradient
        float analytical_grad = gradients_array[i];

        // Compare gradients
        float numerator = fabsf(analytical_grad - numerical_grad);
        float denominator = fmaxf(fabsf(analytical_grad), fabsf(numerical_grad));
        if (denominator < 1e-8) { // Handle case where both grads are near zero
            if (numerator < 1e-8) { // Both very small, consider them equal
                relative_error = 0.0f;
            } else { // One is small, other is not, leads to large error
                denominator = 1.0f; 
                relative_error = numerator / denominator;
            }
        } else {
            relative_error = numerator / denominator;
        }


        fprintf(stderr, "DEBUG(check_grad): %s[%d]: an=%.8f, num=%.8f, rel_err=%.8f\n", array_name, i, analytical_grad, numerical_grad, relative_error);


        if (relative_error > GRAD_CHECK_THRESHOLD) {
            if (status) { // Print header only on first failure for this array
                printf("FAILURES (analytical vs. numerical) [rel_error > %.5f]:\n", GRAD_CHECK_THRESHOLD);
            }
            printf("  [%d]: an=%.6f, num=%.6f, err=%.6f\n", i, analytical_grad, numerical_grad, relative_error);
            status = 0; // Failure
        }
    }
    if (status) {
        printf("OK\n");
    }
    return status;
}


int main() {
    // Seed random for reproducibility
    srand(42); // Use a fixed seed for consistent test results

    printf("--- Gradient Checking Test (V3.5 - Output Layer Debug) ---\n");

    // Use a tiny model for feasibility
    int vocab_size = 2; // Smallest possible vocab
    int model_dim = 1;  // Smallest possible model_dim
    int num_blocks = 0; // No transformer blocks, focus on output layer
    int input_token = 0; // Only one input token (must be < vocab_size)
    int target_token = 1; // Target is the other token (must be < vocab_size)

    LegacyLLM* model = create_legacy_llm(vocab_size, model_dim, num_blocks);
    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, model_dim, num_blocks);
    if (!model || !grads) {
        fprintf(stderr, "Failed to create model or gradients.\n");
        return 1;
    }

    // Force all ternary weights to zero to isolate bias gradients
    for(int i=0; i<model->embedding.embedding_weights.rows * model->embedding.embedding_weights.cols; ++i) {
        model->embedding.embedding_weights.data[i] = 0;
    }
    // If model_dim is 1, and vocab_size is 2, this is 2x1 matrix
    for(int i=0; i<model->output.unembedding_weights.rows * model->output.unembedding_weights.cols; ++i) {
        model->output.unembedding_weights.data[i] = 0;
    }


    // 1. Calculate analytical gradients first
    printf("Calculating analytical gradients...\n");
    zero_legacy_llm_gradients(grads);
    float* analytical_probs = forward_llm(model, input_token);
    if (!analytical_probs) {
        fprintf(stderr, "Error during analytical forward pass for gradient check.\n");
        free_legacy_llm_gradients(grads);
        free_legacy_llm(model);
        return 1;
    }
    free_float_array(analytical_probs);
    backward_llm(model, input_token, target_token, grads);
    printf("Analytical gradients calculated.\n\n");

    // 2. Perform gradient checking for continuous parameters (biases, layer norm params)
    int final_status = 1;

    // Check final output layer bias
    final_status &= check_float_array_gradients(model->output.bias, grads->output_grads.bias, vocab_size, "Output Layer Bias", model, input_token, target_token);

    // --- Final Result ---
    printf("\n--------------------------------\n");
    if (final_status) {
        printf("Gradient Check PASSED!\n");
    } else {
        printf("Gradient Check FAILED!\n");
    }
    printf("--------------------------------\n");

    // Cleanup
    free_legacy_llm_gradients(grads);
    free_legacy_llm(model);

    return final_status ? 0 : 1;
}