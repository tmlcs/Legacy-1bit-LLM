#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include necessary project headers
#include "config.h"
// #include "logger.h" // The generic log_info/log_error are not in the logger API
#include "hyperparams.h"
#include "data_utils.h" // For load_text_from_file, tokenize_text, etc.
#include "model.h"
#include "legacy_llm.h"
#include "backward.h" // For backward_llm and gradients
#include "forward.h"  // For forward_llm

/**
 * @brief Main entry point for the memory test executable.
 *
 * This program initializes a model and its gradients, loads a dataset completely
 * into memory, runs a brief training loop for 5 steps, and then meticulously
 * cleans up all allocated resources. Its primary purpose is to be run under a
 * memory profiler like Valgrind to detect memory leaks by mimicking the exact
 * data flow of the main application.
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_data>\n", argv[0]);
        fprintf(stderr, "This test requires a text data file (e.g., data/saioa_stories_sample.txt) to run.\n");
        return 1;
    }

    char* data_path = argv[1];
    int num_steps = 5; // A small number of steps for a quick memory check

    printf("--- Memory Test Suite (V3) ---\n");

    // --- Data Loading and Tokenization ---
    printf("Loading text from %s...\n", data_path);
    char* text_content = load_text_from_file(data_path);
    if (text_content == NULL) {
        fprintf(stderr, "Error: Could not load text from file '%s'.\n", data_path);
        return 1;
    }

    int* tokens = NULL;
    int total_tokens = 0;
    // Assuming a fixed vocab size as seen in main.c
    int vocab_size = MAX_VOCAB_SIZE;
    tokens = tokenize_text(text_content, vocab_size, &total_tokens);

    if (tokens == NULL || total_tokens <= num_steps) {
        fprintf(stderr, "Error: Could not tokenize text or not enough tokens for %d steps.\n", num_steps);
        free_text(text_content);
        return 1;
    }
    printf("Loaded %d tokens for training.\n", total_tokens);

    // --- Model and Gradients Initialization ---
    Hyperparameters params;
    hyperparams_init_defaults(&params);
    
    LegacyLLM* model = create_legacy_llm(vocab_size, params.model_dim, params.num_transformer_blocks);
    if (!model) {
        fprintf(stderr, "Error: Failed to create LLM model.\n");
        free_tokens(tokens);
        free_text(text_content);
        return 1;
    }
    printf("Created LLM model.\n");

    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, params.model_dim, params.num_transformer_blocks);
    if (!grads) {
        fprintf(stderr, "Error: Failed to create LLM gradients.\n");
        free_legacy_llm(model);
        free_tokens(tokens);
        free_text(text_content);
        return 1;
    }
    printf("Created LLM gradients.\n");

    // --- Short Training Loop ---
    printf("Starting training loop for %d steps...\n", num_steps);
    for (int i = 0; i < num_steps; ++i) {
        int input_token = tokens[i];
        int target_token = tokens[i+1];

        zero_legacy_llm_gradients(grads);

        // Forward and backward pass
        float* predicted_probs = forward_llm(model, input_token);
        // In a real scenario, you'd calculate loss here. For memory test, not needed.
        if (predicted_probs) {
            free_float_array(predicted_probs); // IMPORTANT: forward_llm allocates memory
        }
        
        backward_llm(model, input_token, target_token, grads);
        apply_ternary_weight_updates(model, grads, params.learning_rate);
    }
    printf("Training loop finished.\n");

    // --- CRITICAL STEP: Cleanup ---
    printf("Cleaning up resources...\n");
    free_legacy_llm_gradients(grads);
    printf("Freed LLM gradients.\n");
    free_legacy_llm(model);
    printf("Freed LLM model.\n");
    free_tokens(tokens);
    printf("Freed tokens.\n");
    free_text(text_content);
    printf("Freed text content.\n");

    printf("--- Memory Test SUCCEEDED ---\n");
    return 0;
}