#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, rand
#include <math.h>    // For sqrtf, expf, logf

#include "legacy_llm.h"
#include "data_utils.h"
#include "model.h"
#include "math_ops.h"
#include "forward.h"
#include "backward.h"
#include "test_llm.h" // Include test functions

#define LEARNING_RATE 0.01f
#define NUM_EPOCHS 10
#define CHECKPOINT_FILE "llm_model.bin"
#define SAVE_INTERVAL 2 // Save every 2 epochs
// BATCH_SIZE will be implicitly 1 for now, as we process token by token

int main() {
    printf("Legacy-1bit LLM Project\n");

    START_TIMER(total_training_time); // Add this line

    // Optionally run tests
    printf("Running tests...\n");
    run_all_llm_tests();
    printf("Tests finished.\n");

    printf("\nStarting training loop...\n");

    srand(time(NULL));

    int vocab_size = MAX_VOCAB_SIZE; // Use the defined vocab size

    // --- Data Loading and Tokenization ---
    const char* filepath = "data/saioa_stories_sample.txt"; // This file should exist
    char* text_content = load_text_from_file(filepath);

    if (text_content == NULL) {
        fprintf(stderr, "Error: Could not load text from file '%s'.\n", filepath);
        return 1;
    }

    int* tokens = NULL;
    int total_tokens = 0;
    tokens = tokenize_text(text_content, vocab_size, &total_tokens);

    if (tokens == NULL || total_tokens == 0) {
        fprintf(stderr, "Error: Could not tokenize text or text is empty.\n");
        free_text(text_content);
        return 1;
    }
    printf("Loaded %d tokens for training.\n", total_tokens);
    // Ensure we have at least one pair (input, target)
    if (total_tokens < 2) {
        fprintf(stderr, "Error: Not enough tokens for training pairs.\n");
        free_tokens(tokens);
        free_text(text_content);
        return 1;
    }


    // 1. Initialize Model and Gradients
    int num_transformer_blocks = 2; // Keep it small for 2000 laptop
    LegacyLLM* model = NULL;
    
    // Attempt to load model from checkpoint
    printf("Attempting to load model from %s...\n", CHECKPOINT_FILE);
    model = load_model(CHECKPOINT_FILE);

    if (model) {
        printf("Model loaded successfully from checkpoint. Resuming training.\n");
        // Ensure model dimensions match expectations if necessary, or just proceed
        // For simplicity, we assume loaded model matches expected structure.
    } else {
        printf("No checkpoint found or failed to load. Creating new model.\n");
        model = create_legacy_llm(vocab_size, MODEL_DIM, num_transformer_blocks);
        if (!model) {
            fprintf(stderr, "Error: Failed to create LLM model.\n");
            free_tokens(tokens);
            free_text(text_content);
            return 1;
        }
    }
    

    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, MODEL_DIM, num_transformer_blocks);
    if (!grads) {
        fprintf(stderr, "Error: Failed to create LLM gradients.\n");
        free_legacy_llm(model);
        free_tokens(tokens);
        free_text(text_content);
        return 1;
    }

    // Training Loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        printf("Epoch %d/%d\n", epoch + 1, NUM_EPOCHS);
        float epoch_loss = 0.0f;
        int num_batches = total_tokens - 1; // Number of input-target pairs

        for (int i = 0; i < num_batches; ++i) {
            int input_token = tokens[i];
            int target_token = tokens[i+1];

            // Zero gradients before accumulating for the current sample
            zero_legacy_llm_gradients(grads);

            // Forward pass
            float* predicted_probs = forward_llm(model, input_token);
            if (!predicted_probs) {
                fprintf(stderr, "Error: Forward pass failed for token %d.\n", i);
                epoch_loss = -1.0f; // Indicate error
                break;
            }

            // Calculate loss
            float loss = cross_entropy_loss(predicted_probs, target_token, vocab_size);
            epoch_loss += loss;
            free_float_array(predicted_probs);

            // Backward pass
            backward_llm(model, input_token, target_token, grads);

            // Apply weight updates
            apply_ternary_weight_updates(model, grads, LEARNING_RATE);
        }

        if (epoch_loss == -1.0f) { // Check if an error occurred in the inner loop
            fprintf(stderr, "Training interrupted due to an error.\n");
            break;
        }
        float avg_epoch_loss = epoch_loss / num_batches;
        float perplexity = expf(avg_epoch_loss);
        printf("  Average Epoch Loss: %.4f, Perplexity: %.4f\n", avg_epoch_loss, perplexity);

        // Save checkpoint
        if ((epoch + 1) % SAVE_INTERVAL == 0 || epoch == NUM_EPOCHS - 1) {
            printf("Saving model checkpoint to %s...\n", CHECKPOINT_FILE);
            if (save_model(model, CHECKPOINT_FILE)) {
                printf("Checkpoint saved successfully.\n");
            } else {
                fprintf(stderr, "Error: Failed to save model checkpoint.\n");
            }
        }
    }

    printf("Training loop finished.\n");

    STOP_TIMER(total_training_time, "Total Training Time"); // Add this line

    // Clean up
    free_legacy_llm_gradients(grads);
    free_legacy_llm(model);
    free_tokens(tokens);
    free_text(text_content);

    return 0;
}