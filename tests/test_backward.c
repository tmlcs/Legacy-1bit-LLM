#include "test_framework.h"
#include "backward.h"
#include "model.h"
#include "forward.h"
#include <float.h>

// --- Test Functions ---

void test_cross_entropy_loss() {
    TEST_BEGIN("cross_entropy_loss");
    
    // Test with known probabilities
    float probs[] = {0.1f, 0.7f, 0.2f};
    float loss = cross_entropy_loss(probs, 1, 3);  // True class is index 1 with prob 0.7
    float expected_loss = -logf(0.7f);
    ASSERT_EQUALS_FLOAT(expected_loss, loss, 1e-5f, "Cross-entropy loss calculation incorrect");
    
    // Test with very small probability (clamping)
    float probs2[] = {0.0f, 0.5f, 0.5f};
    float loss2 = cross_entropy_loss(probs2, 0, 3);  // True class has prob 0.0
    // Should be clamped to 1e-9f
    float expected_loss2 = -logf(1e-9f);
    ASSERT_EQUALS_FLOAT(expected_loss2, loss2, 1e-5f, "Cross-entropy loss with clamping incorrect");
    
    // Test with invalid input
    float loss3 = cross_entropy_loss(NULL, 0, 3);
    ASSERT_EQUALS_FLOAT(-1.0f, loss3, 0.0f, "Should return -1.0 for NULL input");
    
    float loss4 = cross_entropy_loss(probs, -1, 3);
    ASSERT_EQUALS_FLOAT(-1.0f, loss4, 0.0f, "Should return -1.0 for invalid token_id");
    
    float loss5 = cross_entropy_loss(probs, 5, 3);
    ASSERT_EQUALS_FLOAT(-1.0f, loss5, 0.0f, "Should return -1.0 for out-of-bounds token_id");
    
    TEST_END();
}

void test_d_loss_d_logits() {
    TEST_BEGIN("d_loss_d_logits");
    
    float probs[] = {0.1f, 0.7f, 0.2f};
    float* grad = d_loss_d_logits(probs, 1, 3);  // True class is index 1
    
    ASSERT_NOT_NULL(grad, "Gradient should not be NULL");
    ASSERT_EQUALS_FLOAT(0.1f, grad[0], 1e-6f, "Gradient[0] incorrect");
    ASSERT_EQUALS_FLOAT(-0.3f, grad[1], 1e-6f, "Gradient[1] should be 0.7-1.0 = -0.3");
    ASSERT_EQUALS_FLOAT(0.2f, grad[2], 1e-6f, "Gradient[2] incorrect");
    
    free_float_array(grad);
    
    // Test with invalid input
    float* grad2 = d_loss_d_logits(NULL, 0, 3);
    ASSERT_NULL(grad2, "Should return NULL for NULL input");
    
    float* grad3 = d_loss_d_logits(probs, -1, 3);
    ASSERT_NULL(grad3, "Should return NULL for invalid token_id");
    
    TEST_END();
}

void test_backward_embedding_batch() {
    TEST_BEGIN("backward_embedding_batch");
    
    int vocab_size = 10;
    int model_dim = 4;
    int batch_size = 2;
    
    // Create embedding layer
    EmbeddingLayer layer;
    layer.embedding_weights = create_ternary_matrix(vocab_size, model_dim);
    ASSERT_NOT_NULL(layer.embedding_weights.data, "Failed to create embedding weights");
    
    // Initialize gradients
    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, model_dim, 0);
    ASSERT_NOT_NULL(grads, "Failed to create gradients");
    
    // Zero gradients first
    zero_legacy_llm_gradients(grads);
    
    // Create input batch
    int input_batch[] = {3, 7};
    
    // Create gradient from output (batch_size * model_dim)
    float d_loss_d_embedding[] = {
        0.1f, 0.2f, 0.3f, 0.4f,  // Gradients for token 3
        0.5f, 0.6f, 0.7f, 0.8f   // Gradients for token 7
    };
    
    // Call backward
    backward_embedding_batch(&layer, input_batch, d_loss_d_embedding, batch_size, model_dim, grads);
    
    // Check that gradients were accumulated at the correct positions
    // For token 3, gradients should be at positions 3*model_dim to 3*model_dim+3
    int idx3 = 3 * model_dim;
    ASSERT_EQUALS_FLOAT(0.1f, grads->embedding_grads.embedding_weights[idx3], 1e-6f, "Gradient for token 3[0] incorrect");
    ASSERT_EQUALS_FLOAT(0.2f, grads->embedding_grads.embedding_weights[idx3 + 1], 1e-6f, "Gradient for token 3[1] incorrect");
    ASSERT_EQUALS_FLOAT(0.3f, grads->embedding_grads.embedding_weights[idx3 + 2], 1e-6f, "Gradient for token 3[2] incorrect");
    ASSERT_EQUALS_FLOAT(0.4f, grads->embedding_grads.embedding_weights[idx3 + 3], 1e-6f, "Gradient for token 3[3] incorrect");
    
    // For token 7, gradients should be at positions 7*model_dim to 7*model_dim+3
    int idx7 = 7 * model_dim;
    ASSERT_EQUALS_FLOAT(0.5f, grads->embedding_grads.embedding_weights[idx7], 1e-6f, "Gradient for token 7[0] incorrect");
    ASSERT_EQUALS_FLOAT(0.6f, grads->embedding_grads.embedding_weights[idx7 + 1], 1e-6f, "Gradient for token 7[1] incorrect");
    ASSERT_EQUALS_FLOAT(0.7f, grads->embedding_grads.embedding_weights[idx7 + 2], 1e-6f, "Gradient for token 7[2] incorrect");
    ASSERT_EQUALS_FLOAT(0.8f, grads->embedding_grads.embedding_weights[idx7 + 3], 1e-6f, "Gradient for token 7[3] incorrect");
    
    // Cleanup
    free_ternary_matrix(&layer.embedding_weights);
    free_legacy_llm_gradients(grads);
    
    TEST_END();
}

void test_zero_legacy_llm_gradients() {
    TEST_BEGIN("zero_legacy_llm_gradients");
    
    int vocab_size = 10;
    int model_dim = 8;
    int num_blocks = 2;
    
    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, model_dim, num_blocks);
    ASSERT_NOT_NULL(grads, "Failed to create gradients");
    
    // Set some non-zero values
    grads->embedding_grads.embedding_weights[0] = 1.0f;
    grads->embedding_grads.embedding_weights[5] = 2.0f;
    grads->output_grads.bias[0] = 3.0f;
    grads->transformer_block_grads[0].attention_grads.bq[0] = 4.0f;
    grads->transformer_block_grads[1].ffn_grads.bi[0] = 5.0f;
    
    // Zero gradients
    zero_legacy_llm_gradients(grads);
    
    // Check all are zero
    ASSERT_EQUALS_FLOAT(0.0f, grads->embedding_grads.embedding_weights[0], 1e-10f, "Embedding gradient should be zero");
    ASSERT_EQUALS_FLOAT(0.0f, grads->embedding_grads.embedding_weights[5], 1e-10f, "Embedding gradient should be zero");
    ASSERT_EQUALS_FLOAT(0.0f, grads->output_grads.bias[0], 1e-10f, "Output bias gradient should be zero");
    ASSERT_EQUALS_FLOAT(0.0f, grads->transformer_block_grads[0].attention_grads.bq[0], 1e-10f, "Attention gradient should be zero");
    ASSERT_EQUALS_FLOAT(0.0f, grads->transformer_block_grads[1].ffn_grads.bi[0], 1e-10f, "FFN gradient should be zero");
    
    // Check a few more random positions
    ASSERT_EQUALS_FLOAT(0.0f, grads->transformer_block_grads[0].ffn_grads.Wo[10], 1e-10f, "FFN Wo gradient should be zero");
    ASSERT_EQUALS_FLOAT(0.0f, grads->output_grads.unembedding_weights[20], 1e-10f, "Unembedding gradient should be zero");
    
    free_legacy_llm_gradients(grads);
    
    TEST_END();
}

void test_backward_layer_norm_batch() {
    TEST_BEGIN("backward_layer_norm_batch");
    
    int batch_size = 2;
    int model_dim = 4;
    int block_idx = 0;
    int norm_idx = 1;  // norm1
    
    // Create gradients structure
    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(10, model_dim, 1);
    ASSERT_NOT_NULL(grads, "Failed to create gradients");
    zero_legacy_llm_gradients(grads);
    
    // Create input data
    float d_loss_d_norm_output[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f
    };
    
    float input_batch[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float mean_batch[] = {2.5f, 6.5f};
    float inv_std_dev_batch[] = {1.0f, 1.0f};
    
    // Call backward
    float* d_loss_d_input = backward_layer_norm_batch(
        d_loss_d_norm_output, input_batch, gamma, beta,
        mean_batch, inv_std_dev_batch, batch_size, model_dim,
        block_idx, norm_idx, grads
    );
    
    ASSERT_NOT_NULL(d_loss_d_input, "Should return gradient w.r.t. input");
    
    // Verify gradients for gamma and beta were accumulated
    // Note: The exact calculation depends on the implementation details of backward_layer_norm_batch
    // Here we just verify that gradients are non-zero (indicating accumulation happened)
    ASSERT_TRUE(grads->transformer_block_grads[block_idx].norm1_gamma[0] != 0.0f,
                "Gamma gradient[0] should be non-zero");
    ASSERT_TRUE(grads->transformer_block_grads[block_idx].norm1_beta[0] != 0.0f,
                "Beta gradient[0] should be non-zero");
    
    free_float_array(d_loss_d_input);
    free_legacy_llm_gradients(grads);
    
    TEST_END();
}

// Function to run all backward tests
void run_backward_tests() {
    printf("--- Running Backward Pass Tests ---\n");
    test_cross_entropy_loss();
    test_d_loss_d_logits();
    test_backward_embedding_batch();
    test_zero_legacy_llm_gradients();
    test_backward_layer_norm_batch();
}
