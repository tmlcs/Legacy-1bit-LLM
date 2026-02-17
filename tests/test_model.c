#include "test_framework.h"
#include "model.h"
#include "backward.h"
#include <string.h>

// --- Test Functions ---

void test_save_and_load_model() {
    TEST_BEGIN("save_and_load_model");
    
    int vocab_size = 100;
    int model_dim = 64;
    int num_blocks = 2;
    const char* test_file = "/tmp/test_model.bin";
    
    // Create original model
    LegacyLLM* original = create_legacy_llm(vocab_size, model_dim, num_blocks);
    ASSERT_NOT_NULL(original, "Failed to create original model");
    
    // Modify some weights to non-zero values for verification
    original->embedding.embedding_weights.data[0] = 1;
    original->embedding.embedding_weights.data[10] = -1;
    original->transformer_blocks[0].attention.bq[0] = 0.5f;
    original->transformer_blocks[0].attention.bq[5] = -0.3f;
    original->output.bias[0] = 0.1f;
    original->output.bias[50] = -0.2f;
    
    // Save model
    int save_result = save_model(original, test_file);
    ASSERT_TRUE(save_result == 1, "Failed to save model");
    
    // Load model
    LegacyLLM* loaded = load_model(test_file);
    ASSERT_NOT_NULL(loaded, "Failed to load model");
    
    // Verify model structure
    ASSERT_TRUE(loaded->vocab_size == vocab_size, "Loaded model vocab_size mismatch");
    ASSERT_TRUE(loaded->model_dim == model_dim, "Loaded model model_dim mismatch");
    ASSERT_TRUE(loaded->num_transformer_blocks == num_blocks, "Loaded model num_transformer_blocks mismatch");
    
    // Verify weights
    ASSERT_TRUE(loaded->embedding.embedding_weights.data[0] == 1, "Embedding weight[0] mismatch");
    ASSERT_TRUE(loaded->embedding.embedding_weights.data[10] == -1, "Embedding weight[10] mismatch");
    ASSERT_EQUALS_FLOAT(0.5f, loaded->transformer_blocks[0].attention.bq[0], 1e-6f, "Attention bq[0] mismatch");
    ASSERT_EQUALS_FLOAT(-0.3f, loaded->transformer_blocks[0].attention.bq[5], 1e-6f, "Attention bq[5] mismatch");
    ASSERT_EQUALS_FLOAT(0.1f, loaded->output.bias[0], 1e-6f, "Output bias[0] mismatch");
    ASSERT_EQUALS_FLOAT(-0.2f, loaded->output.bias[50], 1e-6f, "Output bias[50] mismatch");
    
    // Cleanup
    free_legacy_llm(original);
    free_legacy_llm(loaded);
    remove(test_file);  // Clean up test file
    
    TEST_END();
}

void test_load_invalid_file() {
    TEST_BEGIN("load_invalid_file");
    
    // Try to load non-existent file
    LegacyLLM* model = load_model("/nonexistent/path/model.bin");
    ASSERT_NULL(model, "Should return NULL for non-existent file");
    
    // Create a file with invalid magic number
    const char* invalid_file = "/tmp/invalid_model.bin";
    FILE* fp = fopen(invalid_file, "wb");
    ASSERT_NOT_NULL(fp, "Failed to create test file");
    
    unsigned int bad_magic = 0xDEADBEEF;
    fwrite(&bad_magic, sizeof(unsigned int), 1, fp);
    fclose(fp);
    
    // Try to load file with invalid magic
    LegacyLLM* model2 = load_model(invalid_file);
    ASSERT_NULL(model2, "Should return NULL for file with invalid magic number");
    
    remove(invalid_file);
    
    TEST_END();
}

void test_create_and_free_ternary_matrix() {
    TEST_BEGIN("create_and_free_ternary_matrix");
    
    int rows = 10;
    int cols = 20;
    
    TernaryMatrix mat = create_ternary_matrix(rows, cols);
    ASSERT_NOT_NULL(mat.data, "Failed to create ternary matrix");
    ASSERT_TRUE(mat.rows == rows, "Matrix rows mismatch");
    ASSERT_TRUE(mat.cols == cols, "Matrix cols mismatch");
    
    // Check that data is initialized (values should be -1, 0, or 1)
    int valid_count = 0;
    for (int i = 0; i < rows * cols; i++) {
        int8_t val = mat.data[i];
        if (val == -1 || val == 0 || val == 1) {
            valid_count++;
        }
    }
    ASSERT_TRUE(valid_count == rows * cols, "All values should be -1, 0, or 1");
    
    free_ternary_matrix(&mat);
    ASSERT_NULL(mat.data, "Data should be NULL after free");
    
    TEST_END();
}

void test_create_and_free_float_array() {
    TEST_BEGIN("create_and_free_float_array");
    
    int size = 100;
    float* arr = create_float_array(size);
    ASSERT_NOT_NULL(arr, "Failed to create float array");
    
    // All values should be initialized to small random values between -0.01 and 0.01
    int in_range_count = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] >= -0.01f && arr[i] <= 0.01f) {
            in_range_count++;
        }
    }
    ASSERT_TRUE(in_range_count == size, "All values should be initialized to small random values in range [-0.01, 0.01]");
    
    free_float_array(arr);
    
    TEST_END();
}

void test_apply_ternary_weight_updates() {
    TEST_BEGIN("apply_ternary_weight_updates");
    
    int vocab_size = 10;
    int model_dim = 8;
    int num_blocks = 1;
    float learning_rate = 0.1f;
    
    // Create model and gradients
    LegacyLLM* model = create_legacy_llm(vocab_size, model_dim, num_blocks);
    LegacyLLM_Gradients* grads = create_legacy_llm_gradients(vocab_size, model_dim, num_blocks);
    ASSERT_NOT_NULL(model, "Failed to create model");
    ASSERT_NOT_NULL(grads, "Failed to create gradients");
    
    // Set initial weight
    model->embedding.embedding_weights.data[0] = 0;
    
    // Set positive gradient (should push weight toward 1)
    grads->embedding_grads.embedding_weights[0] = 10.0f;
    apply_ternary_weight_updates(model, grads, learning_rate);
    ASSERT_TRUE(model->embedding.embedding_weights.data[0] == 1, "Positive gradient should push weight to 1");
    
    // Reset weight
    model->embedding.embedding_weights.data[0] = 0;
    
    // Set negative gradient (should push weight toward -1)
    grads->embedding_grads.embedding_weights[0] = -10.0f;
    apply_ternary_weight_updates(model, grads, learning_rate);
    ASSERT_TRUE(model->embedding.embedding_weights.data[0] == -1, "Negative gradient should push weight to -1");
    
    // Reset weight
    model->embedding.embedding_weights.data[0] = 1;
    
    // Set zero gradient (weight should stay the same)
    grads->embedding_grads.embedding_weights[0] = 0.0f;
    apply_ternary_weight_updates(model, grads, learning_rate);
    ASSERT_TRUE(model->embedding.embedding_weights.data[0] == 1, "Zero gradient should not change weight");
    
    // Test bias update (should use standard SGD)
    float initial_bias = model->transformer_blocks[0].attention.bq[0];
    grads->transformer_block_grads[0].attention_grads.bq[0] = 1.0f;
    apply_ternary_weight_updates(model, grads, learning_rate);
    float expected_bias = initial_bias - learning_rate * 1.0f;
    ASSERT_EQUALS_FLOAT(expected_bias, model->transformer_blocks[0].attention.bq[0], 1e-6f, "Bias update incorrect");
    
    free_legacy_llm(model);
    free_legacy_llm_gradients(grads);
    
    TEST_END();
}

void test_create_and_free_legacy_llm() {
    TEST_BEGIN("create_and_free_legacy_llm");
    
    int vocab_size = 50;
    int model_dim = 32;
    int num_blocks = 2;
    
    LegacyLLM* model = create_legacy_llm(vocab_size, model_dim, num_blocks);
    ASSERT_NOT_NULL(model, "Failed to create model");
    ASSERT_TRUE(model->vocab_size == vocab_size, "Model vocab_size mismatch");
    ASSERT_TRUE(model->model_dim == model_dim, "Model model_dim mismatch");
    ASSERT_TRUE(model->num_transformer_blocks == num_blocks, "Model num_transformer_blocks mismatch");
    
    // Check that all blocks are allocated
    ASSERT_NOT_NULL(model->transformer_blocks, "Transformer blocks should be allocated");
    ASSERT_NOT_NULL(model->block_contexts, "Block contexts should be allocated");
    
    // Check embedding layer
    ASSERT_NOT_NULL(model->embedding.embedding_weights.data, "Embedding weights should be allocated");
    ASSERT_TRUE(model->embedding.embedding_weights.rows == vocab_size, "Embedding weights rows mismatch");
    ASSERT_TRUE(model->embedding.embedding_weights.cols == model_dim, "Embedding weights cols mismatch");
    
    // Check output layer
    ASSERT_NOT_NULL(model->output.unembedding_weights.data, "Output weights should be allocated");
    ASSERT_NOT_NULL(model->output.bias, "Output bias should be allocated");
    
    free_legacy_llm(model);
    
    TEST_END();
}

// Function to run all model tests
void run_model_tests() {
    printf("--- Running Model Tests ---\n");
    test_save_and_load_model();
    test_load_invalid_file();
    test_create_and_free_ternary_matrix();
    test_create_and_free_float_array();
    test_apply_ternary_weight_updates();
    test_create_and_free_legacy_llm();
}
