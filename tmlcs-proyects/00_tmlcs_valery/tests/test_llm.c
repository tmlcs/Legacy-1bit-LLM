#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, rand
#include <math.h>    // For fabs

#include "test_llm.h"
#include "legacy_llm.h"
#include "data_utils.h"
#include "model.h"
#include "math_ops.h"
#include "forward.h"
#include "backward.h"

// --- Custom Test Framework Macros ---

static unsigned int tests_run = 0;
static unsigned int tests_passed = 0;
static unsigned int tests_failed = 0;
static unsigned int assertions_run = 0;
static unsigned int assertions_failed = 0;

#define RESET_ASSERTION_COUNTERS() assertions_run = 0; assertions_failed = 0;

#define TEST_BEGIN(name) \
    do { \
        tests_run++; \
        RESET_ASSERTION_COUNTERS(); \
        printf("--- Running Test: %s\n", name);

#define TEST_END() \
        if (assertions_failed > 0) { \
            tests_failed++; \
            printf("--- Test FAILED: %s (Assertions Failed: %u)\n\n", __func__, assertions_failed); \
        } else { \
            tests_passed++; \
            printf("--- Test PASSED: %s\n\n", __func__); \
        } \
    } while (0)

#define ASSERT_TRUE(condition, message, ...) \
    do { \
        assertions_run++; \
        if (!(condition)) { \
            assertions_failed++; \
            fprintf(stderr, "    ASSERT FAILED: %s:%d: " message "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define ASSERT_FALSE(condition, message, ...) ASSERT_TRUE(!(condition), message, ##__VA_ARGS__)

#define ASSERT_EQUALS_FLOAT(expected, actual, epsilon, message, ...) \
    do { \
        assertions_run++; \
        if (fabs((expected) - (actual)) > (epsilon)) { \
            assertions_failed++; \
            fprintf(stderr, "    ASSERT FAILED: %s:%d: Expected %.6f, got %.6f. " message "\n", __FILE__, __LINE__, (double)expected, (double)actual, ##__VA_ARGS__); \
        } \
    } while (0)

#define ASSERT_NOT_NULL(ptr, message, ...) ASSERT_TRUE((ptr) != NULL, message, ##__VA_ARGS__)
#define ASSERT_NULL(ptr, message, ...) ASSERT_TRUE((ptr) == NULL, message, ##__VA_ARGS__)

// --- Test Functions Declarations ---
void test_ModelAllocationAndDataLoading();
// Add more test declarations here

// --- Main Test Runner ---
int main() {
    srand(time(NULL)); // Initialize random seed once
    
    printf("Starting custom test suite...\n\n");

    // --- Call Individual Test Functions ---
    test_ModelAllocationAndDataLoading();
    // Call other test functions here

    printf("===================================\n");
    printf("Total Tests Run:    %u\n", tests_run);
    printf("Total Tests Passed: %u\n", tests_passed);
    printf("Total Tests Failed: %u\n", tests_failed);
    printf("Total Assertions Run:    %u\n", assertions_run);
    printf("Total Assertions Failed: %u\n", assertions_failed);
    printf("===================================\n");

    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}

// --- Test Implementations ---

void test_ModelAllocationAndDataLoading() {
    TEST_BEGIN("ModelAllocationAndDataLoadingTest");

    int vocab_size = 0;
    initialize_vocabulary(&vocab_size);
    ASSERT_TRUE(vocab_size > 0, "Failed to initialize vocabulary. vocab_size: %d", vocab_size);

    const char* filepath = "data/saioa_stories_sample.txt";
    char* text_content = load_text_from_file(filepath);
    ASSERT_NOT_NULL(text_content, "Failed to load text from file: %s", filepath);

    int* tokens = NULL;
    int token_count = 0;
    tokens = tokenize_text(text_content, vocab_size, &token_count);
    ASSERT_NOT_NULL(tokens, "Failed to tokenize text.");
    
    ASSERT_TRUE(token_count > 0, "Token count is not greater than 0. Got: %d", token_count);

    free_tokens(tokens);
    free_text(text_content);

    // --- Test Model Allocation ---
    int num_blocks = 4;
    LegacyLLM* model = create_legacy_llm(vocab_size, MODEL_DIM, num_blocks);
    ASSERT_NOT_NULL(model, "Failed to allocate LegacyLLM model.");
    ASSERT_TRUE(model->vocab_size == vocab_size, "Model vocab_size mismatch. Expected %d, got %d", vocab_size, model->vocab_size);
    ASSERT_TRUE(model->model_dim == MODEL_DIM, "Model model_dim mismatch. Expected %d, got %d", MODEL_DIM, model->model_dim);
    ASSERT_TRUE(model->num_transformer_blocks == num_blocks, "Model num_transformer_blocks mismatch. Expected %d, got %d", num_blocks, model->num_transformer_blocks);
    
    // Free model
    free_legacy_llm(model);

    TEST_END();
}