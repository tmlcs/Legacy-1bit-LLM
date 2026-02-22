#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    // For fabs

// Compiler-agnostic unused macro
#if defined(__GNUC__) || defined(__clang__)
    #define UNUSED_ATTR __attribute__((unused))
#elif defined(_MSC_VER)
    #define UNUSED_ATTR __pragma(warning(suppress:4505))
#else
    #define UNUSED_ATTR
#endif

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

// --- Helper function for float array comparison ---
UNUSED_ATTR
static int compare_float_arrays(const float* arr1, const float* arr2, int size, float epsilon) {
    for (int i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > epsilon) {
            return 0; // Not equal
        }
    }
    return 1; // Equal
}

#endif // TEST_FRAMEWORK_H
