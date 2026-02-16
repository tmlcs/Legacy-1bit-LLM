#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "legacy_llm.h" // For float and int8_t types

// --- Performance Measurement Macros ---
#ifdef MEASURE_PERFORMANCE
#include <stdio.h> // For fprintf
#include "data_utils.h" // For get_time_in_ms()

#define START_TIMER(timer_name) \
    long long start_##timer_name = get_time_in_ms();

#define STOP_TIMER(timer_name, function_name) \
    long long end_##timer_name = get_time_in_ms(); \
    fprintf(stderr, "PERF: %s took %lld ms\n", function_name, end_##timer_name - start_##timer_name); \
    fflush(stderr);

#else // MEASURE_PERFORMANCE
#define START_TIMER(timer_name)
#define STOP_TIMER(timer_name, function_name)
#endif // MEASURE_PERFORMANCE

// Core mathematical operations
float* ternary_matrix_vector_mul(const TernaryMatrix* mat, const float* vec, float* output);
void add_vector_inplace(float* vec1, const float* vec2, int size); // vec1 += vec2
float* matrix_transpose_vector_mul(const TernaryMatrix* mat, const float* vec, float* output); // For backward pass
void add_scalar_mul_vector_inplace(float* vec1, float scalar, const float* vec2, int size); // vec1 += scalar * vec2
void multiply_vector_inplace(float* vec1, const float* vec2, int size); // vec1 *= vec2 element-wise
void scalar_mul_vector_inplace(float* vec, float scalar, int size); // vec *= scalar
float vector_sum(const float* vec, int size); // Sum of elements
void vector_sub_scalar_inplace(float* vec, float scalar, int size); // vec -= scalar
void vector_div_scalar_inplace(float* vec, float scalar, int size); // vec /= scalar
void vector_pow_scalar_inplace(float* vec, float scalar, int size); // vec = pow(vec, scalar)

void outer_product_add_inplace(float* matrix_grad, const float* vec1, const float* vec2, int rows, int cols); // matrix_grad += vec1 * vec2.T

// Activation Functions
void relu(float* input, int size);
void softmax(float* input, int size); // Softmax for attention scores
void layer_norm_forward(float* input, const float* gamma, const float* beta, int size, float epsilon, float* out_mean, float* out_inv_std_dev);
void layer_norm(float* input, const float* gamma, const float* beta, int size, float epsilon);

// Vector operations
float dot_product(const float* vec1, const float* vec2, int size); // Dot product for attention scores

#endif // MATH_OPS_H