#include <stdio.h> // For fprintf, perror
#include <stdlib.h> // For malloc, free
#include <string.h> // For memcpy
#include <math.h>   // For sqrtf, expf, logf

// For SSE intrinsics
#ifdef USE_SSE
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#ifdef __SSE4_1__
#include <smmintrin.h> // SSE4.1 for _mm_floor_ps, if available
#endif
#endif // USE_SSE

#include "math_ops.h" // Include its own header for function declarations
#include "legacy_llm.h" // For TernaryMatrix definition

// --- Core mathematical operations ---

#ifdef USE_SSE
// Helper function: Convert 4 int8_t to 4 floats using SSE4.1 if available
static inline __m128 convert_int8_to_float(const int8_t* data) {
#ifdef __SSE4_1__
    // SSE4.1 path: Load 4 int8_t, extend to 32-bit integers, then convert to float
    // Load 4 bytes into lower 32 bits of xmm register
    int32_t temp;
    memcpy(&temp, data, sizeof(int32_t));
    __m128i int_vec = _mm_cvtsi32_si128(temp);
    // Sign-extend int8 to int32
    __m128i int32_vec = _mm_cvtepi8_epi32(int_vec);
    // Convert int32 to float
    return _mm_cvtepi32_ps(int32_vec);
#else
    // SSE2 fallback: Scalar conversion
    float w0 = (float)data[0];
    float w1 = (float)data[1];
    float w2 = (float)data[2];
    float w3 = (float)data[3];
    return _mm_setr_ps(w0, w1, w2, w3);
#endif
}

// SSE-optimized ternary_matrix_vector_mul
float* ternary_matrix_vector_mul(const TernaryMatrix* mat, const float* vec, float* output) {
    START_TIMER(ternary_matrix_vector_mul_sse);
    if (!mat || !mat->data || !vec || !output) {
        fprintf(stderr, "Error: NULL input to ternary_matrix_vector_mul_sse\n");
        STOP_TIMER(ternary_matrix_vector_mul_sse, "ternary_matrix_vector_mul_sse"); // Stop timer even on error
        return NULL;
    }

    for (int i = 0; i < mat->rows; ++i) {
        __m128 sum_vec = _mm_setzero_ps(); // Accumulator for this row's dot product
        const int8_t* row_weights = &mat->data[i * mat->cols];
        int j;

        for (j = 0; j + 3 < mat->cols; j += 4) {
            // Load 4 floats from vec
            __m128 v_elements = _mm_loadu_ps(&vec[j]);

            // Convert 4 int8_t weights to float using optimized conversion
            __m128 w_elements = convert_int8_to_float(&row_weights[j]);

            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(v_elements, w_elements));
        }

        // Horizontal sum of sum_vec
        sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
        _mm_store_ss(&output[i], sum_vec); // Store the first element (which holds the total sum)

        // Handle remaining elements (if mat->cols is not a multiple of 4)
        for (; j < mat->cols; ++j) {
            int8_t weight = row_weights[j];
            if (weight == 1) {
                output[i] += vec[j];
            } else if (weight == -1) {
                output[i] -= vec[j];
            }
        }
    }
    STOP_TIMER(ternary_matrix_vector_mul_sse, "ternary_matrix_vector_mul_sse");
    return output;
}
#else
// Non-SSE ternary_matrix_vector_mul
float* ternary_matrix_vector_mul(const TernaryMatrix* mat, const float* vec, float* output) {
    START_TIMER(ternary_matrix_vector_mul_non_sse);
    if (!mat || !mat->data || !vec || !output) {
        fprintf(stderr, "Error: NULL input to ternary_matrix_vector_mul\n");
        STOP_TIMER(ternary_matrix_vector_mul_non_sse, "ternary_matrix_vector_mul_non_sse"); // Stop timer even on error
        return NULL;
    }
    for (int i = 0; i < mat->rows; ++i) {
        output[i] = 0.0f; // Initialize accumulator
        for (int j = 0; j < mat->cols; ++j) {
            int8_t weight = mat->data[i * mat->cols + j];
            if (weight == 1) {
                output[i] += vec[j];
            }
            else if (weight == -1) {
                output[i] -= vec[j];
            }
        }
    }
    STOP_TIMER(ternary_matrix_vector_mul_non_sse, "ternary_matrix_vector_mul_non_sse");
    return output;
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized matrix_transpose_vector_mul
float* matrix_transpose_vector_mul(const TernaryMatrix* mat, const float* vec, float* output) {
    START_TIMER(matrix_transpose_vector_mul_sse);
    if (!mat || !mat->data || !vec || !output) {
        fprintf(stderr, "Error: NULL input to matrix_transpose_vector_mul_sse\n");
        STOP_TIMER(matrix_transpose_vector_mul_sse, "matrix_transpose_vector_mul_sse"); // Stop timer even on error
        return NULL;
    }

    // Initialize output vector to zeros
    for (int i = 0; i < mat->cols; ++i) {
        output[i] = 0.0f;
    }

    // Iterate through rows of the vector (which corresponds to rows of the original matrix)
    // and distribute the result to the output vector columns
    for (int j = 0; j < mat->rows; ++j) {
        const int8_t* row_weights = &mat->data[j * mat->cols];
        __m128 v_element = _mm_set1_ps(vec[j]); // Splat current vector element across SSE register

        int i;
        for (i = 0; i + 3 < mat->cols; i += 4) {
            // Load 4 output elements
            __m128 out_elements = _mm_loadu_ps(&output[i]);

            // Convert 4 int8_t weights to float using optimized conversion
            __m128 w_elements = convert_int8_to_float(&row_weights[i]);

            // Add product to output
            out_elements = _mm_add_ps(out_elements, _mm_mul_ps(v_element, w_elements));
            _mm_storeu_ps(&output[i], out_elements);
        }

        // Handle remaining elements for the current row
        for (; i < mat->cols; ++i) {
            int8_t weight = row_weights[i];
            if (weight == 1) {
                output[i] += vec[j];
            } else if (weight == -1) {
                output[i] -= vec[j];
            }
        }
    }
    STOP_TIMER(matrix_transpose_vector_mul_sse, "matrix_transpose_vector_mul_sse");
    return output;
}
#else
// Non-SSE matrix_transpose_vector_mul
float* matrix_transpose_vector_mul(const TernaryMatrix* mat, const float* vec, float* output) {
    START_TIMER(matrix_transpose_vector_mul_non_sse);
    if (!mat || !mat->data || !vec || !output) {
        fprintf(stderr, "Error: NULL input to matrix_transpose_vector_mul\n");
        STOP_TIMER(matrix_transpose_vector_mul_non_sse, "matrix_transpose_vector_mul_non_sse"); // Stop timer even on error
        return NULL;
    }
    for (int i = 0; i < mat->cols; ++i) { // Iterates through columns of original matrix (rows of transpose)
        output[i] = 0.0f; // Initialize accumulator
        // Iterate through mat->rows to perform dot product with the i-th column of mat
        for (int j = 0; j < mat->rows; ++j) { // Iterates through rows of original matrix
            int8_t weight = mat->data[j * mat->cols + i]; // Access mat[j][i]
            if (weight == 1) {
                output[i] += vec[j];
            } else if (weight == -1) {
                output[i] -= vec[j];
            }
        }
    }
    STOP_TIMER(matrix_transpose_vector_mul_non_sse, "matrix_transpose_vector_mul_non_sse");
    return output;
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized add_vector_inplace
void add_vector_inplace(float* vec1, const float* vec2, int size) {
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v1 = _mm_loadu_ps(&vec1[i]); // Load 4 floats from vec1
        __m128 v2 = _mm_loadu_ps(&vec2[i]); // Load 4 floats from vec2
        __m128 result = _mm_add_ps(v1, v2); // Add them
        _mm_storeu_ps(&vec1[i], result);    // Store result back to vec1
    }
    // Handle remaining elements (if size is not a multiple of 4)
    for (; i < size; ++i) {
        vec1[i] += vec2[i];
    }
}
#else
// Non-SSE add_vector_inplace
void add_vector_inplace(float* vec1, const float* vec2, int size) {
    for (int i = 0; i < size; ++i) {
        vec1[i] += vec2[i];
    }
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized add_scalar_mul_vector_inplace
void add_scalar_mul_vector_inplace(float* vec1, float scalar, const float* vec2, int size) {
    __m128 s_vec = _mm_set1_ps(scalar); // Splat scalar into all 4 floats of an SSE register
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v1 = _mm_loadu_ps(&vec1[i]); // Load 4 floats from vec1
        __m128 v2 = _mm_loadu_ps(&vec2[i]); // Load 4 floats from vec2
        __m128 product = _mm_mul_ps(s_vec, v2); // scalar * vec2[i]
        __m128 result = _mm_add_ps(v1, product); // vec1[i] += product
        _mm_storeu_ps(&vec1[i], result);       // Store result back to vec1
    }
    // Handle remaining elements (if size is not a multiple of 4)
    for (; i < size; ++i) {
        vec1[i] += scalar * vec2[i];
    }
}
#else
// Non-SSE add_scalar_mul_vector_inplace
void add_scalar_mul_vector_inplace(float* vec1, float scalar, const float* vec2, int size) {
    for (int i = 0; i < size; ++i) {
        vec1[i] += scalar * vec2[i];
    }
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized multiply_vector_inplace
void multiply_vector_inplace(float* vec1, const float* vec2, int size) {
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v1 = _mm_loadu_ps(&vec1[i]); // Load 4 floats from vec1
        __m128 v2 = _mm_loadu_ps(&vec2[i]); // Load 4 floats from vec2
        __m128 result = _mm_mul_ps(v1, v2); // Multiply them
        _mm_storeu_ps(&vec1[i], result);    // Store result back to vec1
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        vec1[i] *= vec2[i];
    }
}
#else
// Non-SSE multiply_vector_inplace
void multiply_vector_inplace(float* vec1, const float* vec2, int size) {
    for (int i = 0; i < size; ++i) {
        vec1[i] *= vec2[i];
    }
}
#endif // USE_SSE
#ifdef USE_SSE
// SSE-optimized vector_pow_scalar_inplace
void vector_pow_scalar_inplace(float* vec, float scalar, int size) {
    START_TIMER(vector_pow_scalar_inplace_sse);
    if (scalar == 2.0f) { // Optimized for squaring
        __m128 v_vec;
        int i;
        for (i = 0; i + 3 < size; i += 4) {
            v_vec = _mm_loadu_ps(&vec[i]);
            v_vec = _mm_mul_ps(v_vec, v_vec); // v*v for squaring
            _mm_storeu_ps(&vec[i], v_vec);
        }
        for (; i < size; ++i) {
            vec[i] *= vec[i]; // Scalar tail for squaring
        }
    } else if (scalar == 0.5f) { // Optimized for square root
        __m128 v_vec;
        int i;
        for (i = 0; i + 3 < size; i += 4) {
            v_vec = _mm_loadu_ps(&vec[i]);
            v_vec = _mm_sqrt_ps(v_vec); // sqrt
            _mm_storeu_ps(&vec[i], v_vec);
        }
        for (; i < size; ++i) {
            vec[i] = sqrtf(vec[i]); // Scalar tail for sqrt
        }
    }
    else { // Fallback to non-SSE version for general powers
        for (int i = 0; i < size; ++i) {
            vec[i] = powf(vec[i], scalar);
        }
    }
    STOP_TIMER(vector_pow_scalar_inplace_sse, "vector_pow_scalar_inplace_sse");
}
#else
// Non-SSE vector_pow_scalar_inplace
void vector_pow_scalar_inplace(float* vec, float scalar, int size) {
    START_TIMER(vector_pow_scalar_inplace_non_sse);
    for (int i = 0; i < size; ++i) {
        vec[i] = powf(vec[i], scalar);
    }
    STOP_TIMER(vector_pow_scalar_inplace_non_sse, "vector_pow_scalar_inplace_non_sse");
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized scalar_mul_vector_inplace
void scalar_mul_vector_inplace(float* vec, float scalar, int size) {
    __m128 s_vec = _mm_set1_ps(scalar); // Splat scalar into all 4 floats of an SSE register
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&vec[i]); // Load 4 floats from vec
        __m128 result = _mm_mul_ps(v, s_vec); // Multiply them
        _mm_storeu_ps(&vec[i], result);       // Store result back to vec
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        vec[i] *= scalar;
    }
}
#else
// Non-SSE scalar_mul_vector_inplace
void scalar_mul_vector_inplace(float* vec, float scalar, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] *= scalar;
    }
}
#endif // USE_SSE
#ifdef USE_SSE
// SSE-optimized vector_sub_scalar_inplace
void vector_sub_scalar_inplace(float* vec, float scalar, int size) {
    __m128 s_vec = _mm_set1_ps(scalar); // Splat scalar into all 4 floats of an SSE register
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&vec[i]); // Load 4 floats from vec
        __m128 result = _mm_sub_ps(v, s_vec); // Subtract scalar
        _mm_storeu_ps(&vec[i], result);       // Store result back to vec
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        vec[i] -= scalar;
    }
}
#else
// Non-SSE vector_sub_scalar_inplace
void vector_sub_scalar_inplace(float* vec, float scalar, int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] -= scalar;
    }
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized vector_div_scalar_inplace
void vector_div_scalar_inplace(float* vec, float scalar, int size) {
    if (!vec || size <= 0 || scalar == 0.0f) {
        fprintf(stderr, "Error: NULL, invalid input, or division by zero in vector_div_scalar_inplace\n");
        return;
    }
    __m128 s_vec = _mm_set1_ps(scalar); // Splat scalar into all 4 floats of an SSE register
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&vec[i]); // Load 4 floats from vec
        __m128 result = _mm_div_ps(v, s_vec); // Divide by scalar
        _mm_storeu_ps(&vec[i], result);       // Store result back to vec
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        vec[i] /= scalar;
    }
}
#else
// Non-SSE vector_div_scalar_inplace
void vector_div_scalar_inplace(float* vec, float scalar, int size) {
    if (!vec || size <= 0 || scalar == 0.0f) {
        fprintf(stderr, "Error: NULL, invalid input, or division by zero in vector_div_scalar_inplace\n");
        return;
    }
    for (int i = 0; i < size; ++i) {
        vec[i] /= scalar;
    }
}
#endif // USE_SSE
#ifdef USE_SSE
// SSE-optimized vector_sum
float vector_sum(const float* vec, int size) {
    START_TIMER(vector_sum_sse);
    __m128 sum_vec = _mm_setzero_ps(); // Initialize sum to zero
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&vec[i]); // Load 4 floats
        sum_vec = _mm_add_ps(sum_vec, v);  // Add to accumulator
    }

    // Horizontal sum: sum_vec = [sum0, sum1, sum2, sum3]
    // sum_vec = [sum0+sum1, sum2+sum3, sum0+sum1, sum2+sum3] (after first shuffle and add)
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    // sum_vec = [sum0+sum1+sum2+sum3, ..., ..., ...] (after second shuffle and add)
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));

    float sse_sum;
    _mm_store_ss(&sse_sum, sum_vec); // Store the first element (which now holds the total sum)

    float remaining_sum = 0.0f;
    for (; i < size; ++i) {
        remaining_sum += vec[i];
    }
    STOP_TIMER(vector_sum_sse, "vector_sum_sse");
    return sse_sum + remaining_sum;
}
#else
// Non-SSE vector_sum
float vector_sum(const float* vec, int size) {
    START_TIMER(vector_sum_non_sse);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += vec[i];
    }
    STOP_TIMER(vector_sum_non_sse, "vector_sum_non_sse");
    return sum;
}
#endif // USE_SSE
#ifdef USE_SSE
// SSE-optimized outer_product_add_inplace
void outer_product_add_inplace(float* matrix_grad, const float* vec1, const float* vec2, int rows, int cols) {
    START_TIMER(outer_product_add_inplace_sse);
    for (int i = 0; i < rows; ++i) {
        __m128 val1_splat = _mm_set1_ps(vec1[i]); // Splat vec1[i] across SSE register
        int j;
        for (j = 0; j + 3 < cols; j += 4) {
            __m128 m_grad_vec = _mm_loadu_ps(&matrix_grad[i * cols + j]); // Load 4 floats from matrix_grad
            __m128 vec2_vec = _mm_loadu_ps(&vec2[j]);                     // Load 4 floats from vec2
            __m128 product = _mm_mul_ps(val1_splat, vec2_vec);          // vec1[i] * vec2[j...j+3]
            __m128 result = _mm_add_ps(m_grad_vec, product);              // matrix_grad[idx] += product
            _mm_storeu_ps(&matrix_grad[i * cols + j], result);            // Store result back
        }
        // Handle remaining elements (tail processing)
        for (; j < cols; ++j) {
            matrix_grad[i * cols + j] += vec1[i] * vec2[j];
        }
    }
    STOP_TIMER(outer_product_add_inplace_sse, "outer_product_add_inplace_sse");
}
#else
// Non-SSE outer_product_add_inplace
void outer_product_add_inplace(float* matrix_grad, const float* vec1, const float* vec2, int rows, int cols) {
    START_TIMER(outer_product_add_inplace_non_sse);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix_grad[i * cols + j] += vec1[i] * vec2[j];
        }
    }
    STOP_TIMER(outer_product_add_inplace_non_sse, "outer_product_add_inplace_non_sse");
}
#endif // USE_SSE

// ReLU Activation Function
#ifdef USE_SSE
void relu(float* input, int size) {
    __m128 zero = _mm_setzero_ps(); // SSE register with all zeros
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&input[i]); // Load 4 floats
        __m128 result = _mm_max_ps(v, zero); // Compute max(v, 0)
        _mm_storeu_ps(&input[i], result);    // Store result back
    }
    // Handle remaining elements
    for (; i < size; ++i) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}
#else
void relu(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}
#endif // USE_SSE
// Softmax Activation Function
#ifdef USE_SSE
void softmax(float* input, int size) {
    START_TIMER(softmax_sse);
    if (!input || size <= 0) {
        fprintf(stderr, "Error: NULL or invalid input to softmax\n");
        STOP_TIMER(softmax_sse, "softmax_sse");
        return;
    }
    // 1. Find max_val for numerical stability (partially SSE-optimized)
    float max_val = input[0];
    int i;
    for (i = 0; i < size; ++i) { // Scalar max_val finding loop
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    __m128 max_vec = _mm_set1_ps(max_val); // Splat max_val for subtraction
    __m128 sum_exp_vec = _mm_setzero_ps(); // Accumulator for sum_exp

    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&input[i]);
        __m128 diff = _mm_sub_ps(v, max_vec); // input[i] - max_val

        // expf is not directly available in SSE intrinsics, perform scalar expf for now
        // A custom vectorized expf approximation would go here for full optimization
        float temp_exp[4];
        _mm_storeu_ps(temp_exp, diff);
        temp_exp[0] = expf(temp_exp[0]);
        temp_exp[1] = expf(temp_exp[1]);
        temp_exp[2] = expf(temp_exp[2]);
        temp_exp[3] = expf(temp_exp[3]);
        v = _mm_loadu_ps(temp_exp); // Load back exp'd values

        _mm_storeu_ps(&input[i], v); // Store exp'd values back to input
        sum_exp_vec = _mm_add_ps(sum_exp_vec, v); // Add to sum_exp accumulator
    }

    // Horizontal sum of sum_exp_vec
    sum_exp_vec = _mm_add_ps(sum_exp_vec, _mm_shuffle_ps(sum_exp_vec, sum_exp_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    sum_exp_vec = _mm_add_ps(sum_exp_vec, _mm_shuffle_ps(sum_exp_vec, sum_exp_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    float sum_exp;
    _mm_store_ss(&sum_exp, sum_exp_vec);

    // Handle remaining elements (scalar) and add to sum_exp
    for (; i < size; ++i) {
        input[i] = expf(input[i] - max_val);
        sum_exp += input[i];
    }
    
    // 4. Divide by sum_exp (SSE-optimized)
    __m128 inv_sum_exp_vec = _mm_set1_ps(1.0f / sum_exp); // Splat 1/sum_exp
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&input[i]);
        __m128 result = _mm_mul_ps(v, inv_sum_exp_vec);
        _mm_storeu_ps(&input[i], result);
    }
    // Handle remaining elements (scalar)
    for (; i < size; ++i) {
        input[i] /= sum_exp;
    }
    STOP_TIMER(softmax_sse, "softmax_sse");
}
#else
void softmax(float* input, int size) {
    START_TIMER(softmax_non_sse);
    if (!input || size <= 0) {
        fprintf(stderr, "Error: NULL or invalid input to softmax\n");
        STOP_TIMER(softmax_non_sse, "softmax_non_sse");
        return;
    }
    float max_val = input[0];
    for (int i = 0; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val);
        sum_exp += input[i];
    }

    for (int i = 0; i < size; ++i) {
        input[i] /= sum_exp;
    }
    STOP_TIMER(softmax_non_sse, "softmax_non_sse");
}
#endif // USE_SSE

// Layer Normalization
#ifdef USE_SSE
void layer_norm_forward(float* input, const float* gamma, const float* beta, int size, float epsilon, float* out_mean, float* out_inv_std_dev) {
    START_TIMER(layer_norm_forward_sse);
    if (!input || !gamma || !beta || size <= 0) {
        fprintf(stderr, "Error: NULL or invalid input to layer_norm_forward\n");
        STOP_TIMER(layer_norm_forward_sse, "layer_norm_forward_sse");
        return;
    }
    // 1. Calculate mean (partially SSE-optimized)
    __m128 sum_vec = _mm_setzero_ps();
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        sum_vec = _mm_add_ps(sum_vec, _mm_loadu_ps(&input[i]));
    }
    // Horizontal sum
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    float mean_sum;
    _mm_store_ss(&mean_sum, sum_vec);
    for (; i < size; ++i) { // Scalar tail
        mean_sum += input[i];
    }
    float mean = mean_sum / size;
    *out_mean = mean; // Store mean

    // 2. Calculate variance (partially SSE-optimized)
    __m128 var_sum_vec = _mm_setzero_ps();
    __m128 mean_vec = _mm_set1_ps(mean);
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&input[i]);
        __m128 diff = _mm_sub_ps(v, mean_vec);
        var_sum_vec = _mm_add_ps(var_sum_vec, _mm_mul_ps(diff, diff));
    }
    // Horizontal sum
    var_sum_vec = _mm_add_ps(var_sum_vec, _mm_shuffle_ps(var_sum_vec, var_sum_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    var_sum_vec = _mm_add_ps(var_sum_vec, _mm_shuffle_ps(var_sum_vec, var_sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    float variance_sum;
    _mm_store_ss(&variance_sum, var_sum_vec);
    for (; i < size; ++i) { // Scalar tail
        float diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / size;

    // 3. Calculate inv_std_dev (SSE-optimized) using _mm_rsqrt_ps with Newton-Raphson refinement
    __m128 epsilon_vec = _mm_set1_ps(epsilon);
    __m128 var_vec = _mm_set1_ps(variance);
    __m128 std_dev_vec = _mm_sqrt_ps(_mm_add_ps(var_vec, epsilon_vec)); // sqrt(variance + epsilon)
    __m128 inv_std_dev_vec = _mm_rcp_ps(std_dev_vec); // 1.0f / std_dev
    
    float inv_std_dev;
    _mm_store_ss(&inv_std_dev, inv_std_dev_vec); // Get scalar inv_std_dev
    *out_inv_std_dev = inv_std_dev; // Store inv_std_dev

    // 4. Normalize and 5. Apply scale and shift (SSE-optimized)
    mean_vec = _mm_set1_ps(mean); // Re-splat mean
    inv_std_dev_vec = _mm_set1_ps(inv_std_dev); // Re-splat inv_std_dev

    for (i = 0; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(&input[i]);
        __m128 g = _mm_loadu_ps(&gamma[i]);
        __m128 b = _mm_loadu_ps(&beta[i]);

        __m128 normalized = _mm_mul_ps(_mm_sub_ps(v, mean_vec), inv_std_dev_vec);
        __m128 result = _mm_add_ps(_mm_mul_ps(g, normalized), b);
        _mm_storeu_ps(&input[i], result);
    }
    for (; i < size; ++i) { // Scalar tail
        input[i] = (input[i] - mean) * inv_std_dev;
        input[i] = gamma[i] * input[i] + beta[i];
    }
    STOP_TIMER(layer_norm_forward_sse, "layer_norm_forward_sse");
}
#else
void layer_norm_forward(float* input, const float* gamma, const float* beta, int size, float epsilon, float* out_mean, float* out_inv_std_dev) {
    START_TIMER(layer_norm_forward_non_sse);
    if (!input || !gamma || !beta || size <= 0) {
        fprintf(stderr, "Error: NULL or invalid input to layer_norm_forward\n");
        STOP_TIMER(layer_norm_forward_non_sse, "layer_norm_forward_non_sse");
        return;
    }
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += input[i];
    }
    mean /= size;

    float variance = 0.0f;
    for (int i = 0; i < size; ++i) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;

    float std_dev = sqrtf(variance + epsilon);
    float inv_std_dev = 1.0f / std_dev;

    for (int i = 0; i < size; ++i) {
        input[i] = (input[i] - mean) * inv_std_dev;
        input[i] = gamma[i] * input[i] + beta[i];
    }
    *out_mean = mean;
    *out_inv_std_dev = inv_std_dev;
    STOP_TIMER(layer_norm_forward_non_sse, "layer_norm_forward_non_sse");
}
#endif // USE_SSE

#ifdef USE_SSE
void layer_norm(float* input, const float* gamma, const float* beta, int size, float epsilon) {
    // Placeholder: call non-SSE version
    float mean, inv_std_dev;
    layer_norm_forward(input, gamma, beta, size, epsilon, &mean, &inv_std_dev);
}
#else
void layer_norm(float* input, const float* gamma, const float* beta, int size, float epsilon) {
    float mean, inv_std_dev;
    layer_norm_forward(input, gamma, beta, size, epsilon, &mean, &inv_std_dev);
}
#endif // USE_SSE

#ifdef USE_SSE
// SSE-optimized dot_product
float dot_product(const float* vec1, const float* vec2, int size) {
    START_TIMER(dot_product_sse);
    __m128 sum_vec = _mm_setzero_ps(); // Accumulator for products
    int i;
    for (i = 0; i + 3 < size; i += 4) {
        __m128 v1 = _mm_loadu_ps(&vec1[i]);
        __m128 v2 = _mm_loadu_ps(&vec2[i]);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(v1, v2)); // Multiply and add to sum
    }

    // Horizontal sum
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(1, 0, 3, 2)));
    sum_vec = _mm_add_ps(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    float sse_sum;
    _mm_store_ss(&sse_sum, sum_vec);

    float remaining_sum = 0.0f;
    for (; i < size; ++i) {
        remaining_sum += vec1[i] * vec2[i];
    }
    STOP_TIMER(dot_product_sse, "dot_product_sse");
    return sse_sum + remaining_sum;
}
#else
float dot_product(const float* vec1, const float* vec2, int size) {
    START_TIMER(dot_product_non_sse);
    float result = 0.0f;
    for (int i = 0; i < size; ++i) {
        result += vec1[i] * vec2[i];
    }
    STOP_TIMER(dot_product_non_sse, "dot_product_non_sse");
    return result;
}
#endif // USE_SSE
