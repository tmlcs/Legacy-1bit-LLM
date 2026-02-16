#include "test_framework.h" // Custom test framework
#include "math_ops.h"       // Functions to be tested
#include <float.h> // For FLT_EPSILON

// --- Helper function for float array comparison ---
static int compare_float_arrays(const float* arr1, const float* arr2, int size, float epsilon) {
    for (int i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > epsilon) {
            return 0; // Not equal
        }
    }
    return 1; // Equal
}

// --- Test Functions ---

void test_add_vector_inplace() {
    TEST_BEGIN("add_vector_inplace");
    float vec1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    int size = 4;
    float epsilon = 1e-6f;

    add_vector_inplace(vec1, vec2, size);
    ASSERT_TRUE(compare_float_arrays(vec1, expected, size, epsilon), "Vector addition failed.");

    // Test with negative numbers
    float vec3[] = {-1.0f, -2.0f, 0.0f};
    float vec4[] = {1.0f, 2.0f, 3.0f};
    float expected2[] = {0.0f, 0.0f, 3.0f};
    size = 3;
    add_vector_inplace(vec3, vec4, size);
    ASSERT_TRUE(compare_float_arrays(vec3, expected2, size, epsilon), "Vector addition with negatives failed.");

    // Test with size 0
    float vec5[] = {};
    float vec6[] = {};
    float expected3[] = {};
    size = 0;
    add_vector_inplace(vec5, vec6, size); // Should not crash
    ASSERT_TRUE(compare_float_arrays(vec5, expected3, size, epsilon), "Vector addition with size 0 failed.");


    TEST_END();
}

void test_scalar_mul_vector_inplace() {
    TEST_BEGIN("scalar_mul_vector_inplace");
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.0f;
    float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
    int size = 4;
    float epsilon = 1e-6f;

    scalar_mul_vector_inplace(vec, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "Scalar multiplication failed.");

    // Test with zero scalar
    float vec2[] = {1.0f, 2.0f, 3.0f};
    scalar = 0.0f;
    float expected2[] = {0.0f, 0.0f, 0.0f};
    size = 3;
    scalar_mul_vector_inplace(vec2, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "Scalar multiplication by zero failed.");

    // Test with negative scalar
    float vec3[] = {1.0f, 2.0f, 3.0f};
    scalar = -1.0f;
    float expected3[] = {-1.0f, -2.0f, -3.0f};
    size = 3;
    scalar_mul_vector_inplace(vec3, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec3, expected3, size, epsilon), "Scalar multiplication by negative failed.");

    // Test with size 0
    float vec4[] = {};
    scalar = 5.0f;
    float expected4[] = {};
    size = 0;
    scalar_mul_vector_inplace(vec4, scalar, size); // Should not crash
    ASSERT_TRUE(compare_float_arrays(vec4, expected4, size, epsilon), "Scalar multiplication with size 0 failed.");

    TEST_END();
}

void test_vector_sum() {
    TEST_BEGIN("vector_sum");
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected = 10.0f;
    int size = 4;
    float epsilon = 1e-6f;

    float result = vector_sum(vec, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Vector sum failed.");

    // Test with negative numbers
    float vec2[] = {-1.0f, -2.0f, 3.0f};
    expected = 0.0f;
    size = 3;
    result = vector_sum(vec2, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Vector sum with negatives failed.");

    // Test with single element
    float vec3[] = {5.5f};
    expected = 5.5f;
    size = 1;
    result = vector_sum(vec3, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Vector sum with single element failed.");

    // Test with size 0
    float vec4[] = {};
    expected = 0.0f; // Sum of an empty vector is 0
    size = 0;
    result = vector_sum(vec4, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Vector sum with size 0 failed.");

    TEST_END();
}

void test_dot_product() {
    TEST_BEGIN("dot_product");
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};
    float expected = 32.0f; // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    int size = 3;
    float epsilon = 1e-6f;

    float result = dot_product(vec1, vec2, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Dot product failed.");

    // Test with negative numbers
    float vec3[] = {-1.0f, 2.0f};
    float vec4[] = {3.0f, -4.0f};
    expected = -11.0f; // (-1)*3 + 2*(-4) = -3 - 8 = -11
    size = 2;
    result = dot_product(vec3, vec4, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Dot product with negatives failed.");

    // Test with zero values
    float vec5[] = {0.0f, 1.0f};
    float vec6[] = {2.0f, 0.0f};
    expected = 0.0f;
    size = 2;
    result = dot_product(vec5, vec6, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Dot product with zeros failed.");

    // Test with size 0
    float vec7[] = {};
    float vec8[] = {};
    expected = 0.0f; // Dot product of empty vectors is 0
    size = 0;
    result = dot_product(vec7, vec8, size);
    ASSERT_EQUALS_FLOAT(expected, result, epsilon, "Dot product with size 0 failed.");

    TEST_END();
}

void test_multiply_vector_inplace() {
    TEST_BEGIN("multiply_vector_inplace");
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {2.0f, 3.0f, 4.0f};
    float expected[] = {2.0f, 6.0f, 12.0f};
    int size = 3;
    float epsilon = 1e-6f;

    multiply_vector_inplace(vec1, vec2, size);
    ASSERT_TRUE(compare_float_arrays(vec1, expected, size, epsilon), "Element-wise multiplication failed.");

    float vec3[] = {-1.0f, 0.0f, 1.0f};
    float vec4[] = {2.0f, 5.0f, -3.0f};
    float expected2[] = {-2.0f, 0.0f, -3.0f};
    size = 3;
    multiply_vector_inplace(vec3, vec4, size);
    ASSERT_TRUE(compare_float_arrays(vec3, expected2, size, epsilon), "Element-wise multiplication with negatives/zeros failed.");

    TEST_END();
}

void test_vector_pow_scalar_inplace() {
    TEST_BEGIN("vector_pow_scalar_inplace");
    float vec[] = {1.0f, 2.0f, 3.0f};
    float scalar = 2.0f;
    float expected[] = {1.0f, 4.0f, 9.0f};
    int size = 3;
    float epsilon = 1e-6f;

    vector_pow_scalar_inplace(vec, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "Vector power scalar failed.");

    float vec2[] = {4.0f, 9.0f, 16.0f};
    scalar = 0.5f; // Square root
    float expected2[] = {2.0f, 3.0f, 4.0f};
    size = 3;
    vector_pow_scalar_inplace(vec2, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "Vector square root failed.");

    TEST_END();
}

void test_vector_sub_scalar_inplace() {
    TEST_BEGIN("vector_sub_scalar_inplace");
    float vec[] = {5.0f, 6.0f, 7.0f};
    float scalar = 2.0f;
    float expected[] = {3.0f, 4.0f, 5.0f};
    int size = 3;
    float epsilon = 1e-6f;

    vector_sub_scalar_inplace(vec, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "Vector sub scalar failed.");

    float vec2[] = {1.0f, 0.0f, -1.0f};
    scalar = -1.0f;
    float expected2[] = {2.0f, 1.0f, 0.0f};
    size = 3;
    vector_sub_scalar_inplace(vec2, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "Vector sub negative scalar failed.");

    TEST_END();
}

void test_vector_div_scalar_inplace() {
    TEST_BEGIN("vector_div_scalar_inplace");
    float vec[] = {6.0f, 8.0f, 10.0f};
    float scalar = 2.0f;
    float expected[] = {3.0f, 4.0f, 5.0f};
    int size = 3;
    float epsilon = 1e-6f;

    vector_div_scalar_inplace(vec, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "Vector div scalar failed.");

    float vec2[] = {-5.0f, 0.0f, 5.0f};
    scalar = -1.0f;
    float expected2[] = {5.0f, -0.0f, -5.0f};
    size = 3;
    vector_div_scalar_inplace(vec2, scalar, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "Vector div negative scalar failed.");

    TEST_END();
}

void test_outer_product_add_inplace() {
    TEST_BEGIN("outer_product_add_inplace");
    float matrix_grad[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // 2x3 matrix
    float vec1[] = {1.0f, 2.0f}; // 2 elements
    float vec2[] = {3.0f, 4.0f, 5.0f}; // 3 elements
    
    // Expected result:
    // [0 0 0]   [1*3 1*4 1*5]   [3  4  5]
    // [0 0 0] + [2*3 2*4 2*5] = [6  8 10]
    float expected[] = {3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 10.0f};
    int rows = 2;
    int cols = 3;
    float epsilon = 1e-6f;

    outer_product_add_inplace(matrix_grad, vec1, vec2, rows, cols);
    ASSERT_TRUE(compare_float_arrays(matrix_grad, expected, rows * cols, epsilon), "Outer product add failed.");

    // Test adding to existing gradient
    float matrix_grad2[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // 2x3 matrix
    float vec3[] = {-1.0f, 0.0f}; // 2 elements
    float vec4[] = {1.0f, 2.0f, 3.0f}; // 3 elements
    
    // Expected result:
    // [1 1 1]   [-1*1 -1*2 -1*3]   [1-1 1-2 1-3]   [0 -1 -2]
    // [1 1 1] + [ 0*1  0*2  0*3] = [1+0 1+0 1+0] = [1  1  1]
    float expected2[] = {0.0f, -1.0f, -2.0f, 1.0f, 1.0f, 1.0f};
    rows = 2;
    cols = 3;
    outer_product_add_inplace(matrix_grad2, vec3, vec4, rows, cols);
    ASSERT_TRUE(compare_float_arrays(matrix_grad2, expected2, rows * cols, epsilon), "Outer product add to existing gradient failed.");

    TEST_END();
}

void test_relu() {
    TEST_BEGIN("relu");
    float vec[] = {-1.0f, 0.0f, 0.5f, -2.0f, 3.0f};
    float expected[] = {0.0f, 0.0f, 0.5f, 0.0f, 3.0f};
    int size = 5;
    float epsilon = 1e-6f;

    relu(vec, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "ReLU failed.");

    float vec2[] = {1.0f, 2.0f, 3.0f};
    float expected2[] = {1.0f, 2.0f, 3.0f};
    size = 3;
    relu(vec2, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "ReLU all positive failed.");

    TEST_END();
}

void test_softmax() {
    TEST_BEGIN("softmax");
    float vec[] = {1.0f, 2.0f, 3.0f};
    // exp(1) approx 2.718, exp(2) approx 7.389, exp(3) approx 20.085
    // Sum = 30.192
    // Expected = {0.090, 0.245, 0.664} (approx)
    float expected[] = {0.09003057f, 0.24472847f, 0.66524096f}; // Calculated with higher precision
    int size = 3;
    float epsilon = 1e-5f; // Softmax can have larger numerical errors

    softmax(vec, size);
    ASSERT_TRUE(compare_float_arrays(vec, expected, size, epsilon), "Softmax failed.");

    float vec2[] = {0.0f, 0.0f, 0.0f};
    float expected2[] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    size = 3;
    softmax(vec2, size);
    ASSERT_TRUE(compare_float_arrays(vec2, expected2, size, epsilon), "Softmax all zeros failed.");

    // Test with values that sum to 1 after softmax
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += vec2[i];
    }
    ASSERT_EQUALS_FLOAT(1.0f, sum, epsilon, "Softmax output does not sum to 1.");

    TEST_END();
}

void test_layer_norm_forward() {
    TEST_BEGIN("layer_norm_forward");
    float input[] = {1.0f, 2.0f, 3.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    int size = 3;
    float epsilon = 1e-2f;
    float out_mean_val, out_inv_std_dev_val;

    // Expected calculation for input {1, 2, 3}:
    // Mean = (1+2+3)/3 = 2
    // Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1 + 0 + 1)/3 = 2/3
    // Std Dev = sqrt(2/3) approx 0.81649658
    // Inv Std Dev = 1 / 0.81649658 approx 1.22474487
    // Normalized: (input - mean) * inv_std_dev = (input - 2) * 1.22474487
    // { (1-2)*1.2247 = -1.2247, (2-2)*1.2247 = 0.0, (3-2)*1.2247 = 1.2247 }
    float expected_input[] = {-1.22474487f, 0.0f, 1.22474487f};
    float expected_mean = 2.0f;
    float expected_inv_std_dev = 1.22474487f;

    layer_norm_forward(input, gamma, beta, size, 1e-5f, &out_mean_val, &out_inv_std_dev_val);

    ASSERT_TRUE(compare_float_arrays(input, expected_input, size, epsilon), "LayerNorm input modification failed.");
    ASSERT_EQUALS_FLOAT(out_mean_val, expected_mean, epsilon, "LayerNorm mean failed.");
    ASSERT_EQUALS_FLOAT(out_inv_std_dev_val, expected_inv_std_dev, epsilon, "LayerNorm inv_std_dev failed.");
    
    // Test with different gamma/beta
    float input2[] = {1.0f, 2.0f, 3.0f};
    float gamma2[] = {2.0f, 2.0f, 2.0f};
    float beta2[] = {1.0f, 1.0f, 1.0f};
    // Expected: (normalized * 2) + 1
    // { (-1.2247 * 2) + 1 = -1.4494, (0.0 * 2) + 1 = 1.0, (1.2247 * 2) + 1 = 3.4494 }
    float expected_input2[] = {-1.4494897f, 1.0f, 3.4494897f};
    layer_norm_forward(input2, gamma2, beta2, size, 1e-5f, &out_mean_val, &out_inv_std_dev_val);
    ASSERT_TRUE(compare_float_arrays(input2, expected_input2, size, epsilon), "LayerNorm with gamma/beta failed.");


    TEST_END();
}

void test_layer_norm() {
    TEST_BEGIN("layer_norm");
    float input[] = {1.0f, 2.0f, 3.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    int size = 3;
    float epsilon = 1e-2f;

    // Expected calculation same as layer_norm_forward with gamma=1, beta=0
    float expected_input[] = {-1.22474487f, 0.0f, 1.22474487f};

    layer_norm(input, gamma, beta, size, 1e-5f);

    ASSERT_TRUE(compare_float_arrays(input, expected_input, size, epsilon), "LayerNorm failed (simplified).");

    float input2[] = {1.0f, 2.0f, 3.0f};
    float gamma2[] = {2.0f, 2.0f, 2.0f};
    float beta2[] = {1.0f, 1.0f, 1.0f};
    float expected_input2[] = {-1.4494897f, 1.0f, 3.4494897f};
    size = 3;
    layer_norm(input2, gamma2, beta2, size, 1e-5f);
    ASSERT_TRUE(compare_float_arrays(input2, expected_input2, size, epsilon), "LayerNorm with gamma/beta failed (simplified).");

    TEST_END();
}

void run_math_ops_tests() {
    printf("--- Running Math Operations Tests ---\n");
    test_add_vector_inplace();
    test_scalar_mul_vector_inplace();
    test_vector_sum();
    test_dot_product();
    test_multiply_vector_inplace();
    test_vector_pow_scalar_inplace();
    test_vector_sub_scalar_inplace();
    test_vector_div_scalar_inplace();
    test_outer_product_add_inplace();
    test_relu();
    test_softmax();
    test_layer_norm_forward();
    test_layer_norm();
}

