#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "legacy_llm.h"
#include "math_ops.h"
#include "model.h" // For create_float_array, create_ternary_matrix, free_float_array, free_ternary_matrix

// Function to print a float array
void print_float_array(const float* arr, int size, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < size; ++i) {
        printf("%.6f%s", arr[i], (i == size - 1) ? "" : ", ");
    }
    printf("]\n");
}

int main() {
    srand(0); // Fixed seed for reproducibility

    // --- Test ternary_matrix_vector_mul ---
    printf("TEST: ternary_matrix_vector_mul\n");
    int mat_rows_tmvm = 2;
    int mat_cols_tmvm = 4;
    TernaryMatrix mat_tmvm = create_ternary_matrix(mat_rows_tmvm, mat_cols_tmvm);
    float* vec_tmvm = create_float_array(mat_cols_tmvm);
    float* output_tmvm = create_float_array(mat_rows_tmvm);

    // Initialize mat_tmvm and vec_tmvm with known values for consistency
    // Note: create_ternary_matrix and create_float_array use rand(), so srand(0) makes them consistent
    
    // Explicitly set values for simple verification
    // mat_tmvm data:
    // 0  1  0  -1
    // 1 -1  0   1
    mat_tmvm.data[0] = 0; mat_tmvm.data[1] = 1; mat_tmvm.data[2] = 0; mat_tmvm.data[3] = -1;
    mat_tmvm.data[4] = 1; mat_tmvm.data[5] = -1; mat_tmvm.data[6] = 0; mat_tmvm.data[7] = 1;
    
    // vec_tmvm data: [1.0, 2.0, 3.0, 4.0]
    vec_tmvm[0] = 1.0f; vec_tmvm[1] = 2.0f; vec_tmvm[2] = 3.0f; vec_tmvm[3] = 4.0f;


    ternary_matrix_vector_mul(&mat_tmvm, vec_tmvm, output_tmvm);
    print_float_array(output_tmvm, mat_rows_tmvm, "ternary_matrix_vector_mul_output");

    free_ternary_matrix(&mat_tmvm);
    free_float_array(vec_tmvm);
    free_float_array(output_tmvm);

    // --- Test matrix_transpose_vector_mul ---
    printf("\nTEST: matrix_transpose_vector_mul\n");
    int mat_rows_mtvm = 2;
    int mat_cols_mtvm = 4;
    TernaryMatrix mat_mtvm = create_ternary_matrix(mat_rows_mtvm, mat_cols_mtvm);
    float* vec_mtvm = create_float_array(mat_rows_mtvm); // Input vec size is mat->rows
    float* output_mtvm = create_float_array(mat_cols_mtvm); // Output vec size is mat->cols

    // Initialize mat_mtvm and vec_mtvm with known values for consistency
    // mat_mtvm data (same as mat_tmvm):
    // 0  1  0  -1
    // 1 -1  0   1
    mat_mtvm.data[0] = 0; mat_mtvm.data[1] = 1; mat_mtvm.data[2] = 0; mat_mtvm.data[3] = -1;
    mat_mtvm.data[4] = 1; mat_mtvm.data[5] = -1; mat_mtvm.data[6] = 0; mat_mtvm.data[7] = 1;
    
    // vec_mtvm data: [5.0, 6.0]
    vec_mtvm[0] = 5.0f; vec_mtvm[1] = 6.0f;


    matrix_transpose_vector_mul(&mat_mtvm, vec_mtvm, output_mtvm);
    print_float_array(output_mtvm, mat_cols_mtvm, "matrix_transpose_vector_mul_output");

    free_ternary_matrix(&mat_mtvm);
    free_float_array(vec_mtvm);
    free_float_array(output_mtvm);


    // --- Test add_vector_inplace ---
    printf("\nTEST: add_vector_inplace\n");
    int vec_size_avi = 5;
    float* vec1_avi = create_float_array(vec_size_avi);
    float* vec2_avi = create_float_array(vec_size_avi);

    // Initialize with known values
    // vec1_avi: [1.0, 2.0, 3.0, 4.0, 5.0]
    // vec2_avi: [0.5, 0.5, 0.5, 0.5, 0.5]
    vec1_avi[0] = 1.0f; vec1_avi[1] = 2.0f; vec1_avi[2] = 3.0f; vec1_avi[3] = 4.0f; vec1_avi[4] = 5.0f;
    vec2_avi[0] = 0.5f; vec2_avi[1] = 0.5f; vec2_avi[2] = 0.5f; vec2_avi[3] = 0.5f; vec2_avi[4] = 0.5f;

    add_vector_inplace(vec1_avi, vec2_avi, vec_size_avi);
    print_float_array(vec1_avi, vec_size_avi, "add_vector_inplace_output");

    free_float_array(vec1_avi);
    free_float_array(vec2_avi);

    return 0;
}
