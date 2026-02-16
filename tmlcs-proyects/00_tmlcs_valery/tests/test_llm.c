#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, rand
#include <math.h>    // For sqrtf, expf, logf

#include "test_llm.h"
#include "legacy_llm.h"
#include "data_utils.h"
#include "model.h"
#include "math_ops.h"
#include "forward.h"
#include "backward.h"

void run_all_llm_tests() {
    printf("Hello, Legacy-1bit LLM!\n");

    srand(time(NULL));

    int vocab_size = 0;
    initialize_vocabulary(&vocab_size);

    const char* filepath = "data/saioa_stories_sample.txt";
    char* text_content = load_text_from_file(filepath);

    if (text_content != NULL) {
        int* tokens = NULL;
        int token_count = 0;
        tokens = tokenize_text(text_content, vocab_size, &token_count);

        if (tokens != NULL) {
            printf("First 10 tokens: ");
            for (int i = 0; i < token_count && i < 10; ++i) {
                printf("%d ", tokens[i]);
            }
            printf("\n");
            free_tokens(tokens);
        }
        free_text(text_content);
    }

    // --- Test Model Allocation ---
    int num_blocks = 4;
    LegacyLLM* model = create_legacy_llm(vocab_size, MODEL_DIM, num_blocks);
    
    if (model) {
        printf("Model allocation test successful.\n");

        // --- Test Ternary Matrix-Vector Multiplication ---
        printf("\nTesting Ternary Matrix-Vector Multiplication...\n");
        TernaryMatrix test_mat = create_ternary_matrix(3, 4); // 3x4 matrix
        float test_vec[] = {1.0f, 2.0f, 3.0f, 4.0f}; // 4-element vector
        float* result_vec = create_float_array(3); // 3-element result vector

        if (test_mat.data && result_vec) {
            ternary_matrix_vector_mul(&test_mat, test_vec, result_vec);

            printf("Result Vector (3 elements): [");
            for(int i=0; i<3; ++i) {
                printf("%.1f%s", result_vec[i], (i == 2) ? "" : ", ");
            }
            printf("\n");
        }
        free_ternary_matrix(&test_mat);
        free_float_array(result_vec);
        printf("Ternary Matrix-Vector Multiplication test complete.\n");

        // --- Test Matrix Transpose Vector Multiplication (for gradients) ---
        printf("\nTesting Matrix Transpose Vector Multiplication...\n");
        TernaryMatrix test_mat_T = create_ternary_matrix(4, 3); // 3x4 matrix transposed
        // For testing, just assume a matrix with some values
        int8_t T_data[] = {1, 0, -1, // col 0
                           0, 1, 0,  // col 1
                           -1, 0, 1, // col 2
                           0, -1, 0}; // col 3
        memcpy(test_mat_T.data, T_data, 4*3*sizeof(int8_t));

        float test_vec_T[] = {1.0f, 2.0f, 3.0f}; // 3-element vector
        float* result_vec_T = create_float_array(4); // 4-element result vector

        if (test_mat_T.data && result_vec_T) {
            printf("Test Matrix (4x3, conceptually 3x4 transposed):\n");
            for(int i=0; i<test_mat_T.rows; ++i) {
                for(int j=0; j<3; ++j) { // Limit to 3 columns for printing example
                    printf("%3d ", test_mat_T.data[i * test_mat_T.cols + j]);
                }
                printf("\n");
            }
            printf("Test Vector (3 elements): [");
            for(int i=0; i<3; ++i) {
                printf("%.1f%s", test_vec_T[i], (i == 2) ? "" : ", ");
            }
            printf("\n");

            matrix_transpose_vector_mul(&test_mat_T, test_vec_T, result_vec_T);

            printf("Result Vector (4 elements): [");
            for(int i=0; i<4; ++i) {
                printf("%.1f%s", result_vec_T[i], (i == 3) ? "" : ", ");
            }
            printf("\n");
        }
        free_ternary_matrix(&test_mat_T);
        free_float_array(result_vec_T);
        printf("Matrix Transpose Vector Multiplication test complete.\n");


        // --- Test Add Vector Inplace ---
        printf("\nTesting Add Vector Inplace...\n");
        float add_vec1[] = {1.0f, 2.0f, 3.0f};
        float add_vec2[] = {0.5f, 1.5f, 2.5f};
        int add_size = 3;
        add_vector_inplace(add_vec1, add_vec2, add_size);
        printf("Result (Vector 1 + Vector 2): ["); for(int i=0; i<add_size; ++i) printf("%.1f ", add_vec1[i]); printf("\n");
        printf("Add Vector Inplace test complete.\n");

        // --- Test Add Scalar Mul Vector Inplace ---
        printf("\nTesting Add Scalar Mul Vector Inplace...\n");
        float asm_vec1[] = {1.0f, 2.0f, 3.0f};
        float asm_scalar = 2.0f;
        float asm_vec2[] = {0.5f, 1.0f, 1.5f};
        int asm_size = 3;
        add_scalar_mul_vector_inplace(asm_vec1, asm_scalar, asm_vec2, asm_size);
        printf("Result (vec1 += scalar * vec2): ["); for(int i=0; i<asm_size; ++i) printf("%.1f ", asm_vec1[i]); printf("\n");
        printf("Add Scalar Mul Vector Inplace test complete.\n");

        // --- Test Multiply Vector Inplace ---
        printf("\nTesting Multiply Vector Inplace...\n");
        float mvi_vec1[] = {1.0f, 2.0f, 3.0f};
        float mvi_vec2[] = {0.5f, 2.0f, 1.0f};
        int mvi_size = 3;
        multiply_vector_inplace(mvi_vec1, mvi_vec2, mvi_size);
        printf("Result (vec1 *= vec2): ["); for(int i=0; i<mvi_size; ++i) printf("%.1f ", mvi_vec1[i]); printf("\n");
        printf("Multiply Vector Inplace test complete.\n");

        // --- Test Scalar Mul Vector Inplace ---
        printf("\nTesting Scalar Mul Vector Inplace...\n");
        float smvi_vec[] = {1.0f, 2.0f, 3.0f};
        float smvi_scalar = 2.5f;
        int smvi_size = 3;
        scalar_mul_vector_inplace(smvi_vec, smvi_scalar, smvi_size);
        printf("Result (vec *= scalar): ["); for(int i=0; i<smvi_size; ++i) printf("%.1f ", smvi_vec[i]); printf("\n");
        printf("Scalar Mul Vector Inplace test complete.\n");

        // --- Test Vector Sum ---
        printf("\nTesting Vector Sum...\n");
        float vs_vec[] = {1.0f, 2.0f, 3.0f};
        int vs_size = 3;
        float vs_sum = vector_sum(vs_vec, vs_size);
        printf("Vector: ["); for(int i=0; i<vs_size; ++i) printf("%.1f ", vs_vec[i]); printf("\n");
        printf("Sum: %.1f\n", vs_sum);
        printf("Vector Sum test complete.\n");

        // --- Test Vector Sub Scalar Inplace ---
        printf("\nTesting Vector Sub Scalar Inplace...\n");
        float vssi_vec[] = {1.0f, 2.0f, 3.0f};
        float vssi_scalar = 0.5f;
        int vssi_size = 3;
        vector_sub_scalar_inplace(vssi_vec, vssi_scalar, vssi_size);
        printf("Result (vec -= scalar): ["); for(int i=0; i<vssi_size; ++i) printf("%.1f ", vssi_vec[i]); printf("\n");
        printf("Vector Sub Scalar Inplace test complete.\n");

        // --- Test Vector Div Scalar Inplace ---
        printf("\nTesting Vector Div Scalar Inplace...\n");
        float vdsi_vec[] = {1.0f, 2.0f, 3.0f};
        float vdsi_scalar = 2.0f;
        int vdsi_size = 3;
        vector_div_scalar_inplace(vdsi_vec, vdsi_scalar, vdsi_size);
        printf("Result (vec /= scalar): ["); for(int i=0; i<vdsi_size; ++i) printf("%.1f ", vdsi_vec[i]); printf("\n");
        printf("Vector Div Scalar Inplace test complete.\n");

        // --- Test Vector Pow Scalar Inplace ---
        printf("\nTesting Vector Pow Scalar Inplace...\n");
        float vpsi_vec[] = {1.0f, 2.0f, 3.0f};
        float vpsi_scalar = 2.0f;
        int vpsi_size = 3;
        vector_pow_scalar_inplace(vpsi_vec, vpsi_scalar, vpsi_size);
        printf("Result (vec = pow(vec, scalar)): ["); for(int i=0; i<vpsi_size; ++i) printf("%.1f ", vpsi_vec[i]); printf("\n");
        printf("Vector Pow Scalar Inplace test complete.\n");

        // --- Test Outer Product Add Inplace ---
        printf("\nTesting Outer Product Add Inplace...\n");
        float op_matrix_grad[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // 2x3 matrix
        float op_vec1[] = {1.0f, 2.0f}; // 2 elements
        float op_vec2[] = {0.5f, 1.0f, 1.5f}; // 3 elements
        outer_product_add_inplace(op_matrix_grad, op_vec1, op_vec2, 2, 3);
        printf("Result (matrix_grad + vec1 * vec2.T):\n");
        for(int i=0; i<2; ++i) {
            for(int j=0; j<3; ++j) {
                printf("%6.1f ", op_matrix_grad[i * 3 + j]);
            }
            printf("\n");
        }
        printf("Outer Product Add Inplace test complete.\n");


        // --- Test ReLU Activation ---
        printf("\nTesting ReLU Activation...\n");
        float relu_test_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        int relu_size = sizeof(relu_test_input) / sizeof(float);
        relu(relu_test_input, relu_size);
        printf("ReLU Output: [");
        for(int i=0; i<relu_size; ++i) printf("%.1f%s", relu_test_input[i], (i == relu_size - 1) ? "" : ", ");
            printf("\n");
        printf("ReLU test complete.\n");

        // --- Test Softmax Activation ---
        printf("\nTesting Softmax Activation...\n");
        float softmax_test_input[] = {1.0f, 2.0f, 3.0f};
        int softmax_size = sizeof(softmax_test_input) / sizeof(float);
        softmax(softmax_test_input, softmax_size);
        printf("Softmax Output: [");
        for(int i=0; i<softmax_size; ++i) printf("%.3f%s", softmax_test_input[i], (i == softmax_size - 1) ? "" : ", ");
        printf("\n");
        printf("Softmax test complete.\n");


        // --- Test Layer Normalization ---
        printf("\nTesting Layer Normalization...\n");
        float ln_test_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float* ln_gamma = create_float_array(5);
        float* ln_beta = create_float_array(5);
        int ln_size = sizeof(ln_test_input) / sizeof(float);
        float epsilon = 1e-5f;

        for(int i=0; i<ln_size; ++i) {
            ln_gamma[i] = 1.0f;
            ln_beta[i] = 0.0f;
        }

        layer_norm(ln_test_input, ln_gamma, ln_beta, ln_size, epsilon);
        printf("LayerNorm Output (gamma=1, beta=0): [");
        for(int i=0; i<ln_size; ++i) printf("%.3f%s", ln_test_input[i], (i == ln_size - 1) ? "" : ", ");
        printf("\n");
        
        free_float_array(ln_gamma);
        free_float_array(ln_beta);
        printf("Layer Normalization test complete.\n");

        // --- Test Dot Product ---
        printf("\nTesting Dot Product...\n");
        float dp_vec1[] = {1.0f, 2.0f, 3.0f};
        float dp_vec2[] = {4.0f, 5.0f, 6.0f};
        int dp_size = 3;
        float dp_result = dot_product(dp_vec1, dp_vec2, dp_size);
        printf("Dot Product Result: %.1f\n", dp_result);
        printf("Dot Product test complete.\n");


        // --- Test Forward Embedding ---
        printf("\nTesting Forward Embedding...\n");
        int test_token_id = 84; // ASCII for 'T'
        float* embedding_output = forward_embedding(&model->embedding, test_token_id, model->model_dim);

        if (embedding_output) {
            printf("Embedding for token ID %d (char '%c'): [", test_token_id, (char)test_token_id);
            for(int i=0; i<5; ++i) { // Print first 5 elements
                printf("%.1f%s", embedding_output[i], (i == 4) ? "" : ", ");
            }
            printf("...]\n");
            free_float_array(embedding_output);
            printf("Forward Embedding test complete.\n");
        } else {
            fprintf(stderr, "Forward Embedding test failed.\n");
        }

        // --- Test Forward Multi-Head Attention (Simplified) ---
        printf("\nTesting Simplified Forward Multi-Head Attention...\n");
        float* mha_input = create_float_array(MODEL_DIM); // Dummy input vector
        if (mha_input) {
            for(int i=0; i<MODEL_DIM; ++i) mha_input[i] = (float)(i % 10) * 0.1f;
            float* mha_output = forward_multi_head_attention(&model->transformer_blocks[0].attention, mha_input, MODEL_DIM);
            if (mha_output) {
                printf("MHA Input (first 5): [");
                for(int i=0; i<5; ++i) printf("%.1f%s", mha_input[i], (i == 4) ? "" : ", ");
                printf("...]\n");
                printf("MHA Output (first 5): [");
                for(int i=0; i<5; ++i) printf("%.3f%s", mha_output[i], (i == 4) ? "" : ", ");
                printf("...]\n");
                free_float_array(mha_output);
                printf("Simplified Forward Multi-Head Attention test complete.\n");
            } else {
                fprintf(stderr, "Simplified Forward Multi-Head Attention test failed.\n");
            }
            free_float_array(mha_input);
        } else {
            fprintf(stderr, "Memory allocation for MHA input failed.\n");
        }

        // --- Test Forward Feed-Forward Network ---
        printf("\nTesting Forward Feed-Forward Network...\n");
        float* ffn_input = create_float_array(MODEL_DIM); // Dummy input vector
        if (ffn_input) {
            for(int i=0; i<MODEL_DIM; ++i) ffn_input[i] = (float)(i % 7) * 0.05f; // Different pattern

            const FeedForwardLayer* ffn_layer = &model->transformer_blocks[0].ffn;
            if (!ffn_layer || !ffn_layer->Wi.data || !ffn_layer->Wo.data || !ffn_layer->bi || !ffn_layer->bo) {
                fprintf(stderr, "Error: FFN layer data is NULL before calling forward_feed_forward. Wi.data=%p, Wo.data=%p, bi=%p, bo=%p\n", (void*)ffn_layer->Wi.data, (void*)ffn_layer->Wo.data, (void*)ffn_layer->bi, (void*)ffn_layer->bo);
                free_float_array(ffn_input);
            } else {
                float* ffn_output = forward_feed_forward(ffn_layer, ffn_input, MODEL_DIM);
                if (ffn_output) {
                    printf("FFN Input (first 5): [");
                    for(int i=0; i<5; ++i) printf("%.2f%s", ffn_input[i], (i == 4) ? "" : ", ");
                    printf("...]\n");
                    printf("FFN Output (first 5): [");
                    for(int i=0; i<5; ++i) printf("%.3f%s", ffn_output[i], (i == 4) ? "" : ", ");
                    printf("...]\n");
                    free_float_array(ffn_output);
                    printf("Forward Feed-Forward Network test complete.\n");
                } else {
                    fprintf(stderr, "Forward Feed-Forward Network test failed.\n");
                }
            }
            free_float_array(ffn_input);
        } else {
            fprintf(stderr, "Memory allocation for FFN input failed.\n");
        }

        // --- Test Forward Transformer Block ---
        printf("\nTesting Forward Transformer Block...\n");
        float* block_input = create_float_array(MODEL_DIM);
        if (block_input) {
            for(int i=0; i<MODEL_DIM; ++i) block_input[i] = (float)(i % 5) * 0.01f; // Another pattern

            float* block_output = forward_transformer_block(&model->transformer_blocks[0], block_input, MODEL_DIM);
            if (block_output) {
                printf("Block Input (first 5): [");
                for(int i=0; i<5; ++i) printf("%.2f%s", block_input[i], (i == 4) ? "" : ", ");
                printf("...]\n");
                printf("Block Output (first 5): [");
                for(int i=0; i<5; ++i) printf("%.3f%s", block_output[i], (i == 4) ? "" : ", ");
                printf("...]\n");
                free_float_array(block_output);
                printf("Forward Transformer Block test complete.\n");
            } else {
                fprintf(stderr, "Forward Transformer Block test failed.\n");
            }
            free_float_array(block_input);
        } else {
            fprintf(stderr, "Memory allocation for Transformer Block input failed.\n");
        }

        // --- Test Full LLM Forward Pass ---
        printf("\nTesting Full LLM Forward Pass...\n");
        int test_llm_token_id = 104; // ASCII for 'h'
        // Need to create a new model as forward_llm now modifies model->final_hidden_state_input and model->block_contexts
        // For testing, we'll allocate a fresh model to ensure no interference with previous tests.
        LegacyLLM* model_for_full_llm_test = create_legacy_llm(vocab_size, MODEL_DIM, num_blocks);
        if (model_for_full_llm_test) {
            float* llm_output_probs = forward_llm(model_for_full_llm_test, test_llm_token_id);
            if (llm_output_probs) {
                printf("LLM Output Probabilities (first 5): [");
                for(int i=0; i<5; ++i) { // Print first 5 probabilities
                    printf("%.3f%s", llm_output_probs[i], (i == 4) ? "" : ", ");
                }
                printf("...]\n");
                free_float_array(llm_output_probs);
                printf("Full LLM Forward Pass test complete.\n");
            } else {
                fprintf(stderr, "Full LLM Forward Pass test failed.\n");
            }
            free_legacy_llm(model_for_full_llm_test);
        } else {
            fprintf(stderr, "Memory allocation for model_for_full_llm_test failed!\n");
        }

        // --- Test Cross-Entropy Loss ---
        printf("\nTesting Cross-Entropy Loss...\n");
        float predicted_probs_ce[] = {0.1f, 0.2f, 0.7f}; // Example probabilities
        int true_token_id_ce = 2; // True token is the third one (index 2)
        int vocab_size_ce = 3;
        float loss = cross_entropy_loss(predicted_probs_ce, true_token_id_ce, vocab_size_ce);
        printf("Predicted Probs: [");
        for(int i=0; i<vocab_size_ce; ++i) printf("%.1f%s", predicted_probs_ce[i], (i == vocab_size_ce - 1) ? "" : ", ");
        printf("], True Token ID: %d, Loss: %.3f\n", true_token_id_ce, loss);
        printf("Cross-Entropy Loss test complete.\n");

        // --- Test d_loss_d_logits ---
        printf("\nTesting d_loss_d_logits...\n");
        float predicted_probs_dldl[] = {0.1f, 0.2f, 0.7f};
        int true_token_id_dldl = 2;
        int vocab_size_dldl = 3;
        float* gradient_dldl = d_loss_d_logits(predicted_probs_dldl, true_token_id_dldl, vocab_size_dldl);
        if (gradient_dldl) {
            printf("Predicted Probs: [");
            for(int i=0; i<vocab_size_dldl; ++i) printf("%.1f%s", predicted_probs_dldl[i], (i == vocab_size_dldl - 1) ? "" : ", ");
            printf("], True Token ID: %d, dLoss/dLogits: [", true_token_id_dldl);
            for(int i=0; i<vocab_size_dldl; ++i) printf("%.1f%s", gradient_dldl[i], (i == vocab_size_dldl - 1) ? "" : ", ");
            printf("]\n");
            free_float_array(gradient_dldl);
            printf("d_loss_d_logits test complete.\n");
        } else {
            fprintf(stderr, "d_loss_d_logits test failed.\n");
        }

        // --- Test Backward Output Layer ---
        printf("\nTesting Backward Output Layer...\n");
        // Mock data for testing backward_output_layer
        float test_hidden_state_input_bol[] = {0.1f, 0.5f, -0.2f}; // model_dim = 3 for this test
        float test_d_loss_d_logits_bol[] = {0.05f, 0.15f, -0.2f}; // vocab_size = 3 for this test

        // Create a temporary OutputLayer and Grads for testing
        OutputLayer temp_output_layer_bol;
        temp_output_layer_bol.unembedding_weights = create_ternary_matrix(3, 3); // model_dim x vocab_size (3x3)
        temp_output_layer_bol.bias = create_float_array(3);
        
        LegacyLLM_Gradients* temp_grads_bol = create_legacy_llm_gradients(3, 3, 0); // vocab, model_dim, num_blocks
        
        if (temp_output_layer_bol.unembedding_weights.data && temp_output_layer_bol.bias && temp_grads_bol) {
            float* d_loss_d_hidden_state_output_bol = backward_output_layer(&temp_output_layer_bol, test_d_loss_d_logits_bol, test_hidden_state_input_bol, 3, 3, temp_grads_bol);

            if (d_loss_d_hidden_state_output_bol) {
                printf("dLoss/dBias (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", temp_grads_bol->output_grads.bias[i], (i == 2) ? "" : ", ");
                printf("]\n");

                printf("dLoss/dUnembeddingWeights (first 3x3 block):\n");
                for(int i=0; i<3; ++i) {
                    for(int j=0; j<3; ++j) {
                        printf("%6.3f ", temp_grads_bol->output_grads.unembedding_weights[i * 3 + j]);
                    }
                    printf("\n");
                }
                
                printf("dLoss/dHiddenState (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", d_loss_d_hidden_state_output_bol[i], (i == 2) ? "" : ", ");
                printf("]\n");

                free_float_array(d_loss_d_hidden_state_output_bol);
                printf("Backward Output Layer test complete.\n");
            } else {
                fprintf(stderr, "Backward Output Layer test failed.\n");
            }
        } else {
            fprintf(stderr, "Memory allocation for Backward Output Layer test failed.\n");
        }

        free_ternary_matrix(&temp_output_layer_bol.unembedding_weights);
        free_float_array(temp_output_layer_bol.bias);
        free_legacy_llm_gradients(temp_grads_bol);


        // --- Test Backward Feed-Forward Network ---
        printf("\nTesting Backward Feed-Forward Network...\n");
        // Mock data
        float test_d_loss_d_ffn_output[] = {0.1f, 0.2f, 0.3f}; // model_dim = 3
        float test_ffn_input[] = {0.5f, 0.6f, 0.7f}; // model_dim = 3
        float test_hidden_pre_relu_output[] = {-0.1f, 0.8f, 1.2f, -0.5f, 0.3f, 0.6f, 0.9f, -0.2f, 0.1f, 0.4f, 0.7f, 1.0f}; // ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER = 3 * 4 = 12
        
        FeedForwardLayer temp_ffn_layer;
        temp_ffn_layer.Wi = create_ternary_matrix(12, 3); // ffn_hidden_dim x model_dim
        temp_ffn_layer.Wo = create_ternary_matrix(3, 12); // model_dim x ffn_hidden_dim
        temp_ffn_layer.bi = create_float_array(12);
        temp_ffn_layer.bo = create_float_array(3);

        LegacyLLM_Gradients* temp_grads_ffn = create_legacy_llm_gradients(3, 3, 1); // vocab, model_dim, num_blocks=1 for ffn
        
        if (temp_ffn_layer.Wi.data && temp_ffn_layer.Wo.data && temp_ffn_layer.bi && temp_ffn_layer.bo && temp_grads_ffn) {
            float* d_loss_d_ffn_input_output = backward_feed_forward(&temp_ffn_layer, test_d_loss_d_ffn_output, test_ffn_input, test_hidden_pre_relu_output, 3, 0, temp_grads_ffn);

            if (d_loss_d_ffn_input_output) {
                printf("dLoss/dWi (first 3x3 block of 12x3):\n");
                for(int i=0; i<3; ++i) { // Print first 3 rows
                    for(int j=0; j<3; ++j) { // Print first 3 cols
                        printf("%6.3f ", temp_grads_ffn->transformer_block_grads[0].ffn_grads.Wi[i * 3 + j]);
                    }
                    printf("...\n");
                }

                printf("dLoss/dbo (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", temp_grads_ffn->transformer_block_grads[0].ffn_grads.bo[i], (i == 2) ? "" : ", ");
                printf("]\n");

                printf("dLoss/dFFN_Input (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", d_loss_d_ffn_input_output[i], (i == 2) ? "" : ", ");
                printf("]\n");

                free_float_array(d_loss_d_ffn_input_output);
                printf("Backward Feed-Forward Network test complete.\n");
            } else {
                fprintf(stderr, "Backward Feed-Forward Network test failed.\n");
            }
        } else {
            fprintf(stderr, "Memory allocation for Backward Feed-Forward Network test failed.\n");
        }

        free_ternary_matrix(&temp_ffn_layer.Wi);
        free_ternary_matrix(&temp_ffn_layer.Wo);
        free_float_array(temp_ffn_layer.bi);
        free_float_array(temp_ffn_layer.bo);
        free_legacy_llm_gradients(temp_grads_ffn);

        // --- Test Backward Layer Normalization ---
        printf("\nTesting Backward Layer Normalization...\n");
        float test_d_loss_d_norm_output[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}; // size = 5
        float test_ln_input_vec[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // size = 5
        float test_norm_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // size = 5
        float test_norm_beta[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // size = 5
        float test_mean = 3.0f;
        float test_inv_std_dev = 1.0f / sqrtf(2.0f + 1e-5f); // mean=3, var=2, std_dev=sqrt(2)
        int test_ln_size = 5;
        
        LegacyLLM_Gradients* temp_grads_ln = create_legacy_llm_gradients(5, 5, 1); // vocab, model_dim, num_blocks
        
        if (temp_grads_ln) {
            float* d_loss_d_ln_input = backward_layer_norm(test_d_loss_d_norm_output, test_ln_input_vec, test_norm_gamma, test_norm_beta, test_mean, test_inv_std_dev, test_ln_size, 0, 1, temp_grads_ln);

            if (d_loss_d_ln_input) {
                printf("dLoss/dGamma (first 5): [");
                for(int i=0; i<5; ++i) printf("%.3f%s", temp_grads_ln->transformer_block_grads[0].norm1_gamma[i], (i == 4) ? "" : ", ");
                printf("]\n");

                printf("dLoss/dBeta (first 1): ["); // Beta is accumulated as a sum, so one value here
                printf("%.3f", temp_grads_ln->transformer_block_grads[0].norm1_beta[0]);
                printf("]\n");

                printf("dLoss/dInput (first 5): [");
                for(int i=0; i<5; ++i) printf("%.3f%s", d_loss_d_ln_input[i], (i == 4) ? "" : ", ");
                printf("]\n");

                free_float_array(d_loss_d_ln_input);
                printf("Backward Layer Normalization test complete.\n");
            } else {
                fprintf(stderr, "Backward Layer Normalization test failed.\n");
            }
        } else {
            fprintf(stderr, "Memory allocation for Backward Layer Normalization test failed.\n");
        }
        free_legacy_llm_gradients(temp_grads_ln);

        // --- Test Backward Multi-Head Attention ---
        printf("\nTesting Backward Multi-Head Attention...\n");
        // Mock data for testing backward_multi_head_attention
        float test_d_loss_d_mha_output[] = {0.1f, 0.2f, 0.3f}; // model_dim = 3
        float test_mha_input[] = {0.4f, 0.5f, 0.6f}; // model_dim = 3
        // For simplified attention, query/key/value pre-bias/post-bias are essentially just inputs transformed.
        // We'll just pass copies of input for now, in a real scenario these would be stored from forward pass
        float test_query_vec_pre_bias[] = {0.1f, 0.1f, 0.1f};
        float test_key_vec_pre_bias[] = {0.1f, 0.1f, 0.1f};
        float test_value_vec_pre_bias[] = {0.1f, 0.1f, 0.1f};
        float test_query_vec[] = {0.2f, 0.2f, 0.2f};
        float test_key_vec[] = {0.2f, 0.2f, 0.2f};
        float test_value_vec[] = {0.2f, 0.2f, 0.2f};
        float test_attention_output_vec[] = {0.25f, 0.25f, 0.25f}; // Simplified, was value_vec copy

        MultiHeadAttentionLayer temp_mha_layer;
        temp_mha_layer.Wq = create_ternary_matrix(3,3); temp_mha_layer.Wk = create_ternary_matrix(3,3); temp_mha_layer.Wv = create_ternary_matrix(3,3); temp_mha_layer.Wo = create_ternary_matrix(3,3);
        temp_mha_layer.bq = create_float_array(3); temp_mha_layer.bk = create_float_array(3); temp_mha_layer.bv = create_float_array(3); temp_mha_layer.bo = create_float_array(3);

        LegacyLLM_Gradients* temp_grads_mha = create_legacy_llm_gradients(3, 3, 1); // vocab, model_dim, num_blocks=1
        
        if (temp_mha_layer.Wq.data && temp_mha_layer.Wk.data && temp_mha_layer.Wv.data && temp_mha_layer.Wo.data &&
            temp_mha_layer.bq && temp_mha_layer.bk && temp_mha_layer.bv && temp_mha_layer.bo && temp_grads_mha) {
            
            float* d_loss_d_mha_input = backward_multi_head_attention(&temp_mha_layer, test_d_loss_d_mha_output, test_mha_input,
                                                          test_query_vec_pre_bias, test_key_vec_pre_bias, test_value_vec_pre_bias,
                                                          test_query_vec, test_key_vec, test_value_vec, test_attention_output_vec,
                                                          3, 0, temp_grads_mha);
            if (d_loss_d_mha_input) {
                printf("dLoss/dbo (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", temp_grads_mha->transformer_block_grads[0].attention_grads.bo[i], (i == 2) ? "" : ", ");
                printf("]\n");

                printf("dLoss/dWo (first 3x3 block):\n");
                for(int i=0; i<3; ++i) {
                    for(int j=0; j<3; ++j) {
                        printf("%6.3f ", temp_grads_mha->transformer_block_grads[0].attention_grads.Wo[i * 3 + j]);
                    }
                    printf("\n");
                }
                
                printf("dLoss/dbv (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", temp_grads_mha->transformer_block_grads[0].attention_grads.bv[i], (i == 2) ? "" : ", ");
                printf("]\n");

                printf("dLoss/dWv (first 3x3 block):\n");
                for(int i=0; i<3; ++i) {
                    for(int j=0; j<3; ++j) {
                        printf("%6.3f ", temp_grads_mha->transformer_block_grads[0].attention_grads.Wv[i * 3 + j]);
                    }
                    printf("\n");
                }

                printf("dLoss/dInput (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", d_loss_d_mha_input[i], (i == 2) ? "" : ", ");
                printf("]\n");
                
                free_float_array(d_loss_d_mha_input);
                printf("Backward Multi-Head Attention test complete.\n");
            } else {
                fprintf(stderr, "Backward Multi-Head Attention test failed.\n");
            }
        } else {
            fprintf(stderr, "Memory allocation for Backward Multi-Head Attention test failed.\n");
        }

        free_ternary_matrix(&temp_mha_layer.Wq); free_ternary_matrix(&temp_mha_layer.Wk); free_ternary_matrix(&temp_mha_layer.Wv); free_ternary_matrix(&temp_mha_layer.Wo);
        free_float_array(temp_mha_layer.bq); free_float_array(temp_mha_layer.bk); free_float_array(temp_mha_layer.bv); free_float_array(temp_mha_layer.bo);
        free_legacy_llm_gradients(temp_grads_mha);

        // --- Test Backward Transformer Block ---
        printf("\nTesting Backward Transformer Block...\n");
        // Mock data
        float test_d_loss_d_block_output[] = {0.1f, 0.2f, 0.3f}; // model_dim = 3
        float test_block_input[] = {0.05f, 0.10f, 0.15f}; // model_dim = 3

        // Create a temporary TransformerBlock and Grads for testing
        TransformerBlock temp_block;
        temp_block.attention.Wq = create_ternary_matrix(3,3); temp_block.attention.Wk = create_ternary_matrix(3,3); temp_block.attention.Wv = create_ternary_matrix(3,3); temp_block.attention.Wo = create_ternary_matrix(3,3);
        temp_block.attention.bq = create_float_array(3); temp_block.attention.bk = create_float_array(3); temp_block.attention.bv = create_float_array(3); temp_block.attention.bo = create_float_array(3);
        temp_block.ffn.Wi = create_ternary_matrix(12,3); temp_block.ffn.Wo = create_ternary_matrix(3,12);
        temp_block.ffn.bi = create_float_array(12); temp_block.ffn.bo = create_float_array(3);
        temp_block.norm1_gamma = create_float_array(3); temp_block.norm1_beta = create_float_array(3);
        temp_block.norm2_gamma = create_float_array(3); temp_block.norm2_beta = create_float_array(3);

        LegacyLLM_Gradients* temp_grads_block = create_legacy_llm_gradients(3, 3, 1); // vocab, model_dim, num_blocks=1
        TransformerBlockContext* temp_context_block = create_transformer_block_context(3, 12); // model_dim, ffn_hidden_dim
        
        if (temp_grads_block && temp_context_block && temp_block.attention.Wq.data && temp_block.ffn.Wi.data) { // Basic checks
            // Populate context (from a dummy forward pass)
            float* dummy_block_output = forward_transformer_block_with_context(&temp_block, test_block_input, 3, temp_context_block);
            if (dummy_block_output) free_float_array(dummy_block_output);
            
            float* d_loss_d_block_input_output = backward_transformer_block(&temp_block, test_d_loss_d_block_output, 3, 0, temp_grads_block, temp_context_block);
            
            if (d_loss_d_block_input_output) {
                printf("dLoss/dInput (first 3): [");
                for(int i=0; i<3; ++i) printf("%.3f%s", d_loss_d_block_input_output[i], (i == 2) ? "" : ", ");
                printf("]\n");

                printf("Backward Transformer Block test complete.\n");
                free_float_array(d_loss_d_block_input_output);
            } else {
                fprintf(stderr, "Backward Transformer Block test failed.\n");
            }
        } else {
            fprintf(stderr, "Memory allocation for Backward Transformer Block test failed.\n");
        }

        free_ternary_matrix(&temp_block.attention.Wq); free_ternary_matrix(&temp_block.attention.Wk); free_ternary_matrix(&temp_block.attention.Wv); free_ternary_matrix(&temp_block.attention.Wo);
        free_float_array(temp_block.attention.bq); free_float_array(temp_block.attention.bk); free_float_array(temp_block.attention.bv); free_float_array(temp_block.attention.bo);
        free_ternary_matrix(&temp_block.ffn.Wi); free_ternary_matrix(&temp_block.ffn.Wo);
        free_float_array(temp_block.ffn.bi); free_float_array(temp_block.ffn.bo);
        free_float_array(temp_block.norm1_gamma); free_float_array(temp_block.norm1_beta);
        free_float_array(temp_block.norm2_gamma); free_float_array(temp_block.norm2_beta);
        free_legacy_llm_gradients(temp_grads_block);
        free_transformer_block_context(temp_context_block);

        // --- Test Backward Embedding ---
        printf("\nTesting Backward Embedding...\n");
        float test_d_loss_d_embedding_output[] = {0.1f, 0.2f, 0.3f}; // model_dim = 3
        int test_token_id_be = 5; // Example token ID
        LegacyLLM_Gradients* temp_grads_be = create_legacy_llm_gradients(256, 3, 0); // vocab=256, model_dim=3, num_blocks=0
        EmbeddingLayer temp_embedding_layer_be;
        temp_embedding_layer_be.embedding_weights = create_ternary_matrix(256, 3); // vocab_size x model_dim

        if (temp_grads_be && temp_embedding_layer_be.embedding_weights.data) {
            backward_embedding(&temp_embedding_layer_be, test_token_id_be, test_d_loss_d_embedding_output, 3, temp_grads_be);
            printf("dLoss/dEmbeddingWeights for token %d (first 3): [", test_token_id_be);
            for(int i=0; i<3; ++i) {
                printf("%.3f%s", temp_grads_be->embedding_grads.embedding_weights[test_token_id_be * 3 + i], (i == 2) ? "" : ", ");
            }
            printf("]\n");
            printf("Backward Embedding test complete.\n");
        } else {
            fprintf(stderr, "Memory allocation for Backward Embedding test failed.\n");
        }
        free_ternary_matrix(&temp_embedding_layer_be.embedding_weights);
        free_legacy_llm_gradients(temp_grads_be);

        // --- Test Full LLM Backward Pass ---
        printf("\nTesting Full LLM Backward Pass...\n");
        int test_token_id_bllm = 84; // ASCII for 'T'
        int test_true_token_id_bllm = 101; // ASCII for 'e' (next token after 'T' in 'The')

        // Need a fresh model and grads for this test to ensure gradients are reset
        LegacyLLM* model_for_bllm_test = create_legacy_llm(vocab_size, MODEL_DIM, num_blocks);
        LegacyLLM_Gradients* grads_for_bllm_test = create_legacy_llm_gradients(vocab_size, MODEL_DIM, num_blocks);

        if (model_for_bllm_test && grads_for_bllm_test) {
            backward_llm(model_for_bllm_test, test_token_id_bllm, test_true_token_id_bllm, grads_for_bllm_test);
            
            // Just print some sample gradients to show it ran
            printf("Sample dLoss/dOutputBias (first 3): [");
            for(int i=0; i<3; ++i) printf("%.3f%s", grads_for_bllm_test->output_grads.bias[i], (i == 2) ? "" : ", ");
            printf("]\n");

            printf("Sample dLoss/dEmbeddingWeights for token %d (first 3): [", test_token_id_bllm);
            for(int i=0; i<3; ++i) {
                printf("%.3f%s", grads_for_bllm_test->embedding_grads.embedding_weights[test_token_id_bllm * MODEL_DIM + i], (i == 2) ? "" : ", ");
            }
            printf("]\n");

            printf("Full LLM Backward Pass test complete.\n");
        } else {
            fprintf(stderr, "Memory allocation for Full LLM Backward Pass test failed.\n");
        }
        free_legacy_llm(model_for_bllm_test);
        free_legacy_llm_gradients(grads_for_bllm_test);


                free_legacy_llm(model);
    } else {
        fprintf(stderr, "Model allocation failed!\n");
    }

    // --- Test Model Save/Load ---
    test_model_save_load();

}

void test_model_save_load() {
    printf("\nTesting Model Save/Load...\n");

    int vocab_size = 256;
    int model_dim = 64;
    int num_blocks = 2;
    const char* test_filepath = "temp_test_model.bin";

    // 1. Create original model
    LegacyLLM* original_model = create_legacy_llm(vocab_size, model_dim, num_blocks);
    if (!original_model) {
        fprintf(stderr, "Error: Failed to create original model for save/load test.\n");
        return;
    }

    // 2. Save original model
    printf("Saving original model to %s...\n", test_filepath);
    if (!save_model(original_model, test_filepath)) {
        fprintf(stderr, "Error: Failed to save original model.\n");
        free_legacy_llm(original_model);
        return;
    }

    // 3. Load model
    printf("Loading model from %s...\n", test_filepath);
    LegacyLLM* loaded_model = load_model(test_filepath);
    if (!loaded_model) {
        fprintf(stderr, "Error: Failed to load model for save/load test.\n");
        free_legacy_llm(original_model);
        remove(test_filepath); // Clean up temp file
        return;
    }

    // 4. Compare models
    int test_failed = 0;
    if (original_model->vocab_size != loaded_model->vocab_size ||
        original_model->model_dim != loaded_model->model_dim ||
        original_model->num_transformer_blocks != loaded_model->num_transformer_blocks) {
        fprintf(stderr, "Mismatch in model dimensions after save/load.\n");
        test_failed = 1;
    }

    // Compare embedding weights (first few elements)
    for (int i = 0; i < 10 && i < vocab_size * model_dim; ++i) {
        if (original_model->embedding.embedding_weights.data[i] != loaded_model->embedding.embedding_weights.data[i]) {
            fprintf(stderr, "Mismatch in embedding weights at index %d: original %d, loaded %d.\n",
                    i, original_model->embedding.embedding_weights.data[i], loaded_model->embedding.embedding_weights.data[i]);
            test_failed = 1;
            break;
        }
    }
    
    // Compare a bias (first few elements of output bias)
    for (int i = 0; i < 5 && i < vocab_size; ++i) {
        if (fabs(original_model->output.bias[i] - loaded_model->output.bias[i]) > 1e-6) {
            fprintf(stderr, "Mismatch in output bias at index %d: original %.6f, loaded %.6f.\n",
                    i, original_model->output.bias[i], loaded_model->output.bias[i]);
            test_failed = 1;
            break;
        }
    }


    if (!test_failed) {
        printf("Model Save/Load test successful.\n");
    } else {
        fprintf(stderr, "Model Save/Load test failed!\n");
    }

    // 5. Clean up
    free_legacy_llm(original_model);
    free_legacy_llm(loaded_model);
    remove(test_filepath); // Delete the temporary model file

    printf("Model Save/Load test complete.\n");
}
