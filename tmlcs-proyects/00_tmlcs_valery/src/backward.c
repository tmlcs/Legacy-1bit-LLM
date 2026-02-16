#include <stdio.h> // For fprintf
#include <stdlib.h> // For calloc, free
#include <string.h> // For memcpy
#include <math.h>   // For logf

#include "backward.h"
#include "legacy_llm.h" // For struct definitions
#include "model.h" // For create_float_array, free_float_array
#include "math_ops.h" // For vector_sum, multiply_vector_inplace, outer_product_add_inplace, matrix_transpose_vector_mul, vector_sub_scalar_inplace
#include "forward.h" // For forward_llm (used in backward_llm)


// --- Training Infrastructure Functions ---

// Calculates cross-entropy loss between predicted probabilities and true label.
float cross_entropy_loss(const float* predicted_probs, int true_token_id, int vocab_size) {
    if (!predicted_probs || true_token_id < 0 || true_token_id >= vocab_size) {
        fprintf(stderr, "Error: Invalid input to cross_entropy_loss\n");
        return -1.0f; // Indicate error
    }
    // Cross-entropy loss for a single sample is -log(predicted_prob_for_true_token)
    // Add a small epsilon to avoid log(0) if probabilities are exactly 0.
    float prob = predicted_probs[true_token_id];
    if (prob < 1e-9f) { // Clamp minimum probability
        prob = 1e-9f;
    }
    return -logf(prob);
}

// Calculates the gradient of cross-entropy loss with respect to the input logits of softmax.
// This is simply (predicted_probs - one_hot_true_label).
// Output is a float array (dLoss/dLogits), caller must free.
float* d_loss_d_logits(const float* predicted_probs, int true_token_id, int vocab_size) {
    if (!predicted_probs || true_token_id < 0 || true_token_id >= vocab_size) {
        fprintf(stderr, "Error: Invalid input to d_loss_d_logits\n");
        return NULL;
    }

    float* gradient = create_float_array(vocab_size);
    if (!gradient) return NULL;

    for (int i = 0; i < vocab_size; ++i) {
        gradient[i] = predicted_probs[i];
        if (i == true_token_id) {
            gradient[i] -= 1.0f; // Subtract 1 from the true class probability
        }
    }
    return gradient;
}

// Backward pass for the Output Layer
// d_loss_d_output_logits: gradient of loss wrt the input to the softmax (same as d_loss_d_logits_vec)
// hidden_state_input: the input vector that was fed into the output layer (from last transformer block)
// Returns: dLoss/dHiddenState (to be propagated backwards), caller must free.
float* backward_output_layer(const OutputLayer* layer, const float* d_loss_d_output_logits, const float* hidden_state_input, int model_dim, int vocab_size, LegacyLLM_Gradients* grads) {
    if (!layer || !d_loss_d_output_logits || !hidden_state_input || !grads) {
        fprintf(stderr, "Error: Invalid input to backward_output_layer\n");
        return NULL;
    }

    // 1. Gradients for Output Bias (dLoss/dBias = dLoss/dLogits)
    // Add to accumulated gradients, do not just assign. For batch processing.
    for (int i = 0; i < vocab_size; ++i) {
        grads->output_grads.bias[i] += d_loss_d_output_logits[i];
    }

    // 2. Gradients for Unembedding Weights (dLoss/dW = dLoss/dLogits * hidden_state_input_transpose)
    // dLoss/d(W_uv) = dLoss/d(Logits_v) * hidden_state_input_u
    // Where layer->unembedding_weights is model_dim rows x vocab_size cols (matrix U)
    // dLoss/dU_jk = dLoss/dLogits_k * hidden_state_input_j
    outer_product_add_inplace(grads->output_grads.unembedding_weights, hidden_state_input, d_loss_d_output_logits, model_dim, vocab_size);

    // 3. Gradients for Hidden State Input (dLoss/dHiddenState = W_transpose * dLoss/dLogits)
    float* d_loss_d_hidden_state = create_float_array(model_dim);
    if (!d_loss_d_hidden_state) return NULL;

    // dLoss/dHiddenState = unembedding_weights_transpose * dLoss/dLogits
    matrix_transpose_vector_mul(&layer->unembedding_weights, d_loss_d_output_logits, d_loss_d_hidden_state);

    return d_loss_d_hidden_state;
}

// Backward pass for the Output Layer (Batched version)
// d_loss_d_output_logits_batch: flattened gradients of loss wrt logits for the batch
// hidden_state_input_batch: flattened hidden state inputs to the output layer for the batch
// Returns: dLoss/dHiddenState (flattened, to be propagated backwards), caller must free.
float* backward_output_layer_batch(const OutputLayer* layer, const float* d_loss_d_output_logits_batch, const float* hidden_state_input_batch, int batch_size, int model_dim, int vocab_size, LegacyLLM_Gradients* grads) {
    if (!layer || !d_loss_d_output_logits_batch || !hidden_state_input_batch || !grads) {
        fprintf(stderr, "Error: Invalid input to backward_output_layer_batch\n");
        return NULL;
    }

    float* d_loss_d_hidden_state_batch = create_float_array(batch_size * model_dim);
    if (!d_loss_d_hidden_state_batch) return NULL;

    for (int b = 0; b < batch_size; ++b) {
        const float* d_loss_d_output_logits_single = &d_loss_d_output_logits_batch[b * vocab_size];
        const float* hidden_state_input_single = &hidden_state_input_batch[b * model_dim];
        float* d_loss_d_hidden_state_single = &d_loss_d_hidden_state_batch[b * model_dim];

        // 1. Gradients for Output Bias (dLoss/dBias = dLoss/dLogits)
        for (int i = 0; i < vocab_size; ++i) {
            grads->output_grads.bias[i] += d_loss_d_output_logits_single[i];
        }

        // 2. Gradients for Unembedding Weights (dLoss/dW = dLoss/dLogits * hidden_state_input_transpose)
        outer_product_add_inplace(grads->output_grads.unembedding_weights, hidden_state_input_single, d_loss_d_output_logits_single, model_dim, vocab_size);

        // 3. Gradients for Hidden State Input (dLoss/dHiddenState = W_transpose * dLoss/dLogits)
        matrix_transpose_vector_mul(&layer->unembedding_weights, d_loss_d_output_logits_single, d_loss_d_hidden_state_single);
    }
    return d_loss_d_hidden_state_batch;
}


// Backward pass for Feed-Forward Network
// d_loss_d_ffn_output: gradient of loss wrt the output of the FFN
// ffn_input: the input vector to the FFN (input to Wi)
// hidden_pre_relu_output: the output of the first linear layer (Wi * input + bi) BEFORE ReLU
// Returns: dLoss/dFFN_Input (to be propagated backwards), caller must free.
float* backward_feed_forward(const FeedForwardLayer* layer, const float* d_loss_d_ffn_output, const float* ffn_input, const float* hidden_pre_relu_output, int model_dim, int block_idx, LegacyLLM_Gradients* grads) {
    if (!layer || !d_loss_d_ffn_output || !ffn_input || !hidden_pre_relu_output || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks) {
        fprintf(stderr, "Error: Invalid input to backward_feed_forward\n");
        return NULL;
    }

    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;

    // 1. Gradients for Output Bias (dLoss/dbo = dLoss/dFFN_Output)
    for (int i = 0; i < model_dim; ++i) {
        grads->transformer_block_grads[block_idx].ffn_grads.bo[i] += d_loss_d_ffn_output[i];
    }

    // 2. Gradients for Wo (dLoss/dWo = dLoss/dFFN_Output * hidden_output_after_relu_transpose)
    // Wo is (model_dim x ffn_hidden_dim)
    // dLoss/dWo_ij = dLoss/dFFN_Output_i * hidden_output_after_relu_j
    // Note: hidden_pre_relu_output is what is typically called hidden_output_after_relu in this context
    // dLoss/dWo: dLoss/dFFN_Output is (model_dim x 1), hidden_pre_relu_output is (ffn_hidden_dim x 1)
    // So dLoss/dWo is (model_dim x ffn_hidden_dim)
    outer_product_add_inplace(grads->transformer_block_grads[block_idx].ffn_grads.Wo, d_loss_d_ffn_output, hidden_pre_relu_output, model_dim, ffn_hidden_dim);
    
    // Calculate dLoss/dHidden_after_relu = Wo_transpose * dLoss/dFFN_Output
    float* d_loss_d_hidden_after_relu = create_float_array(ffn_hidden_dim);
    if (!d_loss_d_hidden_after_relu) return NULL;
    matrix_transpose_vector_mul(&layer->Wo, d_loss_d_ffn_output, d_loss_d_hidden_after_relu);
    
    // 3. Gradients for ReLU (dLoss/dHidden_pre_relu)
    // d(ReLU)/d(input_to_ReLU) is 1 if input > 0, else 0.
    float* d_loss_d_hidden_pre_relu = create_float_array(ffn_hidden_dim);
    if (!d_loss_d_hidden_pre_relu) { free_float_array(d_loss_d_hidden_after_relu); return NULL; }

    for (int i = 0; i < ffn_hidden_dim; ++i) {
        if (hidden_pre_relu_output[i] > 0) { // Only propagate gradient if input was positive
            d_loss_d_hidden_pre_relu[i] = d_loss_d_hidden_after_relu[i];
        } else {
            d_loss_d_hidden_pre_relu[i] = 0.0f;
        }
    }
    free_float_array(d_loss_d_hidden_after_relu);

    // 4. Gradients for Wi (dLoss/dWi = dLoss/dHidden_pre_relu * ffn_input_transpose)
    outer_product_add_inplace(grads->transformer_block_grads[block_idx].ffn_grads.Wi, d_loss_d_hidden_pre_relu, ffn_input, ffn_hidden_dim, model_dim);

    // 5. Gradients for Bi (dLoss/dbi = dLoss/dHidden_pre_relu)
    for (int i = 0; i < ffn_hidden_dim; ++i) {
        grads->transformer_block_grads[block_idx].ffn_grads.bi[i] += d_loss_d_hidden_pre_relu[i];
    }

    // 6. Gradients for FFN Input (dLoss/dFFN_Input = Wi_transpose * dLoss/dHidden_pre_relu)
    float* d_loss_d_ffn_input = create_float_array(model_dim);
    if (!d_loss_d_ffn_input) { free_float_array(d_loss_d_hidden_pre_relu); return NULL; }

    matrix_transpose_vector_mul(&layer->Wi, d_loss_d_hidden_pre_relu, d_loss_d_ffn_input);
    free_float_array(d_loss_d_hidden_pre_relu);

        return d_loss_d_ffn_input;

    }

    

    // Backward pass for Feed-Forward Network (Batched version)

    // d_loss_d_ffn_output_batch: flattened gradients of loss wrt the output of the FFN

    // ffn_input_batch: flattened input vectors to the FFN

    // hidden_pre_relu_output_batch: flattened output of the first linear layer (Wi * input + bi) BEFORE ReLU

    // Returns: dLoss/dFFN_Input (flattened, to be propagated backwards), caller must free.

    float* backward_feed_forward_batch(const FeedForwardLayer* layer, const float* d_loss_d_ffn_output_batch, const float* ffn_input_batch, const float* hidden_pre_relu_output_batch, int batch_size, int model_dim, int block_idx, LegacyLLM_Gradients* grads) {

        if (!layer || !d_loss_d_ffn_output_batch || !ffn_input_batch || !hidden_pre_relu_output_batch || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks) {

            fprintf(stderr, "Error: Invalid input to backward_feed_forward_batch\n");

            return NULL;

        }

    

        int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;

        float* d_loss_d_ffn_input_batch = create_float_array(batch_size * model_dim);

        if (!d_loss_d_ffn_input_batch) return NULL;

    

        for (int b = 0; b < batch_size; ++b) {

            const float* d_loss_d_ffn_output_single = &d_loss_d_ffn_output_batch[b * model_dim];

            const float* ffn_input_single = &ffn_input_batch[b * model_dim];

            const float* hidden_pre_relu_output_single = &hidden_pre_relu_output_batch[b * ffn_hidden_dim];

            float* d_loss_d_ffn_input_single = &d_loss_d_ffn_input_batch[b * model_dim];

    

            // 1. Gradients for Output Bias (dLoss/dbo = dLoss/dFFN_Output)

            for (int i = 0; i < model_dim; ++i) {

                grads->transformer_block_grads[block_idx].ffn_grads.bo[i] += d_loss_d_ffn_output_single[i];

            }

    

            // 2. Gradients for Wo (dLoss/dWo = dLoss/dFFN_Output * hidden_output_after_relu_transpose)

            outer_product_add_inplace(grads->transformer_block_grads[block_idx].ffn_grads.Wo, d_loss_d_ffn_output_single, hidden_pre_relu_output_single, model_dim, ffn_hidden_dim);

            

            // Calculate dLoss/dHidden_after_relu = Wo_transpose * dLoss/dFFN_Output

            float* d_loss_d_hidden_after_relu = create_float_array(ffn_hidden_dim);

            if (!d_loss_d_hidden_after_relu) { free_float_array(d_loss_d_ffn_input_batch); return NULL; }

            matrix_transpose_vector_mul(&layer->Wo, d_loss_d_ffn_output_single, d_loss_d_hidden_after_relu);

            

            // 3. Gradients for ReLU (dLoss/dHidden_pre_relu)

            float* d_loss_d_hidden_pre_relu = create_float_array(ffn_hidden_dim);

            if (!d_loss_d_hidden_pre_relu) { free_float_array(d_loss_d_hidden_after_relu); free_float_array(d_loss_d_ffn_input_batch); return NULL; }

    

            for (int i = 0; i < ffn_hidden_dim; ++i) {

                if (hidden_pre_relu_output_single[i] > 0) {

                    d_loss_d_hidden_pre_relu[i] = d_loss_d_hidden_after_relu[i];

                } else {

                    d_loss_d_hidden_pre_relu[i] = 0.0f;

                }

            }

            free_float_array(d_loss_d_hidden_after_relu);

    

            // 4. Gradients for Wi (dLoss/dWi = dLoss/dHidden_pre_relu * ffn_input_transpose)

            outer_product_add_inplace(grads->transformer_block_grads[block_idx].ffn_grads.Wi, d_loss_d_hidden_pre_relu, ffn_input_single, ffn_hidden_dim, model_dim);

    

            // 5. Gradients for Bi (dLoss/dbi = dLoss/dHidden_pre_relu)

            for (int i = 0; i < ffn_hidden_dim; ++i) {

                grads->transformer_block_grads[block_idx].ffn_grads.bi[i] += d_loss_d_hidden_pre_relu[i];

            }

    

            // 6. Gradients for FFN Input (dLoss/dFFN_Input = Wi_transpose * dLoss/dHidden_pre_relu)

            matrix_transpose_vector_mul(&layer->Wi, d_loss_d_hidden_pre_relu, d_loss_d_ffn_input_single);

            free_float_array(d_loss_d_hidden_pre_relu);

        }

    

        return d_loss_d_ffn_input_batch;

    }

    

    

    // Backward pass for Layer Normalization
float* backward_layer_norm(const float* d_loss_d_norm_output, const float* input_vec, const float* norm_gamma, const float* norm_beta, float mean, float inv_std_dev, int size, int block_idx, int norm_idx, LegacyLLM_Gradients* grads) {
    if (!d_loss_d_norm_output || !input_vec || !norm_gamma || !norm_beta || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks || (norm_idx != 1 && norm_idx != 2)) {
        fprintf(stderr, "Error: Invalid input to backward_layer_norm\n");
        return NULL;
    }

    float* x_hat = create_float_array(size);
    if (!x_hat) return NULL;
    memcpy(x_hat, input_vec, size * sizeof(float));
    vector_sub_scalar_inplace(x_hat, mean, size);
    scalar_mul_vector_inplace(x_hat, inv_std_dev, size);

    float d_beta_sum = vector_sum(d_loss_d_norm_output, size);
    if (norm_idx == 1) { // First LayerNorm in block
        for(int i=0; i<size; ++i) grads->transformer_block_grads[block_idx].norm1_beta[i] += d_beta_sum; // Accumulate
    } else { // Second LayerNorm in block
        for(int i=0; i<size; ++i) grads->transformer_block_grads[block_idx].norm2_beta[i] += d_beta_sum; // Accumulate
    }

    float* d_gamma_vec = create_float_array(size);
    if (!d_gamma_vec) { free_float_array(x_hat); return NULL; }
    memcpy(d_gamma_vec, d_loss_d_norm_output, size * sizeof(float));
    multiply_vector_inplace(d_gamma_vec, x_hat, size);
    if (norm_idx == 1) { // First LayerNorm in block
        for(int i=0; i<size; ++i) grads->transformer_block_grads[block_idx].norm1_gamma[i] += d_gamma_vec[i];
    } else { // Second LayerNorm in block
        for(int i=0; i<size; ++i) grads->transformer_block_grads[block_idx].norm2_gamma[i] += d_gamma_vec[i];
    }
    free_float_array(d_gamma_vec);

    float* d_loss_d_input = create_float_array(size);
    if (!d_loss_d_input) { free_float_array(x_hat); return NULL; }
    
    float inv_size = 1.0f / size;
    float* dy_hat_times_gamma = create_float_array(size);
    if (!dy_hat_times_gamma) { free_float_array(x_hat); free_float_array(d_loss_d_input); return NULL; }
    memcpy(dy_hat_times_gamma, d_loss_d_norm_output, size * sizeof(float));
    multiply_vector_inplace(dy_hat_times_gamma, norm_gamma, size);
    
    float sum_dl_dy_norm_scaled = vector_sum(dy_hat_times_gamma, size); // sum(dL/dy * gamma)

    float sum_dl_dy_norm_scaled_x_hat = 0.0f;
    for(int i=0; i<size; ++i) {
        sum_dl_dy_norm_scaled_x_hat += dy_hat_times_gamma[i] * x_hat[i]; // sum(dL/dy * gamma * x_hat)
    }

    float inv_std_dev_cubed = inv_std_dev * inv_std_dev * inv_std_dev;

    for (int i = 0; i < size; ++i) {
        float term1 = dy_hat_times_gamma[i] * inv_std_dev;
        float term2 = inv_size * sum_dl_dy_norm_scaled * inv_std_dev;
        float term3 = inv_size * (input_vec[i] - mean) * inv_std_dev_cubed * sum_dl_dy_norm_scaled_x_hat;
        d_loss_d_input[i] = term1 - term2 - term3;
    }

    free_float_array(x_hat);
    free_float_array(dy_hat_times_gamma);

    return d_loss_d_input;
}

// Backward pass for Layer Normalization (Batched version)
float* backward_layer_norm_batch(const float* d_loss_d_norm_output_batch, const float* input_batch, const float* norm_gamma, const float* norm_beta, const float* mean_batch, const float* inv_std_dev_batch, int batch_size, int model_dim, int block_idx, int norm_idx, LegacyLLM_Gradients* grads) {
    if (!d_loss_d_norm_output_batch || !input_batch || !norm_gamma || !norm_beta || !mean_batch || !inv_std_dev_batch || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks || (norm_idx != 1 && norm_idx != 2)) {
        fprintf(stderr, "Error: Invalid input to backward_layer_norm_batch\n");
        return NULL;
    }

    float* d_loss_d_input_batch = create_float_array(batch_size * model_dim);
    if (!d_loss_d_input_batch) return NULL;

    for (int b = 0; b < batch_size; ++b) {
        const float* d_loss_d_norm_output_single = &d_loss_d_norm_output_batch[b * model_dim];
        const float* input_single = &input_batch[b * model_dim];
        float mean_single = mean_batch[b];
        float inv_std_dev_single = inv_std_dev_batch[b];
        float* d_loss_d_input_single = &d_loss_d_input_batch[b * model_dim];

        float* x_hat = create_float_array(model_dim);
        if (!x_hat) { free_float_array(d_loss_d_input_batch); return NULL; }
        memcpy(x_hat, input_single, model_dim * sizeof(float));
        vector_sub_scalar_inplace(x_hat, mean_single, model_dim);
        scalar_mul_vector_inplace(x_hat, inv_std_dev_single, model_dim);

        float d_beta_sum = vector_sum(d_loss_d_norm_output_single, model_dim);
        if (norm_idx == 1) { // First LayerNorm in block
            for(int i=0; i<model_dim; ++i) grads->transformer_block_grads[block_idx].norm1_beta[i] += d_beta_sum; // Accumulate
        } else { // Second LayerNorm in block
            for(int i=0; i<model_dim; ++i) grads->transformer_block_grads[block_idx].norm2_beta[i] += d_beta_sum; // Accumulate
        }

        float* d_gamma_vec = create_float_array(model_dim);
        if (!d_gamma_vec) { free_float_array(x_hat); free_float_array(d_loss_d_input_batch); return NULL; }
        memcpy(d_gamma_vec, d_loss_d_norm_output_single, model_dim * sizeof(float));
        multiply_vector_inplace(d_gamma_vec, x_hat, model_dim);
        if (norm_idx == 1) { // First LayerNorm in block
            for(int i=0; i<model_dim; ++i) grads->transformer_block_grads[block_idx].norm1_gamma[i] += d_gamma_vec[i];
        } else { // Second LayerNorm in block
            for(int i=0; i<model_dim; ++i) grads->transformer_block_grads[block_idx].norm2_gamma[i] += d_gamma_vec[i];
        }
        free_float_array(d_gamma_vec);

        float inv_size = 1.0f / model_dim;
        float* dy_hat_times_gamma = create_float_array(model_dim);
        if (!dy_hat_times_gamma) { free_float_array(x_hat); free_float_array(d_loss_d_input_batch); return NULL; }
        memcpy(dy_hat_times_gamma, d_loss_d_norm_output_single, model_dim * sizeof(float));
        multiply_vector_inplace(dy_hat_times_gamma, norm_gamma, model_dim);
        
        float sum_dl_dy_norm_scaled = vector_sum(dy_hat_times_gamma, model_dim); // sum(dL/dy * gamma)

        float sum_dl_dy_norm_scaled_x_hat = 0.0f;
        for(int i=0; i<model_dim; ++i) {
            sum_dl_dy_norm_scaled_x_hat += dy_hat_times_gamma[i] * x_hat[i]; // sum(dL/dy * gamma * x_hat)
        }

        float inv_std_dev_cubed = inv_std_dev_single * inv_std_dev_single * inv_std_dev_single;

        for (int i = 0; i < model_dim; ++i) {
            float term1 = dy_hat_times_gamma[i] * inv_std_dev_single;
            float term2 = inv_size * sum_dl_dy_norm_scaled * inv_std_dev_single;
            float term3 = inv_size * (input_single[i] - mean_single) * inv_std_dev_cubed * sum_dl_dy_norm_scaled_x_hat;
            d_loss_d_input_single[i] = term1 - term2 - term3;
        }

        free_float_array(x_hat);
        free_float_array(dy_hat_times_gamma);
    }

    return d_loss_d_input_batch;
}


// Backward pass for Multi-Head Attention
float* backward_multi_head_attention(const MultiHeadAttentionLayer* layer, const float* d_loss_d_mha_output, const float* mha_input, const float* query_vec_pre_bias, const float* key_vec_pre_bias, const float* value_vec_pre_bias, const float* query_vec, const float* key_vec, const float* value_vec, const float* attention_output_vec, int model_dim, int block_idx, LegacyLLM_Gradients* grads) {
    if (!layer || !d_loss_d_mha_output || !mha_input || !query_vec || !key_vec || !value_vec || !attention_output_vec || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks) {
        fprintf(stderr, "Error: Invalid input to backward_multi_head_attention\n");
        return NULL;
    }

    (void)query_vec_pre_bias;
    (void)key_vec_pre_bias;
    (void)value_vec_pre_bias;

    // 1. Gradients for Output Bias (dLoss/dbo)
    for (int i = 0; i < model_dim; ++i) {
        grads->transformer_block_grads[block_idx].attention_grads.bo[i] += d_loss_d_mha_output[i];
    }

    // 2. Gradients for Wo (dLoss/dWo = dLoss/dMHA_Output * attention_output_vec.T)
    // Wo is (model_dim x model_dim)
    outer_product_add_inplace(grads->transformer_block_grads[block_idx].attention_grads.Wo, d_loss_d_mha_output, attention_output_vec, model_dim, model_dim);

    // 3. Gradients for attention_output_vec (dLoss/dAttention_Output_vec = Wo.T * dLoss/dMHA_Output)
    float* d_loss_d_attention_output_vec = create_float_array(model_dim);
    if (!d_loss_d_attention_output_vec) return NULL;
    matrix_transpose_vector_mul(&layer->Wo, d_loss_d_mha_output, d_loss_d_attention_output_vec);

    // This is where simplified attention comes in:
    // attention_output_vec was a copy of value_vec. So dLoss/dValue_vec = dLoss/dAttention_Output_vec
    float* d_loss_d_value_vec = create_float_array(model_dim);
    if (!d_loss_d_value_vec) { free_float_array(d_loss_d_attention_output_vec); return NULL; }
    memcpy(d_loss_d_value_vec, d_loss_d_attention_output_vec, model_dim * sizeof(float));
    free_float_array(d_loss_d_attention_output_vec); // No longer needed

    // 4. Gradients for Value Biases (dLoss/dbv = dLoss/dValue_vec)
    for (int i = 0; i < model_dim; ++i) {
        grads->transformer_block_grads[block_idx].attention_grads.bv[i] += d_loss_d_value_vec[i];
    }

    // 5. Gradients for Wv (dLoss/dWv = dLoss/dValue_vec * mha_input.T)
    // Wv is (model_dim x model_dim)
    outer_product_add_inplace(grads->transformer_block_grads[block_idx].attention_grads.Wv, d_loss_d_value_vec, mha_input, model_dim, model_dim);
    
    // 6. Gradients for mha_input (propagated through V projection)
    float* d_loss_d_mha_input_from_v = create_float_array(model_dim);
    if (!d_loss_d_mha_input_from_v) { free_float_array(d_loss_d_value_vec); return NULL; }
    matrix_transpose_vector_mul(&layer->Wv, d_loss_d_value_vec, d_loss_d_mha_input_from_v);
    free_float_array(d_loss_d_value_vec);

    // For simplified attention, Query and Key don't affect the forward pass result,
    // so their gradients with respect to the output are zero.
    // However, if we were doing proper self-attention, we'd have dLoss/dQ and dLoss/dK here.
    // For this basic setup, dLoss/dQ_vec and dLoss/dK_vec are zero.
    // So, dLoss/dWq, dLoss/dbq, dLoss/dWk, dLoss/dbk and their contributions to dLoss/dInput will be zero.
    // This part will need to be expanded for true training.

    // Sum all input gradients
    return d_loss_d_mha_input_from_v; // Only from V path for simplified
}

// Backward pass for a Transformer Block
// Returns: dLoss/dInput (to be propagated backwards), caller must free.
float* backward_transformer_block(const TransformerBlock* block, const float* d_loss_d_block_output, int model_dim, int block_idx, LegacyLLM_Gradients* grads, const TransformerBlockContext* context) {
    if (!block || !d_loss_d_block_output || !context || !grads || block_idx < 0 || block_idx >= grads->num_transformer_blocks) {
        fprintf(stderr, "Error: Invalid input to backward_transformer_block\n");
        return NULL;
    }

    // --- Recompute forward pass intermediates ---
    // block_input is stored in context->block_input
    float* input_vec_LN1 = &context->block_input_batch[0 * model_dim]; 

    // Recompute norm1_output_buffer
    float* norm1_output_buffer_recomputed = create_float_array(model_dim);
    if (!norm1_output_buffer_recomputed) return NULL;
    memcpy(norm1_output_buffer_recomputed, input_vec_LN1, model_dim * sizeof(float));
    // Apply LayerNorm (only forward pass is needed to get normalized values for backprop calculation)
    // Use stored mean/inv_std_dev
    // Note: layer_norm_forward modifies its input, so we use a temporary copy.
    float temp_ln1_mean, temp_ln1_inv_std_dev; // dummy variables as mean/inv_std_dev are stored in context
    layer_norm_forward(norm1_output_buffer_recomputed, block->norm1_gamma, block->norm1_beta, model_dim, 1e-5f, &temp_ln1_mean, &temp_ln1_inv_std_dev);

    // Recompute attn_output
    float* attn_output_recomputed = forward_multi_head_attention(&block->attention, norm1_output_buffer_recomputed, model_dim);
    if (!attn_output_recomputed) { free_float_array(norm1_output_buffer_recomputed); return NULL; }
    
    // Recompute current_output after attention residual
    float* current_output_after_attn_residual_recomputed = create_float_array(model_dim);
    if (!current_output_after_attn_residual_recomputed) { 
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        return NULL; 
    }
    memcpy(current_output_after_attn_residual_recomputed, input_vec_LN1, model_dim * sizeof(float)); // Residual connection
    add_vector_inplace(current_output_after_attn_residual_recomputed, attn_output_recomputed, model_dim);

    // Recompute norm2_output_buffer
    float* norm2_output_buffer_recomputed = create_float_array(model_dim);
    if (!norm2_output_buffer_recomputed) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        return NULL;
    }
    memcpy(norm2_output_buffer_recomputed, current_output_after_attn_residual_recomputed, model_dim * sizeof(float));
    // Apply LayerNorm (only forward pass is needed)
    float temp_ln2_mean, temp_ln2_inv_std_dev; // dummy variables
    layer_norm_forward(norm2_output_buffer_recomputed, block->norm2_gamma, block->norm2_beta, model_dim, 1e-5f, &temp_ln2_mean, &temp_ln2_inv_std_dev);

    // Recompute ffn_output (not strictly needed for backward pass, but for consistency if we wanted to check final output)
    float* ffn_output_recomputed = forward_feed_forward(&block->ffn, norm2_output_buffer_recomputed, model_dim);
    if (!ffn_output_recomputed) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        return NULL;
    }

    // --- Start Backward Pass ---
    float* d_loss_d_current_output = create_float_array(model_dim);
    if (!d_loss_d_current_output) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        return NULL;
    }
    memcpy(d_loss_d_current_output, d_loss_d_block_output, model_dim * sizeof(float)); // dL/dOutput of residual stream

    // Gradients for FFN (and its residual connection)
    float* d_loss_d_ffn_output_for_ffn_back = create_float_array(model_dim);
    if (!d_loss_d_ffn_output_for_ffn_back) { free_float_array(d_loss_d_current_output); return NULL; }
    memcpy(d_loss_d_ffn_output_for_ffn_back, d_loss_d_current_output, model_dim * sizeof(float));
    
    // Recomputed inputs to backward_feed_forward
    float* ffn_input_recomputed = norm2_output_buffer_recomputed; // Input to FFN
    float* hidden_pre_relu_output_recomputed = create_float_array(model_dim * FFN_DIM_MULTIPLIER); // Recompute this
    if (!hidden_pre_relu_output_recomputed) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        free_float_array(d_loss_d_current_output);
        free_float_array(d_loss_d_ffn_output_for_ffn_back);
        return NULL;
    }
    // Recompute the pre-ReLU output from the FFN forward pass
    ternary_matrix_vector_mul(&block->ffn.Wi, ffn_input_recomputed, hidden_pre_relu_output_recomputed);
    add_vector_inplace(hidden_pre_relu_output_recomputed, block->ffn.bi, model_dim * FFN_DIM_MULTIPLIER);


    float* d_loss_d_norm2_output = backward_feed_forward(&block->ffn, d_loss_d_ffn_output_for_ffn_back, ffn_input_recomputed, hidden_pre_relu_output_recomputed, model_dim, block_idx, grads);
    free_float_array(d_loss_d_ffn_output_for_ffn_back);
    free_float_array(hidden_pre_relu_output_recomputed); // Free the recomputed pre-ReLU output
    if (!d_loss_d_norm2_output) { 
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        free_float_array(d_loss_d_current_output);
        return NULL;
    }

    add_vector_inplace(d_loss_d_current_output, d_loss_d_norm2_output, model_dim);
    free_float_array(d_loss_d_norm2_output);
    
    // Gradients for LayerNorm 2
    float* d_loss_d_block_output_before_ffn_residual = backward_layer_norm(d_loss_d_current_output, current_output_after_attn_residual_recomputed, block->norm2_gamma, block->norm2_beta, context->ln2_mean_batch[0], context->ln2_inv_std_dev_batch[0], model_dim, block_idx, 2, grads);
    free_float_array(d_loss_d_current_output);
    if (!d_loss_d_block_output_before_ffn_residual) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        return NULL;
    }

    // Gradients for MHA (and its residual connection)
    float* d_loss_d_attn_output_for_mha_back = create_float_array(model_dim);
    if (!d_loss_d_attn_output_for_mha_back) { free_float_array(d_loss_d_block_output_before_ffn_residual); return NULL; }
    memcpy(d_loss_d_attn_output_for_mha_back, d_loss_d_block_output_before_ffn_residual, model_dim * sizeof(float));

    // Recomputed inputs for backward_multi_head_attention
    float* mha_input_recomputed = norm1_output_buffer_recomputed;

    // Need to recompute query_vec, key_vec, value_vec, attention_output_vec for backward MHA
    float* recomputed_query_vec_pre_bias = create_float_array(model_dim);
    float* recomputed_key_vec_pre_bias = create_float_array(model_dim);
    float* recomputed_value_vec_pre_bias = create_float_array(model_dim);
    float* recomputed_query_vec = create_float_array(model_dim);
    float* recomputed_key_vec = create_float_array(model_dim);
    float* recomputed_value_vec = create_float_array(model_dim);
    float* recomputed_attention_output_vec = create_float_array(model_dim);

    if (!recomputed_query_vec_pre_bias || !recomputed_key_vec_pre_bias || !recomputed_value_vec_pre_bias ||
        !recomputed_query_vec || !recomputed_key_vec || !recomputed_value_vec || !recomputed_attention_output_vec) {
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        free_float_array(d_loss_d_block_output_before_ffn_residual);
        free_float_array(d_loss_d_attn_output_for_mha_back);
        free_float_array(recomputed_query_vec_pre_bias);
        free_float_array(recomputed_key_vec_pre_bias);
        free_float_array(recomputed_value_vec_pre_bias);
        free_float_array(recomputed_query_vec);
        free_float_array(recomputed_key_vec);
        free_float_array(recomputed_value_vec);
        free_float_array(recomputed_attention_output_vec);
        return NULL;
    }

    // Perform linear projections to get Q, K, V
    ternary_matrix_vector_mul(&block->attention.Wq, mha_input_recomputed, recomputed_query_vec_pre_bias);
    memcpy(recomputed_query_vec, recomputed_query_vec_pre_bias, model_dim * sizeof(float));
    add_vector_inplace(recomputed_query_vec, block->attention.bq, model_dim);

    ternary_matrix_vector_mul(&block->attention.Wk, mha_input_recomputed, recomputed_key_vec_pre_bias);
    memcpy(recomputed_key_vec, recomputed_key_vec_pre_bias, model_dim * sizeof(float));
    add_vector_inplace(recomputed_key_vec, block->attention.bk, model_dim);

    ternary_matrix_vector_mul(&block->attention.Wv, mha_input_recomputed, recomputed_value_vec_pre_bias);
    memcpy(recomputed_value_vec, recomputed_value_vec_pre_bias, model_dim * sizeof(float));
    add_vector_inplace(recomputed_value_vec, block->attention.bv, model_dim);

    // Simplified Attention Placeholder for a single token
    memcpy(recomputed_attention_output_vec, recomputed_value_vec, model_dim * sizeof(float));


    float* d_loss_d_norm1_output = backward_multi_head_attention(&block->attention, d_loss_d_attn_output_for_mha_back, mha_input_recomputed, 
                                                                  recomputed_query_vec_pre_bias, recomputed_key_vec_pre_bias, recomputed_value_vec_pre_bias,
                                                                  recomputed_query_vec, recomputed_key_vec, recomputed_value_vec, recomputed_attention_output_vec,
                                                                  model_dim, block_idx, grads);
    free_float_array(d_loss_d_attn_output_for_mha_back);
    // Free recomputed MHA internals
    free_float_array(recomputed_query_vec_pre_bias);
    free_float_array(recomputed_key_vec_pre_bias);
    free_float_array(recomputed_value_vec_pre_bias);
    free_float_array(recomputed_query_vec);
    free_float_array(recomputed_key_vec);
    free_float_array(recomputed_value_vec);
    free_float_array(recomputed_attention_output_vec);

    if (!d_loss_d_norm1_output) { 
        free_float_array(norm1_output_buffer_recomputed);
        free_float_array(attn_output_recomputed);
        free_float_array(current_output_after_attn_residual_recomputed);
        free_float_array(norm2_output_buffer_recomputed);
        free_float_array(ffn_output_recomputed);
        free_float_array(d_loss_d_block_output_before_ffn_residual);
        return NULL;
    }
    
    add_vector_inplace(d_loss_d_block_output_before_ffn_residual, d_loss_d_norm1_output, model_dim);
    free_float_array(d_loss_d_norm1_output);

    float* d_loss_d_block_input = backward_layer_norm(d_loss_d_block_output_before_ffn_residual, input_vec_LN1, block->norm1_gamma, block->norm1_beta, context->ln1_mean_batch[0], context->ln1_inv_std_dev_batch[0], model_dim, block_idx, 1, grads);
    free_float_array(d_loss_d_block_output_before_ffn_residual);

    // Free all recomputed intermediate buffers
    free_float_array(norm1_output_buffer_recomputed);
    free_float_array(attn_output_recomputed);
    free_float_array(current_output_after_attn_residual_recomputed);
    free_float_array(norm2_output_buffer_recomputed);
    free_float_array(ffn_output_recomputed);
    
    if (!d_loss_d_block_input) return NULL;

    return d_loss_d_block_input;
}

// Backward pass for Embedding Layer
void backward_embedding(const EmbeddingLayer* layer, int token_id, const float* d_loss_d_embedding_output, int model_dim, LegacyLLM_Gradients* grads) {
    if (!layer || token_id < 0 || token_id >= layer->embedding_weights.rows || !d_loss_d_embedding_output || !grads) {
        fprintf(stderr, "Error: Invalid input to backward_embedding\n");
        return;
    }

    // dLoss/dEmbeddingWeights_ij = dLoss/dEmbeddingOutput_j (if token_id is row i)
    // Add dLoss/dEmbeddingOutput to the gradient of the row corresponding to token_id
    for (int j = 0; j < model_dim; ++j) {
        grads->embedding_grads.embedding_weights[token_id * model_dim + j] += d_loss_d_embedding_output[j];
    }
}

// Backward pass for Embedding Layer (Batched version)
void backward_embedding_batch(const EmbeddingLayer* layer, const int* input_batch, const float* d_loss_d_embedding_output_batch, int batch_size, int model_dim, LegacyLLM_Gradients* grads) {
    if (!layer || !input_batch || !d_loss_d_embedding_output_batch || !grads) {
        fprintf(stderr, "Error: Invalid input to backward_embedding_batch\n");
        return;
    }

    for (int b = 0; b < batch_size; ++b) {
        int token_id = input_batch[b];
        const float* d_loss_d_embedding_output_single = &d_loss_d_embedding_output_batch[b * model_dim];

        if (token_id == PAD_TOKEN) { // Don't accumulate gradients for PAD_TOKEN
            continue;
        }

        // Add dLoss/dEmbeddingOutput to the gradient of the row corresponding to token_id
        for (int j = 0; j < model_dim; ++j) {
            grads->embedding_grads.embedding_weights[token_id * model_dim + j] += d_loss_d_embedding_output_single[j];
        }
    }
}

// Full backward pass through the LLM for a batch of tokens
void backward_llm_batch(LegacyLLM* model, const int* input_batch, const int* target_batch, int batch_size, LegacyLLM_Gradients* grads) {
    if (!model || !input_batch || !target_batch || !grads) {
        fprintf(stderr, "Error: Invalid input to backward_llm_batch\n");
        return;
    }

    // 1. Calculate dLoss/dLogits for the entire batch
    float* predicted_probs_batch = forward_llm_batch(model, input_batch, batch_size); // Re-run forward pass to get probabilities for this batch
    if (!predicted_probs_batch) return;

    // Allocate array for dLoss/dLogits for the batch
    float* d_loss_d_logits_batch = create_float_array(batch_size * model->vocab_size);
    if (!d_loss_d_logits_batch) { free_float_array(predicted_probs_batch); return; }

    for (int b = 0; b < batch_size; ++b) {
        float* predicted_probs_single = &predicted_probs_batch[b * model->vocab_size];
        int true_token_id = target_batch[b];
        
        // Skip gradient calculation for PAD_TOKEN targets
        if (true_token_id == PAD_TOKEN) {
            for (int j = 0; j < model->vocab_size; ++j) {
                d_loss_d_logits_batch[b * model->vocab_size + j] = 0.0f;
            }
            continue;
        }

        float* d_loss_d_logits_single = d_loss_d_logits(predicted_probs_single, true_token_id, model->vocab_size);
        if (!d_loss_d_logits_single) { free_float_array(predicted_probs_batch); free_float_array(d_loss_d_logits_batch); return; }
        memcpy(&d_loss_d_logits_batch[b * model->vocab_size], d_loss_d_logits_single, model->vocab_size * sizeof(float));
        free_float_array(d_loss_d_logits_single);
    }
    free_float_array(predicted_probs_batch);

    // 2. Backward pass through Output Layer (Batched)
    // We need the hidden_state_input_batch that was fed into the output layer in forward_llm_batch
    float* d_loss_d_hidden_state_batch = backward_output_layer_batch(&model->output, d_loss_d_logits_batch, model->final_hidden_state_input_batch, batch_size, model->model_dim, model->vocab_size, grads);
    free_float_array(d_loss_d_logits_batch);
    if (!d_loss_d_hidden_state_batch) return;

    // 3. Backward pass through Transformer Blocks (in reverse order, Batched)
    for (int i = model->num_transformer_blocks - 1; i >= 0; --i) {
        float* d_loss_d_block_input_batch = backward_transformer_block_batch(&model->transformer_blocks[i], d_loss_d_hidden_state_batch, batch_size, model->model_dim, i, grads, model->block_contexts[i]);
        free_float_array(d_loss_d_hidden_state_batch); // Free previous d_loss_d_hidden_state_batch
        if (!d_loss_d_block_input_batch) return;
        d_loss_d_hidden_state_batch = d_loss_d_block_input_batch;
    }

    // 4. Backward pass through Embedding Layer (Batched)
    backward_embedding_batch(&model->embedding, input_batch, d_loss_d_hidden_state_batch, batch_size, model->model_dim, grads);
    free_float_array(d_loss_d_hidden_state_batch); // Free final d_loss_d_hidden_state_batch
}