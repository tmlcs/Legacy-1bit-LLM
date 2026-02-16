#include <stdio.h> // For fprintf
#include <stdlib.h> // For calloc, free
#include <string.h> // For memcpy

#include "forward.h"
#include "legacy_llm.h" // For struct definitions
#include "model.h" // For create_float_array, free_float_array
#include "math_ops.h" // For ternary_matrix_vector_mul, add_vector_inplace, relu, softmax, layer_norm_forward


// Forward pass for Embedding Layer
float* forward_embedding(const EmbeddingLayer* layer, int token_id, int model_dim) {
    if (!layer || !layer->embedding_weights.data || token_id < 0 || token_id >= layer->embedding_weights.rows) {
        fprintf(stderr, "Error: Invalid input to forward_embedding\n");
        return NULL;
    }

    float* embedding = create_float_array(model_dim);
    if (!embedding) {
        return NULL;
    }

    // Embedding lookup is essentially getting the row corresponding to the token_id
    // and converting its ternary values to floats.
    for (int i = 0; i < model_dim; ++i) {
        embedding[i] = (float)layer->embedding_weights.data[token_id * model_dim + i];
    }
    return embedding;
}

// Forward pass for Embedding Layer (Batched version)
// input_batch: an array of token_ids of size batch_size
// Returns: a flat array of embeddings (batch_size * model_dim), caller must free.
float* forward_embedding_batch(const EmbeddingLayer* layer, const int* input_batch, int batch_size, int model_dim) {
    if (!layer || !layer->embedding_weights.data || !input_batch) {
        fprintf(stderr, "Error: Invalid input to forward_embedding_batch\n");
        return NULL;
    }

    float* embeddings_batch = create_float_array(batch_size * model_dim);
    if (!embeddings_batch) {
        return NULL;
    }

    for (int b = 0; b < batch_size; ++b) {
        int token_id = input_batch[b];
        // Handle PAD_TOKEN: If it's a PAD_TOKEN, set embedding to zeros
        if (token_id == PAD_TOKEN) {
            for (int i = 0; i < model_dim; ++i) {
                embeddings_batch[b * model_dim + i] = 0.0f;
            }
        } else if (token_id < 0 || token_id >= layer->embedding_weights.rows) {
            fprintf(stderr, "Error: Invalid token_id %d in batch at index %d for forward_embedding_batch\n", token_id, b);
            free_float_array(embeddings_batch);
            return NULL;
        } else {
            // Embedding lookup for valid tokens
            for (int i = 0; i < model_dim; ++i) {
                embeddings_batch[b * model_dim + i] = (float)layer->embedding_weights.data[token_id * model_dim + i];
            }
        }
    }
    return embeddings_batch;
}

// Forward pass for Multi-Head Attention for a single input vector
float* forward_multi_head_attention(const MultiHeadAttentionLayer* layer, const float* input_vec, int model_dim) {
    // This function is for testing only, will be replaced by forward_multi_head_attention_with_context
    if (!layer || !input_vec) {
        fprintf(stderr, "Error: Invalid input to forward_multi_head_attention\n");
        return NULL;
    }

    float* query_vec = create_float_array(model_dim);
    float* key_vec = create_float_array(model_dim);
    float* value_vec = create_float_array(model_dim);
    float* attention_output_vec = create_float_array(model_dim); // Output before final projection

    if (!query_vec || !key_vec || !value_vec || !attention_output_vec) {
        free_float_array(query_vec); free_float_array(key_vec); free_float_array(value_vec); free_float_array(attention_output_vec);
        fprintf(stderr, "Error: Memory allocation failed in forward_multi_head_attention\n");
        return NULL;
    }

    // 1. Linear Projections for Q, K, V
    ternary_matrix_vector_mul(&layer->Wq, input_vec, query_vec);
    add_vector_inplace(query_vec, layer->bq, model_dim);

    ternary_matrix_vector_mul(&layer->Wk, input_vec, key_vec);
    add_vector_inplace(key_vec, layer->bk, model_dim);

    ternary_matrix_vector_mul(&layer->Wv, input_vec, value_vec);
    add_vector_inplace(value_vec, layer->bv, model_dim);

    // 2. Simplified Attention Placeholder for a single token
    memcpy(attention_output_vec, value_vec, model_dim * sizeof(float)); // Copy value_vec to attention_output_vec

    // 3. Output Projection
    float* final_output = create_float_array(model_dim);
    if (!final_output) {
        free_float_array(query_vec); free_float_array(key_vec); free_float_array(value_vec); free_float_array(attention_output_vec);
        fprintf(stderr, "Error: Memory allocation failed for final_output in MHA\n");
        return NULL;
    }
    ternary_matrix_vector_mul(&layer->Wo, attention_output_vec, final_output);
    add_vector_inplace(final_output, layer->bo, model_dim);

    // Free intermediate vectors
    free_float_array(query_vec);
    free_float_array(key_vec);
    free_float_array(value_vec);
    free_float_array(attention_output_vec);

        return final_output;

    }

    

    // Forward pass for Multi-Head Attention for a batch of input vectors

    // input_batch: a flattened array of input vectors (batch_size * model_dim)

    // Returns: a flattened array of output vectors (batch_size * model_dim), caller must free.

    float* forward_multi_head_attention_batch(const MultiHeadAttentionLayer* layer, const float* input_batch, int batch_size, int model_dim) {

        if (!layer || !input_batch) {

            fprintf(stderr, "Error: Invalid input to forward_multi_head_attention_batch\n");

            return NULL;

        }

    

        float* output_batch = create_float_array(batch_size * model_dim);

        if (!output_batch) {

            fprintf(stderr, "Error: Memory allocation failed for MHA output batch\n");

            return NULL;

        }

    

        for (int b = 0; b < batch_size; ++b) {

            const float* input_vec = &input_batch[b * model_dim];

            float* current_output_vec = &output_batch[b * model_dim];

    

            float* query_vec = create_float_array(model_dim);

            float* key_vec = create_float_array(model_dim);

            float* value_vec = create_float_array(model_dim);

            float* attention_output_vec = create_float_array(model_dim); // Output before final projection

    

            if (!query_vec || !key_vec || !value_vec || !attention_output_vec) {

                free_float_array(query_vec); free_float_array(key_vec); free_float_array(value_vec); free_float_array(attention_output_vec);

                free_float_array(output_batch);

                fprintf(stderr, "Error: Memory allocation failed in forward_multi_head_attention_batch for intermediate vectors\n");

                return NULL;

            }

    

            // 1. Linear Projections for Q, K, V

            ternary_matrix_vector_mul(&layer->Wq, input_vec, query_vec);

            add_vector_inplace(query_vec, layer->bq, model_dim);

    

            ternary_matrix_vector_mul(&layer->Wk, input_vec, key_vec);

            add_vector_inplace(key_vec, layer->bk, model_dim);

    

            ternary_matrix_vector_mul(&layer->Wv, input_vec, value_vec);

            add_vector_inplace(value_vec, layer->bv, model_dim);

    

            // 2. Simplified Attention Placeholder for a single token

            memcpy(attention_output_vec, value_vec, model_dim * sizeof(float)); // Copy value_vec to attention_output_vec

    

            // 3. Output Projection directly into the output_batch

            ternary_matrix_vector_mul(&layer->Wo, attention_output_vec, current_output_vec);

            add_vector_inplace(current_output_vec, layer->bo, model_dim);

    

            // Free intermediate vectors for this batch item

            free_float_array(query_vec);

            free_float_array(key_vec);

            free_float_array(value_vec);

            free_float_array(attention_output_vec);

        }

    

        return output_batch;

    }

    

    

    // Forward pass for Feed-Forward Network
float* forward_feed_forward(const FeedForwardLayer* layer, const float* input_vec, int model_dim) {
    // This function is for testing only, will be replaced by forward_feed_forward_with_context
    if (!layer || !input_vec) {
        fprintf(stderr, "Error: Invalid input to forward_feed_forward\n");
        return NULL;
    }

    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;
    float* hidden_output = create_float_array(ffn_hidden_dim);
    if (!hidden_output) {
        fprintf(stderr, "Error: Memory allocation failed for FFN hidden_output\n");
        return NULL;
    }

    // 1. First Linear Layer (Wi)
    ternary_matrix_vector_mul(&layer->Wi, input_vec, hidden_output);
    add_vector_inplace(hidden_output, layer->bi, ffn_hidden_dim);

    // 2. Activation Function (ReLU)
    relu(hidden_output, ffn_hidden_dim);

    // 3. Second Linear Layer (Wo)
    float* final_output = create_float_array(model_dim);
    if (!final_output) {
        free_float_array(hidden_output);
        fprintf(stderr, "Error: Memory allocation failed for FFN final_output\n");
        return NULL;
    }
    ternary_matrix_vector_mul(&layer->Wo, hidden_output, final_output);
    add_vector_inplace(final_output, layer->bo, model_dim);

    // Free intermediate vector
    free_float_array(hidden_output);

    return final_output;
}

// Forward pass for Feed-Forward Network (Batched version)
// input_batch: a flattened array of input vectors (batch_size * model_dim)
// Returns: a flattened array of output vectors (batch_size * model_dim), caller must free.
float* forward_feed_forward_batch(const FeedForwardLayer* layer, const float* input_batch, int batch_size, int model_dim) {
    if (!layer || !input_batch) {
        fprintf(stderr, "Error: Invalid input to forward_feed_forward_batch\n");
        return NULL;
    }

    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;
    float* output_batch = create_float_array(batch_size * model_dim);
    if (!output_batch) {
        fprintf(stderr, "Error: Memory allocation failed for FFN output batch\n");
        return NULL;
    }

    for (int b = 0; b < batch_size; ++b) {
        const float* input_vec = &input_batch[b * model_dim];
        float* current_output_vec = &output_batch[b * model_dim];

        float* hidden_output = create_float_array(ffn_hidden_dim);
        if (!hidden_output) {
            free_float_array(output_batch);
            fprintf(stderr, "Error: Memory allocation failed for FFN hidden_output in batch\n");
            return NULL;
        }

        // 1. First Linear Layer (Wi)
        ternary_matrix_vector_mul(&layer->Wi, input_vec, hidden_output);
        add_vector_inplace(hidden_output, layer->bi, ffn_hidden_dim);

        // 2. Activation Function (ReLU)
        relu(hidden_output, ffn_hidden_dim);

        // 3. Second Linear Layer (Wo)
        ternary_matrix_vector_mul(&layer->Wo, hidden_output, current_output_vec);
        add_vector_inplace(current_output_vec, layer->bo, model_dim);

        // Free intermediate vector for this batch item
        free_float_array(hidden_output);
    }

    return output_batch;
}


// Forward pass for a single Transformer Block
float* forward_transformer_block(const TransformerBlock* block, const float* input_vec, int model_dim) {
    // This function is for testing only, will be replaced by forward_transformer_block_with_context
    if (!block || !input_vec) {
        fprintf(stderr, "Error: Invalid input to forward_transformer_block\n");
        return NULL;
    }

    float* current_output = create_float_array(model_dim); // To hold residual connections
    if (!current_output) return NULL;
    memcpy(current_output, input_vec, model_dim * sizeof(float)); // Initialize with input for residual

    // LayerNorm 1 and Multi-Head Attention
    float* norm1_output_buffer = create_float_array(model_dim);
    if (!norm1_output_buffer) { free_float_array(current_output); return NULL; }
    memcpy(norm1_output_buffer, input_vec, model_dim * sizeof(float)); // Copy input for normalization
    
    float ln1_mean, ln1_inv_std_dev; // Store for backward pass
    layer_norm_forward(norm1_output_buffer, block->norm1_gamma, block->norm1_beta, model_dim, 1e-5f, &ln1_mean, &ln1_inv_std_dev);

    float* attn_output = forward_multi_head_attention(&block->attention, norm1_output_buffer, model_dim);
    free_float_array(norm1_output_buffer); // Free intermediate norm1_output_buffer
    if (!attn_output) { free_float_array(current_output); return NULL; }
    
    // Residual connection for attention: current_output += attn_output
    add_vector_inplace(current_output, attn_output, model_dim);
    free_float_array(attn_output); // Free attention output

    // LayerNorm 2 and Feed-Forward Network
    float* norm2_output_buffer = create_float_array(model_dim);
    if (!norm2_output_buffer) { free_float_array(current_output); return NULL; }
    memcpy(norm2_output_buffer, current_output, model_dim * sizeof(float)); // Copy current_output for normalization
    
    float ln2_mean, ln2_inv_std_dev; // Store for backward pass
    layer_norm_forward(norm2_output_buffer, block->norm2_gamma, block->norm2_beta, model_dim, 1e-5f, &ln2_mean, &ln2_inv_std_dev);

    float* ffn_output = forward_feed_forward(&block->ffn, norm2_output_buffer, model_dim);
    free_float_array(norm2_output_buffer); // Free intermediate norm2_output_buffer
    if (!ffn_output) { free_float_array(current_output); return NULL; }

    // Residual connection for FFN: current_output += ffn_output
    add_vector_inplace(current_output, ffn_output, model_dim);
    free_float_array(ffn_output); // Free FFN output

    return current_output;
}

// Forward pass for a single Transformer Block, storing context for backward pass
float* forward_transformer_block_with_context(const TransformerBlock* block, const float* input_vec, int model_dim, TransformerBlockContext* context) {
    if (!block || !input_vec || !context) {
        fprintf(stderr, "Error: Invalid input to forward_transformer_block_with_context\n");
        return NULL;
    }

    // Store original input to the block for recomputation during backward pass (for the first item in batch)
    memcpy(&context->block_input_batch[0], input_vec, model_dim * sizeof(float));

    float* current_output = create_float_array(model_dim); // To hold residual connections
    if (!current_output) return NULL;
    memcpy(current_output, input_vec, model_dim * sizeof(float)); // Initialize with input for residual

    // LayerNorm 1 and Multi-Head Attention
    float* norm1_output_buffer = create_float_array(model_dim);
    if (!norm1_output_buffer) { free_float_array(current_output); return NULL; }
    memcpy(norm1_output_buffer, input_vec, model_dim * sizeof(float));
    
    layer_norm_forward(norm1_output_buffer, block->norm1_gamma, block->norm1_beta, model_dim, 1e-5f, &context->ln1_mean_batch[0], &context->ln1_inv_std_dev_batch[0]);

    // Call non-context version of MHA forward, as internal intermediates are recomputed in backward_transformer_block
    float* attn_output = forward_multi_head_attention(&block->attention, norm1_output_buffer, model_dim);
    free_float_array(norm1_output_buffer);
    if (!attn_output) { free_float_array(current_output); return NULL; }
    
    // Residual connection for attention: current_output += attn_output
    add_vector_inplace(current_output, attn_output, model_dim);
    free_float_array(attn_output);

    // LayerNorm 2 and Feed-Forward Network
    float* norm2_output_buffer = create_float_array(model_dim);
    if (!norm2_output_buffer) { free_float_array(current_output); return NULL; }
    memcpy(norm2_output_buffer, current_output, model_dim * sizeof(float));
    
    layer_norm_forward(norm2_output_buffer, block->norm2_gamma, block->norm2_beta, model_dim, 1e-5f, &context->ln2_mean_batch[0], &context->ln2_inv_std_dev_batch[0]);

    // Call non-context version of FFN forward, as internal intermediates are recomputed in backward_transformer_block
    float* ffn_output = forward_feed_forward(&block->ffn, norm2_output_buffer, model_dim);
    free_float_array(norm2_output_buffer);
    if (!ffn_output) { free_float_array(current_output); return NULL; }

    // Residual connection for FFN: current_output += ffn_output
    add_vector_inplace(current_output, ffn_output, model_dim);
    free_float_array(ffn_output);

    return current_output;
}

// Forward pass for a single Transformer Block, storing context for backward pass (Batched version)
float* forward_transformer_block_batch_with_context(const TransformerBlock* block, const float* input_batch, int batch_size, int model_dim, TransformerBlockContext* context) {
    if (!block || !input_batch || !context) {
        fprintf(stderr, "Error: Invalid input to forward_transformer_block_batch_with_context\n");
        return NULL;
    }

    // Store original input batch to the block for recomputation during backward pass
    memcpy(context->block_input_batch, input_batch, batch_size * model_dim * sizeof(float));

    float* current_output_batch = create_float_array(batch_size * model_dim); // To hold residual connections
    if (!current_output_batch) return NULL;
    memcpy(current_output_batch, input_batch, batch_size * model_dim * sizeof(float)); // Initialize with input for residual

    // LayerNorm 1 and Multi-Head Attention
    float* norm1_output_buffer_batch = create_float_array(batch_size * model_dim);
    if (!norm1_output_buffer_batch) { free_float_array(current_output_batch); return NULL; }
    memcpy(norm1_output_buffer_batch, input_batch, batch_size * model_dim * sizeof(float)); // Copy input for normalization

    // Apply LayerNorm to each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        float* input_vec_single = &norm1_output_buffer_batch[b * model_dim];
        layer_norm_forward(input_vec_single, block->norm1_gamma, block->norm1_beta, model_dim, 1e-5f, &context->ln1_mean_batch[b], &context->ln1_inv_std_dev_batch[b]);
    }
    
    float* attn_output_batch = forward_multi_head_attention_batch(&block->attention, norm1_output_buffer_batch, batch_size, model_dim);
    free_float_array(norm1_output_buffer_batch); // Free intermediate norm1_output_buffer
    if (!attn_output_batch) { free_float_array(current_output_batch); return NULL; }
    
    // Residual connection for attention: current_output_batch += attn_output_batch
    for (int i = 0; i < batch_size * model_dim; ++i) {
        current_output_batch[i] += attn_output_batch[i];
    }
    free_float_array(attn_output_batch); // Free attention output

    // LayerNorm 2 and Feed-Forward Network
    float* norm2_output_buffer_batch = create_float_array(batch_size * model_dim);
    if (!norm2_output_buffer_batch) { free_float_array(current_output_batch); return NULL; }
    memcpy(norm2_output_buffer_batch, current_output_batch, batch_size * model_dim * sizeof(float)); // Copy current_output for normalization
    
    // Apply LayerNorm to each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        float* input_vec_single = &norm2_output_buffer_batch[b * model_dim];
        layer_norm_forward(input_vec_single, block->norm2_gamma, block->norm2_beta, model_dim, 1e-5f, &context->ln2_mean_batch[b], &context->ln2_inv_std_dev_batch[b]);
    }

    float* ffn_output_batch = forward_feed_forward_batch(&block->ffn, norm2_output_buffer_batch, batch_size, model_dim);
    free_float_array(norm2_output_buffer_batch); // Free intermediate norm2_output_buffer
    if (!ffn_output_batch) { free_float_array(current_output_batch); return NULL; }

    // Residual connection for FFN: current_output_batch += ffn_output_batch
    for (int i = 0; i < batch_size * model_dim; ++i) {
        current_output_batch[i] += ffn_output_batch[i];
    }
    free_float_array(ffn_output_batch); // Free FFN output

    return current_output_batch;
}

// Full forward pass through the LLM for a batch of tokens
// Returns: a flattened array of probabilities (batch_size * vocab_size), caller must free.
float* forward_llm_batch(LegacyLLM* model, const int* input_batch, int batch_size) {
    if (!model || !input_batch) {
        fprintf(stderr, "Error: Invalid input to forward_llm_batch\n");
        return NULL;
    }

    // 1. Input Embedding
    float* hidden_state_batch = forward_embedding_batch(&model->embedding, input_batch, batch_size, model->model_dim);
    if (!hidden_state_batch) return NULL;

    // 2. Iterate through Transformer Blocks, storing contexts
    for (int i = 0; i < model->num_transformer_blocks; ++i) {
        float* next_hidden_state_batch = forward_transformer_block_batch_with_context(&model->transformer_blocks[i], hidden_state_batch, batch_size, model->model_dim, model->block_contexts[i]);
        free_float_array(hidden_state_batch); // Free previous hidden_state_batch
        if (!next_hidden_state_batch) return NULL;
        hidden_state_batch = next_hidden_state_batch;
    }

    // Note: model->final_hidden_state_input is a single vector, not suitable for batching.
    // For now, we store the first item of the batch for backward pass compatibility,
    // but this needs to be properly addressed when backward_llm is batched.
    memcpy(model->final_hidden_state_input_batch, &hidden_state_batch[0 * model->model_dim], model->model_dim * sizeof(float));


    // 3. Final Output Layer (Unembedding + Bias)
    float* output_probs_batch = forward_output_layer_batch(&model->output, hidden_state_batch, batch_size, model->model_dim, model->vocab_size);
    free_float_array(hidden_state_batch); // Free final hidden_state_batch
    return output_probs_batch; // Return probabilities
}

// Full forward pass through the LLM for a single token
float* forward_llm(LegacyLLM* model, int token_id) {
    if (!model) {
        fprintf(stderr, "Error: Invalid input to forward_llm\n");
        return NULL;
    }
    
    // Create a single-element batch for input token
    int input_batch_single[1] = {token_id};
    
    // Call the batched version with batch_size = 1
    float* output_probs = forward_llm_batch(model, input_batch_single, 1);

    return output_probs; // Return probabilities
}

// Forward pass for the Output Layer (Batched version)
// hidden_state_batch: a flattened array of hidden states (batch_size * model_dim)
// Returns: a flattened array of probabilities (batch_size * vocab_size), caller must free.
float* forward_output_layer_batch(const OutputLayer* layer, const float* hidden_state_batch, int batch_size, int model_dim, int vocab_size) {
    if (!layer || !hidden_state_batch) {
        fprintf(stderr, "Error: Invalid input to forward_output_layer_batch\n");
        return NULL;
    }

    float* output_probs_batch = create_float_array(batch_size * vocab_size);
    if (!output_probs_batch) {
        fprintf(stderr, "Error: Memory allocation failed for output probabilities batch\n");
        return NULL;
    }

    for (int b = 0; b < batch_size; ++b) {
        const float* hidden_state = &hidden_state_batch[b * model_dim];
        float* logits = create_float_array(vocab_size); // Logits for this batch item
        if (!logits) {
            free_float_array(output_probs_batch);
            fprintf(stderr, "Error: Memory allocation failed for logits in output layer batch\n");
            return NULL;
        }

        // 1. Unembedding Weights multiplication
        ternary_matrix_vector_mul(&layer->unembedding_weights, hidden_state, logits);
        
        // 2. Add Bias
        add_vector_inplace(logits, layer->bias, vocab_size);

        // 3. Softmax to get probabilities
        softmax(logits, vocab_size);

        // Copy probabilities to the output batch
        memcpy(&output_probs_batch[b * vocab_size], logits, vocab_size * sizeof(float));
        free_float_array(logits); // Free logits for this batch item
    }

    return output_probs_batch;
}

