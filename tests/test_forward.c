#include "test_framework.h"
#include "forward.h"
#include "model.h" // For TernaryMatrix, create_float_array, free_float_array
#include "legacy_llm.h" // For EmbeddingLayer, MODEL_DIM, PAD_TOKEN

// Helper for float array comparison (copied from test_math_ops.c, can be moved to test_framework.h)
static int compare_float_arrays(const float* arr1, const float* arr2, int size, float epsilon) {
    for (int i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > epsilon) {
            return 0; // Not equal
        }
    }
    return 1; // Equal
}


// Function to create a dummy EmbeddingLayer for testing
EmbeddingLayer create_dummy_embedding_layer(int vocab_size, int model_dim) {
    EmbeddingLayer layer;
    layer.embedding_weights.rows = vocab_size;
    layer.embedding_weights.cols = model_dim;
    layer.embedding_weights.data = (int8_t*)calloc(vocab_size * model_dim, sizeof(int8_t));
    ASSERT_NOT_NULL(layer.embedding_weights.data, "Failed to allocate dummy embedding weights");

    // Initialize with some known values (-1, 0, 1)
    // For simplicity, let's make token_id == i correspond to embedding values where each value is (i % 3 - 1)
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < model_dim; ++j) {
            layer.embedding_weights.data[i * model_dim + j] = (int8_t)((i + j) % 3 - 1);
        }
    }
    return layer;
}

void free_dummy_embedding_layer(EmbeddingLayer* layer) {
    free(layer->embedding_weights.data);
    layer->embedding_weights.data = NULL;
}

// Function to create a dummy MultiHeadAttentionLayer for testing
MultiHeadAttentionLayer create_dummy_mha_layer(int model_dim) {
    MultiHeadAttentionLayer layer;

    // Initialize TernaryMatrix weights
    layer.Wq = create_ternary_matrix(model_dim, model_dim);
    layer.Wk = create_ternary_matrix(model_dim, model_dim);
    layer.Wv = create_ternary_matrix(model_dim, model_dim);
    layer.Wo = create_ternary_matrix(model_dim, model_dim);

    // Initialize float biases
    layer.bq = create_float_array(model_dim);
    layer.bk = create_float_array(model_dim);
    layer.bv = create_float_array(model_dim);
    layer.bo = create_float_array(model_dim);

    // For simplicity, let's set all weights to 1 and biases to 0 for predictable calculations
    for (int i = 0; i < model_dim * model_dim; ++i) {
        layer.Wq.data[i] = 1;
        layer.Wk.data[i] = 1;
        layer.Wv.data[i] = 1;
        layer.Wo.data[i] = 1;
    }
    for (int i = 0; i < model_dim; ++i) {
        layer.bq[i] = 0.0f;
        layer.bk[i] = 0.0f;
        layer.bv[i] = 0.0f;
        layer.bo[i] = 0.0f;
    }
    
    return layer;
}

void free_dummy_mha_layer(MultiHeadAttentionLayer* layer) {
    free_ternary_matrix(&layer->Wq);
    free_ternary_matrix(&layer->Wk);
    free_ternary_matrix(&layer->Wv);
    free_ternary_matrix(&layer->Wo);
    free_float_array(layer->bq);
    free_float_array(layer->bk);
    free_float_array(layer->bv);
    free_float_array(layer->bo);
}

// Function to create a dummy FeedForwardLayer for testing
FeedForwardLayer create_dummy_ffn_layer(int model_dim) {
    FeedForwardLayer layer;
    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;

    // Initialize TernaryMatrix weights
    layer.Wi = create_ternary_matrix(ffn_hidden_dim, model_dim);
    layer.Wo = create_ternary_matrix(model_dim, ffn_hidden_dim);

    // Initialize float biases
    layer.bi = create_float_array(ffn_hidden_dim);
    layer.bo = create_float_array(model_dim);

    // For simplicity, let's set all weights to 1 and biases to 0 for predictable calculations
    for (int i = 0; i < ffn_hidden_dim * model_dim; ++i) {
        layer.Wi.data[i] = 1;
    }
    for (int i = 0; i < model_dim * ffn_hidden_dim; ++i) {
        layer.Wo.data[i] = 1;
    }
    for (int i = 0; i < ffn_hidden_dim; ++i) {
        layer.bi[i] = 0.0f;
    }
    for (int i = 0; i < model_dim; ++i) {
        layer.bo[i] = 0.0f;
    }
    
    return layer;
}

void free_dummy_ffn_layer(FeedForwardLayer* layer) {
    free_ternary_matrix(&layer->Wi);
    free_ternary_matrix(&layer->Wo);
    free_float_array(layer->bi);
    free_float_array(layer->bo);
}

// Function to create a dummy TransformerBlock for testing
TransformerBlock create_dummy_transformer_block(int model_dim) {
    TransformerBlock block;
    block.attention = create_dummy_mha_layer(model_dim);
    block.ffn = create_dummy_ffn_layer(model_dim);

    block.norm1_gamma = create_float_array(model_dim);
    block.norm1_beta = create_float_array(model_dim);
    block.norm2_gamma = create_float_array(model_dim);
    block.norm2_beta = create_float_array(model_dim);

    // Set gamma to 1s and beta to 0s for predictable LayerNorm
    for (int i = 0; i < model_dim; ++i) {
        block.norm1_gamma[i] = 1.0f;
        block.norm1_beta[i] = 0.0f;
        block.norm2_gamma[i] = 1.0f;
        block.norm2_beta[i] = 0.0f;
    }
    return block;
}

void free_dummy_transformer_block(TransformerBlock* block) {
    free_dummy_mha_layer(&block->attention);
    free_dummy_ffn_layer(&block->ffn);
    free_float_array(block->norm1_gamma);
    free_float_array(block->norm1_beta);
    free_float_array(block->norm2_gamma);
    free_float_array(block->norm2_beta);
}

// Function to create a dummy TransformerBlockContext for testing
TransformerBlockContext* create_dummy_transformer_block_context(int model_dim, int test_batch_size) {
    TransformerBlockContext* context = (TransformerBlockContext*)calloc(1, sizeof(TransformerBlockContext));
    ASSERT_NOT_NULL(context, "Failed to allocate dummy TransformerBlockContext");

    context->block_input_batch = create_float_array(test_batch_size * model_dim);
    context->ln1_mean_batch = create_float_array(test_batch_size);
    context->ln1_inv_std_dev_batch = create_float_array(test_batch_size);
    context->ln2_mean_batch = create_float_array(test_batch_size);
    context->ln2_inv_std_dev_batch = create_float_array(test_batch_size);
    
    return context;
}

void free_dummy_transformer_block_context(TransformerBlockContext* context) {
    if (context) {
        free_float_array(context->block_input_batch);
        free_float_array(context->ln1_mean_batch);
        free_float_array(context->ln1_inv_std_dev_batch);
        free_float_array(context->ln2_mean_batch);
        free_float_array(context->ln2_inv_std_dev_batch);
        free(context);
    }
}

// Function to create a dummy OutputLayer for testing
OutputLayer create_dummy_output_layer(int model_dim, int vocab_size) {
    OutputLayer layer;

    layer.unembedding_weights = create_ternary_matrix(vocab_size, model_dim);
    layer.bias = create_float_array(vocab_size);

    // Set weights to 1 and biases to 0 for predictable calculations
    for (int i = 0; i < model_dim * vocab_size; ++i) {
        layer.unembedding_weights.data[i] = 1;
    }
    for (int i = 0; i < vocab_size; ++i) {
        layer.bias[i] = 0.0f;
    }
    return layer;
}

void free_dummy_output_layer(OutputLayer* layer) {
    free_ternary_matrix(&layer->unembedding_weights);
    free_float_array(layer->bias);
}

// Function to create a dummy LegacyLLM for testing with predictable weights/biases
LegacyLLM* create_dummy_llm(int vocab_size, int model_dim, int num_transformer_blocks, int test_batch_size) {
    LegacyLLM* model = (LegacyLLM*)calloc(1, sizeof(LegacyLLM));
    ASSERT_NOT_NULL(model, "Failed to allocate dummy LegacyLLM model");

    model->vocab_size = vocab_size;
    model->model_dim = model_dim;
    model->num_transformer_blocks = num_transformer_blocks;

    // Embedding Layer
    model->embedding.embedding_weights = create_ternary_matrix(vocab_size, model_dim);
    for (int i = 0; i < vocab_size * model_dim; ++i) {
        model->embedding.embedding_weights.data[i] = 1; // All 1s for predictability
    }

    // Transformer Blocks
    model->transformer_blocks = (TransformerBlock*)calloc(num_transformer_blocks, sizeof(TransformerBlock));
    ASSERT_NOT_NULL(model->transformer_blocks, "Failed to allocate dummy transformer blocks");

    // Allocate contexts array
    model->block_contexts = (TransformerBlockContext**)calloc(num_transformer_blocks, sizeof(TransformerBlockContext*));
    ASSERT_NOT_NULL(model->block_contexts, "Failed to allocate dummy block_contexts");
    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;
    for (int i = 0; i < num_transformer_blocks; ++i) {
        model->block_contexts[i] = create_dummy_transformer_block_context(model_dim, test_batch_size);
        ASSERT_NOT_NULL(model->block_contexts[i], "Failed to create dummy TransformerBlockContext");

        // Set up each TransformerBlock
        model->transformer_blocks[i].attention = create_dummy_mha_layer(model_dim);
        for (int k = 0; k < model_dim * model_dim; ++k) { // All 1s weights
            model->transformer_blocks[i].attention.Wq.data[k] = 1;
            model->transformer_blocks[i].attention.Wk.data[k] = 1;
            model->transformer_blocks[i].attention.Wv.data[k] = 1;
            model->transformer_blocks[i].attention.Wo.data[k] = 1;
        }
        for (int k = 0; k < model_dim; ++k) { // All 0s biases
            model->transformer_blocks[i].attention.bq[k] = 0.0f;
            model->transformer_blocks[i].attention.bk[k] = 0.0f;
            model->transformer_blocks[i].attention.bv[k] = 0.0f;
            model->transformer_blocks[i].attention.bo[k] = 0.0f;
        }

        model->transformer_blocks[i].ffn = create_dummy_ffn_layer(model_dim);
        for (int k = 0; k < ffn_hidden_dim * model_dim; ++k) { // All 1s weights
            model->transformer_blocks[i].ffn.Wi.data[k] = 1;
        }
        for (int k = 0; k < model_dim * ffn_hidden_dim; ++k) { // All 1s weights
            model->transformer_blocks[i].ffn.Wo.data[k] = 1;
        }
        for (int k = 0; k < ffn_hidden_dim; ++k) { // All 0s biases
            model->transformer_blocks[i].ffn.bi[k] = 0.0f;
        }
        for (int k = 0; k < model_dim; ++k) { // All 0s biases
            model->transformer_blocks[i].ffn.bo[k] = 0.0f;
        }

        // Layer Normalization
        model->transformer_blocks[i].norm1_gamma = create_float_array(model_dim);
        model->transformer_blocks[i].norm1_beta = create_float_array(model_dim);
        model->transformer_blocks[i].norm2_gamma = create_float_array(model_dim);
        model->transformer_blocks[i].norm2_beta = create_float_array(model_dim);
        for (int k = 0; k < model_dim; ++k) {
            model->transformer_blocks[i].norm1_gamma[k] = 1.0f; // All 1s
            model->transformer_blocks[i].norm1_beta[k] = 0.0f;  // All 0s
            model->transformer_blocks[i].norm2_gamma[k] = 1.0f; // All 1s
            model->transformer_blocks[i].norm2_beta[k] = 0.0f;  // All 0s
        }
    }

    // Output Layer
    model->output = create_dummy_output_layer(model_dim, vocab_size);
    for (int i = 0; i < model_dim * vocab_size; ++i) {
        model->output.unembedding_weights.data[i] = 1; // All 1s
    }
    for (int i = 0; i < vocab_size; ++i) {
        model->output.bias[i] = 0.0f; // All 0s
    }

    // Allocate final_hidden_state_input_batch
    model->final_hidden_state_input_batch = create_float_array(test_batch_size * model_dim);
    ASSERT_NOT_NULL(model->final_hidden_state_input_batch, "Failed to allocate dummy final_hidden_state_input_batch");


    return model;
}

void free_dummy_llm(LegacyLLM* model) {
    if (!model) return;

    // Free Embedding Layer
    free_ternary_matrix(&model->embedding.embedding_weights);

    // Free Transformer Blocks
    if (model->transformer_blocks) {
        for (int i = 0; i < model->num_transformer_blocks; ++i) {
            free_dummy_mha_layer(&model->transformer_blocks[i].attention);
            free_dummy_ffn_layer(&model->transformer_blocks[i].ffn);
            free_float_array(model->transformer_blocks[i].norm1_gamma);
            free_float_array(model->transformer_blocks[i].norm1_beta);
            free_float_array(model->transformer_blocks[i].norm2_gamma);
            free_float_array(model->transformer_blocks[i].norm2_beta);
        }
        free(model->transformer_blocks);
    }

    // Free contexts
    if (model->block_contexts) {
        for (int i = 0; i < model->num_transformer_blocks; ++i) {
            free_dummy_transformer_block_context(model->block_contexts[i]);
        }
        free(model->block_contexts);
    }

    // Free Output Layer
    free_ternary_matrix(&model->output.unembedding_weights);
    free_float_array(model->output.bias);

    // Free final_hidden_state_input_batch
    free_float_array(model->final_hidden_state_input_batch);

    free(model);
}


void test_forward_embedding_batch() {
    TEST_BEGIN("forward_embedding_batch");

    int vocab_size = 5;
    int model_dim = 4;
    int batch_size = 3;
    float epsilon = 1e-6f;

    EmbeddingLayer embedding_layer = create_dummy_embedding_layer(vocab_size, model_dim);
    ASSERT_NOT_NULL(embedding_layer.embedding_weights.data, "Dummy embedding layer creation failed.");

    // Input batch: token IDs {0, 1, PAD_TOKEN}
    int input_batch[] = {0, 1, PAD_TOKEN};

    // Expected output for input_batch {0, 1, PAD_TOKEN}
    // Token 0: (0%3-1, 1%3-1, 2%3-1, 3%3-1) = (-1, 0, 1, -1)  (based on (i+j)%3-1 for i=0) -> (0%3-1 = -1, 1%3-1=0, 2%3-1=1, 3%3-1=-1)
    // Token 1: (1%3-1, 2%3-1, 3%3-1, 4%3-1) = (0, 1, -1, 0)   (based on (i+j)%3-1 for i=1) -> (1%3-1=0, 2%3-1=1, 3%3-1=-1, 4%3-1=0)
    // PAD_TOKEN: (0, 0, 0, 0)
    float expected_output[] = {
        -1.0f,  0.0f,  1.0f, -1.0f, // Embedding for token 0 (0+0)%3-1, (0+1)%3-1, (0+2)%3-1, (0+3)%3-1
         0.0f,  1.0f, -1.0f,  0.0f, // Embedding for token 1 (1+0)%3-1, (1+1)%3-1, (1+2)%3-1, (1+3)%3-1
         0.0f,  0.0f,  0.0f,  0.0f  // Embedding for PAD_TOKEN
    };

    float* output_embeddings = forward_embedding_batch(&embedding_layer, input_batch, batch_size, model_dim);
    ASSERT_NOT_NULL(output_embeddings, "forward_embedding_batch returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_embeddings, expected_output, batch_size * model_dim, epsilon), "Batched embedding output mismatch.");

    free_float_array(output_embeddings);
    free_dummy_embedding_layer(&embedding_layer);

    TEST_END();
}

void test_forward_multi_head_attention_batch() {
    TEST_BEGIN("forward_multi_head_attention_batch");

    int model_dim = 4;
    int batch_size = 2;
    float epsilon = 1e-6f;

    MultiHeadAttentionLayer mha_layer = create_dummy_mha_layer(model_dim);
    ASSERT_NOT_NULL(mha_layer.Wq.data, "Dummy MHA layer creation failed.");

    // Input batch: 2 vectors, each of model_dim
    // input_batch_0 = {1.0, 2.0, 3.0, 4.0}
    // input_batch_1 = {5.0, 6.0, 7.0, 8.0}
    float input_batch[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    // Expected output calculation (simplified weights = 1, biases = 0, so output is just sum of input)
    // For MHA: output = Wo * (V) + bo. Since Wo is all 1s, and bo is 0, V is copied.
    // And V is Wv * input + bv. Since Wv is all 1s and bv is 0, V is sum of input elements repeated model_dim times.
    // However, the current simplified attention in forward.c does:
    // 1. Q, K, V projections
    // 2. memcpy(attention_output_vec, value_vec, model_dim * sizeof(float)); // Copies value_vec
    // 3. Output Projection: ternary_matrix_vector_mul(&layer->Wo, attention_output_vec, current_output_vec);
    //    add_vector_inplace(current_output_vec, layer->bo, model_dim);
    //
    // So, with Wq, Wk, Wv, Wo all 1s, and all biases 0:
    // value_vec elements will be sum(input_vec)
    // attention_output_vec will be value_vec
    // current_output_vec will be sum(attention_output_vec) = sum(value_vec) = sum(sum(input_vec))
    // This is incorrect based on actual transformer attention. The implementation in forward.c for MHA is highly simplified.
    // It should be (query @ key.T) * value.
    // Let's re-evaluate based on the `src/forward.c` code snippet provided earlier by the user.

    // From src/forward.c:
    // ternary_matrix_vector_mul(&layer->Wq, input_vec, query_vec); add_vector_inplace(query_vec, layer->bq, model_dim);
    // ternary_matrix_vector_mul(&layer->Wk, input_vec, key_vec); add_vector_inplace(key_vec, layer->bk, model_dim);
    // ternary_matrix_vector_mul(&layer->Wv, input_vec, value_vec); add_vector_inplace(value_vec, layer->bv, model_dim);
    // memcpy(attention_output_vec, value_vec, model_dim * sizeof(float)); // Copies value_vec
    // ternary_matrix_vector_mul(&layer->Wo, attention_output_vec, final_output); add_vector_inplace(final_output, layer->bo, model_dim);

    // Given Wq, Wk, Wv, Wo all 1s, and all biases 0:
    // ternary_matrix_vector_mul with all 1s weights (TernaryMatrix) effectively performs a sum of input vector elements
    // For input_vec = {1, 2, 3, 4}, model_dim = 4:
    // Wq * input_vec = {sum(input_vec), sum(input_vec), sum(input_vec), sum(input_vec)} = {10, 10, 10, 10}
    // So, query_vec = {10, 10, 10, 10}
    // key_vec = {10, 10, 10, 10}
    // value_vec = {10, 10, 10, 10}
    // attention_output_vec = {10, 10, 10, 10} (due to memcpy from value_vec)
    // final_output = Wo * attention_output_vec = {sum(attention_output_vec), ..., sum(attention_output_vec)}
    // = {10+10+10+10, ..., 10+10+10+10} = {40, 40, 40, 40}



    float expected_output[] = {
        40.0f, 40.0f, 40.0f, 40.0f,
        104.0f, 104.0f, 104.0f, 104.0f
    };

    float* output_mha_batch = forward_multi_head_attention_batch(&mha_layer, input_batch, batch_size, model_dim);
    ASSERT_NOT_NULL(output_mha_batch, "forward_multi_head_attention_batch returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_mha_batch, expected_output, batch_size * model_dim, epsilon), "Batched MHA output mismatch.");

    free_float_array(output_mha_batch);
    free_dummy_mha_layer(&mha_layer);

    TEST_END();
}

void test_forward_feed_forward_batch() {
    TEST_BEGIN("forward_feed_forward_batch");

    int model_dim = 4;
    int batch_size = 2;
    float epsilon = 1e-6f;

    FeedForwardLayer ffn_layer = create_dummy_ffn_layer(model_dim);
    ASSERT_NOT_NULL(ffn_layer.Wi.data, "Dummy FFN layer creation failed.");

    // Input batch: 2 vectors, each of model_dim
    // input_batch_0 = {1.0, -2.0, 3.0, -4.0}
    // input_batch_1 = {5.0, 6.0, -7.0, 8.0}
    float input_batch[] = {
        1.0f, -2.0f, 3.0f, -4.0f,
        5.0f, 6.0f, -7.0f, 8.0f
    };

    // Expected output calculation (simplified weights = 1, biases = 0)
    // 1. First Linear Layer (Wi) + bi
    // hidden_output elements = sum(input_vec) (due to Wi all 1s, bi all 0s)
    // For input_batch_0 = {1,-2,3,-4}, sum = -2. hidden_output_0 = {-2, -2, ..., -2} (ffn_hidden_dim times)
    // For input_batch_1 = {5,6,-7,8}, sum = 12. hidden_output_1 = {12, 12, ..., 12} (ffn_hidden_dim times)

    // 2. Activation Function (ReLU)
    // For hidden_output_0 = {-2, ...}, ReLU makes it {0, 0, ..., 0}
    // For hidden_output_1 = {12, ...}, ReLU keeps it {12, 12, ..., 12}

    // 3. Second Linear Layer (Wo) + bo
    // final_output elements = sum(relu_output) (due to Wo all 1s, bo all 0s)
    // For input_batch_0, relu_output_0 is all zeros. So, final_output_0 = {0, 0, 0, 0}
    // For input_batch_1, relu_output_1 is all 12s (ffn_hidden_dim times).
    // sum(relu_output_1) = ffn_hidden_dim * 12 = 16 * 12 = 192.0f
    // So, final_output_1 = {192.0f, 192.0f, 192.0f, 192.0f}

    float expected_output[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        192.0f, 192.0f, 192.0f, 192.0f
    };

    float* output_ffn_batch = forward_feed_forward_batch(&ffn_layer, input_batch, batch_size, model_dim);
    ASSERT_NOT_NULL(output_ffn_batch, "forward_feed_forward_batch returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_ffn_batch, expected_output, batch_size * model_dim, epsilon), "Batched FFN output mismatch.");

    free_float_array(output_ffn_batch);
    free_dummy_ffn_layer(&ffn_layer);

    TEST_END();
}

void test_forward_transformer_block_batch_with_context() {
    TEST_BEGIN("forward_transformer_block_batch_with_context");

    int model_dim = 4;
    int batch_size = 1; // Start with batch_size 1 for simpler manual calculation
    float epsilon = 1e-2f; // Relax epsilon due to multiple floating point operations

    TransformerBlock block = create_dummy_transformer_block(model_dim);
    ASSERT_NOT_NULL(block.attention.Wq.data, "Dummy TransformerBlock creation failed.");

    TransformerBlockContext* context = create_dummy_transformer_block_context(model_dim, batch_size);
    ASSERT_NOT_NULL(context, "Dummy TransformerBlockContext creation failed.");

    float input_batch[] = {1.0f, 2.0f, 3.0f, 4.0f}; // sum = 10

    // Manual calculation of expected output:
    // With all-1 weights and all-0 biases for MHA and FFN, and gamma=1, beta=0 for LayerNorm.
    // 1. LayerNorm 1:
    //    input = {1,2,3,4} -> mean = 2.5, var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = (2.25+0.25+0.25+2.25)/4 = 5/4 = 1.25
    //    std_dev = sqrt(1.25) approx 1.118034
    //    inv_std_dev = 1/1.118034 approx 0.894427
    //    Normalized: (input - 2.5) * 0.894427
    //    input_normed_1 = {(-1.5)*0.8944, (-0.5)*0.8944, (0.5)*0.8944, (1.5)*0.8944}
    //                   = {-1.3416, -0.4472, 0.4472, 1.3416}
    //    Since gamma=1, beta=0, norm1_output = input_normed_1

    // 2. MHA (Simplified: Wo*(Wv*input+bv)+bo):
    //    Wv (all 1s) * input_normed_1 = sum(input_normed_1) = 0
    //    value_vec = {0,0,0,0} (since bv=0)
    //    attention_output_vec = {0,0,0,0} (memcpy from value_vec)
    //    Wo (all 1s) * attention_output_vec = {0,0,0,0} (since bo=0)
    //    attn_output = {0,0,0,0}

    // 3. Residual 1: current_output = input_batch + attn_output = {1,2,3,4} + {0,0,0,0} = {1,2,3,4}

    // 4. LayerNorm 2: (Applied to current_output = {1,2,3,4})
    //    input_normed_2 = {-1.3416, -0.4472, 0.4472, 1.3416} (same as input_normed_1 since input is same)
    //    Since gamma=1, beta=0, norm2_output = input_normed_2

    // 5. FFN:
    //    hidden_output = sum(input_normed_2) = 0 (since Wi all 1s, bi all 0s) -> {0,0,..,0}
    //    ReLU(hidden_output) = {0,0,..,0}
    //    final_output = Wo (all 1s) * ReLU_output = {0,0,0,0} (since bo=0)
    //    ffn_output = {0,0,0,0}

    // 6. Residual 2: final_output = current_output + ffn_output = {1,2,3,4} + {0,0,0,0} = {1,2,3,4}

    // So, with these simplified weights, the expected output should essentially be the input itself after all transformations.
    // However, the ternary_matrix_vector_mul with all 1s weights (TernaryMatrix) actually sums elements.
    // Let's re-calculate `value_vec` (intermediate in MHA) and `hidden_output` (intermediate in FFN) based on `math_ops.c`.
    // In ternary_matrix_vector_mul, if matrix is all 1s, output vector element `k` is `sum(input_vector)` if input_vector is `input_vec`.
    // So, for input_normed_1 = {-1.3416, -0.4472, 0.4472, 1.3416}, sum is ~0.
    // -> value_vec (from Wv*input_normed_1) would be {0,0,0,0} (if Wv all 1s)
    // -> attn_output (from Wo*value_vec) would be {0,0,0,0} (if Wo all 1s)
    // -> ffn_hidden_output (from Wi*norm2_output) would be {0,0,..,0} (if Wi all 1s and norm2_output sum is 0)
    // -> ffn_output (from Wo*ReLU(ffn_hidden_output)) would be {0,0,0,0} (if Wo all 1s)

    // This means for this specific setup (all 1s weights, all 0s biases for MHA/FFN, and gamma=1, beta=0 for LN)
    // the output should indeed be very close to the input, due to sums cancelling out and ReLU on negative sums becoming zero.

    float expected_output[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Expected to be approximately the input

    float* output_block_batch = forward_transformer_block_batch_with_context(&block, input_batch, batch_size, model_dim, context);
    ASSERT_NOT_NULL(output_block_batch, "forward_transformer_block_batch_with_context returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_block_batch, expected_output, batch_size * model_dim, epsilon), "Batched Transformer Block output mismatch.");

    free_float_array(output_block_batch);
    free_dummy_transformer_block_context(context);
    free_dummy_transformer_block(&block);

    TEST_END();
}

void test_forward_output_layer_batch() {
    TEST_BEGIN("forward_output_layer_batch");

    int model_dim = 4;
    int vocab_size = 3;
    int batch_size = 2;
    float epsilon = 1e-6f;

    OutputLayer output_layer = create_dummy_output_layer(model_dim, vocab_size);
    ASSERT_NOT_NULL(output_layer.unembedding_weights.data, "Dummy OutputLayer creation failed.");

    // Input hidden state batch: 2 vectors, each of model_dim
    // hidden_state_0 = {1.0, 2.0, 3.0, 4.0}
    // hidden_state_1 = {-1.0, 0.0, 1.0, 2.0}
    float hidden_state_batch[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f
    };

    // Expected output calculation (simplified weights = 1, biases = 0, then softmax)
    // For unembedding_weights (model_dim x vocab_size) all 1s, bias all 0s:
    // logits_for_item_k = (sum(hidden_state_item_k) for all vocab_size elements)
    //
    // For hidden_state_0 = {1,2,3,4}, sum = 10.
    // logits_0 = {10.0, 10.0, 10.0}
    // softmax({10,10,10}) = {1/3, 1/3, 1/3} = {0.3333, 0.3333, 0.3333} approx
    //
    // For hidden_state_1 = {-1,0,1,2}, sum = 2.
    // logits_1 = {2.0, 2.0, 2.0}
    // softmax({2,2,2}) = {1/3, 1/3, 1/3} = {0.3333, 0.3333, 0.3333} approx

    float expected_output[] = {
        0.33333333f, 0.33333333f, 0.33333333f,
        0.33333333f, 0.33333333f, 0.33333333f
    };
    epsilon = 1e-5f; // Relax epsilon for softmax

    float* output_probs_batch = forward_output_layer_batch(&output_layer, hidden_state_batch, batch_size, model_dim, vocab_size);
    ASSERT_NOT_NULL(output_probs_batch, "forward_output_layer_batch returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_probs_batch, expected_output, batch_size * vocab_size, epsilon), "Batched OutputLayer output mismatch.");

    free_float_array(output_probs_batch);
    free_dummy_output_layer(&output_layer);

    TEST_END();
}

void test_forward_llm_batch() {
    TEST_BEGIN("forward_llm_batch");

    int vocab_size = 5;
    int model_dim = 4;
    int num_transformer_blocks = 1; // Use 1 for simpler calculation
    int batch_size = 1; // Test with batch_size 1 for simpler manual calculation
    float epsilon = 1e-2f; // Relax epsilon for multiple floating point operations

    LegacyLLM* model = create_dummy_llm(vocab_size, model_dim, num_transformer_blocks, batch_size);
    ASSERT_NOT_NULL(model, "Dummy LegacyLLM creation failed.");

    // Input batch: token ID {1}
    int input_batch[] = {1};

    // Manual calculation of expected output for input_token_id = 1:
    // With all-1 weights and all-0 biases for all layers, and gamma=1, beta=0 for LayerNorm.

    // 1. Embedding Layer: for token_id=1, (i+j)%3-1 for i=1
    // embedding_weights.data[1*4+j] = (1+j)%3-1
    // j=0: (1+0)%3-1 = 0
    // j=1: (1+1)%3-1 = 1
    // j=2: (1+2)%3-1 = -1
    // j=3: (1+3)%3-1 = 0
    // So, initial_hidden_state = {0.0f, 1.0f, -1.0f, 0.0f}
    // sum(initial_hidden_state) = 0

    // 2. Transformer Block (with num_transformer_blocks = 1):
    // Based on previous test_forward_transformer_block_batch_with_context logic (simplified weights = 1, biases = 0, LN gamma=1, beta=0),
    // if input is {0, 1, -1, 0}, its sum is 0.
    // LayerNorm 1: mean=(0+1-1+0)/4 = 0. var= (0^2+1^2+(-1)^2+0^2)/4 = 2/4 = 0.5. std=sqrt(0.5)=0.707. inv_std=1.414.
    // input_normed = (input - 0)*1.414 = {0, 1.414, -1.414, 0}
    // MHA: sum(input_normed) = 0. Attn output will be {0,0,0,0}.
    // Residual 1: {0, 1, -1, 0} + {0,0,0,0} = {0, 1, -1, 0}
    // LayerNorm 2: same as LN1. input_normed again.
    // FFN: sum(input_normed) = 0. FFN output will be {0,0,0,0}.
    // Residual 2: {0, 1, -1, 0} + {0,0,0,0} = {0, 1, -1, 0}
    // So, hidden_state_after_transformer = {0.0f, 1.0f, -1.0f, 0.0f} (same as input)
    // sum(hidden_state_after_transformer) = 0

    // 3. Output Layer:
    // hidden_state_batch = {0.0f, 1.0f, -1.0f, 0.0f}
    // unembedding_weights all 1s, bias all 0s.
    // logits_for_item = sum(hidden_state_batch) (for all vocab_size elements)
    // sum = 0. So, logits = {0.0, 0.0, 0.0, 0.0, 0.0} (vocab_size=5)
    // softmax({0,0,0,0,0}) = {1/5, 1/5, 1/5, 1/5, 1/5} = {0.2, 0.2, 0.2, 0.2, 0.2}

    float expected_output[] = {
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f
    };

    float* output_probs_batch = forward_llm_batch(model, input_batch, batch_size);
    ASSERT_NOT_NULL(output_probs_batch, "forward_llm_batch returned NULL.");

    ASSERT_TRUE(compare_float_arrays(output_probs_batch, expected_output, batch_size * vocab_size, epsilon), "Batched LLM forward pass output mismatch.");

    free_float_array(output_probs_batch);
    free_dummy_llm(model);

    TEST_END();
}


void run_forward_tests() {
    printf("--- Running Forward Pass Tests ---\n");
    test_forward_embedding_batch();
    test_forward_multi_head_attention_batch(); // Call new test
    test_forward_feed_forward_batch(); // Call new test
    test_forward_transformer_block_batch_with_context(); // Call new test
    test_forward_output_layer_batch(); // Call new test
    test_forward_llm_batch(); // Call new test
}
