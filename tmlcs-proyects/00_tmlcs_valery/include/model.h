#ifndef MODEL_H
#define MODEL_H

#include "legacy_llm.h" // For global defines and forward declarations

// --- Data Structures for Ternary Weights ---
struct TernaryMatrix {
    int rows;
    int cols;
    int8_t* data;
};

// --- Model Layer Structures ---
struct EmbeddingLayer {
    TernaryMatrix embedding_weights; // vocab_size x MODEL_DIM
};

struct MultiHeadAttentionLayer {
    TernaryMatrix Wq; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wk; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wv; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wo; // MODEL_DIM x MODEL_DIM (Output projection)
    float* bq; // MODEL_DIM
    float* bk; // MODEL_DIM
    float* bv; // MODEL_DIM
    float* bo; // MODEL_DIM
};

struct FeedForwardLayer {
    TernaryMatrix Wi; // (MODEL_DIM * FFN_DIM_MULTIPLIER) x MODEL_DIM (Output x Input)
    TernaryMatrix Wo; // MODEL_DIM x (MODEL_DIM * FFN_DIM_MULTIPLIER) (Output x Input)
    float* bi; // (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bo; // MODEL_DIM
};

struct TransformerBlock {
    MultiHeadAttentionLayer attention;
    FeedForwardLayer ffn;
    float* norm1_gamma; // MODEL_DIM
    float* norm1_beta;  // MODEL_DIM
    float* norm2_gamma; // MODEL_DIM
    float* norm2_beta;  // MODEL_DIM
};

struct OutputLayer {
    TernaryMatrix unembedding_weights; // MODEL_DIM x vocab_size
    float* bias;                       // vocab_size
};

// Full Legacy-1bit LLM Model (Forward-Pass only structure)
struct LegacyLLM {
    EmbeddingLayer embedding;
    TransformerBlock* transformer_blocks; // Array of transformer blocks
    OutputLayer output;
    int num_transformer_blocks;
    int vocab_size;
    int model_dim;
    // Contexts to store intermediate activations for backward pass
    TransformerBlockContext** block_contexts; // Array of pointers to contexts
    float* final_hidden_state_input; // Hidden state before final output layer
};


// --- Gradient Structures (mirrors model structure but holds float gradients) ---
struct EmbeddingLayerGradients {
    float* embedding_weights; // vocab_size x MODEL_DIM
};

struct MultiHeadAttentionLayerGradients {
    float* Wq; // MODEL_DIM x MODEL_DIM
    float* Wk; // MODEL_DIM x MODEL_DIM
    float* Wv; // MODEL_DIM x MODEL_DIM
    float* Wo; // MODEL_DIM x MODEL_DIM
    float* bq; // MODEL_DIM
    float* bk; // MODEL_DIM
    float* bv; // MODEL_DIM
    float* bo; // MODEL_DIM
};

struct FeedForwardLayerGradients {
    float* Wi; // (MODEL_DIM * FFN_DIM_MULTIPLIER) x MODEL_DIM
    float* Wo; // MODEL_DIM x (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bi; // (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bo; // MODEL_DIM
};

struct TransformerBlockGradients {
    MultiHeadAttentionLayerGradients attention_grads;
    FeedForwardLayerGradients ffn_grads;
    float* norm1_gamma; // MODEL_DIM
    float* norm1_beta;  // MODEL_DIM
    float* norm2_gamma; // MODEL_DIM
    float* norm2_beta;  // MODEL_DIM
};

struct OutputLayerGradients {
    float* unembedding_weights; // MODEL_DIM x vocab_size
    float* bias;               // vocab_size
};

struct LegacyLLM_Gradients {
    EmbeddingLayerGradients embedding_grads;
    TransformerBlockGradients* transformer_block_grads;
    OutputLayerGradients output_grads;
    int num_transformer_blocks;
    int vocab_size;
    int model_dim;
};




// --- Model Operations Functions (Allocation and Initialization) ---
TernaryMatrix create_ternary_matrix(int rows, int cols);
void free_ternary_matrix(TernaryMatrix* mat);
float* create_float_array(int size);
void free_float_array(float* arr);
LegacyLLM* create_legacy_llm(int vocab_size, int model_dim, int num_transformer_blocks);
void free_legacy_llm(LegacyLLM* model);
LegacyLLM_Gradients* create_legacy_llm_gradients(int vocab_size, int model_dim, int num_transformer_blocks);
void free_legacy_llm_gradients(LegacyLLM_Gradients* grads);
void zero_legacy_llm_gradients(LegacyLLM_Gradients* grads);
void apply_ternary_weight_updates(LegacyLLM* model, LegacyLLM_Gradients* grads, float learning_rate);

// --- Model Persistence Functions ---
int save_model(const LegacyLLM* model, const char* filepath);
LegacyLLM* load_model(const char* filepath);

// --- Context management ---
TransformerBlockContext* create_transformer_block_context(int model_dim, int ffn_hidden_dim);
void free_transformer_block_context(TransformerBlockContext* context);

#endif // MODEL_H