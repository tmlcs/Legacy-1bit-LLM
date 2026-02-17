#ifndef LEGACY_LLM_H
#define LEGACY_LLM_H

#include <stdint.h> // For int8_t

// Max vocabulary size for character-level tokenization
#define MAX_VOCAB_SIZE 256 // ASCII character set
#define MODEL_DIM 256     // As specified: 256 or 512
#define NUM_HEADS 4       // Example number of attention heads (MODEL_DIM % NUM_HEADS must be 0)
#define FFN_DIM_MULTIPLIER 4 // Multiplier for feed-forward network dimension
#define MAX_SEQUENCE_LENGTH 128 // Max input sequence length for model
#define BATCH_SIZE 8 // Number of sequences in a batch
#define PAD_TOKEN MAX_VOCAB_SIZE  // Special token for padding, using MAX_VOCAB_SIZE to avoid conflict with actual char codes
#define LAYER_NORM_EPSILON 1e-5f // Epsilon for layer normalization to prevent division by zero

// --- Data Structures for Ternary Weights ---

// A simple structure to represent a matrix of int8_t values (for ternary weights)
// Values are expected to be -1, 0, or 1.
typedef struct {
    int rows;
    int cols;
    int8_t* data;
} TernaryMatrix;

// --- Model Layer Structures ---

// Input Embedding Layer
typedef struct {
    TernaryMatrix embedding_weights; // vocab_size x MODEL_DIM
} EmbeddingLayer;

// Multi-Head Self-Attention Block
typedef struct {
    // Query, Key, Value weights
    TernaryMatrix Wq; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wk; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wv; // MODEL_DIM x MODEL_DIM
    TernaryMatrix Wo; // MODEL_DIM x MODEL_DIM (Output projection)

    // Biases (can be float as they are "high-precision accumulators" or small floats)
    float* bq; // MODEL_DIM
    float* bk; // MODEL_DIM
    float* bv; // MODEL_DIM
    float* bo; // MODEL_DIM
} MultiHeadAttentionLayer;

// Feed-Forward Network Block
typedef struct {
    // Wi: (MODEL_DIM * FFN_DIM_MULTIPLIER) x MODEL_DIM (Output x Input)
    TernaryMatrix Wi;
    // Wo: MODEL_DIM x (MODEL_DIM * FFN_DIM_MULTIPLIER) (Output x Input)
    TernaryMatrix Wo;
    
    // Biases
    float* bi; // (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bo; // MODEL_DIM
} FeedForwardLayer;

// Transformer Block (contains Attention and FFN)
typedef struct {
    MultiHeadAttentionLayer attention;
    FeedForwardLayer ffn;
    // Layer normalization parameters (could be float)
    float* norm1_gamma; // MODEL_DIM
    float* norm1_beta;  // MODEL_DIM
    float* norm2_gamma; // MODEL_DIM
    float* norm2_beta;  // MODEL_DIM
} TransformerBlock;

// Final Output Layer (Unembedding)
typedef struct {
    TernaryMatrix unembedding_weights; // vocab_size x MODEL_DIM
    float* bias;                       // vocab_size
} OutputLayer;

// Full Legacy-1bit LLM Model (Forward-Pass only structure)
typedef struct {
    EmbeddingLayer embedding;
    TransformerBlock* transformer_blocks; // Array of transformer blocks
    OutputLayer output;
    int num_transformer_blocks;
    int vocab_size;
    int model_dim;
    // Contexts to store intermediate activations for backward pass
    struct TransformerBlockContext** block_contexts; // Array of pointers to contexts
    float* final_hidden_state_input_batch; // Hidden state batch before final output layer
} LegacyLLM;


// --- Gradient Structures (mirrors model structure but holds float gradients) ---
typedef struct {
    float* embedding_weights; // vocab_size x MODEL_DIM
} EmbeddingLayerGradients;

typedef struct {
    float* Wq; // MODEL_DIM x MODEL_DIM
    float* Wk; // MODEL_DIM x MODEL_DIM
    float* Wv; // MODEL_DIM x MODEL_DIM
    float* Wo; // MODEL_DIM x MODEL_DIM
    float* bq; // MODEL_DIM
    float* bk; // MODEL_DIM
    float* bv; // MODEL_DIM
    float* bo; // MODEL_DIM
} MultiHeadAttentionLayerGradients;

typedef struct {
    float* Wi; // (MODEL_DIM * FFN_DIM_MULTIPLIER) x MODEL_DIM
    float* Wo; // MODEL_DIM x (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bi; // (MODEL_DIM * FFN_DIM_MULTIPLIER)
    float* bo; // MODEL_DIM
} FeedForwardLayerGradients;

typedef struct {
    MultiHeadAttentionLayerGradients attention_grads;
    FeedForwardLayerGradients ffn_grads;
    float* norm1_gamma; // MODEL_DIM
    float* norm1_beta;  // MODEL_DIM
    float* norm2_gamma; // MODEL_DIM
    float* norm2_beta;  // MODEL_DIM
} TransformerBlockGradients;

typedef struct {
    float* unembedding_weights; // MODEL_DIM x vocab_size
    float* bias;               // vocab_size
} OutputLayerGradients;

typedef struct {
    EmbeddingLayerGradients embedding_grads;
    TransformerBlockGradients* transformer_block_grads;
    OutputLayerGradients output_grads;
    int num_transformer_blocks;
    int vocab_size;
    int model_dim;
} LegacyLLM_Gradients;


// --- Context Structures (to store intermediate activations for backward pass) ---
// Structure to store context for a single transformer block in a batched scenario
typedef struct TransformerBlockContext {
    // Input to the block for each item in the batch
    float* block_input_batch; // BATCH_SIZE * MODEL_DIM

    // LayerNorm means and inverse standard deviations for each item in the batch
    float* ln1_mean_batch;       // BATCH_SIZE
    float* ln1_inv_std_dev_batch; // BATCH_SIZE
    float* ln2_mean_batch;       // BATCH_SIZE
    float* ln2_inv_std_dev_batch; // BATCH_SIZE

    // Add other intermediate values needed for backward pass, now batched
    // For example, if attention mechanism stores Q, K, V or attention scores, they would be batched here.
    // For simplicity, for now, we only batch LN context. Other intermediate values will be recomputed.
} TransformerBlockContext;




#endif // LEGACY_LLM_H