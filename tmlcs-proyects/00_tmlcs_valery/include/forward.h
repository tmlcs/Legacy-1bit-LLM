#ifndef FORWARD_H
#define FORWARD_H

#include "legacy_llm.h" // For structs and types

// Forward pass functions
float* forward_embedding(const EmbeddingLayer* layer, int token_id, int model_dim); // Deprecated: Use batched version
float* forward_embedding_batch(const EmbeddingLayer* layer, const int* input_batch, int batch_size, int model_dim);

float* forward_multi_head_attention(const MultiHeadAttentionLayer* layer, const float* input_vec, int model_dim); // Deprecated: Use batched version
float* forward_multi_head_attention_batch(const MultiHeadAttentionLayer* layer, const float* input_batch, int batch_size, int model_dim);

float* forward_feed_forward(const FeedForwardLayer* layer, const float* input_vec, int model_dim); // Deprecated: Use batched version
float* forward_feed_forward_batch(const FeedForwardLayer* layer, const float* input_batch, int batch_size, int model_dim);
float* forward_transformer_block_with_context(const TransformerBlock* block, const float* input_vec, int model_dim, TransformerBlockContext* context); // Deprecated: Use batched version
float* forward_transformer_block_batch_with_context(const TransformerBlock* block, const float* input_batch, int batch_size, int model_dim, TransformerBlockContext* context);
float* forward_transformer_block(const TransformerBlock* block, const float* input_vec, int model_dim); // Original for testing
float* forward_llm(LegacyLLM* model, int token_id); // Deprecated: Use batched version
float* forward_llm_batch(LegacyLLM* model, const int* input_batch, int batch_size);
float* forward_output_layer_batch(const OutputLayer* layer, const float* hidden_state_batch, int batch_size, int model_dim, int vocab_size);

#endif // FORWARD_H