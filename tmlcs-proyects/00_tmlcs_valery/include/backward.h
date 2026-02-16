#ifndef BACKWARD_H
#define BACKWARD_H

#include "legacy_llm.h" // For structs and types

// Training Infrastructure Functions
float cross_entropy_loss(const float* predicted_probs, int true_token_id, int vocab_size);
float* d_loss_d_logits(const float* predicted_probs, int true_token_id, int vocab_size);

// Backward pass functions
float* backward_output_layer(const OutputLayer* layer, const float* d_loss_d_output_logits, const float* hidden_state_input, int model_dim, int vocab_size, LegacyLLM_Gradients* grads); // Deprecated
float* backward_output_layer_batch(const OutputLayer* layer, const float* d_loss_d_output_logits_batch, const float* hidden_state_input_batch, int batch_size, int model_dim, int vocab_size, LegacyLLM_Gradients* grads);

float* backward_feed_forward(const FeedForwardLayer* layer, const float* d_loss_d_ffn_output, const float* ffn_input, const float* hidden_pre_relu_output, int model_dim, int block_idx, LegacyLLM_Gradients* grads); // Deprecated
float* backward_feed_forward_batch(const FeedForwardLayer* layer, const float* d_loss_d_ffn_output_batch, const float* ffn_input_batch, const float* hidden_pre_relu_output_batch, int batch_size, int model_dim, int block_idx, LegacyLLM_Gradients* grads);

float* backward_layer_norm(const float* d_loss_d_norm_output, const float* input_vec, const float* norm_gamma, const float* norm_beta, float mean, float inv_std_dev, int size, int block_idx, int norm_idx, LegacyLLM_Gradients* grads); // Deprecated
float* backward_layer_norm_batch(const float* d_loss_d_norm_output_batch, const float* input_batch, const float* norm_gamma, const float* norm_beta, const float* mean_batch, const float* inv_std_dev_batch, int batch_size, int model_dim, int block_idx, int norm_idx, LegacyLLM_Gradients* grads);

float* backward_multi_head_attention(const MultiHeadAttentionLayer* layer, const float* d_loss_d_mha_output, const float* mha_input, const float* query_vec_pre_bias, const float* key_vec_pre_bias, const float* value_vec_pre_bias, const float* query_vec, const float* key_vec, const float* value_vec, const float* attention_output_vec, int model_dim, int block_idx, LegacyLLM_Gradients* grads); // Deprecated
float* backward_multi_head_attention_batch(const MultiHeadAttentionLayer* layer, const float* d_loss_d_mha_output_batch, const float* mha_input_batch, const float* query_vec_pre_bias_batch, const float* key_vec_pre_bias_batch, const float* value_vec_pre_bias_batch, const float* query_vec_batch, const float* key_vec_batch, const float* value_vec_batch, const float* attention_output_vec_batch, int batch_size, int model_dim, int block_idx, LegacyLLM_Gradients* grads);


float* backward_transformer_block(const TransformerBlock* block, const float* d_loss_d_block_output, int model_dim, int block_idx, LegacyLLM_Gradients* grads, const TransformerBlockContext* context); // Deprecated
float* backward_transformer_block_batch(const TransformerBlock* block, const float* d_loss_d_block_output_batch, int batch_size, int model_dim, int block_idx, LegacyLLM_Gradients* grads, const TransformerBlockContext* context);

void backward_embedding(const EmbeddingLayer* layer, int token_id, const float* d_loss_d_embedding_output, int model_dim, LegacyLLM_Gradients* grads); // Deprecated
void backward_embedding_batch(const EmbeddingLayer* layer, const int* input_batch, const float* d_loss_d_embedding_output_batch, int batch_size, int model_dim, LegacyLLM_Gradients* grads);
void backward_llm(LegacyLLM* model, int token_id, int true_token_id, LegacyLLM_Gradients* grads); // Deprecated: Use batched version
void backward_llm_batch(LegacyLLM* model, const int* input_batch, const int* target_batch, int batch_size, LegacyLLM_Gradients* grads);

#endif // BACKWARD_H