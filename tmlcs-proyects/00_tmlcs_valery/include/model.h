#ifndef MODEL_H
#define MODEL_H

#include "legacy_llm.h" // For global defines and forward declarations

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