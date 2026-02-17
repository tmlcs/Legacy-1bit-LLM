#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, rand
#include <math.h>    // For sqrtf, expf, logf

#include "model.h"
#include "legacy_llm.h" // For struct definitions
#include "math_ops.h" // For ternary_matrix_vector_mul etc.

// Define a magic number and version for model files
#define MODEL_MAGIC 0x1B1B1 // "1-Bit LLM"
#define MODEL_VERSION 1

// --- Helper functions for saving/loading ---

// Function to write a TernaryMatrix to file
static int write_ternary_matrix(FILE* fp, const TernaryMatrix* mat) {
    size_t total_elements = (size_t)mat->rows * mat->cols;
    if (fwrite(&mat->rows, sizeof(int), 1, fp) != 1 ||
        fwrite(&mat->cols, sizeof(int), 1, fp) != 1 ||
        fwrite(mat->data, sizeof(int8_t), total_elements, fp) != total_elements) {
        return 0; // Error
    }
    return 1; // Success
}

// Function to read a TernaryMatrix from file
static int read_ternary_matrix(FILE* fp, TernaryMatrix* mat) {
    if (fread(&mat->rows, sizeof(int), 1, fp) != 1 ||
        fread(&mat->cols, sizeof(int), 1, fp) != 1) {
        return 0; // Error
    }
    size_t total_elements = (size_t)mat->rows * mat->cols;
    mat->data = (int8_t*)malloc(total_elements * sizeof(int8_t));
    if (!mat->data) {
        perror("Error allocating memory for TernaryMatrix data during load");
        return 0;
    }
    if (fread(mat->data, sizeof(int8_t), total_elements, fp) != total_elements) {
        free(mat->data);
        mat->data = NULL;
        return 0; // Error
    }
    return 1; // Success
}

// Function to write a float array to file
static int write_float_array(FILE* fp, const float* arr, int size) {
    size_t total_elements = (size_t)size;
    if (fwrite(&size, sizeof(int), 1, fp) != 1 ||
        fwrite(arr, sizeof(float), total_elements, fp) != total_elements) {
        return 0; // Error
    }
    return 1; // Success
}

// Function to read a float array from file
static int read_float_array(FILE* fp, float** arr_ptr) {
    int size;
    if (fread(&size, sizeof(int), 1, fp) != 1) {
        return 0; // Error
    }
    size_t total_elements = (size_t)size;
    *arr_ptr = (float*)malloc(total_elements * sizeof(float));
    if (!(*arr_ptr)) {
        perror("Error allocating memory for float array during load");
        return 0;
    }
    if (fread(*arr_ptr, sizeof(float), total_elements, fp) != total_elements) {
        free(*arr_ptr);
        *arr_ptr = NULL;
        return 0; // Error
    }
    return 1; // Success
}


// --- save_model implementation ---
int save_model(const LegacyLLM* model, const char* filepath) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        perror("Error opening file for saving model");
        return 0;
    }

    // Write magic number, version, and model config
    unsigned int magic = MODEL_MAGIC;
    unsigned int version = MODEL_VERSION;
    if (fwrite(&magic, sizeof(unsigned int), 1, fp) != 1 ||
        fwrite(&version, sizeof(unsigned int), 1, fp) != 1 ||
        fwrite(&model->vocab_size, sizeof(int), 1, fp) != 1 ||
        fwrite(&model->model_dim, sizeof(int), 1, fp) != 1 ||
        fwrite(&model->num_transformer_blocks, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    // Embedding Layer
    if (!write_ternary_matrix(fp, &model->embedding.embedding_weights)) { fclose(fp); return 0; }

    // Transformer Blocks
    for (int i = 0; i < model->num_transformer_blocks; ++i) {
        // Attention
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wq)) { fclose(fp); return 0; }
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wk)) { fclose(fp); return 0; }
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wv)) { fclose(fp); return 0; }
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wo)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].attention.bq, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].attention.bk, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].attention.bv, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].attention.bo, model->model_dim)) { fclose(fp); return 0; }
        
        // FFN
        int ffn_hidden_dim = model->model_dim * FFN_DIM_MULTIPLIER;
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].ffn.Wi)) { fclose(fp); return 0; }
        if (!write_ternary_matrix(fp, &model->transformer_blocks[i].ffn.Wo)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].ffn.bi, ffn_hidden_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].ffn.bo, model->model_dim)) { fclose(fp); return 0; }

        // Layer Normalization
        if (!write_float_array(fp, model->transformer_blocks[i].norm1_gamma, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].norm1_beta, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].norm2_gamma, model->model_dim)) { fclose(fp); return 0; }
        if (!write_float_array(fp, model->transformer_blocks[i].norm2_beta, model->model_dim)) { fclose(fp); return 0; }
    }

    // Output Layer
    if (!write_ternary_matrix(fp, &model->output.unembedding_weights)) { fclose(fp); return 0; }
    if (!write_float_array(fp, model->output.bias, model->vocab_size)) { fclose(fp); return 0; }

    fclose(fp);
    return 1; // Success
}

// --- load_model implementation ---
LegacyLLM* load_model(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        perror("Error opening file for loading model");
        return NULL;
    }

    unsigned int magic, version;
    int vocab_size, model_dim, num_transformer_blocks;

    if (fread(&magic, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&version, sizeof(unsigned int), 1, fp) != 1 ||
        fread(&vocab_size, sizeof(int), 1, fp) != 1 ||
        fread(&model_dim, sizeof(int), 1, fp) != 1 ||
        fread(&num_transformer_blocks, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "Error reading model metadata.\n");
        return NULL;
    }

    if (magic != MODEL_MAGIC) {
        fclose(fp);
        fprintf(stderr, "Error: Invalid model file magic number.\n");
        return NULL;
    }
    if (version != MODEL_VERSION) {
        fclose(fp);
        fprintf(stderr, "Error: Unsupported model file version.\n");
        return NULL;
    }

    // Create a new model based on loaded dimensions
    // This will allocate and initialize with random data, which we will then overwrite
    LegacyLLM* model = create_legacy_llm(vocab_size, model_dim, num_transformer_blocks);
    if (!model) {
        fclose(fp);
        return NULL;
    }

    // Now read and overwrite data for each component
    // Embedding Layer
    // Free the randomly initialized data and load directly
    free_ternary_matrix(&model->embedding.embedding_weights); // Free random data
    if (!read_ternary_matrix(fp, &model->embedding.embedding_weights)) { fclose(fp); free_legacy_llm(model); return NULL; }

    // Transformer Blocks
    for (int i = 0; i < model->num_transformer_blocks; ++i) {
        // Attention
        free_ternary_matrix(&model->transformer_blocks[i].attention.Wq);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wq)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_ternary_matrix(&model->transformer_blocks[i].attention.Wk);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wk)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_ternary_matrix(&model->transformer_blocks[i].attention.Wv);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wv)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_ternary_matrix(&model->transformer_blocks[i].attention.Wo);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].attention.Wo)) { fclose(fp); free_legacy_llm(model); return NULL; }

        free_float_array(model->transformer_blocks[i].attention.bq);
        if (!read_float_array(fp, &model->transformer_blocks[i].attention.bq)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].attention.bk);
        if (!read_float_array(fp, &model->transformer_blocks[i].attention.bk)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].attention.bv);
        if (!read_float_array(fp, &model->transformer_blocks[i].attention.bv)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].attention.bo);
        if (!read_float_array(fp, &model->transformer_blocks[i].attention.bo)) { fclose(fp); free_legacy_llm(model); return NULL; }
        
        // FFN
        free_ternary_matrix(&model->transformer_blocks[i].ffn.Wi);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].ffn.Wi)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_ternary_matrix(&model->transformer_blocks[i].ffn.Wo);
        if (!read_ternary_matrix(fp, &model->transformer_blocks[i].ffn.Wo)) { fclose(fp); free_legacy_llm(model); return NULL; }

        free_float_array(model->transformer_blocks[i].ffn.bi);
        if (!read_float_array(fp, &model->transformer_blocks[i].ffn.bi)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].ffn.bo);
        if (!read_float_array(fp, &model->transformer_blocks[i].ffn.bo)) { fclose(fp); free_legacy_llm(model); return NULL; }

        // Layer Normalization
        free_float_array(model->transformer_blocks[i].norm1_gamma);
        if (!read_float_array(fp, &model->transformer_blocks[i].norm1_gamma)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].norm1_beta);
        if (!read_float_array(fp, &model->transformer_blocks[i].norm1_beta)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].norm2_gamma);
        if (!read_float_array(fp, &model->transformer_blocks[i].norm2_gamma)) { fclose(fp); free_legacy_llm(model); return NULL; }
        free_float_array(model->transformer_blocks[i].norm2_beta);
        if (!read_float_array(fp, &model->transformer_blocks[i].norm2_beta)) { fclose(fp); free_legacy_llm(model); return NULL; }
    }

    // Output Layer
    free_ternary_matrix(&model->output.unembedding_weights);
    if (!read_ternary_matrix(fp, &model->output.unembedding_weights)) { fclose(fp); free_legacy_llm(model); return NULL; }
    free_float_array(model->output.bias);
    if (!read_float_array(fp, &model->output.bias)) { fclose(fp); free_legacy_llm(model); return NULL; }

    fclose(fp);
    printf("Model loaded from %s.\n", filepath);
    return model;
}

// --- Model Operations Functions (Allocation and Initialization) ---

void initialize_ternary_data(int8_t* data, int size) {
    for (int i = 0; i < size; ++i) {
        int r = rand() % 3; // 0, 1, 2
        data[i] = (int8_t)(r - 1); // Maps to -1, 0, 1
    }
}

TernaryMatrix create_ternary_matrix(int rows, int cols) {
    TernaryMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (int8_t*)calloc(rows * cols, sizeof(int8_t));
    if (mat.data == NULL) {
        perror("Error allocating memory for TernaryMatrix data");
        mat.rows = 0;
        mat.cols = 0;
        return mat;
    }
    initialize_ternary_data(mat.data, rows * cols);
    return mat;
}

void free_ternary_matrix(TernaryMatrix* mat) {
    if (mat && mat->data) {
        free(mat->data);
        mat->data = NULL;
    }
}

float* create_float_array(int size) {
    float* arr = (float*)calloc(size, sizeof(float));
    if (arr == NULL) {
        perror("Error allocating memory for float array");
        return NULL;
    }
    for (int i = 0; i < size; ++i) {
        arr[i] = (float)rand() / (float)RAND_MAX * 0.02f - 0.01f; // Small random init around 0
    }
    return arr;
}

void free_float_array(float* arr) {
    if (arr) {
        free(arr);
    }
}

LegacyLLM* create_legacy_llm(int vocab_size, int model_dim, int num_transformer_blocks) {
    LegacyLLM* model = (LegacyLLM*)calloc(1, sizeof(LegacyLLM));
    if (model == NULL) {
        perror("Error allocating memory for LegacyLLM model");
        return NULL;
    }

    model->vocab_size = vocab_size;
    model->model_dim = model_dim;
    model->num_transformer_blocks = num_transformer_blocks;

    model->embedding.embedding_weights = create_ternary_matrix(vocab_size, model_dim);
    if (!model->embedding.embedding_weights.data) { free_legacy_llm(model); return NULL; }

    model->transformer_blocks = (TransformerBlock*)calloc(num_transformer_blocks, sizeof(TransformerBlock));
    if (model->transformer_blocks == NULL) {
        perror("Error allocating memory for transformer blocks");
        free_legacy_llm(model);
        return NULL;
    }

    // Allocate contexts array
    model->block_contexts = (TransformerBlockContext**)calloc(num_transformer_blocks, sizeof(TransformerBlockContext*));
    if (!model->block_contexts) {
        perror("Error allocating memory for block_contexts");
        free_legacy_llm(model);
        return NULL;
    }
    int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;
    for (int i = 0; i < num_transformer_blocks; ++i) {
        model->block_contexts[i] = create_transformer_block_context(model_dim, ffn_hidden_dim);
        if (!model->block_contexts[i]) {
            fprintf(stderr, "Error: Failed to create TransformerBlockContext for block %d\n", i);
            free_legacy_llm(model);
            return NULL;
        }
    }


    for (int i = 0; i < num_transformer_blocks; ++i) {
        // Multi-Head Attention Layer
        model->transformer_blocks[i].attention.Wq = create_ternary_matrix(model_dim, model_dim);
        if (!model->transformer_blocks[i].attention.Wq.data) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.Wk = create_ternary_matrix(model_dim, model_dim);
        if (!model->transformer_blocks[i].attention.Wk.data) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.Wv = create_ternary_matrix(model_dim, model_dim);
        if (!model->transformer_blocks[i].attention.Wv.data) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.Wo = create_ternary_matrix(model_dim, model_dim);
        if (!model->transformer_blocks[i].attention.Wo.data) { free_legacy_llm(model); return NULL; }

        model->transformer_blocks[i].attention.bq = create_float_array(model_dim);
        if (!model->transformer_blocks[i].attention.bq) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.bk = create_float_array(model_dim);
        if (!model->transformer_blocks[i].attention.bk) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.bv = create_float_array(model_dim);
        if (!model->transformer_blocks[i].attention.bv) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].attention.bo = create_float_array(model_dim);
        if (!model->transformer_blocks[i].attention.bo) { free_legacy_llm(model); return NULL; }
        
        // Feed-Forward Layer
        model->transformer_blocks[i].ffn.Wi = create_ternary_matrix(ffn_hidden_dim, model_dim);
        if (!model->transformer_blocks[i].ffn.Wi.data) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].ffn.Wo = create_ternary_matrix(model_dim, ffn_hidden_dim);
        if (!model->transformer_blocks[i].ffn.Wo.data) { free_legacy_llm(model); return NULL; }

        model->transformer_blocks[i].ffn.bi = create_float_array(ffn_hidden_dim);
        if (!model->transformer_blocks[i].ffn.bi) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].ffn.bo = create_float_array(model_dim);
        if (!model->transformer_blocks[i].ffn.bo) { free_legacy_llm(model); return NULL; }

        // Layer Normalization
        model->transformer_blocks[i].norm1_gamma = create_float_array(model_dim);
        if (!model->transformer_blocks[i].norm1_gamma) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].norm1_beta = create_float_array(model_dim);
        if (!model->transformer_blocks[i].norm1_beta) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].norm2_gamma = create_float_array(model_dim);
        if (!model->transformer_blocks[i].norm2_gamma) { free_legacy_llm(model); return NULL; }
        model->transformer_blocks[i].norm2_beta = create_float_array(model_dim);
        if (!model->transformer_blocks[i].norm2_beta) { free_legacy_llm(model); return NULL; }
    }

    // Output Layer
    model->output.unembedding_weights = create_ternary_matrix(model_dim, vocab_size);
    if (!model->output.unembedding_weights.data) { free_legacy_llm(model); return NULL; }
    model->output.bias = create_float_array(vocab_size);
    if (!model->output.bias) { free_legacy_llm(model); return NULL; }

    // Allocate final_hidden_state_input_batch
    model->final_hidden_state_input_batch = create_float_array(BATCH_SIZE * model_dim);
    if (!model->final_hidden_state_input_batch) { free_legacy_llm(model); return NULL; }


    printf("LegacyLLM model created with %d transformer blocks, vocab_size=%d, model_dim=%d.\n",
           num_transformer_blocks, vocab_size, model_dim);
    return model;
}

void free_legacy_llm(LegacyLLM* model) {
    if (model == NULL) return;

    // Free Embedding Layer
    free_ternary_matrix(&model->embedding.embedding_weights);

    // Free Transformer Blocks
    if (model->transformer_blocks) {
        for (int i = 0; i < model->num_transformer_blocks; ++i) {
            // Attention
            free_ternary_matrix(&model->transformer_blocks[i].attention.Wq);
            free_ternary_matrix(&model->transformer_blocks[i].attention.Wk);
            free_ternary_matrix(&model->transformer_blocks[i].attention.Wv);
            free_ternary_matrix(&model->transformer_blocks[i].attention.Wo);
            free_float_array(model->transformer_blocks[i].attention.bq);
            free_float_array(model->transformer_blocks[i].attention.bk);
            free_float_array(model->transformer_blocks[i].attention.bv);
            free_float_array(model->transformer_blocks[i].attention.bo);

            // FFN
            free_ternary_matrix(&model->transformer_blocks[i].ffn.Wi);
            free_ternary_matrix(&model->transformer_blocks[i].ffn.Wo);
            free_float_array(model->transformer_blocks[i].ffn.bi);
            free_float_array(model->transformer_blocks[i].ffn.bo);

            // Layer Normalization
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
            free_transformer_block_context(model->block_contexts[i]);
        }
        free(model->block_contexts);
    }

    // Free Output Layer
    free_ternary_matrix(&model->output.unembedding_weights);
    free_float_array(model->output.bias);

    // Free final_hidden_state_input_batch
    free_float_array(model->final_hidden_state_input_batch);

    free(model);
    printf("LegacyLLM model freed.\n");
}

// --- Gradient Structures Allocation/Deallocation ---

LegacyLLM_Gradients* create_legacy_llm_gradients(int vocab_size, int model_dim, int num_transformer_blocks) {
    LegacyLLM_Gradients* grads = (LegacyLLM_Gradients*)calloc(1, sizeof(LegacyLLM_Gradients));
    if (!grads) {
        perror("Error allocating memory for LegacyLLM_Gradients");
        return NULL;
    }
    grads->vocab_size = vocab_size;
    grads->model_dim = model_dim;
    grads->num_transformer_blocks = num_transformer_blocks;

    grads->embedding_grads.embedding_weights = create_float_array(vocab_size * model_dim);
    if (!grads->embedding_grads.embedding_weights) { free_legacy_llm_gradients(grads); return NULL; }

    grads->transformer_block_grads = (TransformerBlockGradients*)calloc(num_transformer_blocks, sizeof(TransformerBlockGradients));
    if (!grads->transformer_block_grads) { free_legacy_llm_gradients(grads); return NULL; }

    for (int i = 0; i < num_transformer_blocks; ++i) {
        grads->transformer_block_grads[i].attention_grads.Wq = create_float_array(model_dim * model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.Wq) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.Wk = create_float_array(model_dim * model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.Wk) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.Wv = create_float_array(model_dim * model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.Wv) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.Wo = create_float_array(model_dim * model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.Wo) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.bq = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.bq) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.bk = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.bk) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.bv = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.bv) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].attention_grads.bo = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].attention_grads.bo) { free_legacy_llm_gradients(grads); return NULL; }

        int ffn_hidden_dim = model_dim * FFN_DIM_MULTIPLIER;
        grads->transformer_block_grads[i].ffn_grads.Wi = create_float_array(ffn_hidden_dim * model_dim);
        if (!grads->transformer_block_grads[i].ffn_grads.Wi) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].ffn_grads.Wo = create_float_array(model_dim * ffn_hidden_dim);
        if (!grads->transformer_block_grads[i].ffn_grads.Wo) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].ffn_grads.bi = create_float_array(ffn_hidden_dim);
        if (!grads->transformer_block_grads[i].ffn_grads.bi) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].ffn_grads.bo = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].ffn_grads.bo) { free_legacy_llm_gradients(grads); return NULL; }

        grads->transformer_block_grads[i].norm1_gamma = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].norm1_gamma) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].norm1_beta = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].norm1_beta) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].norm2_gamma = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].norm2_gamma) { free_legacy_llm_gradients(grads); return NULL; }
        grads->transformer_block_grads[i].norm2_beta = create_float_array(model_dim);
        if (!grads->transformer_block_grads[i].norm2_beta) { free_legacy_llm_gradients(grads); return NULL; }
    }

    grads->output_grads.unembedding_weights = create_float_array(model_dim * vocab_size);
    if (!grads->output_grads.unembedding_weights) { free_legacy_llm_gradients(grads); return NULL; }
    grads->output_grads.bias = create_float_array(vocab_size);
    if (!grads->output_grads.bias) { free_legacy_llm_gradients(grads); return NULL; }

    return grads;
}

void free_legacy_llm_gradients(LegacyLLM_Gradients* grads) {
    if (!grads) return;

    free_float_array(grads->embedding_grads.embedding_weights);

    if (grads->transformer_block_grads) {
        for (int i = 0; i < grads->num_transformer_blocks; ++i) {
            free_float_array(grads->transformer_block_grads[i].attention_grads.Wq);
            free_float_array(grads->transformer_block_grads[i].attention_grads.Wk);
            free_float_array(grads->transformer_block_grads[i].attention_grads.Wv);
            free_float_array(grads->transformer_block_grads[i].attention_grads.Wo);
            free_float_array(grads->transformer_block_grads[i].attention_grads.bq);
            free_float_array(grads->transformer_block_grads[i].attention_grads.bk);
            free_float_array(grads->transformer_block_grads[i].attention_grads.bv);
            free_float_array(grads->transformer_block_grads[i].attention_grads.bo);

            free_float_array(grads->transformer_block_grads[i].ffn_grads.Wi);
            free_float_array(grads->transformer_block_grads[i].ffn_grads.Wo);
            free_float_array(grads->transformer_block_grads[i].ffn_grads.bi);
            free_float_array(grads->transformer_block_grads[i].ffn_grads.bo);

            free_float_array(grads->transformer_block_grads[i].norm1_gamma);
            free_float_array(grads->transformer_block_grads[i].norm1_beta);
            free_float_array(grads->transformer_block_grads[i].norm2_gamma);
            free_float_array(grads->transformer_block_grads[i].norm2_beta);
        }
        free(grads->transformer_block_grads);
    }

    free_float_array(grads->output_grads.unembedding_weights);
    free_float_array(grads->output_grads.bias);

    free(grads);
}

void zero_float_array(float* arr, int size) {
    if (arr) {
        for (int i = 0; i < size; ++i) {
            arr[i] = 0.0f;
        }
    }
}

void zero_legacy_llm_gradients(LegacyLLM_Gradients* grads) {
    if (!grads) return;

    // Zero Embedding Gradients
    zero_float_array(grads->embedding_grads.embedding_weights, grads->vocab_size * grads->model_dim);

    // Zero Transformer Block Gradients
    if (grads->transformer_block_grads) {
        for (int i = 0; i < grads->num_transformer_blocks; ++i) {
            // Attention Gradients
            zero_float_array(grads->transformer_block_grads[i].attention_grads.Wq, grads->model_dim * grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.Wk, grads->model_dim * grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.Wv, grads->model_dim * grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.Wo, grads->model_dim * grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.bq, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.bk, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.bv, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].attention_grads.bo, grads->model_dim);

            // FFN Gradients
            int ffn_hidden_dim = grads->model_dim * FFN_DIM_MULTIPLIER;
            zero_float_array(grads->transformer_block_grads[i].ffn_grads.Wi, ffn_hidden_dim * grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].ffn_grads.Wo, grads->model_dim * ffn_hidden_dim);
            zero_float_array(grads->transformer_block_grads[i].ffn_grads.bi, ffn_hidden_dim);
            zero_float_array(grads->transformer_block_grads[i].ffn_grads.bo, grads->model_dim);

            // Layer Normalization Gradients
            zero_float_array(grads->transformer_block_grads[i].norm1_gamma, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].norm1_beta, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].norm2_gamma, grads->model_dim);
            zero_float_array(grads->transformer_block_grads[i].norm2_beta, grads->model_dim);
        }
    }

    // Zero Output Layer Gradients
    zero_float_array(grads->output_grads.unembedding_weights, grads->model_dim * grads->vocab_size);
    zero_float_array(grads->output_grads.bias, grads->vocab_size);
}

void apply_ternary_weight_updates(LegacyLLM* model, LegacyLLM_Gradients* grads, float learning_rate) {
    if (!model || !grads) return;

    // Apply updates to Embedding Layer
    // Ternary weights update
    for (int i = 0; i < model->embedding.embedding_weights.rows * model->embedding.embedding_weights.cols; ++i) {
        if (grads->embedding_grads.embedding_weights[i] > 0) {
            model->embedding.embedding_weights.data[i] = 1;
        } else if (grads->embedding_grads.embedding_weights[i] < 0) {
            model->embedding.embedding_weights.data[i] = -1;
        }
        // If gradient is 0, weight remains unchanged, which is default for ternary SGD.
    }

    // Apply updates to Transformer Blocks
    for (int i = 0; i < model->num_transformer_blocks; ++i) {
        // Multi-Head Attention Layer
        // Ternary weights update
        for (int j = 0; j < model->model_dim * model->model_dim; ++j) {
            if (grads->transformer_block_grads[i].attention_grads.Wq[j] > 0) {
                model->transformer_blocks[i].attention.Wq.data[j] = 1;
            } else if (grads->transformer_block_grads[i].attention_grads.Wq[j] < 0) {
                model->transformer_blocks[i].attention.Wq.data[j] = -1;
            }

            if (grads->transformer_block_grads[i].attention_grads.Wk[j] > 0) {
                model->transformer_blocks[i].attention.Wk.data[j] = 1;
            } else if (grads->transformer_block_grads[i].attention_grads.Wk[j] < 0) {
                model->transformer_blocks[i].attention.Wk.data[j] = -1;
            }

            if (grads->transformer_block_grads[i].attention_grads.Wv[j] > 0) {
                model->transformer_blocks[i].attention.Wv.data[j] = 1;
            } else if (grads->transformer_block_grads[i].attention_grads.Wv[j] < 0) {
                model->transformer_blocks[i].attention.Wv.data[j] = -1;
            }

            if (grads->transformer_block_grads[i].attention_grads.Wo[j] > 0) {
                model->transformer_blocks[i].attention.Wo.data[j] = 1;
            } else if (grads->transformer_block_grads[i].attention_grads.Wo[j] < 0) {
                model->transformer_blocks[i].attention.Wo.data[j] = -1;
            }
        }

        // Float biases update (Standard SGD)
        for (int j = 0; j < model->model_dim; ++j) {
            model->transformer_blocks[i].attention.bq[j] -= learning_rate * grads->transformer_block_grads[i].attention_grads.bq[j];
            model->transformer_blocks[i].attention.bk[j] -= learning_rate * grads->transformer_block_grads[i].attention_grads.bk[j];
            model->transformer_blocks[i].attention.bv[j] -= learning_rate * grads->transformer_block_grads[i].attention_grads.bv[j];
            model->transformer_blocks[i].attention.bo[j] -= learning_rate * grads->transformer_block_grads[i].attention_grads.bo[j];
        }

        // Feed-Forward Layer
        int ffn_hidden_dim = model->model_dim * FFN_DIM_MULTIPLIER;
        // Ternary weights update
        for (int j = 0; j < ffn_hidden_dim * model->model_dim; ++j) {
            if (grads->transformer_block_grads[i].ffn_grads.Wi[j] > 0) {
                model->transformer_blocks[i].ffn.Wi.data[j] = 1;
            } else if (grads->transformer_block_grads[i].ffn_grads.Wi[j] < 0) {
                model->transformer_blocks[i].ffn.Wi.data[j] = -1;
            }
        }
        for (int j = 0; j < model->model_dim * ffn_hidden_dim; ++j) {
            if (grads->transformer_block_grads[i].ffn_grads.Wo[j] > 0) {
                model->transformer_blocks[i].ffn.Wo.data[j] = 1;
            } else if (grads->transformer_block_grads[i].ffn_grads.Wo[j] < 0) {
                model->transformer_blocks[i].ffn.Wo.data[j] = -1;
            }
        }

        // Float biases update (Standard SGD)
        for (int j = 0; j < ffn_hidden_dim; ++j) {
            model->transformer_blocks[i].ffn.bi[j] -= learning_rate * grads->transformer_block_grads[i].ffn_grads.bi[j];
        }
        for (int j = 0; j < model->model_dim; ++j) {
            model->transformer_blocks[i].ffn.bo[j] -= learning_rate * grads->transformer_block_grads[i].ffn_grads.bo[j];
        }

        // Layer Normalization Parameters update (Standard SGD)
        for (int j = 0; j < model->model_dim; ++j) {
            model->transformer_blocks[i].norm1_gamma[j] -= learning_rate * grads->transformer_block_grads[i].norm1_gamma[j];
            model->transformer_blocks[i].norm1_beta[j] -= learning_rate * grads->transformer_block_grads[i].norm1_beta[j];
            model->transformer_blocks[i].norm2_gamma[j] -= learning_rate * grads->transformer_block_grads[i].norm2_gamma[j];
            model->transformer_blocks[i].norm2_beta[j] -= learning_rate * grads->transformer_block_grads[i].norm2_beta[j];
        }
    }

    // Apply updates to Output Layer
    // Ternary weights update
    for (int i = 0; i < model->output.unembedding_weights.rows * model->output.unembedding_weights.cols; ++i) {
        if (grads->output_grads.unembedding_weights[i] > 0) {
            model->output.unembedding_weights.data[i] = 1;
        } else if (grads->output_grads.unembedding_weights[i] < 0) {
            model->output.unembedding_weights.data[i] = -1;
        }
    }
    // Float biases update (Standard SGD)
    for (int i = 0; i < model->vocab_size; ++i) {
        model->output.bias[i] -= learning_rate * grads->output_grads.bias[i];
    }
}



// --- Context Management ---
TransformerBlockContext* create_transformer_block_context(int model_dim, int ffn_hidden_dim) {
    (void)ffn_hidden_dim; // ffn_hidden_dim might not be directly used for context, but keeping for compatibility
    TransformerBlockContext* context = (TransformerBlockContext*)calloc(1, sizeof(TransformerBlockContext));
    if (!context) {
        perror("Error allocating memory for TransformerBlockContext");
        return NULL;
    }

    context->block_input_batch = create_float_array(BATCH_SIZE * model_dim);
    if (!context->block_input_batch) {
        free(context);
        return NULL;
    }

    context->ln1_mean_batch = create_float_array(BATCH_SIZE);
    if (!context->ln1_mean_batch) { free_float_array(context->block_input_batch); free(context); return NULL; }
    context->ln1_inv_std_dev_batch = create_float_array(BATCH_SIZE);
    if (!context->ln1_inv_std_dev_batch) { free_float_array(context->ln1_mean_batch); free_float_array(context->block_input_batch); free(context); return NULL; }
    
    context->ln2_mean_batch = create_float_array(BATCH_SIZE);
    if (!context->ln2_mean_batch) { free_float_array(context->ln1_inv_std_dev_batch); free_float_array(context->ln1_mean_batch); free_float_array(context->block_input_batch); free(context); return NULL; }
    context->ln2_inv_std_dev_batch = create_float_array(BATCH_SIZE);
    if (!context->ln2_inv_std_dev_batch) { free_float_array(context->ln2_mean_batch); free_float_array(context->ln1_inv_std_dev_batch); free_float_array(context->ln1_mean_batch); free_float_array(context->block_input_batch); free(context); return NULL; }


    return context;
}

void free_transformer_block_context(TransformerBlockContext* context) {
    if (!context) return;

    free_float_array(context->block_input_batch);
    free_float_array(context->ln1_mean_batch);
    free_float_array(context->ln1_inv_std_dev_batch);
    free_float_array(context->ln2_mean_batch);
    free_float_array(context->ln2_inv_std_dev_batch);

    free(context);
}
