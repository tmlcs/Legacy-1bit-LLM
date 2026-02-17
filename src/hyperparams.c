#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hyperparams.h"

// Initialize with default hyperparameters
void hyperparams_init_defaults(Hyperparameters* params) {
    if (!params) return;
    
    // Learning rate
    params->learning_rate = 0.01f;
    params->min_learning_rate = 0.0001f;
    params->max_learning_rate = 0.1f;
    params->learning_rate_decay = 0.95f;
    
    // Model architecture
    params->model_dim = 256;
    params->num_transformer_blocks = 2;
    params->num_heads = 4;
    params->ffn_dim_multiplier = 4;
    
    // Training
    params->num_epochs = 10;
    params->batch_size = 8;
    params->max_sequence_length = 128;
    params->warmup_steps = 100;
    params->gradient_clip_value = 1;
    
    // Optimization
    params->use_adam = 0;  // SGD by default
    params->beta1 = 0.9f;
    params->beta2 = 0.999f;
    params->epsilon = 1e-8f;
    params->weight_decay = 0.0001f;
    
    // Quantization
    params->quantization_scheme = 0;  // Ternary (-1, 0, 1)
    params->straight_through_estimator = 1.0f;
    
    // Data
    params->chunk_size = 8192;
    params->shuffle_data = 1;
    params->train_split = 0.9f;
    
    // Logging
    params->log_interval = 10;
    params->save_interval = 2;
    params->validate_interval = 1;
}

// Load from config file (simple key=value format)
int hyperparams_load_from_file(Hyperparameters* params, const char* filepath) {
    if (!params || !filepath) return 0;
    
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Warning: Could not open config file '%s'. Using defaults.\n", filepath);
        hyperparams_init_defaults(params);
        return 1; // Not a critical error, using defaults
    }
    
    // Initialize with defaults first
    hyperparams_init_defaults(params);
    
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') continue;
        
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Parse key=value
        char key[64];
        char value[192];
        if (sscanf(line, "%63[^=]=%191s", key, value) == 2) {
            // Trim whitespace from key
            char* k = key;
            while (*k == ' ' || *k == '\t') k++;
            
            // Parse values
            if (strcmp(k, "learning_rate") == 0) {
                params->learning_rate = atof(value);
            } else if (strcmp(k, "model_dim") == 0) {
                params->model_dim = atoi(value);
            } else if (strcmp(k, "num_transformer_blocks") == 0) {
                params->num_transformer_blocks = atoi(value);
            } else if (strcmp(k, "num_epochs") == 0) {
                params->num_epochs = atoi(value);
            } else if (strcmp(k, "batch_size") == 0) {
                params->batch_size = atoi(value);
            } else if (strcmp(k, "quantization_scheme") == 0) {
                params->quantization_scheme = atoi(value);
            } else if (strcmp(k, "use_adam") == 0) {
                params->use_adam = atoi(value);
            }
            // Add more parameters as needed
        }
    }
    
    fclose(fp);
    printf("[Hyperparams] Loaded configuration from '%s'\n", filepath);
    return 1;
}

// Save hyperparameters to file
int hyperparams_save_to_file(const Hyperparameters* params, const char* filepath) {
    if (!params || !filepath) return 0;
    
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not create config file '%s'\n", filepath);
        return 0;
    }
    
    fprintf(fp, "# Legacy-1bit LLM Hyperparameters\n");
    fprintf(fp, "# Auto-generated configuration\n\n");
    
    fprintf(fp, "# Learning Rate\n");
    fprintf(fp, "learning_rate=%.6f\n", params->learning_rate);
    fprintf(fp, "learning_rate_decay=%.4f\n", params->learning_rate_decay);
    fprintf(fp, "min_learning_rate=%.6f\n\n", params->min_learning_rate);
    
    fprintf(fp, "# Model Architecture\n");
    fprintf(fp, "model_dim=%d\n", params->model_dim);
    fprintf(fp, "num_transformer_blocks=%d\n", params->num_transformer_blocks);
    fprintf(fp, "num_heads=%d\n", params->num_heads);
    fprintf(fp, "ffn_dim_multiplier=%d\n\n", params->ffn_dim_multiplier);
    
    fprintf(fp, "# Training\n");
    fprintf(fp, "num_epochs=%d\n", params->num_epochs);
    fprintf(fp, "batch_size=%d\n", params->batch_size);
    fprintf(fp, "max_sequence_length=%d\n\n", params->max_sequence_length);
    
    fprintf(fp, "# Quantization\n");
    fprintf(fp, "quantization_scheme=%d\n", params->quantization_scheme);
    fprintf(fp, "# 0=Ternary(-1,0,1), 1=Binary(-1,1), 2=2-bit(-2,-1,1,2)\n\n");
    
    fprintf(fp, "# Optimization\n");
    fprintf(fp, "use_adam=%d\n", params->use_adam);
    fprintf(fp, "weight_decay=%.6f\n", params->weight_decay);
    
    fclose(fp);
    printf("[Hyperparams] Saved configuration to '%s'\n", filepath);
    return 1;
}

// Print hyperparameters
void hyperparams_print(const Hyperparameters* params) {
    if (!params) return;
    
    printf("\n=== Hyperparameters ===\n");
    printf("Learning Rate: %.6f (decay: %.4f)\n", 
           params->learning_rate, params->learning_rate_decay);
    printf("Model: %d dim, %d blocks, %d heads\n",
           params->model_dim, params->num_transformer_blocks, params->num_heads);
    printf("Training: %d epochs, batch size %d\n",
           params->num_epochs, params->batch_size);
    printf("Quantization: %s\n",
           params->quantization_scheme == 0 ? "Ternary" :
           params->quantization_scheme == 1 ? "Binary" : "2-bit");
    printf("=======================\n\n");
}

// Grid search implementation
GridSearchConfig* grid_search_create(const float* learning_rates, int n_lr,
                                     const int* model_dims, int n_dim,
                                     const int* num_blocks, int n_blocks,
                                     const int* batch_sizes, int n_batch) {
    GridSearchConfig* config = (GridSearchConfig*)malloc(sizeof(GridSearchConfig));
    if (!config) return NULL;
    
    // Copy learning rates
    config->num_learning_rates = n_lr;
    config->learning_rates = (float*)malloc(n_lr * sizeof(float));
    memcpy(config->learning_rates, learning_rates, n_lr * sizeof(float));
    
    // Copy model dimensions
    config->num_model_dims = n_dim;
    config->model_dims = (int*)malloc(n_dim * sizeof(int));
    memcpy(config->model_dims, model_dims, n_dim * sizeof(int));
    
    // Copy number of blocks
    config->num_num_blocks = n_blocks;
    config->num_blocks = (int*)malloc(n_blocks * sizeof(int));
    memcpy(config->num_blocks, num_blocks, n_blocks * sizeof(int));
    
    // Copy batch sizes
    config->num_batch_sizes = n_batch;
    config->batch_sizes = (int*)malloc(n_batch * sizeof(int));
    memcpy(config->batch_sizes, batch_sizes, n_batch * sizeof(int));
    
    config->current_config = 0;
    config->total_configs = n_lr * n_dim * n_blocks * n_batch;
    
    return config;
}

void grid_search_destroy(GridSearchConfig* config) {
    if (!config) return;
    
    if (config->learning_rates) free(config->learning_rates);
    if (config->model_dims) free(config->model_dims);
    if (config->num_blocks) free(config->num_blocks);
    if (config->batch_sizes) free(config->batch_sizes);
    
    free(config);
}

int grid_search_next_config(GridSearchConfig* config, Hyperparameters* params) {
    if (!config || !params || config->current_config >= config->total_configs) {
        return 0; // No more configurations
    }
    
    // Calculate indices for each parameter
    int idx = config->current_config;
    
    int i_batch = idx % config->num_batch_sizes;
    idx /= config->num_batch_sizes;
    
    int i_blocks = idx % config->num_num_blocks;
    idx /= config->num_num_blocks;
    
    int i_dim = idx % config->num_model_dims;
    idx /= config->num_model_dims;
    
    int i_lr = idx % config->num_learning_rates;
    
    // Set parameters
    hyperparams_init_defaults(params);
    params->learning_rate = config->learning_rates[i_lr];
    params->model_dim = config->model_dims[i_dim];
    params->num_transformer_blocks = config->num_blocks[i_blocks];
    params->batch_size = config->batch_sizes[i_batch];
    
    config->current_config++;
    return 1;
}

void grid_search_reset(GridSearchConfig* config) {
    if (config) {
        config->current_config = 0;
    }
}

// Learning rate schedulers
float lr_scheduler_step(int epoch, float initial_lr, float decay_factor, int decay_epochs) {
    int num_decays = epoch / decay_epochs;
    float lr = initial_lr;
    for (int i = 0; i < num_decays; i++) {
        lr *= decay_factor;
    }
    return lr;
}

float lr_scheduler_cosine(int step, int total_steps, float initial_lr, float min_lr) {
    if (step >= total_steps) return min_lr;
    float progress = (float)step / (float)total_steps;
    return min_lr + (initial_lr - min_lr) * 0.5f * (1.0f + cosf(3.14159f * progress));
}

float lr_scheduler_warmup(int step, int warmup_steps, float initial_lr) {
    if (step >= warmup_steps) return initial_lr;
    return initial_lr * ((float)step / (float)warmup_steps);
}
