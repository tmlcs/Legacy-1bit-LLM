#ifndef HYPERPARAMS_H
#define HYPERPARAMS_H

// Hyperparameter configuration structure
typedef struct {
    // Learning rate
    float learning_rate;
    float min_learning_rate;
    float max_learning_rate;
    float learning_rate_decay;
    
    // Model architecture
    int model_dim;
    int num_transformer_blocks;
    int num_heads;
    int ffn_dim_multiplier;
    
    // Training
    int num_epochs;
    int batch_size;
    int max_sequence_length;
    int warmup_steps;
    int gradient_clip_value;
    
    // Optimization
    int use_adam;  // 0 for SGD, 1 for Adam-like
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    
    // Quantization
    int quantization_scheme;  // 0: Ternary (-1,0,1), 1: Binary (-1,1), 2: 2-bit (-2,-1,1,2)
    float straight_through_estimator;  // STE coefficient
    
    // Data
    int chunk_size;
    int shuffle_data;
    float train_split;  // Fraction for training (rest for validation)
    
    // Logging
    int log_interval;
    int save_interval;
    int validate_interval;
} Hyperparameters;

// Grid search configuration
typedef struct {
    float* learning_rates;
    int num_learning_rates;
    
    int* model_dims;
    int num_model_dims;
    
    int* num_blocks;
    int num_num_blocks;
    
    int* batch_sizes;
    int num_batch_sizes;
    
    int current_config;
    int total_configs;
} GridSearchConfig;

// Initialize with default hyperparameters
void hyperparams_init_defaults(Hyperparameters* params);

// Initialize from config file
int hyperparams_load_from_file(Hyperparameters* params, const char* filepath);

// Save hyperparameters to file
int hyperparams_save_to_file(const Hyperparameters* params, const char* filepath);

// Print hyperparameters
void hyperparams_print(const Hyperparameters* params);

// Grid search functions
GridSearchConfig* grid_search_create(const float* learning_rates, int n_lr,
                                     const int* model_dims, int n_dim,
                                     const int* num_blocks, int n_blocks,
                                     const int* batch_sizes, int n_batch);
void grid_search_destroy(GridSearchConfig* config);
int grid_search_next_config(GridSearchConfig* config, Hyperparameters* params);
void grid_search_reset(GridSearchConfig* config);

// Learning rate schedulers
float lr_scheduler_step(int epoch, float initial_lr, float decay_factor, int decay_epochs);
float lr_scheduler_cosine(int step, int total_steps, float initial_lr, float min_lr);
float lr_scheduler_warmup(int step, int warmup_steps, float initial_lr);

#endif // HYPERPARAMS_H
