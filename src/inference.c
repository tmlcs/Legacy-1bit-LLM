#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "legacy_llm.h"
#include "data_utils.h"
#include "model.h"
#include "forward.h"

#define DEFAULT_MAX_TOKENS 100
#define DEFAULT_TEMPERATURE 1.0f
#define DEFAULT_TOP_K 0  // 0 means disabled
#define MODEL_FILE "llm_model.bin"

typedef enum {
    SAMPLING_GREEDY,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K
} SamplingStrategy;

typedef struct {
    SamplingStrategy strategy;
    float temperature;
    int top_k;
    int max_tokens;
    int seed;
} GenerationConfig;

// Sample next token using specified strategy
static int sample_token(float* probs, int vocab_size, const GenerationConfig* config) {
    if (config->strategy == SAMPLING_GREEDY) {
        // Greedy: pick the token with highest probability
        int best_token = 0;
        float best_prob = probs[0];
        for (int i = 1; i < vocab_size; i++) {
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best_token = i;
            }
        }
        return best_token;
    }
    else if (config->strategy == SAMPLING_TEMPERATURE) {
        // Temperature sampling: apply temperature and sample
        float scaled_probs[256]; // Assuming vocab_size <= 256
        float sum = 0.0f;
        
        for (int i = 0; i < vocab_size; i++) {
            // Apply temperature: p_i^(1/T) / Z
            if (config->temperature != 1.0f) {
                scaled_probs[i] = powf(probs[i], 1.0f / config->temperature);
            } else {
                scaled_probs[i] = probs[i];
            }
            sum += scaled_probs[i];
        }
        
        // Normalize
        for (int i = 0; i < vocab_size; i++) {
            scaled_probs[i] /= sum;
        }
        
        // Sample from distribution
        float r = (float)rand() / (float)RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += scaled_probs[i];
            if (r <= cumsum) {
                return i;
            }
        }
        return vocab_size - 1;
    }
    else if (config->strategy == SAMPLING_TOP_K) {
        // Top-k sampling: sample only from top k tokens
        typedef struct {
            int token;
            float prob;
        } TokenProb;
        
        TokenProb tokens[256];
        for (int i = 0; i < vocab_size; i++) {
            tokens[i].token = i;
            tokens[i].prob = probs[i];
        }
        
        // Sort by probability (simple bubble sort, vocab_size is small)
        int k = config->top_k < vocab_size ? config->top_k : vocab_size;
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < vocab_size; j++) {
                if (tokens[j].prob > tokens[i].prob) {
                    TokenProb temp = tokens[i];
                    tokens[i] = tokens[j];
                    tokens[j] = temp;
                }
            }
        }
        
        // Renormalize top-k probabilities
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += tokens[i].prob;
        }
        
        // Sample from top-k
        float r = (float)rand() / (float)RAND_MAX * sum;
        float cumsum = 0.0f;
        for (int i = 0; i < k; i++) {
            cumsum += tokens[i].prob;
            if (r <= cumsum) {
                return tokens[i].token;
            }
        }
        return tokens[k-1].token;
    }
    
    return 0; // Fallback
}

// Generate text from prompt
static void generate_text(LegacyLLM* model, const char* prompt, const GenerationConfig* config) {
    int vocab_size = model->vocab_size;
    int max_tokens = config->max_tokens;
    
    printf("Prompt: %s", prompt);
    fflush(stdout);
    
    // Tokenize prompt
    int prompt_length = strlen(prompt);
    int* input_tokens = (int*)malloc(prompt_length * sizeof(int));
    int num_tokens = 0;
    
    for (int i = 0; i < prompt_length && prompt[i] != '\0'; i++) {
        unsigned char c = (unsigned char)prompt[i];
        if (c < vocab_size) {
            input_tokens[num_tokens++] = c;
        }
    }
    
    if (num_tokens == 0) {
        printf("\nError: Prompt contains no valid tokens.\n");
        free(input_tokens);
        return;
    }
    
    // Generate tokens autoregressively
    int current_token = input_tokens[num_tokens - 1];
    free(input_tokens);
    
    int generated_count = 0;
    while (generated_count < max_tokens) {
        // Forward pass
        float* probs = forward_llm(model, current_token);
        if (!probs) {
            printf("\nError: Forward pass failed.\n");
            break;
        }
        
        // Sample next token
        current_token = sample_token(probs, vocab_size, config);
        free_float_array(probs);
        
        // Print token as character
        if (current_token < 256 && current_token >= 0) {
            char c = (char)current_token;
            printf("%c", c);
            fflush(stdout);
            
            // Stop on newline or end of text
            if (c == '\n' || c == '\0') {
                break;
            }
        }
        
        generated_count++;
    }
    
    if (generated_count >= max_tokens) {
        printf("\n[Max tokens reached]\n");
    } else {
        printf("\n");
    }
}

static void print_usage(const char* program_name) {
    printf("Usage: %s [options] [prompt]\n", program_name);
    printf("\nOptions:\n");
    printf("  -m, --model <file>      Model file (default: %s)\n", MODEL_FILE);
    printf("  -n, --max-tokens <n>    Maximum tokens to generate (default: %d)\n", DEFAULT_MAX_TOKENS);
    printf("  -s, --strategy <type>   Sampling strategy: greedy, temperature, topk (default: greedy)\n");
    printf("  -t, --temperature <t>   Temperature for sampling (default: %.1f)\n", DEFAULT_TEMPERATURE);
    printf("  -k, --top-k <k>         Top-k value for topk sampling (default: %d)\n", DEFAULT_TOP_K);
    printf("  --seed <n>              Random seed (default: current time)\n");
    printf("  -h, --help              Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s \"Once upon a time\"\n", program_name);
    printf("  %s -s temperature -t 0.8 \"Hello world\"\n", program_name);
    printf("  %s -s topk -k 40 -n 200 \"In 1492,\"\n", program_name);
}

int main(int argc, char* argv[]) {
    GenerationConfig config = {
        .strategy = SAMPLING_GREEDY,
        .temperature = DEFAULT_TEMPERATURE,
        .top_k = DEFAULT_TOP_K,
        .max_tokens = DEFAULT_MAX_TOKENS,
        .seed = (int)time(NULL)
    };
    
    const char* model_file = MODEL_FILE;
    char prompt[1024] = "";
    int prompt_set = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_file = argv[++i];
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            config.max_tokens = atoi(argv[++i]);
        }
        else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--strategy") == 0) && i + 1 < argc) {
            const char* strat = argv[++i];
            if (strcmp(strat, "greedy") == 0) config.strategy = SAMPLING_GREEDY;
            else if (strcmp(strat, "temperature") == 0) config.strategy = SAMPLING_TEMPERATURE;
            else if (strcmp(strat, "topk") == 0) config.strategy = SAMPLING_TOP_K;
            else {
                fprintf(stderr, "Error: Unknown strategy '%s'\n", strat);
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temperature") == 0) && i + 1 < argc) {
            config.temperature = atof(argv[++i]);
        }
        else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0) && i + 1 < argc) {
            config.top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        }
        else if (!prompt_set) {
            // First non-option argument is the prompt
            strncpy(prompt, argv[i], sizeof(prompt) - 1);
            prompt[sizeof(prompt) - 1] = '\0';
            prompt_set = 1;
        }
        else {
            // Append to prompt
            strncat(prompt, " ", sizeof(prompt) - strlen(prompt) - 1);
            strncat(prompt, argv[i], sizeof(prompt) - strlen(prompt) - 1);
        }
    }
    
    if (!prompt_set) {
        fprintf(stderr, "Error: No prompt provided.\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize random seed
    srand(config.seed);
    
    printf("Legacy-1bit LLM Inference\n");
    printf("=========================\n");
    printf("Loading model from %s...\n", model_file);
    
    // Load model
    LegacyLLM* model = load_model(model_file);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_file);
        fprintf(stderr, "Make sure you have trained a model first by running: ./legacy_llm_sse\n");
        return 1;
    }
    
    printf("Model loaded successfully.\n");
    printf("Vocabulary size: %d, Model dimension: %d, Blocks: %d\n",
           model->vocab_size, model->model_dim, model->num_transformer_blocks);
    
    // Print configuration
    printf("\nGeneration config:\n");
    printf("  Strategy: ");
    switch (config.strategy) {
        case SAMPLING_GREEDY: printf("greedy\n"); break;
        case SAMPLING_TEMPERATURE: printf("temperature (T=%.2f)\n", config.temperature); break;
        case SAMPLING_TOP_K: printf("top-k (k=%d)\n", config.top_k); break;
    }
    printf("  Max tokens: %d\n", config.max_tokens);
    printf("  Random seed: %d\n", config.seed);
    printf("\n");
    
    // Generate text
    generate_text(model, prompt, &config);
    
    // Cleanup
    free_legacy_llm(model);
    
    return 0;
}
