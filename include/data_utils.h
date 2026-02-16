#ifndef DATA_UTILS_H
#define DATA_UTILS_H

// Function for data loading and tokenization
void initialize_vocabulary(int *vocab_size);
char* load_text_from_file(const char* filepath);
int* tokenize_text(const char* text, int vocab_size, int *token_count);
void free_text(char* text);
void free_tokens(int* tokens);

// Batching utility
int* create_padded_sequence(const int* all_tokens, int total_tokens, int start_idx);

// Timing utilities
long long get_time_in_ms();

#endif // DATA_UTILS_H