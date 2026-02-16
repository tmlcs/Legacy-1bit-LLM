#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> // For gettimeofday

#include "data_utils.h" // Include its own header
#include "legacy_llm.h" // For MAX_VOCAB_SIZE

// --- Data Handling Functions ---

void initialize_vocabulary(int *vocab_size) {
    *vocab_size = MAX_VOCAB_SIZE;
}

char* load_text_from_file(const char* filepath) {
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char*)malloc(length + 1);
    if (buffer == NULL) {
        perror("Error allocating memory for file content");
        fclose(file);
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, length, file);
    if (read_bytes != (size_t)length) {
        perror("Error reading file");
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[length] = '\0';

    fclose(file);
    return buffer;
}

int* tokenize_text(const char* text, int vocab_size, int *token_count) {
    if (text == NULL) {
        *token_count = 0;
        return NULL;
    }

    size_t len = strlen(text);
    int *tokens = (int*)malloc(len * sizeof(int));
    if (tokens == NULL) {
        perror("Error allocating memory for tokens");
        *token_count = 0;
        return NULL;
    }

    *token_count = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c < vocab_size) {
            tokens[*token_count] = (int)c;
            (*token_count)++;
        } else {
            fprintf(stderr, "Warning: Unknown character '%c' (ASCII %d) skipped during tokenization.\n", c, c);
        }
    }
    return tokens;
}

void free_text(char* text) {
    if (text) {
        free(text);
    }
}

void free_tokens(int* tokens) {
    if (tokens) {
        free(tokens);
    }
}

// Batching utility
int* create_padded_sequence(const int* all_tokens, int total_tokens, int start_idx) {
    int* sequence = (int*)malloc(MAX_SEQUENCE_LENGTH * sizeof(int));
    if (!sequence) {
        perror("Error allocating memory for padded sequence");
        return NULL;
    }

    for (int i = 0; i < MAX_SEQUENCE_LENGTH; ++i) {
        int token_idx = start_idx + i;
        if (token_idx < total_tokens) {
            sequence[i] = all_tokens[token_idx];
        } else {
            sequence[i] = PAD_TOKEN; // Pad with PAD_TOKEN
        }
    }
    return sequence;
}

// --- Timing Utilities ---
long long get_time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}
