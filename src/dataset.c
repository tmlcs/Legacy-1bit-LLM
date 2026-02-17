#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"

// Initialize dataset stream for large files
DatasetStream* dataset_stream_init(const char* filepath, int vocab_size, 
                                   int chunk_size, int max_seq_len) {
    DatasetStream* stream = (DatasetStream*)malloc(sizeof(DatasetStream));
    if (!stream) {
        fprintf(stderr, "Error: Failed to allocate dataset stream\n");
        return NULL;
    }
    
    stream->file = fopen(filepath, "r");
    if (!stream->file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filepath);
        free(stream);
        return NULL;
    }
    
    // Get file size
    fseek(stream->file, 0, SEEK_END);
    stream->file_size = ftell(stream->file);
    fseek(stream->file, 0, SEEK_SET);
    
    stream->filepath = strdup(filepath);
    stream->current_position = 0;
    stream->vocab_size = vocab_size;
    stream->chunk_size = chunk_size > 0 ? chunk_size : 8192; // Default 8KB chunks
    stream->max_sequence_length = max_seq_len;
    stream->pad_token = vocab_size; // Assuming PAD_TOKEN is vocab_size
    
    // Allocate buffer
    stream->buffer_size = stream->chunk_size + max_seq_len; // Extra space for overlap
    stream->buffer = (char*)malloc(stream->buffer_size);
    if (!stream->buffer) {
        fprintf(stderr, "Error: Failed to allocate buffer\n");
        fclose(stream->file);
        free(stream->filepath);
        free(stream);
        return NULL;
    }
    
    stream->buffer_pos = 0;
    stream->buffer_filled = 0;
    
    printf("[DatasetStream] Initialized for '%s' (%.2f MB)\n", 
           filepath, stream->file_size / (1024.0 * 1024.0));
    
    return stream;
}

// Close dataset stream
void dataset_stream_close(DatasetStream* stream) {
    if (!stream) return;
    
    if (stream->file) {
        fclose(stream->file);
        stream->file = NULL;
    }
    
    if (stream->buffer) {
        free(stream->buffer);
        stream->buffer = NULL;
    }
    
    if (stream->filepath) {
        free(stream->filepath);
        stream->filepath = NULL;
    }
    
    free(stream);
    printf("[DatasetStream] Closed\n");
}

// Read next chunk from file
int dataset_stream_next_chunk(DatasetStream* stream) {
    if (!stream || !stream->file) return 0;
    
    // Keep last max_sequence_length characters for context overlap
    int overlap = stream->buffer_filled - stream->buffer_pos;
    if (overlap > stream->max_sequence_length) {
        overlap = stream->max_sequence_length;
    }
    
    if (overlap > 0) {
        // Move overlap to beginning of buffer
        memmove(stream->buffer, stream->buffer + stream->buffer_pos - overlap, overlap);
    }
    
    // Read new chunk
    size_t to_read = stream->chunk_size;
    size_t actually_read = fread(stream->buffer + overlap, 1, to_read, stream->file);
    
    stream->buffer_filled = overlap + actually_read;
    stream->buffer_pos = overlap;
    stream->current_position = ftell(stream->file);
    
    return actually_read > 0 ? 1 : 0;
}

// Get next sequence and target from stream
int dataset_stream_get_sequence(DatasetStream* stream, int* sequence, 
                                int seq_length, int* target_token) {
    if (!stream || !sequence || !target_token) return 0;
    
    // Check if we need more data
    if (stream->buffer_pos + seq_length + 1 >= stream->buffer_filled) {
        if (!dataset_stream_next_chunk(stream)) {
            return 0; // End of file
        }
    }
    
    // Extract sequence
    for (int i = 0; i < seq_length && stream->buffer_pos < stream->buffer_filled; i++) {
        unsigned char c = (unsigned char)stream->buffer[stream->buffer_pos++];
        sequence[i] = (c < stream->vocab_size) ? c : stream->pad_token;
    }
    
    // Get target (next character)
    if (stream->buffer_pos < stream->buffer_filled) {
        unsigned char c = (unsigned char)stream->buffer[stream->buffer_pos];
        *target_token = (c < stream->vocab_size) ? c : stream->pad_token;
    } else {
        *target_token = stream->pad_token;
    }
    
    return 1;
}

// Create batch configuration
BatchConfig* batch_config_create(int batch_size, int sequence_length, 
                                 int num_samples, int shuffle) {
    BatchConfig* config = (BatchConfig*)malloc(sizeof(BatchConfig));
    if (!config) return NULL;
    
    config->batch_size = batch_size;
    config->sequence_length = sequence_length;
    config->num_batches = (num_samples + batch_size - 1) / batch_size;
    config->current_batch = 0;
    config->shuffle = shuffle;
    
    // Create indices
    config->batch_indices = (int*)malloc(config->num_batches * sizeof(int));
    if (!config->batch_indices) {
        free(config);
        return NULL;
    }
    
    for (int i = 0; i < config->num_batches; i++) {
        config->batch_indices[i] = i;
    }
    
    if (shuffle) {
        batch_config_shuffle(config);
    }
    
    return config;
}

// Destroy batch configuration
void batch_config_destroy(BatchConfig* config) {
    if (!config) return;
    if (config->batch_indices) {
        free(config->batch_indices);
    }
    free(config);
}

// Shuffle batch indices
void batch_config_shuffle(BatchConfig* config) {
    if (!config || !config->batch_indices) return;
    
    srand(time(NULL));
    for (int i = config->num_batches - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = config->batch_indices[i];
        config->batch_indices[i] = config->batch_indices[j];
        config->batch_indices[j] = temp;
    }
}

// Get batch data (returns array of sequence indices)
int* batch_config_get_batch(BatchConfig* config, int batch_idx, 
                            const int* all_data, int data_size) {
    if (!config || !all_data || batch_idx >= config->num_batches) return NULL;
    
    int actual_batch_idx = config->batch_indices[batch_idx];
    int start_idx = actual_batch_idx * config->batch_size;
    int end_idx = start_idx + config->batch_size;
    if (end_idx > data_size) end_idx = data_size;
    
    int batch_size = end_idx - start_idx;
    int* batch = (int*)malloc(batch_size * config->sequence_length * sizeof(int));
    if (!batch) return NULL;
    
    // Copy sequences
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < config->sequence_length; j++) {
            int data_idx = start_idx + i + j;
            if (data_idx < data_size) {
                batch[i * config->sequence_length + j] = all_data[data_idx];
            } else {
                batch[i * config->sequence_length + j] = 0; // PAD
            }
        }
    }
    
    return batch;
}

// Advanced tokenization with subword support (placeholder)
int* tokenize_text_advanced(const char* text, int vocab_size, 
                            int* token_count, int use_subword) {
    // For now, same as basic tokenization
    // TODO: Implement BPE or WordPiece tokenization
    (void)use_subword; // Unused for now
    
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    if (!tokens) return NULL;
    
    *token_count = 0;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c < vocab_size) {
            tokens[(*token_count)++] = c;
        }
    }
    
    return tokens;
}

// Data augmentation (placeholder)
char* augment_text(const char* text, float swap_prob, float delete_prob) {
    // TODO: Implement text augmentation
    // - Character swap
    // - Character deletion
    // - Character insertion
    (void)swap_prob;
    (void)delete_prob;
    
    // For now, return copy of original
    return strdup(text);
}
