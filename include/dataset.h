#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>

// Dataset stream structure for handling large files
typedef struct {
    FILE* file;
    char* filepath;
    long file_size;
    long current_position;
    int vocab_size;
    
    // Buffer for streaming
    char* buffer;
    int buffer_size;
    int buffer_pos;
    int buffer_filled;
    
    // Configuration
    int chunk_size;        // Size of chunks to read at a time
    int max_sequence_length;
    int pad_token;
} DatasetStream;

// Dynamic batch configuration
typedef struct {
    int batch_size;
    int sequence_length;
    int num_batches;
    int current_batch;
    int* batch_indices;    // For shuffling
    int shuffle;
} BatchConfig;

// Streaming dataset functions
DatasetStream* dataset_stream_init(const char* filepath, int vocab_size, 
                                   int chunk_size, int max_seq_len);
void dataset_stream_close(DatasetStream* stream);
int dataset_stream_next_chunk(DatasetStream* stream);
int dataset_stream_get_sequence(DatasetStream* stream, int* sequence, 
                                int seq_length, int* target_token);

// Batch configuration
BatchConfig* batch_config_create(int batch_size, int sequence_length, 
                                 int num_samples, int shuffle);
void batch_config_destroy(BatchConfig* config);
void batch_config_shuffle(BatchConfig* config);
int* batch_config_get_batch(BatchConfig* config, int batch_idx, 
                            const int* all_data, int data_size);

// Advanced tokenization with subword support (BPE placeholder)
int* tokenize_text_advanced(const char* text, int vocab_size, 
                            int* token_count, int use_subword);

// Data augmentation (placeholder for future)
char* augment_text(const char* text, float swap_prob, float delete_prob);

#endif // DATASET_H
