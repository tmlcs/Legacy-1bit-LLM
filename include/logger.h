#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <time.h>

// Logger structure
typedef struct {
    FILE* json_file;
    int epoch;
    float learning_rate;
    int is_first_entry;
} Logger;

// Initialize logger
Logger* logger_init(const char* log_dir, float learning_rate);

// Close logger and free resources
void logger_close(Logger* logger);

// Log training metrics for an epoch
void logger_log_epoch(Logger* logger, int epoch, float loss, float perplexity, 
                      float top1_accuracy, float top3_accuracy, float top5_accuracy,
                      int samples_processed, double epoch_time);

// Log batch metrics
void logger_log_batch(Logger* logger, int epoch, int batch, float loss, 
                      int samples_processed);

// Generate timestamp string
void logger_get_timestamp(char* buffer, size_t buffer_size);

#endif // LOGGER_H
