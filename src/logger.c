#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "logger.h"

Logger* logger_init(const char* log_dir, float learning_rate) {
    Logger* logger = (Logger*)malloc(sizeof(Logger));
    if (!logger) {
        fprintf(stderr, "Error: Failed to allocate logger\n");
        return NULL;
    }
    
    // Create log filename with timestamp
    char filename[256];
    char timestamp[64];
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);
    
    snprintf(filename, sizeof(filename), "%s/training_%s.json", 
             log_dir ? log_dir : ".", timestamp);
    
    logger->json_file = fopen(filename, "w");
    if (!logger->json_file) {
        fprintf(stderr, "Error: Failed to create log file: %s\n", filename);
        free(logger);
        return NULL;
    }
    
    // Write JSON header
    fprintf(logger->json_file, "{\n");
    fprintf(logger->json_file, "  \"training_session\": {\n");
    fprintf(logger->json_file, "    \"start_time\": \"%s\",\n", timestamp);
    fprintf(logger->json_file, "    \"learning_rate\": %.6f,\n", learning_rate);
    fprintf(logger->json_file, "    \"epochs\": [\n");
    
    logger->epoch = 0;
    logger->learning_rate = learning_rate;
    logger->is_first_entry = 1;
    
    printf("[Logger] Initialized. Logging to: %s\n", filename);
    return logger;
}

void logger_close(Logger* logger) {
    if (!logger) return;
    
    if (logger->json_file) {
        // Close JSON array and object
        fprintf(logger->json_file, "\n    ]\n");
        fprintf(logger->json_file, "  }\n");
        fprintf(logger->json_file, "}\n");
        
        fclose(logger->json_file);
        logger->json_file = NULL;
    }
    
    free(logger);
    printf("[Logger] Closed successfully.\n");
}

void logger_log_epoch(Logger* logger, int epoch, float loss, float perplexity,
                      float top1_accuracy, float top3_accuracy, float top5_accuracy,
                      int samples_processed, double epoch_time) {
    if (!logger || !logger->json_file) return;
    
    // Add comma if not first entry
    if (!logger->is_first_entry) {
        fprintf(logger->json_file, ",\n");
    }
    logger->is_first_entry = 0;
    
    // Write epoch data
    fprintf(logger->json_file, "      {\n");
    fprintf(logger->json_file, "        \"epoch\": %d,\n", epoch);
    fprintf(logger->json_file, "        \"loss\": %.6f,\n", loss);
    fprintf(logger->json_file, "        \"perplexity\": %.6f,\n", perplexity);
    fprintf(logger->json_file, "        \"top1_accuracy\": %.4f,\n", top1_accuracy);
    fprintf(logger->json_file, "        \"top3_accuracy\": %.4f,\n", top3_accuracy);
    fprintf(logger->json_file, "        \"top5_accuracy\": %.4f,\n", top5_accuracy);
    fprintf(logger->json_file, "        \"samples_processed\": %d,\n", samples_processed);
    fprintf(logger->json_file, "        \"epoch_time_seconds\": %.3f\n", epoch_time);
    fprintf(logger->json_file, "      }");
    
    fflush(logger->json_file);
}

void logger_log_batch(Logger* logger, int epoch, int batch, float loss,
                      int samples_processed) {
    // For now, we only log batch info to console
    // Could be extended to log to a separate file or include in main JSON
    (void)logger; // Unused for now
    (void)epoch;
    (void)batch;
    (void)loss;
    (void)samples_processed;
}

void logger_get_timestamp(char* buffer, size_t buffer_size) {
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    strftime(buffer, buffer_size, "%Y-%m-%d %H:%M:%S", tm_info);
}
