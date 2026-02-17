#ifndef CONFIG_H
#define CONFIG_H

// --- Model Architecture Parameters ---
#define MAX_VOCAB_SIZE 256        // Max vocabulary size for character-level tokenization
#define MODEL_DIM 256             // Dimensionality of model embeddings and transformer layers
#define NUM_HEADS 4               // Number of attention heads (MODEL_DIM % NUM_HEADS must be 0)
#define FFN_DIM_MULTIPLIER 4      // Multiplier for feed-forward network hidden dimension (e.g., MODEL_DIM * FFN_DIM_MULTIPLIER)
#define MAX_SEQUENCE_LENGTH 128   // Maximum input sequence length for the model
#define BATCH_SIZE 8              // Number of sequences processed in parallel in a batch
#define PAD_TOKEN MAX_VOCAB_SIZE  // Special token for padding, using MAX_VOCAB_SIZE to avoid conflict with actual char codes

// --- Training Parameters ---
#define LEARNING_RATE 0.01f       // Learning rate for the optimizer
#define NUM_EPOCHS 10             // Number of training epochs
#define SAVE_INTERVAL 2           // Save model checkpoint every N epochs
#define CHECKPOINT_FILE "llm_model.bin" // Filename for saving/loading model checkpoints

#endif // CONFIG_H
