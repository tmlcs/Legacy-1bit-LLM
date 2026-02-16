# Project Architecture: Legacy-1bit LLM

This document provides a high-level overview of the architecture, key algorithms, and design decisions behind the Legacy-1bit Large Language Model (LLM) project.

## 1. Core Principles

The primary design principle of this LLM is to explore extreme quantization by utilizing **ternary weights** (-1, 0, 1). This choice aims to significantly reduce model size and potentially improve inference efficiency, albeit with potential trade-offs in model capacity and accuracy. The project is implemented in C for direct memory control and performance optimization.

## 2. Model Architecture

The LLM broadly follows a simplified Transformer-like architecture, adapted for ternary weights. It consists of the following main components:

### 2.1. Embedding Layer
*   **Purpose:** Converts input token IDs into dense vector representations (embeddings).
*   **Implementation:** Uses a `TernaryMatrix` for its embedding weights (`vocab_size x MODEL_DIM`).

### 2.2. Transformer Blocks
The model comprises multiple identical `TransformerBlock` layers, each consisting of:

#### 2.2.1. Multi-Head Attention (MHA) Layer
*   **Purpose:** Allows the model to weigh the importance of different tokens in the input sequence when processing each token.
*   **Implementation:**
    *   Uses `TernaryMatrix` for query (Wq), key (Wk), value (Wv), and output projection (Wo) weights.
    *   Biases (bq, bk, bv, bo) are implemented as standard floating-point arrays.
    *   A simplified attention mechanism (likely single-head for simplicity, or with an implicit multi-head structure that is not explicitly detailed at a high level).

#### 2.2.2. Feed-Forward Network (FFN) Layer
*   **Purpose:** A simple two-layer neural network applied independently to each position, enhancing the model's ability to learn complex patterns.
*   **Implementation:**
    *   Uses `TernaryMatrix` for input (Wi) and output (Wo) weights.
    *   Biases (bi, bo) are standard floating-point arrays.
    *   ReLU activation is typically applied between the two linear transformations.

#### 2.2.3. Layer Normalization
*   **Purpose:** Stabilizes learning by normalizing the activations of each layer, making training faster and more stable.
*   **Implementation:** Applied before (or after, depending on pre/post-norm design) both the MHA and FFN sub-layers. Uses trainable `gamma` and `beta` parameters (floating-point arrays).

### 2.3. Output Layer
*   **Purpose:** Transforms the final hidden state into logits, which are then converted into probability distributions over the vocabulary.
*   **Implementation:**
    *   Uses a `TernaryMatrix` for unembedding weights (`MODEL_DIM x vocab_size`).
    *   A floating-point `bias` vector is added.
    *   Softmax activation is applied to convert logits into probabilities.

## 3. Training and Optimization

### 3.1. Loss Function
*   **Cross-Entropy Loss:** Standard for classification tasks like next-token prediction.

### 3.2. Backward Pass and Gradients
*   **Implementation:** A manual backpropagation implementation computes gradients for all parameters.
*   **Gradient Structures:** Mirror the model's structure, but hold floating-point gradients (`LegacyLLM_Gradients`).

### 3.3. Ternary Weight Updates
*   **Algorithm:** Unlike standard SGD where weights are updated by `weight -= learning_rate * gradient`, ternary weights are updated based on the *sign* of their gradients.
    *   If `gradient > 0`, weight tends towards `1`.
    *   If `gradient < 0`, weight tends towards `-1`.
    *   If `gradient == 0`, weight remains unchanged.
*   **Biases and LayerNorm Parameters:** Updated using standard Stochastic Gradient Descent (SGD) with a `learning_rate`.

### 3.4. Model Persistence
*   **Save/Load:** The model state can be saved to and loaded from a binary file (`llm_model.bin`). This includes integrity checks using a magic number and version.

## 4. Key Design Decisions & Challenges

*   **Ternary Quantization:**
    *   **Pros:** Significant model size reduction, potential for hardware-accelerated inference.
    *   **Cons:** Reduced model capacity, complex training (non-differentiable weight updates), potentially lower accuracy compared to full-precision models. The chosen update rule is a common heuristic for training quantized networks.
*   **C Language Implementation:**
    *   **Pros:** Fine-grained control over memory, high performance, no runtime overhead from virtual machines.
    *   **Cons:** Manual memory management (prone to errors like leaks/double-frees), steeper learning curve, absence of high-level ML libraries.
*   **Simplified Transformer:** The architecture is a simplification of modern Transformers, likely omitting aspects like positional encoding, advanced normalization techniques, or complex multi-head attention mechanisms for pedagogical or resource-constrained reasons.
*   **Test Methodology:** Initial tests were manual `printf`-based. Refactoring to automated assertions is crucial for reliability.

## 5. Future Work (Implicit)

*   **Performance Optimization:** Further optimize core math operations (e.g., SIMD instructions, custom kernels).
*   **Advanced Quantization:** Explore more sophisticated quantization-aware training techniques.
*   **Model Capacity:** Investigate ways to improve the model's representational power within the ternary constraint.
*   **Deployment:** Consider integration with specialized hardware for efficient inference.
