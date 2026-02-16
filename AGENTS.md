# AGENTS.md - Legacy-1bit LLM

C99 project implementing a ternary (-1, 0, 1) quantized Large Language Model optimized for 2000-era hardware.

## Build Commands

```bash
# Build everything (both SSE and non-SSE variants)
make

# Build with SSE optimization enabled
make USE_SSE_BUILD=1

# Build specific variants
make legacy_llm_sse      # SSE optimized
make legacy_llm_no_sse   # Non-SSE fallback

# Clean build artifacts
make clean

# Run performance comparison
make perf
```

## Run Commands

```bash
# Run the executable (runs tests, then training)
./legacy_llm_sse
./legacy_llm_no_sse

# Check performance logs
cat sse_perf_log.txt
cat no_sse_perf_log.txt
```

## Code Style Guidelines

### Language Standard
- **C99 only** - no C++ features, no GNU extensions
- Compile with `-std=c99 -Wall -Wextra`

### Naming Conventions
- **Structs**: `snake_case` (e.g., `TransformerBlock`, `TernaryMatrix`)
- **Functions**: `snake_case` with module prefix (e.g., `create_ternary_matrix`, `free_legacy_llm`)
- **Constants/Macros**: `UPPER_CASE` (e.g., `MAX_VOCAB_SIZE`, `MODEL_DIM`)
- **Files**: `snake_case.c` / `snake_case.h`
- **Typedefs**: Use typedef for structs (e.g., `typedef struct { ... } TernaryMatrix;`)

### File Organization
```
include/       # Public headers only
  legacy_llm.h # Main struct definitions, global macros
  model.h      # Model creation, persistence, management
  math_ops.h   # Mathematical operations (SSE conditional)
  forward.h    # Forward pass functions
  backward.h   # Backward pass functions
  data_utils.h # Data loading and tokenization
  test_llm.h   # Test declarations

src/           # Implementation files
  main.c       # Entry point, training loop
  model.c      # Model implementation
  math_ops.c   # Math ops with SSE intrinsics
  forward.c    # Forward pass
  backward.c   # Backward pass
  data_utils.c # Data utilities

tests/         # Test suite
  test_llm.c   # Unit tests

AUDIT.md       # Detailed quality audit report
```

### Headers
```c
// Include guards - ALWAYS use
#ifndef FILENAME_H
#define FILENAME_H

// Standard library first
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Then project headers
#include "legacy_llm.h"

#endif // FILENAME_H
```

### Memory Management
- **ALWAYS** check malloc/calloc return values
- **ALWAYS** provide corresponding free functions
- Use `perror()` for system-level allocation failures
- Free in reverse order of allocation

```c
// Pattern: Create
TernaryMatrix* mat = create_ternary_matrix(rows, cols);
if (!mat) {
    perror("Error allocating matrix");
    return NULL;
}

// Pattern: Free
free_ternary_matrix(mat);
```

### SSE Optimization
```c
// Conditional SSE compilation
#ifdef USE_SSE
  #include <xmmintrin.h>
  // SSE implementation
#else
  // Fallback scalar implementation
#endif
```

### Error Handling
```c
// Return 0 for error, 1 for success in internal functions
static int write_to_file(FILE* fp, const void* data, size_t len) {
    if (fwrite(data, 1, len, fp) != len) {
        return 0; // Error
    }
    return 1; // Success
}

// Check file operations
if (!write_to_file(fp, data, len)) {
    fprintf(stderr, "Error: Failed to write data\n");
    return 0;
}
```

### Types
- Weights: `int8_t` (ternary: -1, 0, 1)
- Activations/Gradients: `float`
- Sizes/Indices: `int` (not size_t for consistency)
- Explicit casting for malloc: `(int8_t*)malloc(...)`

### Comments
- Use `//` for single-line comments
- Use `/* */` for multi-line or file headers
- Comment complex algorithms, not obvious code
- Document units and constraints in struct definitions

### Testing
- Tests are compiled into the main executable
- Run automatically at startup via `run_all_llm_tests()`
- Tests use `printf()` for output visibility
- No separate test runner - use main binary

### Compiler Flags
```bash
# Standard flags (from Makefile)
-Wall -Wextra -std=c99 -Iinclude

# SSE builds add:
-DUSE_SSE -msse -msse2

# Performance measurement:
-DMEASURE_PERFORMANCE
```

### Key Constraints
- Target: 2000-era laptops (limited RAM/CPU)
- Memory-efficient: Gradient checkpointing, sparse storage
- Ternary weights only: -1, 0, 1
- No external dependencies (math library only with `-lm`)
