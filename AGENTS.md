# AGENTS.md - Legacy-1bit LLM

C99 project implementing a ternary (-1, 0, 1) quantized Large Language Model optimized for 2000-era hardware.

## Build Commands

```bash
make                           # Build main application (non-SSE)
make legacy_llm_no_sse         # Non-SSE build
make legacy_llm_sse            # SSE optimized build
make inference_no_sse          # Inference binary (non-SSE)
make inference_sse             # Inference binary (SSE)
make clean                     # Clean build artifacts
```

## Test Commands

```bash
make test                      # Run all tests (builds and runs both SSE/non-SSE)
./test_runner_no_sse           # Run non-SSE tests only
./test_runner_sse              # Run SSE tests only
make test_sse_correctness      # Compare SSE vs non-SSE correctness
make test_memory               # Memory leak test (requires Valgrind)
make test_grad_check           # Gradient verification test
make perf                      # Performance comparison with timing logs
```

## Lint/Typecheck

```bash
# No separate lint tool - use compiler warnings
make clean && make CFLAGS="-Wall -Wextra -Werror -std=c99"
```

## Run Commands

```bash
./legacy_llm_no_sse            # Run main application
./inference_no_sse             # Run inference
```

## Code Style Guidelines

### Language Standard
- **C99 only** - no C++ features, no GNU extensions
- Compile with `-std=c99 -Wall -Wextra`

### Naming Conventions
- **Structs**: `PascalCase` (e.g., `TransformerBlock`, `TernaryMatrix`)
- **Functions**: `snake_case` with module prefix (e.g., `create_ternary_matrix`, `forward_llm_batch`)
- **Constants/Macros**: `UPPER_CASE` (e.g., `MAX_VOCAB_SIZE`, `MODEL_DIM`)
- **Files**: `snake_case.c` / `snake_case.h`
- **Typedefs**: Use typedef for structs

### File Organization
```
include/           # Public headers
  legacy_llm.h     # Core structs, macros, global defines
  model.h          # Model creation, persistence
  math_ops.h       # Math operations, timer macros
  forward.h        # Forward pass functions
  backward.h       # Backward pass functions
  config.h         # Training/architecture parameters
  test_framework.h # Test macros (ASSERT_*, TEST_*)

src/               # Implementation files
tests/             # Test suite
```

### Headers
```c
#ifndef FILENAME_H
#define FILENAME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "legacy_llm.h"

#endif // FILENAME_H
```

### Types
- **Weights**: `int8_t` (ternary values: -1, 0, 1)
- **Activations/Gradients**: `float`
- **Sizes/Indices**: `int` (not size_t)
- **Explicit casting**: `(int8_t*)malloc(...)`, `(float*)calloc(...)`

### Memory Management
- **ALWAYS** check malloc/calloc return values
- **ALWAYS** provide corresponding free functions
- Use `perror()` for allocation failures
- Free in reverse order of allocation

```c
TernaryMatrix* mat = create_ternary_matrix(rows, cols);
if (!mat) {
    perror("Error allocating matrix");
    return NULL;
}
free_ternary_matrix(mat);
```

### Error Handling
- Return `NULL` for pointer failures, `0` for int failures
- Return `1` for success in internal helper functions
- Use `fprintf(stderr, ...)` for error messages

### SSE Optimization
```c
#ifdef USE_SSE
  #include <xmmintrin.h>
  #include <emmintrin.h>
  // SSE implementation
#else
  // Scalar fallback
#endif
```

### Performance Timing
```c
// Wrap performance-critical functions with timers
START_TIMER(my_function_timer);
// ... function body ...
STOP_TIMER(my_function_timer, "my_function_name");

// Enable with: make perf (adds -DMEASURE_PERFORMANCE)
```

### Testing
- Test files in `tests/` directory
- Use framework macros from `test_framework.h`:
```c
void test_my_feature() {
    TEST_BEGIN("MyFeatureTest");
    ASSERT_TRUE(condition, "Description");
    ASSERT_EQUALS_FLOAT(expected, actual, 0.0001f, "Description");
    ASSERT_NOT_NULL(ptr, "Description");
    TEST_END();
}
```
- Register tests in `tests/test_llm.c` main runner

### Comments
- Use `//` for single-line comments
- Use `/* */` for multi-line or file headers
- No comments unless explaining non-obvious logic

### Key Constraints
- Target: 2000-era hardware (limited RAM/CPU)
- Ternary weights only: -1, 0, 1
- No external dependencies (only `-lm`)
- Use batched operations (`*_batch` functions) over deprecated single-item versions
