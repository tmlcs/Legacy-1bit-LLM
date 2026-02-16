# Makefile for Legacy-1bit LLM
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -Iinclude $(CFLAGS_EXTRA) # -std=c99 para compatibilidad con compiladores antiguos

# Conditional compilation for SSE
ifdef USE_SSE_BUILD
    CFLAGS += -DUSE_SSE -msse -msse2 # Define USE_SSE macro and enable SSE/SSE2 instructions
endif

# Conditional compilation for Performance Measurement
ifdef MEASURE_PERF
    CFLAGS += -DMEASURE_PERFORMANCE
endif

SRCS = src/main.c src/math_ops.c src/data_utils.c src/model.c src/forward.c src/backward.c tests/test_llm.c
OBJS = $(SRCS:.c=.o)

all: legacy_llm_no_sse

# Default build (non-SSE)
legacy_llm_no_sse: CFLAGS_EXTRA=
legacy_llm_no_sse: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ -lm

# SSE optimized build
legacy_llm_sse: CFLAGS_EXTRA=-DUSE_SSE -msse -msse2
legacy_llm_sse: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ -lm

# Target to run performance analysis
perf: clean
	@echo "--- Building for performance measurement (Non-SSE) ---"
	$(MAKE) legacy_llm_no_sse MEASURE_PERF=1
	@echo "--- Running Non-SSE build and capturing logs ---"
	./legacy_llm_no_sse 2> no_sse_perf_log.txt

	@echo "--- Building for performance measurement (SSE) ---"
	$(MAKE) legacy_llm_sse MEASURE_PERF=1 USE_SSE_BUILD=1
	@echo "--- Running SSE build and capturing logs ---"
	./legacy_llm_sse 2> sse_perf_log.txt

	@echo "--- Analyzing performance logs ---"
	./analyze_perf.sh

.PHONY: all perf clean legacy_llm_no_sse legacy_llm_sse

clean:
	rm -f $(OBJS) legacy_llm_no_sse legacy_llm_sse *.o sse_perf_log.txt no_sse_perf_log.txt