# Makefile for Legacy-1bit LLM
CC = gcc
COMMON_CFLAGS = -Wall -Wextra -std=c99 -Iinclude

# Core source files (common to both main app and tests, excluding main and test files)
CORE_SRCS = src/math_ops.c src/data_utils.c src/model.c src/forward.c src/backward.c
MAIN_SRCS = src/main.c
TEST_SRCS = tests/test_llm.c tests/test_math_ops.c tests/test_forward.c tests/test_backward.c tests/test_model.c

# Object files for non-SSE build
CORE_OBJS_NO_SSE = $(addprefix obj_no_sse/,$(CORE_SRCS:.c=.o))
MAIN_OBJS_NO_SSE = $(addprefix obj_no_sse/,$(MAIN_SRCS:.c=.o))
TEST_OBJS_NO_SSE = $(addprefix obj_no_sse/,$(TEST_SRCS:.c=.o))

# Object files for SSE build
CORE_OBJS_SSE = $(addprefix obj_sse/,$(CORE_SRCS:.c=.o))
MAIN_OBJS_SSE = $(addprefix obj_sse/,$(MAIN_SRCS:.c=.o))
TEST_OBJS_SSE = $(addprefix obj_sse/,$(TEST_SRCS:.c=.o))

# Define object directories
OBJ_DIRS = obj_no_sse obj_sse

.PHONY: all test perf clean $(OBJ_DIRS) # Add $(OBJ_DIRS) to PHONY for clean up, but they are actual directories

# Default target
all: legacy_llm_no_sse

# Rule to create object directories - this is now handled by generic rules' mkdir -p
# $(OBJ_DIRS):
# 	mkdir -p $@

# Generic rule to compile .c files into .o files for non-SSE
obj_no_sse/%.o: %.c
	@mkdir -p $(dir $@) # Create the directory for the object file
	$(CC) $(COMMON_CFLAGS) $(MEASURE_PERF_FLAG) -c $< -o $@

# Generic rule to compile .c files into .o files for SSE
obj_sse/%.o: %.c
	@mkdir -p $(dir $@) # Create the directory for the object file
	$(CC) $(COMMON_CFLAGS) -DUSE_SSE -msse -msse2 $(MEASURE_PERF_FLAG) -c $< -o $@

# --- Main Application Builds ---

# Default build (non-SSE for main application)
legacy_llm_no_sse: $(MAIN_OBJS_NO_SSE) $(CORE_OBJS_NO_SSE)
	$(CC) $(COMMON_CFLAGS) $(MEASURE_PERF_FLAG) $(MAIN_OBJS_NO_SSE) $(CORE_OBJS_NO_SSE) -o $@ -lm

# SSE optimized build (for main application)
legacy_llm_sse: $(MAIN_OBJS_SSE) $(CORE_OBJS_SSE)
	$(CC) $(COMMON_CFLAGS) -DUSE_SSE -msse -msse2 $(MEASURE_PERF_FLAG) $(MAIN_OBJS_SSE) $(CORE_OBJS_SSE) -o $@ -lm

# --- Test Runner Builds ---

# Test runner build (non-SSE)
test_runner_no_sse: $(TEST_OBJS_NO_SSE) $(CORE_OBJS_NO_SSE)
	$(CC) $(COMMON_CFLAGS) $(TEST_OBJS_NO_SSE) $(CORE_OBJS_NO_SSE) -o $@ -lm

# Test runner build (SSE optimized)
test_runner_sse: $(TEST_OBJS_SSE) $(CORE_OBJS_SSE)
	$(CC) $(COMMON_CFLAGS) -DUSE_SSE -msse -msse2 $(TEST_OBJS_SSE) $(CORE_OBJS_SSE) -o $@ -lm

# --- Main Targets ---

# Run tests
test: clean
	@echo "--- Building Test Runner (Non-SSE) ---"
	$(MAKE) test_runner_no_sse
	@echo "--- Running Non-SSE Tests ---"
	./test_runner_no_sse

	@echo "--- Building Test Runner (SSE) ---"
	$(MAKE) test_runner_sse
	@echo "--- Running SSE Tests ---"
	./test_runner_sse

# Run performance analysis
perf: clean
	@echo "--- Building for performance measurement (Non-SSE) ---"
	$(MAKE) legacy_llm_no_sse MEASURE_PERF_FLAG="-DMEASURE_PERFORMANCE"
	@echo "--- Running Non-SSE build and capturing logs ---"
	./legacy_llm_no_sse 2> no_sse_perf_log.txt

	@echo "--- Building for performance measurement (SSE) ---"
	$(MAKE) legacy_llm_sse MEASURE_PERF_FLAG="-DMEASURE_PERFORMANCE"
	@echo "--- Running SSE build and capturing logs ---"
	./legacy_llm_sse 2> sse_perf_log.txt

	@echo "--- Analyzing performance logs ---"
	./analyze_perf.sh

clean:
	rm -rf $(OBJ_DIRS)
	rm -f legacy_llm_no_sse legacy_llm_sse
	rm -f test_runner_no_sse test_runner_sse
	rm -f sse_perf_log.txt no_sse_perf_log.txt