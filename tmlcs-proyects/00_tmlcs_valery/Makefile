# Makefile for Legacy-1bit LLM
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -Iinclude $(CFLAGS_EXTRA) # -std=c99 para compatibilidad con compiladores antiguos
TARGET = legacy_llm

# Conditional compilation for SSE
ifdef USE_SSE_BUILD
    CFLAGS += -DUSE_SSE -msse -msse2 # Define USE_SSE macro and enable SSE/SSE2 instructions
endif

SRCS = src/main.c src/math_ops.c src/data_utils.c src/model.c src/forward.c src/backward.c tests/test_llm.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)