CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=gnu11 -fopenmp

all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) -o matmul matmul.c

clean:
	rm -f matmul

.PHONY: all clean
