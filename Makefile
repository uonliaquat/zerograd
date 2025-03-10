# Compiler
CC = gcc

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Compiler Flags
CFLAGS = -Wall -Wextra -I$(INC_DIR)

# Find all .c files in src/
SRCS = $(wildcard $(SRC_DIR)/*.c) main.c
OBJS = $(SRCS:.c=.o)

# Output binary
TARGET = $(BUILD_DIR)/main

# Default rule
all: $(TARGET)

# Link objects into executable
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJS) -o $(TARGET)

# Compile .c to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) *.o $(SRC_DIR)/*.o main.o

# Run the compiled program
run: all
	./$(TARGET)