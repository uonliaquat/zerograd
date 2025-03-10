# Compiler
CC = gcc

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Compiler Flags
CFLAGS = -Wall -Wextra -I$(INC_DIR)

# Find all .c files in src/ and main.c
SRCS = $(wildcard $(SRC_DIR)/*.c) main.c

# Output binary
TARGET = $(BUILD_DIR)/main

# Default rule (compile everything)
all:
	@mkdir -p $(BUILD_DIR)  # Create build directory if it doesn't exist
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

# Run the compiled program
run: all
	./$(TARGET)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) *.o $(SRC_DIR)/*.o main.o