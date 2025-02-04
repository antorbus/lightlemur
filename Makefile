CC     = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = -Wall -Wextra -fPIC -O3 -march=native -ftree-vectorize -fopenmp -Iinclude $(CPPFLAGS)
CPPFLAGS += -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib -lomp

LIB_NAME = lightlemur
SRC_DIR = backend/src
SRCS = $(SRC_DIR)/tensor.c \
       $(SRC_DIR)/ops.c \
       $(SRC_DIR)/interface.c \
       $(SRC_DIR)/lemurinit.c \
       $(SRC_DIR)/compiler.c \
       $(SRC_DIR)/parameter.c \
       $(SRC_DIR)/kernels/binaryops.c \
       $(SRC_DIR)/kernels/unaryops.c \
       $(SRC_DIR)/kernels/reduceops.c \
       $(SRC_DIR)/kernels/shapeops.c \
       $(SRC_DIR)/kernels/matmulops.c \
       $(SRC_DIR)/kernels/otherops.c \

OBJS     = $(SRCS:.c=.o)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
    TARGET_EXT = so
    LEAK_CHECK = valgrind --leak-check=full --show-leak-kinds=all
else ifeq ($(UNAME_S), Darwin)
    TARGET_EXT = dylib
    LEAK_CHECK = leaks -atExit --
    CPPFLAGS += -I/opt/homebrew/opt/libomp/include
else ifeq ($(OS), Windows_NT)
    TARGET_EXT = dll
    LEAK_CHECK = echo "Memory leak checking is not supported on Windows."
else
    $(error Unsupported platform)
endif

TARGET = lib$(LIB_NAME).$(TARGET_EXT)
TEST_BIN = tests/tests.out
TEST_SRC = tests/tests.c

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run-tests: $(TARGET) $(TEST_BIN)
	./$(TEST_BIN)

run-tests-leak-check: $(TARGET) $(TEST_BIN)
	$(LEAK_CHECK) ./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CC) -o $@ $< -L$(shell pwd) -l$(LIB_NAME) -lm $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_BIN)

clean-compiled:
	find . -type d -name lemurcompiled -exec rm -rf {} +
