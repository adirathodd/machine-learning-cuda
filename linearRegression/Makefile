# Makefile for CUDA linearRegression project

# Compiler and flags
NVCC       := nvcc
GCC        := gcc
G+CC        := g++
CFLAGS     := -O2 -I./include
SRC_DIR    := src
TARGET     := linearRegression

# Source files
SRCS       := $(SRC_DIR)/main.cu $(SRC_DIR)/lr_kernels.cu $(SRC_DIR)/csv.cu $(SRC_DIR)/linearRegression.cu
OBJS       := $(SRC_DIR)/main.o $(SRC_DIR)/lr_kernels.o $(SRC_DIR)/csv.o $(SRC_DIR)/linearRegression.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(OBJS) -o $(TARGET)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

$(SRC_DIR)/lr_kernels.o: $(SRC_DIR)/lr_kernels.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/lr_kernels.cu -o $(SRC_DIR)/lr_kernels.o

$(SRC_DIR)/csv.o: $(SRC_DIR)/csv.cpp
	$(G+CC) $(CFLAGS) -c $(SRC_DIR)/csv.cpp -o $(SRC_DIR)/csv.o

$(SRC_DIR)/linearRegression.o: $(SRC_DIR)/linearRegression.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/linearRegression.cu -o $(SRC_DIR)/linearRegression.o

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

