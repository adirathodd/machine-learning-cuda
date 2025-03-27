# Makefile for CUDA kNN project

# Compiler and flags
NVCC       := nvcc
CFLAGS     := -O2 -I./include
SRC_DIR    := src
TARGET     := knn

# Source files
SRCS       := $(SRC_DIR)/main.cu $(SRC_DIR)/knn_kernels.cu $(SRC_DIR)/readCSV.c
OBJS       := $(SRC_DIR)/main.o $(SRC_DIR)/knn_kernels.o $(SRC_DIR)/readCSV.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(OBJS) -o $(TARGET)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

$(SRC_DIR)/knn_kernels.o: $(SRC_DIR)/knn_kernels.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/knn_kernels.cu -o $(SRC_DIR)/knn_kernels.o

$(SRC_DIR)/readCSV.o: $(SRC_DIR)/readCSV.c
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/readCSV.c -o $(SRC_DIR)/readCSV.o

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

