# Makefile for CUDA logisticRegression project

# Compiler and flags
NVCC       := nvcc
CFLAGS     := -O2 -I./include
SRC_DIR    := src
TARGET     := logisticRegression

# Source files
SRCS       := $(SRC_DIR)/main.cu $(SRC_DIR)/lr_kernels.cu $(SRC_DIR)/csv.cu $(SRC_DIR)/logisticRegression.cu
OBJS       := $(SRC_DIR)/main.o $(SRC_DIR)/lr_kernels.o $(SRC_DIR)/csv.o $(SRC_DIR)/logisticRegression.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(OBJS) -o $(TARGET)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

$(SRC_DIR)/lr_kernels.o: $(SRC_DIR)/lr_kernels.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/lr_kernels.cu -o $(SRC_DIR)/lr_kernels.o

$(SRC_DIR)/csv.o: $(SRC_DIR)/csv.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/csv.cu -o $(SRC_DIR)/csv.o

$(SRC_DIR)/logisticRegression.o: $(SRC_DIR)/logisticRegression.cu
	$(NVCC) $(CFLAGS) -c $(SRC_DIR)/logisticRegression.cu -o $(SRC_DIR)/logisticRegression.o

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

