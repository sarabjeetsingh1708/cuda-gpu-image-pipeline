# Build file for the CUDA batch image pipeline.
#
#   make                 # build ./build/gpu_pipeline
#   make ARCH=sm_75      # pick a GPU architecture explicitly
#   make lint            # run cpplint (pip install cpplint)
#   make run             # build, generate data, run everything (run.sh)
#   make clean

NVCC ?= nvcc
# "native" targets the GPU installed in this machine (CUDA 11.6+). If your
# nvcc is older, pass e.g. ARCH=sm_75.
ARCH ?= native

NVCCFLAGS := -std=c++17 -O2 -arch=$(ARCH) -Isrc -Ithird_party \
             -Xcompiler -Wall
BUILD_DIR := build
TARGET := $(BUILD_DIR)/gpu_pipeline

OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/gpu_pipeline.o \
           $(BUILD_DIR)/cpu_reference.o $(BUILD_DIR)/image_io.o \
           $(BUILD_DIR)/logger.o

.PHONY: all clean lint run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cc | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

lint:
	cpplint --repository=. --extensions=cc,h,cu \
	  --filter=-legal/copyright,-build/include_subdir,-build/c++17 \
	  --linelength=80 src/*

run:
	./run.sh

clean:
	rm -rf $(BUILD_DIR)
