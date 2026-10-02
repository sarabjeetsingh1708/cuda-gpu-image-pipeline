// GPU implementation of the batch image pipeline:
//   histogram equalization -> 5x5 Gaussian blur -> Sobel edge detection.
#ifndef SRC_GPU_PIPELINE_H_
#define SRC_GPU_PIPELINE_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "image_io.h"

namespace gpu_pipeline {

// How the 256-bin histogram is accumulated on the GPU.
enum class HistogramMode {
  kShared,  // Per-block shared-memory histograms merged into global memory.
  kGlobal,  // Every pixel does an atomicAdd directly on global memory.
};

// Average per-image stage durations in milliseconds (measured with events).
struct StageTimes {
  float h2d_ms = 0.0f;
  float histogram_ms = 0.0f;
  float lut_ms = 0.0f;
  float equalize_ms = 0.0f;
  float blur_ms = 0.0f;
  float sobel_ms = 0.0f;
  float d2h_ms = 0.0f;
  float total_ms = 0.0f;
};

// Returns a human-readable description of the active CUDA device.
std::string DescribeDevice();

// Processes many images using several CUDA streams with pinned host memory so
// that copies and kernels of different images overlap.
class GpuPipeline {
 public:
  GpuPipeline(int num_streams, int block_size, HistogramMode histogram_mode);
  ~GpuPipeline();
  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Processes every image in one call. `edges` receives the final output for
  // every image; `equalized` is filled only for the first `keep_equalized`
  // images. `elapsed_ms` is the wall-clock time including copies.
  bool Run(const std::vector<Image>& images, int keep_equalized,
           std::vector<Image>* edges, std::vector<Image>* equalized,
           double* elapsed_ms);

  // Runs the pipeline `repetitions` times (after one warm-up run) on a single
  // image and reports per-stage averages.
  bool ProfileStages(const Image& image, int repetitions, StageTimes* times);

 private:
  struct StreamContext {
    cudaStream_t stream = nullptr;
    uint8_t* host_input = nullptr;  // Pinned.
    uint8_t* host_edges = nullptr;  // Pinned.
    uint8_t* host_equalized = nullptr;  // Pinned.
    uint8_t* device_input = nullptr;
    uint8_t* device_equalized = nullptr;
    uint8_t* device_blurred = nullptr;
    uint8_t* device_edges = nullptr;
    uint8_t* device_lut = nullptr;
    unsigned int* device_histogram = nullptr;
    int pending_index = -1;
    bool pending_copy_equalized = false;
  };

  void AllocateBuffers(size_t max_bytes);
  void FreeBuffers();
  void Enqueue(StreamContext* context, int width, int height,
               bool copy_equalized, cudaEvent_t* events);
  void Harvest(StreamContext* context, const std::vector<Image>& images,
               std::vector<Image>* edges, std::vector<Image>* equalized);

  int block_size_;
  HistogramMode histogram_mode_;
  size_t capacity_bytes_ = 0;
  std::vector<StreamContext> contexts_;
};

}  // namespace gpu_pipeline

#endif  // SRC_GPU_PIPELINE_H_
