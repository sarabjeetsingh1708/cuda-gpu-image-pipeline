#include "gpu_pipeline.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GP_CUDA_CHECK(call)                                              \
  do {                                                                   \
    cudaError_t gp_status = (call);                                      \
    if (gp_status != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                   cudaGetErrorName(gp_status), __FILE__, __LINE__,      \
                   cudaGetErrorString(gp_status));                       \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

namespace gpu_pipeline {
namespace {

constexpr int kHistogramBins = 256;
constexpr int kHistogramThreads = 256;
constexpr int kMaxHistogramBlocks = 128;
constexpr int kGaussianRadius = 2;
constexpr int kSobelRadius = 1;
constexpr int kNumEvents = 8;

__constant__ int kGaussianWeights[5] = {1, 4, 6, 4, 1};

__device__ __forceinline__ int ClampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// Cooperatively loads the (blockDim + 2 * kRadius)^2 tile of `src` that this
// block needs into shared memory. Out-of-range reads clamp to the border.
template <int kRadius>
__device__ void LoadTile(const uint8_t* src, int width, int height,
                         uint8_t* tile) {
  const int tile_width = blockDim.x + 2 * kRadius;
  const int tile_height = blockDim.y + 2 * kRadius;
  const int origin_x = blockIdx.x * blockDim.x - kRadius;
  const int origin_y = blockIdx.y * blockDim.y - kRadius;
  for (int ty = threadIdx.y; ty < tile_height; ty += blockDim.y) {
    const int sy = ClampInt(origin_y + ty, 0, height - 1);
    for (int tx = threadIdx.x; tx < tile_width; tx += blockDim.x) {
      const int sx = ClampInt(origin_x + tx, 0, width - 1);
      tile[ty * tile_width + tx] = src[sy * width + sx];
    }
  }
}

// Shared-memory histogram: each block builds a private histogram with cheap
// shared-memory atomics, then merges it into the global one.
__global__ void HistogramSharedKernel(const uint8_t* src, int num_pixels,
                                      unsigned int* histogram) {
  __shared__ unsigned int local[kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) local[i] = 0;
  __syncthreads();
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += stride) {
    atomicAdd(&local[src[i]], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    if (local[i] > 0) atomicAdd(&histogram[i], local[i]);
  }
}

// Naive histogram: every pixel contends on 256 global counters.
__global__ void HistogramGlobalKernel(const uint8_t* src, int num_pixels,
                                      unsigned int* histogram) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += stride) {
    atomicAdd(&histogram[src[i]], 1u);
  }
}

// Single-block kernel (256 threads): inclusive scan of the histogram and
// construction of the equalization lookup table. Integer math only, so the
// result is bit-exact with the CPU reference.
__global__ void BuildEqualizationLutKernel(const unsigned int* histogram,
                                           int num_pixels, uint8_t* lut) {
  __shared__ unsigned int cdf[kHistogramBins];
  __shared__ unsigned int cdf_min;
  const int t = threadIdx.x;
  const unsigned int count = histogram[t];
  cdf[t] = count;
  if (t == 0) cdf_min = 0xFFFFFFFFu;
  __syncthreads();
  for (int offset = 1; offset < kHistogramBins; offset <<= 1) {
    const unsigned int addend = (t >= offset) ? cdf[t - offset] : 0u;
    __syncthreads();
    cdf[t] += addend;
    __syncthreads();
  }
  if (count > 0) atomicMin(&cdf_min, cdf[t]);
  __syncthreads();
  const uint64_t denominator = static_cast<uint64_t>(num_pixels) - cdf_min;
  if (denominator == 0) {
    lut[t] = static_cast<uint8_t>(t);
  } else if (cdf[t] <= cdf_min) {
    lut[t] = 0;
  } else {
    lut[t] = static_cast<uint8_t>(
        ((static_cast<uint64_t>(cdf[t]) - cdf_min) * 255 + denominator / 2) /
        denominator);
  }
}

__global__ void ApplyLutKernel(const uint8_t* src, uint8_t* dst,
                               int num_pixels, const uint8_t* lut) {
  __shared__ uint8_t local_lut[kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    local_lut[i] = lut[i];
  }
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pixels) dst[i] = local_lut[src[i]];
}

// 5x5 Gaussian blur (binomial weights, sum 256) using a shared-memory tile.
__global__ void GaussianBlurKernel(const uint8_t* src, uint8_t* dst,
                                   int width, int height) {
  extern __shared__ uint8_t blur_tile[];
  LoadTile<kGaussianRadius>(src, width, height, blur_tile);
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int tile_width = blockDim.x + 2 * kGaussianRadius;
  int sum = 0;
  for (int dy = -kGaussianRadius; dy <= kGaussianRadius; ++dy) {
    const int row = (threadIdx.y + kGaussianRadius + dy) * tile_width;
    for (int dx = -kGaussianRadius; dx <= kGaussianRadius; ++dx) {
      sum += kGaussianWeights[dy + kGaussianRadius] *
             kGaussianWeights[dx + kGaussianRadius] *
             blur_tile[row + threadIdx.x + kGaussianRadius + dx];
    }
  }
  dst[y * width + x] = static_cast<uint8_t>((sum + 128) >> 8);
}

// Sobel gradient magnitude (|gx| + |gy|, clamped to 255) with a
// shared-memory tile.
__global__ void SobelKernel(const uint8_t* src, uint8_t* dst, int width,
                            int height) {
  extern __shared__ uint8_t sobel_tile[];
  LoadTile<kSobelRadius>(src, width, height, sobel_tile);
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int tile_width = blockDim.x + 2 * kSobelRadius;
  const uint8_t* center =
      &sobel_tile[(threadIdx.y + kSobelRadius) * tile_width + threadIdx.x +
                  kSobelRadius];
  const int a = center[-tile_width - 1];
  const int b = center[-tile_width];
  const int c = center[-tile_width + 1];
  const int d = center[-1];
  const int f = center[1];
  const int g = center[tile_width - 1];
  const int h = center[tile_width];
  const int i = center[tile_width + 1];
  const int gx = -a + c - 2 * d + 2 * f - g + i;
  const int gy = -a - 2 * b - c + g + 2 * h + i;
  const int magnitude = abs(gx) + abs(gy);
  dst[y * width + x] = static_cast<uint8_t>(magnitude > 255 ? 255 : magnitude);
}

void Mark(cudaEvent_t* events, int index, cudaStream_t stream) {
  if (events != nullptr) GP_CUDA_CHECK(cudaEventRecord(events[index], stream));
}

}  // namespace

std::string DescribeDevice() {
  int device = 0;
  GP_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  GP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int runtime_version = 0;
  int driver_version = 0;
  GP_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  GP_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer),
                "%s | compute capability %d.%d | %d SMs | %.1f GiB | "
                "CUDA runtime %d.%d, driver %d.%d",
                properties.name, properties.major, properties.minor,
                properties.multiProcessorCount,
                static_cast<double>(properties.totalGlobalMem) /
                    (1024.0 * 1024.0 * 1024.0),
                runtime_version / 1000, (runtime_version % 1000) / 10,
                driver_version / 1000, (driver_version % 1000) / 10);
  return std::string(buffer);
}

GpuPipeline::GpuPipeline(int num_streams, int block_size,
                         HistogramMode histogram_mode)
    : block_size_(block_size),
      histogram_mode_(histogram_mode),
      contexts_(num_streams) {
  GP_CUDA_CHECK(cudaFree(nullptr));  // Forces CUDA context creation.
  for (StreamContext& context : contexts_) {
    GP_CUDA_CHECK(cudaStreamCreate(&context.stream));
  }
}

GpuPipeline::~GpuPipeline() {
  FreeBuffers();
  for (StreamContext& context : contexts_) {
    if (context.stream != nullptr) cudaStreamDestroy(context.stream);
  }
}

void GpuPipeline::AllocateBuffers(size_t max_bytes) {
  if (max_bytes <= capacity_bytes_) return;
  FreeBuffers();
  for (StreamContext& c : contexts_) {
    GP_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&c.host_input),
                                 max_bytes));
    GP_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&c.host_edges),
                                 max_bytes));
    GP_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&c.host_equalized),
                                 max_bytes));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_input),
                             max_bytes));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_equalized),
                             max_bytes));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_blurred),
                             max_bytes));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_edges),
                             max_bytes));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_lut),
                             kHistogramBins));
    GP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.device_histogram),
                             kHistogramBins * sizeof(unsigned int)));
  }
  capacity_bytes_ = max_bytes;
}

void GpuPipeline::FreeBuffers() {
  if (capacity_bytes_ == 0) return;
  GP_CUDA_CHECK(cudaDeviceSynchronize());
  for (StreamContext& c : contexts_) {
    cudaFreeHost(c.host_input);
    cudaFreeHost(c.host_edges);
    cudaFreeHost(c.host_equalized);
    cudaFree(c.device_input);
    cudaFree(c.device_equalized);
    cudaFree(c.device_blurred);
    cudaFree(c.device_edges);
    cudaFree(c.device_lut);
    cudaFree(c.device_histogram);
    c.host_input = c.host_edges = c.host_equalized = nullptr;
    c.device_input = c.device_equalized = nullptr;
    c.device_blurred = c.device_edges = c.device_lut = nullptr;
    c.device_histogram = nullptr;
    c.pending_index = -1;
  }
  capacity_bytes_ = 0;
}

void GpuPipeline::Enqueue(StreamContext* context, int width, int height,
                          bool copy_equalized, cudaEvent_t* events) {
  const size_t bytes = static_cast<size_t>(width) * height;
  const int num_pixels = static_cast<int>(bytes);
  cudaStream_t stream = context->stream;
  const dim3 block(block_size_, block_size_);
  const dim3 grid((width + block_size_ - 1) / block_size_,
                  (height + block_size_ - 1) / block_size_);

  Mark(events, 0, stream);
  GP_CUDA_CHECK(cudaMemcpyAsync(context->device_input, context->host_input,
                                bytes, cudaMemcpyHostToDevice, stream));
  Mark(events, 1, stream);

  GP_CUDA_CHECK(cudaMemsetAsync(context->device_histogram, 0,
                                kHistogramBins * sizeof(unsigned int),
                                stream));
  const int histogram_blocks = std::max(
      1, std::min(kMaxHistogramBlocks,
                  (num_pixels + kHistogramThreads - 1) / kHistogramThreads));
  if (histogram_mode_ == HistogramMode::kShared) {
    HistogramSharedKernel<<<histogram_blocks, kHistogramThreads, 0, stream>>>(
        context->device_input, num_pixels, context->device_histogram);
  } else {
    HistogramGlobalKernel<<<histogram_blocks, kHistogramThreads, 0, stream>>>(
        context->device_input, num_pixels, context->device_histogram);
  }
  GP_CUDA_CHECK(cudaGetLastError());
  Mark(events, 2, stream);

  BuildEqualizationLutKernel<<<1, kHistogramBins, 0, stream>>>(
      context->device_histogram, num_pixels, context->device_lut);
  GP_CUDA_CHECK(cudaGetLastError());
  Mark(events, 3, stream);

  const int lut_blocks = (num_pixels + 255) / 256;
  ApplyLutKernel<<<lut_blocks, 256, 0, stream>>>(
      context->device_input, context->device_equalized, num_pixels,
      context->device_lut);
  GP_CUDA_CHECK(cudaGetLastError());
  Mark(events, 4, stream);

  const size_t blur_shared =
      static_cast<size_t>(block_size_ + 2 * kGaussianRadius) *
      (block_size_ + 2 * kGaussianRadius);
  GaussianBlurKernel<<<grid, block, blur_shared, stream>>>(
      context->device_equalized, context->device_blurred, width, height);
  GP_CUDA_CHECK(cudaGetLastError());
  Mark(events, 5, stream);

  const size_t sobel_shared =
      static_cast<size_t>(block_size_ + 2 * kSobelRadius) *
      (block_size_ + 2 * kSobelRadius);
  SobelKernel<<<grid, block, sobel_shared, stream>>>(
      context->device_blurred, context->device_edges, width, height);
  GP_CUDA_CHECK(cudaGetLastError());
  Mark(events, 6, stream);

  GP_CUDA_CHECK(cudaMemcpyAsync(context->host_edges, context->device_edges,
                                bytes, cudaMemcpyDeviceToHost, stream));
  if (copy_equalized) {
    GP_CUDA_CHECK(cudaMemcpyAsync(context->host_equalized,
                                  context->device_equalized, bytes,
                                  cudaMemcpyDeviceToHost, stream));
  }
  Mark(events, 7, stream);
}

void GpuPipeline::Harvest(StreamContext* context,
                          const std::vector<Image>& images,
                          std::vector<Image>* edges,
                          std::vector<Image>* equalized) {
  if (context->pending_index < 0) return;
  GP_CUDA_CHECK(cudaStreamSynchronize(context->stream));
  const int index = context->pending_index;
  const Image& source = images[index];
  const size_t bytes = source.pixels.size();

  Image& edge_image = (*edges)[index];
  edge_image.name = source.name;
  edge_image.width = source.width;
  edge_image.height = source.height;
  edge_image.pixels.assign(context->host_edges, context->host_edges + bytes);

  if (context->pending_copy_equalized) {
    Image& eq_image = (*equalized)[index];
    eq_image.name = source.name;
    eq_image.width = source.width;
    eq_image.height = source.height;
    eq_image.pixels.assign(context->host_equalized,
                           context->host_equalized + bytes);
  }
  context->pending_index = -1;
}

bool GpuPipeline::Run(const std::vector<Image>& images, int keep_equalized,
                      std::vector<Image>* edges,
                      std::vector<Image>* equalized, double* elapsed_ms) {
  if (images.empty()) return false;
  size_t max_bytes = 0;
  for (const Image& image : images) {
    max_bytes = std::max(max_bytes, image.pixels.size());
  }
  AllocateBuffers(max_bytes);
  edges->assign(images.size(), Image());
  equalized->assign(images.size(), Image());

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < images.size(); ++i) {
    StreamContext* context = &contexts_[i % contexts_.size()];
    // Reusing a stream context: finish and collect its previous image first.
    Harvest(context, images, edges, equalized);
    const Image& image = images[i];
    std::memcpy(context->host_input, image.pixels.data(),
                image.pixels.size());
    const bool copy_equalized = static_cast<int>(i) < keep_equalized;
    Enqueue(context, image.width, image.height, copy_equalized, nullptr);
    context->pending_index = static_cast<int>(i);
    context->pending_copy_equalized = copy_equalized;
  }
  for (StreamContext& context : contexts_) {
    Harvest(&context, images, edges, equalized);
  }
  const auto end = std::chrono::steady_clock::now();
  *elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return true;
}

bool GpuPipeline::ProfileStages(const Image& image, int repetitions,
                                StageTimes* times) {
  if (repetitions <= 0 || image.pixels.empty()) return false;
  AllocateBuffers(image.pixels.size());
  StreamContext* context = &contexts_[0];
  std::memcpy(context->host_input, image.pixels.data(), image.pixels.size());

  cudaEvent_t events[kNumEvents];
  for (cudaEvent_t& event : events) GP_CUDA_CHECK(cudaEventCreate(&event));

  float sums[kNumEvents - 1] = {0};
  float total = 0.0f;
  for (int rep = -1; rep < repetitions; ++rep) {  // rep == -1 is a warm-up.
    Enqueue(context, image.width, image.height, /*copy_equalized=*/true,
            events);
    GP_CUDA_CHECK(cudaEventSynchronize(events[kNumEvents - 1]));
    if (rep < 0) continue;
    for (int e = 0; e + 1 < kNumEvents; ++e) {
      float ms = 0.0f;
      GP_CUDA_CHECK(cudaEventElapsedTime(&ms, events[e], events[e + 1]));
      sums[e] += ms;
    }
    float ms = 0.0f;
    GP_CUDA_CHECK(cudaEventElapsedTime(&ms, events[0], events[kNumEvents - 1]));
    total += ms;
  }
  for (cudaEvent_t& event : events) cudaEventDestroy(event);

  const float n = static_cast<float>(repetitions);
  times->h2d_ms = sums[0] / n;
  times->histogram_ms = sums[1] / n;
  times->lut_ms = sums[2] / n;
  times->equalize_ms = sums[3] / n;
  times->blur_ms = sums[4] / n;
  times->sobel_ms = sums[5] / n;
  times->d2h_ms = sums[6] / n;
  times->total_ms = total / n;
  return true;
}

}  // namespace gpu_pipeline
