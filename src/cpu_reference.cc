#include "cpu_reference.h"

#include <algorithm>
#include <cstdlib>

namespace gpu_pipeline {
namespace {

constexpr int kHistogramBins = 256;
constexpr int kGaussianRadius = 2;
constexpr int kGaussianWeights[5] = {1, 4, 6, 4, 1};

int Clamp(int value, int low, int high) {
  return std::min(std::max(value, low), high);
}

void EqualizeHistogram(const Image& input, Image* output) {
  const uint64_t num_pixels = input.pixels.size();
  uint64_t histogram[kHistogramBins] = {0};
  for (uint8_t pixel : input.pixels) ++histogram[pixel];

  uint64_t cdf[kHistogramBins];
  uint64_t running = 0;
  uint64_t cdf_min = 0;
  bool found_min = false;
  for (int i = 0; i < kHistogramBins; ++i) {
    running += histogram[i];
    cdf[i] = running;
    if (!found_min && histogram[i] > 0) {
      cdf_min = running;
      found_min = true;
    }
  }

  uint8_t lut[kHistogramBins];
  const uint64_t denominator = num_pixels - cdf_min;
  for (int i = 0; i < kHistogramBins; ++i) {
    if (denominator == 0) {
      lut[i] = static_cast<uint8_t>(i);
    } else if (cdf[i] <= cdf_min) {
      lut[i] = 0;
    } else {
      lut[i] = static_cast<uint8_t>(
          ((cdf[i] - cdf_min) * 255 + denominator / 2) / denominator);
    }
  }

  output->width = input.width;
  output->height = input.height;
  output->name = input.name;
  output->pixels.resize(input.pixels.size());
  for (size_t i = 0; i < input.pixels.size(); ++i) {
    output->pixels[i] = lut[input.pixels[i]];
  }
}

void GaussianBlur(const Image& input, Image* output) {
  const int width = input.width;
  const int height = input.height;
  output->width = width;
  output->height = height;
  output->name = input.name;
  output->pixels.resize(input.pixels.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sum = 0;
      for (int dy = -kGaussianRadius; dy <= kGaussianRadius; ++dy) {
        const int sy = Clamp(y + dy, 0, height - 1);
        for (int dx = -kGaussianRadius; dx <= kGaussianRadius; ++dx) {
          const int sx = Clamp(x + dx, 0, width - 1);
          sum += kGaussianWeights[dy + kGaussianRadius] *
                 kGaussianWeights[dx + kGaussianRadius] *
                 input.pixels[static_cast<size_t>(sy) * width + sx];
        }
      }
      output->pixels[static_cast<size_t>(y) * width + x] =
          static_cast<uint8_t>((sum + 128) >> 8);
    }
  }
}

void Sobel(const Image& input, Image* output) {
  const int width = input.width;
  const int height = input.height;
  output->width = width;
  output->height = height;
  output->name = input.name;
  output->pixels.resize(input.pixels.size());
  auto at = [&](int x, int y) {
    return static_cast<int>(input.pixels[static_cast<size_t>(
                                             Clamp(y, 0, height - 1)) *
                                             width +
                                         Clamp(x, 0, width - 1)]);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int gx = -at(x - 1, y - 1) + at(x + 1, y - 1) -
                     2 * at(x - 1, y) + 2 * at(x + 1, y) -
                     at(x - 1, y + 1) + at(x + 1, y + 1);
      const int gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) -
                     at(x + 1, y - 1) + at(x - 1, y + 1) +
                     2 * at(x, y + 1) + at(x + 1, y + 1);
      output->pixels[static_cast<size_t>(y) * width + x] =
          static_cast<uint8_t>(std::min(255, std::abs(gx) + std::abs(gy)));
    }
  }
}

}  // namespace

void RunCpuPipeline(const Image& input, Image* equalized, Image* edges) {
  Image blurred;
  EqualizeHistogram(input, equalized);
  GaussianBlur(*equalized, &blurred);
  Sobel(blurred, edges);
}

}  // namespace gpu_pipeline
