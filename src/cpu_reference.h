// Single-threaded CPU implementation of the pipeline. It is bit-exact with
// the GPU version and serves as both a correctness check and a timing
// baseline.
#ifndef SRC_CPU_REFERENCE_H_
#define SRC_CPU_REFERENCE_H_

#include "image_io.h"

namespace gpu_pipeline {

// Runs histogram equalization -> 5x5 Gaussian blur -> Sobel on `input`.
// `equalized` and `edges` receive the intermediate and final images.
void RunCpuPipeline(const Image& input, Image* equalized, Image* edges);

}  // namespace gpu_pipeline

#endif  // SRC_CPU_REFERENCE_H_
