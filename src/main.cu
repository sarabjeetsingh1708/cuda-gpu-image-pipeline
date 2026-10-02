// Command-line driver for the batch GPU image pipeline.
//
// Example:
//   ./build/gpu_pipeline --input_dir data --output_dir out --num_images 1000
#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cpu_reference.h"
#include "gpu_pipeline.h"
#include "image_io.h"
#include "logger.h"

namespace gpu_pipeline {
namespace {

struct Options {
  std::string input_dir;
  std::string output_dir = "out";
  std::string log_file;
  int num_images = 0;  // 0 means "all images in input_dir".
  int block_size = 16;
  int num_streams = 4;
  int save_limit = 20;  // Number of images whose outputs are written; -1: all.
  int profile_reps = 20;
  HistogramMode histogram_mode = HistogramMode::kShared;
  bool cpu_baseline = false;
};

void PrintUsage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s --input_dir DIR [options]\n"
      "  --input_dir DIR          Directory with png/jpg/bmp/pgm/tga.\n"
      "  --output_dir DIR         Where results are written (default: out).\n"
      "  --num_images N           Process at most N images (default: all).\n"
      "  --block_size {8,16,32}   Thread-block edge length (default: 16).\n"
      "  --num_streams N          CUDA streams used for overlap (default: 4).\n"
      "  --histogram_mode MODE    'shared' or 'global' (default: shared).\n"
      "  --save_limit N           Save outputs of the first N images; -1\n"
      "                           saves all, 0 none (default: 20).\n"
      "  --profile_reps N         Repetitions for the per-stage timing pass;\n"
      "                           0 disables it (default: 20).\n"
      "  --cpu_baseline           Also run the CPU pipeline, compare results\n"
      "                           and report the speedup.\n"
      "  --log_file FILE          Also write the log to FILE.\n"
      "  --help                   Show this message.\n",
      program);
}

bool ParseInt(const std::string& flag, const std::string& text, int* value) {
  char* end = nullptr;
  const int64_t parsed = std::strtoll(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    std::fprintf(stderr, "Invalid integer for %s: '%s'\n", flag.c_str(),
                 text.c_str());
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") return false;
    if (flag == "--cpu_baseline") {
      options->cpu_baseline = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value for %s\n", flag.c_str());
      return false;
    }
    const std::string value = argv[++i];
    bool ok = true;
    if (flag == "--input_dir") {
      options->input_dir = value;
    } else if (flag == "--output_dir") {
      options->output_dir = value;
    } else if (flag == "--log_file") {
      options->log_file = value;
    } else if (flag == "--num_images") {
      ok = ParseInt(flag, value, &options->num_images);
    } else if (flag == "--block_size") {
      ok = ParseInt(flag, value, &options->block_size);
    } else if (flag == "--num_streams") {
      ok = ParseInt(flag, value, &options->num_streams);
    } else if (flag == "--save_limit") {
      ok = ParseInt(flag, value, &options->save_limit);
    } else if (flag == "--profile_reps") {
      ok = ParseInt(flag, value, &options->profile_reps);
    } else if (flag == "--histogram_mode") {
      if (value == "shared") {
        options->histogram_mode = HistogramMode::kShared;
      } else if (value == "global") {
        options->histogram_mode = HistogramMode::kGlobal;
      } else {
        std::fprintf(stderr, "--histogram_mode must be shared or global\n");
        ok = false;
      }
    } else {
      std::fprintf(stderr, "Unknown flag: %s\n", flag.c_str());
      ok = false;
    }
    if (!ok) return false;
  }
  if (options->input_dir.empty()) {
    std::fprintf(stderr, "--input_dir is required\n");
    return false;
  }
  if (options->block_size != 8 && options->block_size != 16 &&
      options->block_size != 32) {
    std::fprintf(stderr, "--block_size must be 8, 16 or 32\n");
    return false;
  }
  if (options->num_streams < 1 || options->num_streams > 32) {
    std::fprintf(stderr, "--num_streams must be in [1, 32]\n");
    return false;
  }
  return true;
}

size_t CountMismatches(const std::vector<uint8_t>& a,
                       const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  size_t mismatches = 0;
  for (size_t i = 0; i < a.size(); ++i) mismatches += (a[i] != b[i]);
  return mismatches;
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int Main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (!InitLog(options.log_file)) {
    std::fprintf(stderr, "Cannot open log file %s\n",
                 options.log_file.c_str());
    return 1;
  }

  Log("=== GPU batch image pipeline: equalize -> Gaussian 5x5 -> Sobel ===\n");
  Log("[config] input_dir=%s output_dir=%s num_images=%d block_size=%d "
      "num_streams=%d histogram=%s save_limit=%d cpu_baseline=%s\n",
      options.input_dir.c_str(), options.output_dir.c_str(),
      options.num_images, options.block_size, options.num_streams,
      options.histogram_mode == HistogramMode::kShared ? "shared" : "global",
      options.save_limit, options.cpu_baseline ? "yes" : "no");
  Log("[device] %s\n", DescribeDevice().c_str());

  // ---- Load data -------------------------------------------------------
  const std::vector<std::string> paths =
      ListImageFiles(options.input_dir, options.num_images);
  if (paths.empty()) {
    Log("No images found in %s\n", options.input_dir.c_str());
    return 1;
  }
  const auto load_start = std::chrono::steady_clock::now();
  std::vector<Image> images;
  images.reserve(paths.size());
  double total_megapixels = 0.0;
  for (const std::string& path : paths) {
    Image image;
    if (!LoadGrayscale(path, &image)) {
      Log("Skipping unreadable image: %s\n", path.c_str());
      continue;
    }
    total_megapixels += static_cast<double>(image.pixels.size()) / 1e6;
    images.push_back(std::move(image));
  }
  if (images.empty()) return 1;
  Log("[data] loaded %zu images (%.1f MPixels, first: %s %dx%d) in %.1f ms\n",
      images.size(), total_megapixels, images[0].name.c_str(),
      images[0].width, images[0].height, ElapsedMs(load_start));

  GpuPipeline pipeline(options.num_streams, options.block_size,
                       options.histogram_mode);

  // ---- Per-stage timing on one image (also warms up the GPU) -----------
  if (options.profile_reps > 0) {
    StageTimes t;
    pipeline.ProfileStages(images[0], options.profile_reps, &t);
    Log("[stages] mean over %d reps on %s (%dx%d), milliseconds per image:\n",
        options.profile_reps, images[0].name.c_str(), images[0].width,
        images[0].height);
    Log("[stages]   H2D copy      %8.4f\n", t.h2d_ms);
    Log("[stages]   histogram     %8.4f\n", t.histogram_ms);
    Log("[stages]   build LUT     %8.4f\n", t.lut_ms);
    Log("[stages]   apply LUT     %8.4f\n", t.equalize_ms);
    Log("[stages]   Gaussian blur %8.4f\n", t.blur_ms);
    Log("[stages]   Sobel         %8.4f\n", t.sobel_ms);
    Log("[stages]   D2H copy      %8.4f\n", t.d2h_ms);
    Log("[stages]   total         %8.4f\n", t.total_ms);
  }

  // ---- Batch run on all images in a single execution -------------------
  const int num_images = static_cast<int>(images.size());
  const int keep_equalized =
      options.save_limit < 0 ? num_images
                             : std::min(options.save_limit, num_images);
  std::vector<Image> edges;
  std::vector<Image> equalized;
  double gpu_ms = 0.0;
  pipeline.Run(images, keep_equalized, &edges, &equalized, &gpu_ms);
  Log("[gpu] processed %d images (%.1f MPixels) in a single run: %.2f ms "
      "total, %.1f images/s, %.1f MPixels/s, %.4f ms/image\n",
      num_images, total_megapixels, gpu_ms, num_images / (gpu_ms / 1e3),
      total_megapixels / (gpu_ms / 1e3), gpu_ms / num_images);

  // ---- Optional CPU baseline + verification ----------------------------
  if (options.cpu_baseline) {
    size_t edge_mismatches = 0;
    size_t equalized_mismatches = 0;
    double cpu_ms = 0.0;
    for (int i = 0; i < num_images; ++i) {
      Image cpu_equalized;
      Image cpu_edges;
      const auto t0 = std::chrono::steady_clock::now();
      RunCpuPipeline(images[i], &cpu_equalized, &cpu_edges);
      cpu_ms += ElapsedMs(t0);
      edge_mismatches += CountMismatches(cpu_edges.pixels, edges[i].pixels);
      if (i < keep_equalized) {
        equalized_mismatches +=
            CountMismatches(cpu_equalized.pixels, equalized[i].pixels);
      }
    }
    Log("[cpu] processed %d images single-threaded: %.2f ms total, "
        "%.4f ms/image\n",
        num_images, cpu_ms, cpu_ms / num_images);
    Log("[speedup] GPU is %.1fx faster than the single-thread CPU baseline "
        "(GPU time includes H2D/D2H copies)\n",
        cpu_ms / gpu_ms);
    Log("[verify] mismatching pixels vs CPU reference: edges=%zu "
        "equalized(first %d images)=%zu -> %s\n",
        edge_mismatches, keep_equalized, equalized_mismatches,
        (edge_mismatches == 0 && equalized_mismatches == 0) ? "PASS"
                                                            : "FAIL");
  }

  // ---- Save outputs ------------------------------------------------------
  if (keep_equalized > 0) {
    std::error_code error;
    std::filesystem::create_directories(options.output_dir, error);
    int saved = 0;
    for (int i = 0; i < keep_equalized; ++i) {
      const std::string base = options.output_dir + "/" + images[i].name;
      const bool ok = SaveGrayscalePng(base + "_input.png", images[i]) &&
                      SaveGrayscalePng(base + "_equalized.png", equalized[i]) &&
                      SaveGrayscalePng(base + "_edges.png", edges[i]);
      if (!ok) {
        Log("Failed to write outputs for %s\n", images[i].name.c_str());
        continue;
      }
      ++saved;
    }
    Log("[output] wrote input/equalized/edges PNGs for %d images to %s\n",
        saved, options.output_dir.c_str());
  }
  Log("[done]\n");
  CloseLog();
  return 0;
}

}  // namespace gpu_pipeline

int main(int argc, char** argv) { return gpu_pipeline::Main(argc, argv); }
