// Image loading, saving and directory listing helpers (backed by stb).
#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_pipeline {

// An 8-bit, single-channel, row-major image.
struct Image {
  std::string name;  // File stem; used to name output files.
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

// Returns image files (png/jpg/jpeg/bmp/pgm/tga) in `directory`, sorted by
// name. If `max_images` > 0, at most that many paths are returned.
std::vector<std::string> ListImageFiles(const std::string& directory,
                                        int max_images);

// Loads `path` and converts it to grayscale. Returns false on failure.
bool LoadGrayscale(const std::string& path, Image* image);

// Writes `image` as a PNG. Returns false on failure.
bool SaveGrayscalePng(const std::string& path, const Image& image);

}  // namespace gpu_pipeline

#endif  // SRC_IMAGE_IO_H_
