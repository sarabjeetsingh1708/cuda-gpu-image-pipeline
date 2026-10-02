#include "image_io.h"

#include <algorithm>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace gpu_pipeline {

std::vector<std::string> ListImageFiles(const std::string& directory,
                                        int max_images) {
  namespace fs = std::filesystem;
  static const std::set<std::string> kExtensions = {
      ".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tga"};
  std::vector<std::string> paths;
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(directory, error)) {
    if (!entry.is_regular_file()) continue;
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (kExtensions.count(extension) > 0) {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (max_images > 0 && static_cast<int>(paths.size()) > max_images) {
    paths.resize(max_images);
  }
  return paths;
}

bool LoadGrayscale(const std::string& path, Image* image) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &channels, /*desired=*/1);
  if (data == nullptr) return false;
  image->width = width;
  image->height = height;
  image->pixels.assign(data, data + static_cast<size_t>(width) * height);
  image->name = std::filesystem::path(path).stem().string();
  stbi_image_free(data);
  return true;
}

bool SaveGrayscalePng(const std::string& path, const Image& image) {
  return stbi_write_png(path.c_str(), image.width, image.height, 1,
                        image.pixels.data(), image.width) != 0;
}

}  // namespace gpu_pipeline
