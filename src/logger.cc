#include "logger.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace gpu_pipeline {
namespace {

FILE* log_file = nullptr;

}  // namespace

bool InitLog(const std::string& path) {
  if (path.empty()) return true;
  log_file = std::fopen(path.c_str(), "w");
  return log_file != nullptr;
}

void CloseLog() {
  if (log_file != nullptr) {
    std::fclose(log_file);
    log_file = nullptr;
  }
}

void Log(const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  std::vprintf(format, args);
  std::fflush(stdout);
  if (log_file != nullptr) {
    std::vfprintf(log_file, format, args_copy);
    std::fflush(log_file);
  }
  va_end(args_copy);
  va_end(args);
}

}  // namespace gpu_pipeline
