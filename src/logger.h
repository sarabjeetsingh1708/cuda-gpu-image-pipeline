// Minimal logger that writes to stdout and, optionally, to a log file.
#ifndef SRC_LOGGER_H_
#define SRC_LOGGER_H_

#include <string>

namespace gpu_pipeline {

// Opens `path` for logging. An empty path disables file logging.
bool InitLog(const std::string& path);

// Closes the log file, if any.
void CloseLog();

// printf-style logging to stdout and the log file.
void Log(const char* format, ...) __attribute__((format(printf, 1, 2)));

}  // namespace gpu_pipeline

#endif  // SRC_LOGGER_H_
