// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Read-only POSIX mmap of a checkpoint (MAP_PRIVATE, MADV_SEQUENTIAL).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace bonsai {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);  // throws std::runtime_error
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

std::string read_text_file(const std::string& path);

}  // namespace bonsai
